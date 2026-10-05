# Multi-label Disentangled VAE with Embedding + Attention Condition Encoder
# Predicts perturbed gene sets (multi-hot), supports multi-gene combinatorial perturbations
import os
import gc
import json
import warnings
import logging
import scanpy as sc
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from scipy.stats import pearsonr
import torch.nn.functional as F
from scvi.distributions import NegativeBinomial
from torch import nn
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, precision_score, recall_score
import scipy.sparse

warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
os.environ["PYTORCH_LIGHTNING_SUPPRESS_WARNINGS"] = "1"


# ==========================================
# 0. Utility functions
# ==========================================
class MultiLabelContrastiveLoss(nn.Module):
    """Multi-label contrastive learning with Jaccard similarity as affinity matrix"""
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, multi_hot_labels):
        features = F.normalize(features, dim=1)
        sim_matrix = torch.matmul(features, features.T) / self.temperature

        # Jaccard similarity
        intersection = torch.matmul(multi_hot_labels, multi_hot_labels.T)
        cardinality = multi_hot_labels.sum(1, keepdim=True)
        union = cardinality + cardinality.T - intersection
        jaccard_mask = intersection / (union + 1e-6)

        # Control pairs (all-zero) serve as positive samples
        is_both_control = (union == 0).float()
        jaccard_mask = jaccard_mask + is_both_control

        # Exclude diagonal
        logits_mask = torch.ones_like(sim_matrix) - torch.eye(sim_matrix.shape[0], device=sim_matrix.device)
        jaccard_mask = jaccard_mask * logits_mask

        sim_matrix = sim_matrix - sim_matrix.max(dim=1, keepdim=True)[0]  # numerical stability
        exp_sim = torch.exp(sim_matrix) * logits_mask
        log_prob = sim_matrix - torch.log(exp_sim.sum(1, keepdim=True) + 1e-6)
        loss = -(jaccard_mask * log_prob).sum(1) / (jaccard_mask.sum(1) + 1e-6)
        return loss.mean()


class MLP(nn.Module):
    def __init__(self, dims, dropout=0.1, use_batch_norm=True):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class GradientReversalFunction(torch.autograd.Function):
    """Gradient reversal: identity forward, negate gradient by -lambda"""
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_=0.2):
        super().__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


# ==========================================
# 1. Condition Encoder: Embedding + Attention
# ==========================================
class AttentionPerturbationEncoder(nn.Module):
    """
    Input: gene_indices [Batch, max_pert_genes] (zero-padded perturbation gene indices)
    Output: 8D condition vector [Batch, 8]
    use_attn=True:  Embedding + Self-Attention + Mean Pooling
    use_attn=False: Embedding + Mean Pooling (no interaction)
    """
    def __init__(self, n_genes, emb_dim=8, n_heads=2, dropout=0.1, use_attn=True):
        super().__init__()
        self.use_attn = use_attn
        self.emb_dim = emb_dim
        self.n_heads = n_heads
        self.head_dim = emb_dim // n_heads
        self.gene_embedding = nn.Embedding(n_genes + 1, emb_dim, padding_idx=0)
        if use_attn:
            self.q_proj = nn.Linear(emb_dim, emb_dim)
            self.k_proj = nn.Linear(emb_dim, emb_dim)
            self.v_proj = nn.Linear(emb_dim, emb_dim)
            self.out_proj = nn.Linear(emb_dim, emb_dim)
        self.fc = nn.Linear(emb_dim, emb_dim)

    def forward(self, gene_indices):
        mask = (gene_indices == 0)  # True = padding [B, K]
        emb = self.gene_embedding(gene_indices)  # [B, K, 8]

        if self.use_attn:
            B, K, D = emb.shape
            # Identify samples with all padding (control cells)
            all_pad = mask.all(dim=1)  # [B]

            Q = self.q_proj(emb).view(B, K, self.n_heads, self.head_dim).transpose(1, 2)
            K_ = self.k_proj(emb).view(B, K, self.n_heads, self.head_dim).transpose(1, 2)
            V = self.v_proj(emb).view(B, K, self.n_heads, self.head_dim).transpose(1, 2)
            # [B, n_heads, K, head_dim]

            scores = torch.matmul(Q, K_.transpose(-2, -1)) / (self.head_dim ** 0.5)
            # [B, n_heads, K, K]

            # mask padding: set key positions to -inf
            key_mask = mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, K]
            scores = scores.masked_fill(key_mask, -1e9)

            attn_weights = F.softmax(scores, dim=-1)  # [B, n_heads, K, K]
            # All-padding samples: softmax(-inf,...,-inf) = nan → replace with 0
            attn_weights = attn_weights.masked_fill(all_pad.unsqueeze(1).unsqueeze(2).unsqueeze(3), 0.0)
            attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

            attn_out = torch.matmul(attn_weights, V)  # [B, n_heads, K, head_dim]
            attn_out = attn_out.transpose(1, 2).contiguous().view(B, K, D)
            attn_out = self.out_proj(attn_out)

            emb = emb + attn_out  # residual connection

        # Mean pooling (ignore padding)
        valid = (~mask).float().unsqueeze(-1)  # [B, K, 1]
        valid_count = valid.sum(1).clamp(min=1.0)
        pooled = (emb * valid).sum(1) / valid_count  # [B, 8]

        return self.fc(pooled)


# ==========================================
# 2. Dataset (multi-label)
# ==========================================
class MacCFVAEDataset(Dataset):
    def __init__(self, adata, sep='+', max_pert_genes=5, target_sum=10000):
        if scipy.sparse.issparse(adata.X):
            self.X = torch.tensor(adata.X.toarray(), dtype=torch.float32)
        else:
            self.X = torch.tensor(adata.X, dtype=torch.float32)

        # normalize library size
        library_size = self.X.sum(1, keepdim=True)
        self.X = self.X / (library_size + 1e-6) * target_sum
        self.X_norm = torch.log1p(self.X)

        # auto-detect perturbation column
        cond_key = 'perturbation' if 'perturbation' in adata.obs else 'condition_name'
        if cond_key not in adata.obs:
            raise ValueError(f"Cannot find perturbation column in {adata.obs.columns}")

        raw_conditions = adata.obs[cond_key].values

        # parse perturbation: A+B → ['a', 'b'], control → []
        def parse_cond(c):
            s = str(c)
            if s.lower() in ['control', 'ctrl', 'unperturbed', 'nan']:
                return []
            return [p.strip().lower() for p in s.split(sep)]

        cond_list = [parse_cond(c) for c in raw_conditions]

        # MultiLabelBinarizer: whether each gene is perturbed
        self.mlb = MultiLabelBinarizer()
        self.multi_hot_matrix = self.mlb.fit_transform(cond_list)
        self.condition_vecs = torch.tensor(self.multi_hot_matrix, dtype=torch.float32)
        self.n_unique_pert = len(self.mlb.classes_)

        # save gene names (for reconstructing gene_to_idx at inference)
        self.gene_names = list(self.mlb.classes_)

        # build gene index sequence (zero-padded to max_pert_genes)
        gene_to_idx = {g: i + 1 for i, g in enumerate(self.mlb.classes_)}  # 0 = padding
        self.gene_indices = np.zeros((len(cond_list), max_pert_genes), dtype=np.int64)
        for i, genes in enumerate(cond_list):
            for j, g in enumerate(genes[:max_pert_genes]):
                self.gene_indices[i, j] = gene_to_idx[g]
        self.gene_indices = torch.tensor(self.gene_indices, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {
            "X": self.X[idx],
            "X_norm": self.X_norm[idx],
            "condition_vec": self.condition_vecs[idx],      # multi-hot [n_unique_pert]
            "gene_indices": self.gene_indices[idx],          # [max_pert_genes]
        }


# ==========================================
# 3. Model (multi-label disentanglement)
# ==========================================
class MacCFVAE(pl.LightningModule):
    def __init__(self, in_dim, n_unique_pert, max_pert_genes=5,
                 encoder_kwargs=None, learning_rate=5e-4,
                 cell_latent_dim=60, batch_latent_dim=8,
                 use_attn=True, use_adversarial=True, target_sum=10000,
                 w_kl_cell_ratio=0.05, w_clf_ratio=0.25,
                 w_supcon_ratio=0.15, w_align_ratio=0.10,
                 w_adv_ratio=0.50):
        super().__init__()
        if encoder_kwargs is None:
            encoder_kwargs = {"dims": [512, 128], "dropout": 0.1}

        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.cell_dim = cell_latent_dim
        self.batch_dim = batch_latent_dim
        self.total_dim = cell_latent_dim + batch_latent_dim
        self.target_sum = target_sum
        self.n_unique_pert = n_unique_pert
        self.w_kl_cell_ratio = w_kl_cell_ratio
        self.w_clf_ratio = w_clf_ratio
        self.w_supcon_ratio = w_supcon_ratio
        self.w_align_ratio = w_align_ratio
        self.w_adv_ratio = w_adv_ratio
        self.use_adversarial = use_adversarial

        # RNA encoder
        enc_dims = [in_dim, *encoder_kwargs["dims"]]
        self.rna_encoder_body = MLP(enc_dims, dropout=encoder_kwargs.get("dropout", 0.1))
        self.rna_z_mean = nn.Linear(enc_dims[-1], self.total_dim)
        self.rna_z_log_var = nn.Linear(enc_dims[-1], self.total_dim)

        # Condition encoder: Embedding + [Attention | Sum] → 8D
        self.cond_encoder = AttentionPerturbationEncoder(
            n_genes=n_unique_pert, emb_dim=batch_latent_dim, n_heads=2,
            use_attn=use_attn
        )

        # Decoder: split into two independent pathways
        # 1) cell_decoder: 60D → gene space (baseline expression, no perturbation)
        cell_dec_dims = [self.cell_dim] + enc_dims[1:][::-1]
        self.cell_decoder = MLP(cell_dec_dims, dropout=encoder_kwargs.get("dropout", 0.1))
        self.cell_output = nn.Linear(cell_dec_dims[-1], in_dim)

        # 2) pert_decoder: 8D → gene space (perturbation effect Δ, sole source)
        self.pert_decoder = nn.Sequential(
            nn.Linear(self.batch_dim, 64),
            nn.ReLU(),
            nn.Linear(64, in_dim)
        )
        # Zero initialization: no perturbation effect initially, learns from baseline
        nn.init.normal_(self.pert_decoder[-1].weight, std=0.01)
        nn.init.normal_(self.pert_decoder[-1].bias, std=0.01)

        self.theta = nn.Parameter(torch.ones(in_dim), requires_grad=True)

        # Multi-label classifier: 8D → n_unique_pert (independent sigmoid)
        self.condition_classifier = nn.Sequential(
            nn.Linear(self.batch_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.n_unique_pert)
        )

        # Adversarial classifier: 60D → n_unique_pert (gradient reversal, forces 60D to exclude perturbation info)
        if self.use_adversarial:
            self.grl = GradientReversalLayer(lambda_=0.2)
            self.adv_classifier = nn.Sequential(
                nn.Linear(self.cell_dim, 64),
                nn.ReLU(),
                nn.Linear(64, self.n_unique_pert)
            )

        # BCE with pos_weight for class imbalance (most genes are 0)
        self.register_buffer('pos_weight', torch.ones(self.n_unique_pert))
        self.supcon_loss_fn = MultiLabelContrastiveLoss(temperature=0.1)


    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, batch):
        X, X_norm, cond_vec, gene_idx = (
            batch["X"], batch["X_norm"], batch["condition_vec"], batch["gene_indices"]
        )
        size_factor = self.target_sum

        # 1. RNA encode → 60D cell + 8D condition
        h_rna = self.rna_encoder_body(X_norm)
        mu_rna = self.rna_z_mean(h_rna)
        logvar_rna = self.rna_z_log_var(h_rna)

        mu_rna_cell = mu_rna[:, :self.cell_dim]
        logvar_rna_cell = logvar_rna[:, :self.cell_dim]

        mu_rna_cond = mu_rna[:, self.cell_dim:]
        logvar_rna_cond = logvar_rna[:, self.cell_dim:]

        # 2. Condition encoder: Embedding + [Attention | Sum] → 8D
        mu_cond_attn = self.cond_encoder(gene_idx)  # [B, 8]

        # 3. Reconstruction: split decode
        z_cell = self.reparameterize(mu_rna_cell, logvar_rna_cell)
        # Baseline: 60D → gene expression (no perturbation info)
        logits_baseline = self.cell_output(self.cell_decoder(z_cell))
        # Perturbation effect: 8D → Δ (sole perturbation source)
        logits_pert = self.pert_decoder(mu_cond_attn)
        # Merge
        logits_gene = logits_baseline + logits_pert
        mu_hat = F.softmax(logits_gene, dim=1) * size_factor

        # 4. Classifier (on attention-encoded 8D)
        logits_cond = self.condition_classifier(mu_cond_attn)

        # 5. Adversarial classifier (on 60D，gradient reversal forces 60D to exclude perturbation info)
        if self.use_adversarial:
            logits_adv = self.adv_classifier(self.grl(mu_rna_cell))
        else:
            logits_adv = None

        return (mu_hat, mu_rna_cell, logvar_rna_cell,
                mu_rna_cond, logvar_rna_cond,
                mu_cond_attn, logits_cond, logits_adv, X, cond_vec)

    @torch.no_grad()
    def get_cond_8d(self, gene_names_list, gene_to_idx, max_pert_genes=5, fallback_8d=None):
        """
        Compute fixed 8D condition latent for a perturbation.

        Args:
            gene_names_list: list of gene names (lowercase), e.g. ['tp53', 'kras']
            gene_to_idx: dict mapping gene name -> index (1-based, 0=padding)
            max_pert_genes: max number of perturbation genes
            fallback_8d: numpy array [8], used when no genes are in gene_to_idx

        Returns:
            numpy array [8]
        """
        device = next(self.parameters()).device
        # Build gene_indices
        indices = []
        for g in gene_names_list:
            g_lower = g.strip().lower()
            if g_lower in gene_to_idx:
                indices.append(gene_to_idx[g_lower])
        if not indices:
            # Completely unseen: return fallback
            if fallback_8d is not None:
                return fallback_8d.copy()
            return np.zeros(8, dtype=np.float32)
        # Pad to max_pert_genes
        indices = indices[:max_pert_genes]
        indices = indices + [0] * (max_pert_genes - len(indices))
        gene_idx_tensor = torch.tensor([indices], dtype=torch.long, device=device)
        vec = self.cond_encoder(gene_idx_tensor)  # [1, 8]
        return vec.cpu().numpy().flatten()

    def _compute_losses(self, forward_out):
        (mu_hat, mu_rna_cell, logvar_rna_cell,
         mu_rna_cond, logvar_rna_cond,
         mu_cond_attn, logits_cond, logits_adv, X, cond_vec) = forward_out

        # 1. Reconstruction loss (NB)
        theta = torch.exp(self.theta)
        px = NegativeBinomial(mu=mu_hat, theta=theta)
        recon_loss = -px.log_prob(X).sum(1).mean()

        # 2. KL prior (cell)
        kl_cell = -0.5 * torch.sum(
            1 + logvar_rna_cell - mu_rna_cell.pow(2) - logvar_rna_cell.exp(), dim=1
        ).mean()

        # 3. Alignment loss: RNA encoder 8D → cond_encoder 8D
        kl_align = F.mse_loss(mu_rna_cond, mu_cond_attn)

        # 4. Multi-label classification loss (BCE + pos_weight)
        clf_loss = F.binary_cross_entropy_with_logits(
            logits_cond, cond_vec, pos_weight=self.pos_weight
        )

        # 5. Adversarial classification loss (on 60D, gradient reversal in forward)
        if self.use_adversarial and logits_adv is not None:
            adv_loss = F.binary_cross_entropy_with_logits(
                logits_adv, cond_vec, pos_weight=self.pos_weight
            )
        else:
            adv_loss = torch.tensor(0.0, device=recon_loss.device)

        # 6. Contrastive learning (Jaccard)
        supcon_loss = self.supcon_loss_fn(mu_cond_attn, cond_vec)

        # 7. Dynamic weights: different loss ratios relative to recon
        with torch.no_grad():
            # Unified dynamic scaling for all auxiliary losses
            w_kl_cell = (self.w_kl_cell_ratio * recon_loss.detach()) / (kl_cell.detach() + 1e-6)
            w_clf = (self.w_clf_ratio * recon_loss.detach()) / (clf_loss.detach() + 1e-6)
            w_supcon = (self.w_supcon_ratio * recon_loss.detach()) / (supcon_loss.detach() + 1e-6)
            w_align = (self.w_align_ratio * recon_loss.detach()) / (kl_align.detach() + 1e-6)
            if self.use_adversarial:
                w_adv = (self.w_adv_ratio * recon_loss.detach()) / (adv_loss.detach() + 1e-6)
            else:
                w_adv = torch.tensor(0.0, device=recon_loss.device)
            # clamp to prevent extreme weights
            w_kl_cell = w_kl_cell.clamp(0.1, 10000.0)
            w_clf = w_clf.clamp(0.1, 10000.0)
            w_supcon = w_supcon.clamp(0.1, 10000.0)
            w_align = w_align.clamp(0.1, 10000.0)
            if self.use_adversarial:
                w_adv = w_adv.clamp(0.1, 10000.0)

        return (recon_loss, kl_cell, kl_align, clf_loss, adv_loss, supcon_loss,
                w_kl_cell, w_clf, w_supcon, w_align, w_adv)

    def _update_pos_weight(self, dataloader):
        """Compute pos_weight from training set label distribution"""
        all_labels = []
        for batch in dataloader:
            all_labels.append(batch['condition_vec'])
        all_labels = torch.cat(all_labels, dim=0)
        n_total = all_labels.shape[0]
        n_pos = all_labels.sum(dim=0)
        n_neg = n_total - n_pos
        # pos_weight = neg / pos, clamped to prevent extremes
        pw = (n_neg / (n_pos + 1e-6)).clamp(0.1, 100.0)
        self.pos_weight = pw.to(self.device)

    def training_step(self, batch, batch_idx):
        losses = self._compute_losses(self(batch))
        (recon_loss, kl_cell, kl_align, clf_loss, adv_loss, supcon_loss,
         w_kl_cell, w_clf, w_supcon, w_align, w_adv) = losses

        kl_weight = min(1.0, self.current_epoch / 10.0)

        total_loss = (recon_loss
                      + kl_weight * (w_kl_cell * kl_cell + w_align * kl_align)
                      + w_clf * clf_loss
                      + w_adv * adv_loss
                      + w_supcon * supcon_loss)

        self.log('train_loss', total_loss, on_epoch=True, prog_bar=False, logger=False)
        return total_loss

    def validation_step(self, batch, batch_idx):
        losses = self._compute_losses(self(batch))
        (recon_loss, kl_cell, kl_align, clf_loss, adv_loss, supcon_loss,
         w_kl_cell, w_clf, w_supcon, w_align, w_adv) = losses

        kl_weight = min(1.0, self.current_epoch / 10.0)

        total_val_loss = (recon_loss
                          + kl_weight * (w_kl_cell * kl_cell + w_align * kl_align)
                          + w_clf * clf_loss
                          + w_adv * adv_loss
                          + w_supcon * supcon_loss)

        self.log('val_loss', total_val_loss, on_epoch=True, prog_bar=False, logger=False)
        return total_val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)
        return [optimizer], [scheduler]


# ==========================================
# 4. Single dataset processing pipeline
# ==========================================
def train_maccf_vae(h5ad_path, save_dir, seed=42, use_adversarial=False):
    dataset_name = os.path.splitext(os.path.basename(h5ad_path))[0]

    # Skip completed datasets
    ckpt_path = os.path.join(save_dir, dataset_name, 'best-model.ckpt')
    # if os.path.exists(ckpt_path):
    #     print(f"\n⏭ Skip: {dataset_name} (checkpoint exists)")
    #     return

    print(f"\n▶ Processing: {dataset_name} | save path: {save_dir}")

    pl.seed_everything(seed, workers=True)

    adata = sc.read_h5ad(h5ad_path)
    dataset = MacCFVAEDataset(adata, target_sum=10000)

    split_generator = torch.Generator().manual_seed(seed)
    total_len = len(dataset)
    train_size = int(0.8 * total_len)
    val_size = int(0.1 * total_len)
    test_size = total_len - train_size - val_size

    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size], generator=split_generator)

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True,
                              num_workers=1, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False,
                            num_workers=1, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False,
                             num_workers=1, pin_memory=True)

    model = MacCFVAE(
        in_dim=dataset.X.shape[1],
        n_unique_pert=dataset.n_unique_pert,
        max_pert_genes=5,
        encoder_kwargs={"dims": [512, 128]},
        use_adversarial=use_adversarial,
        target_sum=10000,
        w_kl_cell_ratio=0.05,
        w_clf_ratio=0.25,
        w_supcon_ratio=0.15,
        w_align_ratio=0.2,
        w_adv_ratio=0.2,
    )

    # Compute pos_weight (from training set label distribution)
    model._update_pos_weight(train_loader)

    checkpoint = ModelCheckpoint(
        dirpath=os.path.join(save_dir, dataset_name),
        filename='best-model',
        monitor='val_loss',
        save_top_k=1,
        mode='min',
        enable_version_counter=False
    )
    early_stop = EarlyStopping(monitor="val_loss", patience=8, mode="min")

    trainer = pl.Trainer(
        max_epochs=500,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        precision='bf16-mixed' if torch.cuda.is_available() else '32-true',
        callbacks=[checkpoint, early_stop],
        gradient_clip_val=1.0,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False
    )

    print(f"  [data] Train: {train_size} | Val: {val_size} | Test: {test_size}")
    print(f"  [Perturbation gene] {dataset.n_unique_pert}  types")

    trainer.fit(model, train_loader, val_loader)

    # save gene names (for reconstructing gene_to_idx at inference)
    gene_names_path = os.path.join(save_dir, dataset_name, 'gene_names.txt')
    with open(gene_names_path, 'w') as f:
        for g in dataset.gene_names:
            f.write(g + '\n')
    print(f"  [Save] Gene names list: {gene_names_path}")

    # Save gene_to_idx (for unseen perturbation prediction)
    gene_to_idx = {g: i + 1 for i, g in enumerate(dataset.gene_names)}
    gene_to_idx_path = os.path.join(save_dir, dataset_name, 'gene_to_idx.json')
    with open(gene_to_idx_path, 'w') as f:
        json.dump(gene_to_idx, f)
    print(f"  [save] gene_to_idx: {gene_to_idx_path}")

    # Compute and save mean 8D of all training perturbations (fallback for completely unseen perturbations)
    best_model_for_8d = MacCFVAE.load_from_checkpoint(checkpoint.best_model_path)
    best_model_for_8d.eval()
    device_8d = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_model_for_8d.to(device_8d)
    all_train_8d = []
    max_pg = dataset.gene_indices.shape[1]
    seen_conditions = set()
    with torch.no_grad():
        for i in range(len(dataset)):
            cond_tuple = tuple(sorted(dataset.mlb.inverse_transform(dataset.multi_hot_matrix[i:i+1])[0]))
            if cond_tuple in seen_conditions or len(cond_tuple) == 0:
                continue
            seen_conditions.add(cond_tuple)
            gi = dataset.gene_indices[i:i+1].to(device_8d)
            vec = best_model_for_8d.cond_encoder(gi).cpu().numpy().flatten()
            all_train_8d.append(vec)
    mean_train_8d = np.mean(all_train_8d, axis=0) if all_train_8d else np.zeros(8, dtype=np.float32)
    mean_8d_path = os.path.join(save_dir, dataset_name, 'mean_train_8d.npy')
    np.save(mean_8d_path, mean_train_8d)
    print(f"  [save] mean_train_8d ({len(all_train_8d)} perturbations): {mean_8d_path}")
    del best_model_for_8d

    # ---------------------------------------------------------
    # Final evaluation
    # ---------------------------------------------------------
    best_model = MacCFVAE.load_from_checkpoint(checkpoint.best_model_path)
    best_model.eval()
    best_model.to('cuda' if torch.cuda.is_available() else 'cpu')

    true_counts, pred_means = [], []
    all_preds, all_targets = [], []

    with torch.no_grad():
        for batch in test_loader:
            batch_gpu = {k: v.to(best_model.device) for k, v in batch.items()}
            out = best_model(batch_gpu)
            mu_hat, logits_cond = out[0], out[6]

            true_counts.append(batch_gpu["X"].cpu().numpy())
            pred_means.append(mu_hat.cpu().numpy())

            # Multi-label prediction: sigmoid > 0.5
            preds = (torch.sigmoid(logits_cond) > 0.5).float()
            all_preds.append(preds.cpu().numpy())
            all_targets.append(batch_gpu['condition_vec'].cpu().numpy())

    true_counts = np.vstack(true_counts)
    pred_means = np.vstack(pred_means)

    cell_pccs = [pearsonr(true_counts[i], pred_means[i])[0] for i in range(true_counts.shape[0])]
    avg_cell_pcc = np.mean(cell_pccs)

    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)

    # Metrics: per-gene F1 on perturbed cells only
    pert_mask = (all_targets.sum(axis=1) > 0)
    if pert_mask.sum() > 0:
        pert_f1 = f1_score(all_targets[pert_mask], all_preds[pert_mask], average='micro')
        pert_prec = precision_score(all_targets[pert_mask], all_preds[pert_mask], average='micro')
        pert_rec = recall_score(all_targets[pert_mask], all_preds[pert_mask], average='micro')
    else:
        pert_f1 = pert_prec = pert_rec = 0.0

    # All-zero prediction ratio
    all_zero_rate = (all_preds.sum(axis=1) == 0).mean()

    print(f"  [Done] Best model: {checkpoint.best_model_path}")
    print(f"  [test set] cells PCC: {avg_cell_pcc:.4f}")
    print(f"  [Classification] Perturbed F1: {pert_f1:.4f} | Precision: {pert_prec:.4f} | Recall: {pert_rec:.4f}")
    print(f"  [Classification] All-zero prediction ratio: {all_zero_rate:.4f} (lower is better)")

    del model, best_model, trainer, adata, dataset, train_loader, val_loader, test_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ==========================================
# 5. Batch execution entry point
# ==========================================
if __name__ == "__main__":
    DATA_BASE_DIR = "./data/"
    GENE_DATA_DIR = os.path.join(DATA_BASE_DIR, "gene")
    DRUG_DATA_DIR = os.path.join(DATA_BASE_DIR, "drug")

    GENE_MODEL_DIR = "./output/maccf_vae"
    DRUG_MODEL_DIR = "./output/drug/maccf_vae"

    tasks = [
        (GENE_DATA_DIR, GENE_MODEL_DIR),
        # (DRUG_DATA_DIR, DRUG_MODEL_DIR)
    ]
    train_maccf_vae('./data/your_dataset.h5ad', './output/combos', seed=42)
    # for data_dir, model_dir in tasks:
    #     if not os.path.exists(data_dir):
    #         continue
    #
    #     for file in os.listdir(data_dir):
    #         # file = 'TianKampmann2019_day7neuron_CRISPR.h5ad'
    #         if file.endswith(".h5ad"):
    #             file_path = os.path.join(data_dir, file)
    #             try:
    #                 train_maccf_vae(file_path, model_dir, seed=42)
    #                 # break
    #             except Exception as e:
    #                 print(f"  [Failed] processing {file}: {e}")
