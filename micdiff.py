"""
scPRISM SDEdit variant: SDEdit-based Diffusion Pipeline

Key changes (vs diffusion_pipeline_new.py):
  1. z_ctrl injected via SDEdit starting point only, not channel concat
  2. Inference starts from noised z_ctrl (SDEdit style), not pure noise
  3. Training objective: directly predict z_pert (absolute state), no residual

SDEdit principle (Meng et al. 2021):
  x_{T0} = sqrt(alpha_bar_{T0}) * x_0 + sqrt(1 - alpha_bar_{T0}) * epsilon
  Denoise from x_{T0}, preserving source structure

In scPRISM:
  z_{T0} = sqrt(alpha_bar_{T0}) * z_ctrl + sqrt(1 - alpha_bar_{T0}) * epsilon
  Denoise from z_{T0}, preserving control cell state, applying perturbation effect
"""
import argparse
import json
import os
import torch
from torch.utils.data import DataLoader, Dataset
import scanpy as sc
import anndata as ad
import numpy as np
import pandas as pd
from collections import Counter
import ot
from tqdm import tqdm
import gc
import scipy
from sklearn.preprocessing import LabelEncoder

from micdiff_backbone import MicDiffUNet
from Gaussian_diffusion import GaussianDiffusion, get_named_beta_schedule, ModelMeanType, ModelVarType, LossType
from train_util import TrainLoop
import dist_util
import logger
from maccf_vae import MacCFVAE, MacCFVAEDataset

import re

_QUIET = False

def qprint(*args, **kwargs):
    if not _QUIET:
        print(*args, **kwargs)


# ==========================================
# 1. Dataset (SDEdit: always uses absolute state)
# ==========================================
class MicDiffDataset(Dataset):
    """
    SDEdit variant: always learns absolute state z_pert (no residual)
    x_ctrl stored for constructing starting point during inference
    """
    def __init__(self, target_60d, ctrl_60d, scale_factor=1.0,
                 cond_vec=None, pert_id=None, celltype_id=None, vae_cond_8d=None,
                 is_train=False, batch_dim=8):
        super().__init__()
        self.target = torch.tensor(target_60d, dtype=torch.float32) * scale_factor
        self.ctrl = torch.tensor(ctrl_60d, dtype=torch.float32) * scale_factor
        self.scale_factor = scale_factor
        self.is_train = is_train

        if cond_vec is not None:
            self.cond_vec = torch.tensor(cond_vec, dtype=torch.float32)
        else:
            self.cond_vec = None

        if pert_id is not None:
            self.pert_id = torch.tensor(pert_id, dtype=torch.long)
        else:
            self.pert_id = None

        if celltype_id is not None:
            self.celltype_id = torch.tensor(celltype_id, dtype=torch.long)
        else:
            self.celltype_id = None

        if vae_cond_8d is not None:
            self.vae_cond_8d = torch.tensor(vae_cond_8d, dtype=torch.float32)
        else:
            self.vae_cond_8d = torch.zeros(len(target_60d), batch_dim, dtype=torch.float32)

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        # SDEdit: always uses absolute state (not residual)
        x = self.target[idx].unsqueeze(0)  # [1, 60] full target
        item = {
            'x': x,
            'x_ctrl': self.ctrl[idx].unsqueeze(0),  # [1, 60] (for constructing starting point during inference)
            'vae_cond_8d': self.vae_cond_8d[idx],   # [8]
        }
        if self.cond_vec is not None:
            item['cond_vec'] = self.cond_vec[idx]
        if self.pert_id is not None:
            item['pert_id'] = self.pert_id[idx]
        if self.celltype_id is not None:
            item['label'] = self.celltype_id[idx]
        return item


# ==========================================
# 2. OT pairing (consistent with original code)
# ==========================================
def pair_cells(z_ctrl, z_pert, mode='ot', seed=42):
    if z_ctrl.shape[0] == 0 or z_pert.shape[0] == 0:
        return None
    if mode == 'ot':
        M = ot.dist(z_pert, z_ctrl, metric='euclidean')
        a = np.ones((z_pert.shape[0],)) / z_pert.shape[0]
        b = np.ones((z_ctrl.shape[0],)) / z_ctrl.shape[0]
        G = ot.emd(a, b, M, numItermax=100000)
        return np.argmax(G, axis=1)
    elif mode == 'random':
        rng = np.random.RandomState(seed)
        return rng.choice(z_ctrl.shape[0], size=z_pert.shape[0], replace=True)
    else:
        np.random.seed(seed)
        return np.random.choice(z_ctrl.shape[0], size=z_pert.shape[0], replace=True)


# ==========================================
# 3. External embedding generation (consistent with original code)
# ==========================================
def load_gene_embedding(pickle_path):
    import pickle
    if pickle_path is None or not os.path.exists(pickle_path):
        return None
    with open(pickle_path, 'rb') as f:
        gene_map = pickle.load(f)
    sample_vec = next(iter(gene_map.values()))
    qprint(f"   Loaded gene embedding: {len(gene_map)}  genes, dim={len(sample_vec)}")
    return gene_map


def precompute_gene_embeddings(pert_names, embed_dim, gene_embedding_path):
    gene_map = load_gene_embedding(gene_embedding_path)
    if gene_map is None:
        qprint("   Warning: gene embedding file not found, using zeros.")
        return np.zeros((len(pert_names), embed_dim), dtype=np.float32)

    sample_vec = next(iter(gene_map.values()))
    actual_dim = len(sample_vec)
    embeddings = np.zeros((len(pert_names), actual_dim), dtype=np.float32)

    gene_map_upper = {k.upper(): v for k, v in gene_map.items()}

    missing_stats = Counter()
    for i, name in enumerate(pert_names):
        if isinstance(name, bytes):
            name = name.decode('utf-8')
        sub_names = re.split(r'[+_]', str(name))
        vecs = []
        for sub in sub_names:
            clean = sub.strip().upper()
            if clean in gene_map_upper:
                vecs.append(gene_map_upper[clean])
        if len(vecs) > 0:
            embeddings[i] = np.mean(vecs, axis=0)
        elif str(name).lower() not in ['control', 'nan']:
            missing_stats[name] += 1

    if missing_stats:
        qprint(f"   Warning:  {sum(missing_stats.values())}  cells missing embedding, involving  {len(missing_stats)}  perturbations.")

    # L2-normalize non-zero rows; keep zero rows as zero (control or missing genes)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    nonzero_mask = (norms.flatten() > 1e-8)
    embeddings[nonzero_mask] = embeddings[nonzero_mask] / norms[nonzero_mask]

    return embeddings


def build_drug_id_mapping(pert_names):
    unique_perts = sorted(set(str(p) for p in pert_names if str(p).lower() not in ['control', 'nan']))
    pert_to_id = {name: idx for idx, name in enumerate(unique_perts)}
    return pert_to_id


# ==========================================
# 4. Data preparation (consistent with original, without residual parameter)
# ==========================================
def prepare_data_in_memory(raw_h5ad_path, vae_ckpt_path, pairing_mode='ot', seed=42,
                           gene_embedding_path=None, embed_dim=1536,
                           task_type='gene', test_pairing_mode='random',
                           identity_ratio=0.0):
    torch.manual_seed(seed)
    np.random.seed(seed)

    adata = sc.read_h5ad(raw_h5ad_path)

    vae_dataset = MacCFVAEDataset(adata, target_sum=10000)
    n_unique_pert = vae_dataset.n_unique_pert

    if n_unique_pert == 0:
        raise ValueError(f"Dataset has no valid perturbations (n_unique_pert=0), skipping")

    has_celltype = 'celltype' in adata.obs
    celltype_id_all = None
    num_classes = 0
    if has_celltype:
        le = LabelEncoder()
        celltype_id_all = le.fit_transform(adata.obs['celltype'].astype(str).values)
        num_classes = len(le.classes_)

    # Load VAE and extract features
    try:
        model = MacCFVAE.load_from_checkpoint(
            vae_ckpt_path, map_location='cuda', target_sum=10000,
        ).eval().cuda()
    except TypeError:
        model = MacCFVAE.load_from_checkpoint(
            vae_ckpt_path, map_location='cuda', target_sum=10000,
            n_unique_pert=n_unique_pert,
            w_kl_cell_ratio=0.05, w_clf_ratio=0.25,
            w_supcon_ratio=0.15, w_align_ratio=0.10, w_adv_ratio=0.50,
        ).eval().cuda()

    # Load VAE vocabulary, remap gene_indices (only when vocabulary mismatches)
    vae_gene_names_path = os.path.join(os.path.dirname(vae_ckpt_path), 'gene_names.txt')
    if os.path.exists(vae_gene_names_path):
        with open(vae_gene_names_path) as f:
            vae_gene_names = [line.strip() for line in f if line.strip()]
        data_gene_names = vae_dataset.gene_names
        if vae_gene_names != data_gene_names:
            qprint(f"  -> Vocabulary mismatch (VAE: {len(vae_gene_names)}, Data: {len(data_gene_names)})，remapping...")
            vae_gene_to_idx = {g: i + 1 for i, g in enumerate(vae_gene_names)}
            remap = np.zeros(len(data_gene_names) + 1, dtype=np.int64)
            for i, g in enumerate(data_gene_names):
                remap[i + 1] = vae_gene_to_idx.get(g, 0)
            old_gene_indices = vae_dataset.gene_indices.numpy()
            new_gene_indices = remap[old_gene_indices]
            vae_dataset.gene_indices = torch.tensor(new_gene_indices, dtype=torch.long)
        else:
            qprint(f"  -> Vocabulary matched ({len(vae_gene_names)}  perturbation genes)，no remapping needed")

    loader = DataLoader(vae_dataset, batch_size=512, shuffle=False)
    mu_cells, mu_conds = [], []

    cell_dim = model.cell_dim
    batch_dim = model.batch_dim
    qprint(f"  -> VAE latent space: cell_dim={cell_dim}, batch_dim={batch_dim}")

    qprint(f"  -> Extracting VAE features...")
    for batch in loader:
        h_rna = model.rna_encoder_body(batch['X_norm'].cuda())
        mu_cells.append(model.rna_z_mean(h_rna)[:, :model.cell_dim].detach().cpu().numpy())
        gene_idx = batch['gene_indices'].cuda()
        mu_conds.append(model.cond_encoder(gene_idx).detach().cpu().numpy())

    adata.obsm['mu_cell_60d'] = np.vstack(mu_cells)
    adata.obsm['mu_cond_8d'] = np.vstack(mu_conds)
    latent_std = np.std(adata.obsm['mu_cell_60d'])
    scale_factor = 1.0 / (latent_std + 1e-6)
    qprint(f"  -> Latent space std: {latent_std:.4f} | Scale factor: {scale_factor:.4f}")

    all_pert_names = adata.obs['perturbation'].values
    cond_vec_all = None
    pert_id_all = None
    pert_to_id = None
    num_perturbations = 0

    if task_type == 'gene':
        qprint(f"  -> Building [GENE] external embedding...")
        cond_vec_all = precompute_gene_embeddings(all_pert_names, embed_dim, gene_embedding_path)
    elif task_type == 'drug':
        qprint(f"  -> Build [DRUG] integer ID mapping...")
        pert_to_id = build_drug_id_mapping(all_pert_names)
        num_perturbations = len(pert_to_id)
        pert_id_all = np.array([pert_to_id.get(str(p), -1) for p in all_pert_names], dtype=np.int64)

    # Data split 8:1:1
    total_len = len(vae_dataset)
    t_size = int(0.8 * total_len)
    v_size = int(0.1 * total_len)
    split_generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(total_len, generator=split_generator).numpy()
    splits = {
        'train': indices[:t_size],
        'val': indices[t_size:t_size + v_size],
        'test': indices[t_size + v_size:]
    }

    cell_types = adata.obs['celltype'].unique() if has_celltype else ["global"]
    perts = [p for p in adata.obs['perturbation'].unique() if str(p).lower() not in ['control', 'nan']]
    test_adata = None

    memory_datasets = {}
    train_sub_adata = None
    for split_name, split_idx in splits.items():
        sub_adata = adata[split_idx].copy()
        if split_name == 'train':
            train_sub_adata = sub_adata
        paired_p_idx, paired_c_idx = [], []
        current_mode = test_pairing_mode if split_name == 'test' else pairing_mode

        for ct in cell_types:
            for pert in perts:
                mask_ct = (sub_adata.obs['celltype'] == ct) if has_celltype else True
                idx_c = np.where((sub_adata.obs['perturbation'].astype(str).str.lower() == 'control') & mask_ct)[0]
                idx_p = np.where((sub_adata.obs['perturbation'] == pert) & mask_ct)[0]

                if len(idx_c) < 3 or len(idx_p) < 3:
                    continue

                z_c = sub_adata.obsm['mu_cell_60d'][idx_c]
                z_p = sub_adata.obsm['mu_cell_60d'][idx_p]
                match_idx = pair_cells(z_c, z_p, mode=current_mode, seed=seed)
                if match_idx is not None:
                    paired_p_idx.extend(idx_p)
                    paired_c_idx.extend(idx_c[match_idx])

        if len(paired_p_idx) == 0:
            qprint(f"  warning: {split_name}  split has no valid pairs.")
            continue

        qprint(f"  {split_name}  split pairing mode: {current_mode} | pairs: {len(paired_p_idx)}")

        # Add ctrl→ctrl identity pairing (train only)
        n_identity = 0
        if split_name == 'train' and identity_ratio > 0:
            ctrl_all_idx = np.where(sub_adata.obs['perturbation'].astype(str).str.lower().isin(['control', 'nan']))[0]
            n_identity = int(len(paired_p_idx) * identity_ratio)
            if len(ctrl_all_idx) > 0 and n_identity > 0:
                rng = np.random.RandomState(seed)
                identity_ctrl = rng.choice(ctrl_all_idx, size=n_identity, replace=True)
                paired_p_idx.extend(identity_ctrl.tolist())
                paired_c_idx.extend(identity_ctrl.tolist())  # target = ctrl, ctrl = ctrl (self)
                qprint(f"  {split_name}  split identity pairing: {n_identity}  pairs (ratio={identity_ratio})")

        target_60d = sub_adata.obsm['mu_cell_60d'][paired_p_idx]
        ctrl_60d = sub_adata.obsm['mu_cell_60d'][paired_c_idx]
        vae_cond_8d = sub_adata.obsm['mu_cond_8d'][paired_p_idx]

        cond_vec = None
        pert_id = None
        celltype_id = None

        if task_type == 'gene' and cond_vec_all is not None:
            cond_vec = cond_vec_all[split_idx][paired_p_idx]
            # Set cond_vec to zero for identity pairing (represents 'no perturbation')
            if n_identity > 0:
                cond_vec[-n_identity:] = 0.0
        elif task_type == 'drug' and pert_id_all is not None:
            pert_id = pert_id_all[split_idx][paired_p_idx]
            if n_identity > 0:
                pert_id[-n_identity:] = -1  # use -1 for 'no perturbation'

        if has_celltype and celltype_id_all is not None:
            celltype_id = celltype_id_all[split_idx][paired_p_idx]

        is_train = (split_name == 'train')

        # SDEdit: No residual needed，use scale_factor directly
        memory_datasets[split_name] = MicDiffDataset(
            target_60d, ctrl_60d, scale_factor=scale_factor,
            cond_vec=cond_vec, pert_id=pert_id, celltype_id=celltype_id,
            vae_cond_8d=vae_cond_8d, is_train=is_train, batch_dim=batch_dim,
        )
        qprint(f"  {split_name}  split ready: {len(paired_p_idx)}  cell pairs")

        if split_name == 'test':
            test_adata = ad.AnnData(
                X=sub_adata.X[paired_p_idx].copy(),
                obs=sub_adata.obs.iloc[paired_p_idx].copy()
            )
            test_adata.layers['X_ctrl_real'] = sub_adata.X[paired_c_idx].copy()
            test_adata.obsm['Z_true_68d'] = np.concatenate([target_60d, vae_cond_8d], axis=1)
            test_adata.obsm['Z_ctrl_68d'] = np.concatenate([ctrl_60d, vae_cond_8d], axis=1)

            # Save training set statistics (for computing true delta during evaluation)
            if train_sub_adata is not None:
                _to_dense = lambda X: X.toarray() if hasattr(X, 'toarray') else np.asarray(X)
                # Training set control mean
                train_ctrl_mask = train_sub_adata.obs['perturbation'].astype(str).str.lower().isin(['control', 'nan']).values
                if train_ctrl_mask.sum() > 0:
                    X_train_ctrl = _to_dense(train_sub_adata.X[train_ctrl_mask]).astype(np.float32)
                    train_ctrl_lib = X_train_ctrl.sum(1, keepdims=True)
                    test_adata.uns['train_ctrl_mean'] = np.mean(X_train_ctrl / (train_ctrl_lib + 1e-8) * 10000, axis=0)
                # Training set perturbed pseudobulk
                train_pert_mask = ~train_ctrl_mask
                if train_pert_mask.sum() > 0:
                    X_train_pert = _to_dense(train_sub_adata.X[train_pert_mask]).astype(np.float32)
                    train_pert_lib = X_train_pert.sum(1, keepdims=True)
                    X_train_pert_norm = X_train_pert / (train_pert_lib + 1e-8) * 10000
                    train_pert_conds = train_sub_adata.obs['perturbation'].astype(str).values[train_pert_mask]
                    train_pert_names = list(set(train_pert_conds))
                    train_pb = np.zeros((len(train_pert_names), X_train_pert_norm.shape[1]), dtype=np.float32)
                    for pi, pname in enumerate(train_pert_names):
                        train_pb[pi] = X_train_pert_norm[train_pert_conds == pname].mean(axis=0)
                    test_adata.uns['train_pert_names'] = np.array(train_pert_names)
                    test_adata.uns['train_pert_pb'] = train_pb

    del adata, model, loader
    torch.cuda.empty_cache()

    return (memory_datasets.get('train'), memory_datasets.get('val'),
            memory_datasets.get('test'), test_adata, scale_factor,
            num_classes, num_perturbations, n_unique_pert, scale_factor,
            cell_dim, batch_dim)


# ==========================================
# 5. Model creation (SDEdit: concat_ctrl=False)
# ==========================================
def create_micdiff(args, num_classes=0, num_perturbations=0):
    rna_dim = [int(i) for i in args.rna_dim.split(',')]
    channel_mult = [int(i) for i in args.channel_mult.split(',')]

    betas = get_named_beta_schedule(args.noise_schedule, args.diffusion_steps)
    model_var_type = ModelVarType.LEARNED_RANGE
    in_channels = rna_dim[0]
    out_channels = in_channels * 2

    model = MicDiffUNet(
        rna_dim=rna_dim,
        model_channels=args.model_channels,
        out_channels=out_channels,
        num_res_blocks=args.num_res_blocks,
        dropout=args.dropout,
        channel_mult=tuple(channel_mult),
        num_classes=num_classes,
        num_perturbations=num_perturbations,
        use_checkpoint=False,
        use_fp16=args.use_fp16,
        num_heads=args.num_heads,
        use_scale_shift_norm=True,
        resblock_updown=True,
        pert_embed_dim=args.pert_embed_dim,
        concat_ctrl=False,   # z_ctrl injected via SDEdit starting point only
    )

    diffusion = GaussianDiffusion(
        betas=betas,
        model_mean_type=ModelMeanType.EPSILON,
        model_var_type=ModelVarType.LEARNED_RANGE,
        loss_type=LossType.MSE,
        rescale_timesteps=False,
    )
    return model, diffusion


# ==========================================
# 6. SDEdit sampling function
# ==========================================
def micdiff_sample(diffusion, model, shape, x_ctrl, noise_ratio, model_kwargs=None, eta=0.0):
    """
    SDEdit-style sampling: denoise from noised z_ctrl to z_pert using DDIM

    Args:
        diffusion: GaussianDiffusion object
        model: UNet model
        shape: (B, 1, cell_dim)
        x_ctrl: [B, 1, cell_dim] control cell latent (scaled)
        noise_ratio: float, 0~1, noise level control (T0 = T * noise_ratio)
        model_kwargs: other conditions (cond_vec, pert_id, etc.)
        eta: float, 0=deterministic DDIM, 1=equivalent to DDPM

    Returns:
        z_pred: [B, 1, cell_dim] predicted perturbed cell latent (scaled)
    """
    device = x_ctrl.device
    T = diffusion.num_timesteps
    T0 = max(1, int(T * noise_ratio))

    # Get alpha_bar_{T0}
    alpha_bar = diffusion.alphas_cumprod  # [T]
    alpha_bar_T0 = torch.tensor(alpha_bar[T0 - 1], device=device, dtype=torch.float32)

    # Construct starting point: z_{T0} = sqrt(alpha_bar_{T0}) * z_ctrl + sqrt(1 - alpha_bar_{T0}) * epsilon
    epsilon = torch.randn_like(x_ctrl)
    z_T0 = torch.sqrt(alpha_bar_T0) * x_ctrl + torch.sqrt(1 - alpha_bar_T0) * epsilon

    # Denoise from T0 using DDIM
    img = z_T0
    indices = list(range(T0))[::-1]

    for i in indices:
        t = torch.tensor([i] * shape[0], device=device)
        with torch.no_grad():
            out = diffusion.ddim_sample(
                model, img, t,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                eta=eta,
            )
            img = out["sample"]

    return img


# ==========================================
# 7. Test set generation
# ==========================================
def generate_and_save_test(model, diffusion, test_loader, test_adata, out_path,
                           scale_factor, noise_ratio=0.5, cell_dim=60, eta=0.0):
    qprint(f"\n  -> Starting SDEdit counterfactual generation...")
    qprint(f"     noise_ratio: {noise_ratio} | Scale factor: {scale_factor:.4f}")
    model.eval()
    device = dist_util.dev()

    all_pred_60d, all_vae_cond_8d = [], []

    with torch.no_grad():
        for batch in (tqdm(test_loader, desc="  SDEdit Sampling", leave=False) if not _QUIET else test_loader):
            x_ctrl = batch['x_ctrl'].to(device)
            vae_cond_8d = batch['vae_cond_8d']

            model_kwargs = {}
            if 'cond_vec' in batch:
                model_kwargs['cond_vec'] = batch['cond_vec'].to(device)
            if 'pert_id' in batch:
                model_kwargs['pert_id'] = batch['pert_id'].to(device)
            if 'label' in batch:
                model_kwargs['label'] = batch['label'].to(device)

            shape = (x_ctrl.shape[0], 1, cell_dim)

            # SDEdit sampling: from noised z_ctrl using DDIM
            pred_target = micdiff_sample(
                diffusion, model, shape, x_ctrl,
                noise_ratio=noise_ratio,
                model_kwargs=model_kwargs,
                eta=eta,
            )

            pred_unscaled = pred_target.detach().cpu().numpy().squeeze(1) / scale_factor
            all_pred_60d.append(pred_unscaled)
            all_vae_cond_8d.append(vae_cond_8d.detach().cpu().numpy())

    X_pred_60d = np.vstack(all_pred_60d)
    X_vae_cond_8d = np.vstack(all_vae_cond_8d)

    Z_pred_68d = np.concatenate([X_pred_60d, X_vae_cond_8d], axis=1)
    test_adata.obsm['Z_pred_68d'] = Z_pred_68d
    test_adata.write_h5ad(out_path, compression='gzip')
    qprint(f"  Test set results saved to: {out_path}\n")


# ==========================================
# 8. Single dataset: train + generate + evaluate
# ==========================================
def run_micdiff(raw_data_path, vae_ckpt_path, output_dir, args, task_type):
    dataset_name = os.path.splitext(os.path.basename(raw_data_path))[0]
    os.makedirs(output_dir, exist_ok=True)
    quiet = getattr(args, 'quiet', False)
    logger.configure(dir=output_dir, quiet=quiet)

    qprint(f"\n{'='*60}")
    qprint(f"  Task: [ {dataset_name} ] | type: {task_type.upper()} | mode: SDEdit")
    qprint(f"{'='*60}")

    # 1. Data preparation
    (train_ds, val_ds, test_ds, test_adata, scale_factor,
     num_classes, num_perturbations, n_unique_pert, _,
     cell_dim, batch_dim) = prepare_data_in_memory(
        raw_h5ad_path=raw_data_path,
        vae_ckpt_path=vae_ckpt_path,
        pairing_mode=args.pairing_mode,
        gene_embedding_path=args.gene_embedding_path,
        embed_dim=args.pert_embed_dim,
        task_type=task_type,
        test_pairing_mode=args.test_pairing_mode,
        identity_ratio=args.identity_ratio,
    )

    if train_ds is None or test_adata is None:
        qprint("  Dataset split/pairing error, skipping。")
        return

    args.rna_dim = f"1,{cell_dim}"

    # 2. Create model (concat_ctrl=False)
    model, diffusion = create_micdiff(args, num_classes, num_perturbations)
    model.to(dist_util.dev())

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    steps_per_epoch = len(train_loader)
    if steps_per_epoch == 0:
        qprint("  Train Loader length is 0, skipping。")
        return

    args.eval_interval = steps_per_epoch
    args.log_interval = steps_per_epoch

    # 3. Clean old checkpoints
    for old_file in ['model_best.pt', 'model_latest.pt', 'ema_0.9999_best.pt', 'ema_0.9999_latest.pt', 'opt_latest.pt']:
        old_path = os.path.join(output_dir, old_file)
        if os.path.exists(old_path):
            os.remove(old_path)

    # 4. Training
    qprint(f"\n  -> Starting SDEdit Diffusion model training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=infinite_loader(train_loader),
        val_data=infinite_loader(val_loader),
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        eval_interval=args.eval_interval,
        patience=args.patience,
        lr_anneal_steps=args.lr_anneal_steps,
        resume_checkpoint=args.resume_checkpoint,
        lr=args.lr,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        weight_decay=args.weight_decay,
        class_cond=(num_classes > 0),
        num_classes=num_classes,
        use_db=False,
        use_tqdm=False,
    ).run_loop()

    # 5. Load best model
    best_ckpt = os.path.join(output_dir, "model_best.pt")
    if os.path.exists(best_ckpt):
        model.load_state_dict(torch.load(best_ckpt, map_location=dist_util.dev()))
        qprint(f"  -> Loaded best checkpoint: {best_ckpt}")
    else:
        qprint(f"  -> No best checkpoint found, using final training weights.")

    # 6. SDEdit generation
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    out_file = os.path.join(output_dir, "test_res_sdedit.h5ad")
    generate_and_save_test(model, diffusion, test_loader, test_adata, out_file,
                           scale_factor, noise_ratio=args.noise_ratio, cell_dim=cell_dim, eta=args.eta)

    # 7. Decode + evaluate
    qprint(f"  -> Starting decoding and evaluation...")
    from decode_eval import evaluate_scprism
    evaluate_scprism(h5ad_path=out_file, vae_ckpt_path=vae_ckpt_path,
                     full_data_path=raw_data_path,
                     condition_key='perturbation', n_unique_pert=n_unique_pert)

    del train_ds, val_ds, test_ds, test_adata, model, diffusion
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def infinite_loader(loader):
    while True:
        yield from loader


# ==========================================
# 9. Main function
# ==========================================
def predict_unseen_perturbations(raw_data_path, vae_ckpt_path, output_dir, args, task_type):
    """
    Predict unseen perturbations using trained VAE + Diffusion models.

    For new combos (A+B): averages individual gene 8D embeddings.
    For completely new genes: uses mean of all training perturbation 8D vectors.
    """
    dataset_name = os.path.splitext(os.path.basename(raw_data_path))[0]
    device = dist_util.dev()

    # Load gene_to_idx and mean_train_8d
    vae_model_dir = os.path.dirname(vae_ckpt_path)
    gene_to_idx_path = os.path.join(vae_model_dir, 'gene_to_idx.json')
    mean_8d_path = os.path.join(vae_model_dir, 'mean_train_8d.npy')

    if not os.path.exists(gene_to_idx_path):
        qprint(f"  gene_to_idx.json not found: {gene_to_idx_path}")
        return
    if not os.path.exists(mean_8d_path):
        qprint(f"  mean_train_8d.npy not found: {mean_8d_path}")
        return

    with open(gene_to_idx_path) as f:
        gene_to_idx = json.load(f)
    mean_train_8d = np.load(mean_8d_path)
    qprint(f"  gene_to_idx: {len(gene_to_idx)} genes | mean_train_8d shape: {mean_train_8d.shape}")

    # Load VAE
    try:
        vae_model = MacCFVAE.load_from_checkpoint(
            vae_ckpt_path, map_location=device, target_sum=10000,
        ).eval().to(device)
    except TypeError:
        vae_model = MacCFVAE.load_from_checkpoint(
            vae_ckpt_path, map_location=device, target_sum=10000,
            n_unique_pert=len(gene_to_idx),
            w_kl_cell_ratio=0.05, w_clf_ratio=0.25,
            w_supcon_ratio=0.15, w_align_ratio=0.10, w_adv_ratio=0.50,
        ).eval().to(device)

    cell_dim = vae_model.cell_dim

    # Load diffusion model
    best_ckpt = os.path.join(output_dir, "model_best.pt")
    if not os.path.exists(best_ckpt):
        qprint(f"  Diffusion checkpoint not found: {best_ckpt}")
        return

    # Need to prepare data once to get model architecture
    (train_ds, val_ds, test_ds, test_adata, scale_factor,
     num_classes, num_perturbations, n_unique_pert, _,
     cell_dim_data, batch_dim) = prepare_data_in_memory(
        raw_h5ad_path=raw_data_path,
        vae_ckpt_path=vae_ckpt_path,
        pairing_mode=args.pairing_mode,
        gene_embedding_path=args.gene_embedding_path,
        embed_dim=args.pert_embed_dim,
        task_type=task_type,
        test_pairing_mode=args.test_pairing_mode,
        identity_ratio=args.identity_ratio,
    )

    if test_adata is None:
        qprint("  No test data available.")
        return

    args.rna_dim = f"1,{cell_dim_data}"
    model, diffusion = create_micdiff(args, num_classes, num_perturbations)
    model.load_state_dict(torch.load(best_ckpt, map_location=device))
    model.to(device)
    model.eval()

    # Load gene2vec
    gene_map = load_gene_embedding(args.gene_embedding_path)
    gene_map_upper = {k.upper(): v for k, v in gene_map.items()} if gene_map else {}
    sample_vec = next(iter(gene_map.values())) if gene_map else None
    actual_dim = len(sample_vec) if sample_vec is not None else args.pert_embed_dim

    # Get control cells from test set
    ctrl_mask_test = np.array([str(p).lower() in ['control', 'ctrl', 'nan']
                                for p in test_adata.obs['perturbation'].values])
    ctrl_adata_test = test_adata[ctrl_mask_test]
    if len(ctrl_adata_test) == 0:
        qprint("  No control cells in test set.")
        return

    ctrl_mu_60d = ctrl_adata_test.obsm.get('mu_cell_60d')
    if ctrl_mu_60d is None:
        qprint("  mu_cell_60d not found in test adata.")
        return

    # Parse unseen perturbations
    unseen_perts_str = args.unseen_perts
    if not unseen_perts_str:
        qprint("  No unseen perturbations specified (--unseen_perts).")
        return
    unseen_perts = [p.strip() for p in unseen_perts_str.split(',') if p.strip()]
    qprint(f"  Predicting {len(unseen_perts)} unseen perturbations...")

    # Predict each unseen perturbation
    all_pred_60d, all_vae_cond_8d, all_pert_labels, all_ctrl_indices = [], [], [], []
    n_samples_per_pert = min(args.unseen_n_samples, len(ctrl_adata_test))

    for pert_name in unseen_perts:
        qprint(f"    Predicting: {pert_name}")

        # 1. Compute 8D condition latent
        gene_names = [g.strip().lower() for g in pert_name.split('+') if g.strip().lower() not in ['ctrl', 'control']]
        cond_8d = vae_model.get_cond_8d(gene_names, gene_to_idx,
                                         fallback_8d=mean_train_8d)  # [8]

        # 2. Compute gene2vec embedding
        cond_vec = np.zeros(actual_dim, dtype=np.float32)
        if gene_map_upper:
            sub_vecs = []
            for g in gene_names:
                g_upper = g.upper()
                if g_upper in gene_map_upper:
                    sub_vecs.append(gene_map_upper[g_upper])
            if sub_vecs:
                cond_vec = np.mean(sub_vecs, axis=0)
                norm = np.linalg.norm(cond_vec)
                if norm > 1e-8:
                    cond_vec = cond_vec / norm

        # 3. Sample control cells
        rng = np.random.RandomState(42)
        sample_idx = rng.choice(len(ctrl_mu_60d), size=n_samples_per_pert, replace=True)
        x_ctrl_batch = ctrl_mu_60d[sample_idx]  # [N, 60]
        x_ctrl_scaled = x_ctrl_batch * scale_factor

        # 4. Prepare condition tensors
        cond_vec_batch = np.tile(cond_vec, (n_samples_per_pert, 1))  # [N, 1536]
        vae_cond_8d_batch = np.tile(cond_8d, (n_samples_per_pert, 1))  # [N, 8]

        x_ctrl_tensor = torch.tensor(x_ctrl_scaled, dtype=torch.float32).unsqueeze(1).to(device)  # [N, 1, 60]
        cond_vec_tensor = torch.tensor(cond_vec_batch, dtype=torch.float32).to(device)  # [N, 1536]

        # 5. SDEdit sampling
        model_kwargs = {'cond_vec': cond_vec_tensor}
        shape = (n_samples_per_pert, 1, cell_dim_data)

        with torch.no_grad():
            pred_target = micdiff_sample(
                diffusion, model, shape, x_ctrl_tensor,
                noise_ratio=args.noise_ratio,
                model_kwargs=model_kwargs,
                eta=args.eta,
            )

        pred_unscaled = pred_target.detach().cpu().numpy().squeeze(1) / scale_factor
        all_pred_60d.append(pred_unscaled)
        all_vae_cond_8d.append(vae_cond_8d_batch)
        all_pert_labels.extend([pert_name] * n_samples_per_pert)
        all_ctrl_indices.extend(sample_idx.tolist())

    # Save results
    X_pred_60d = np.vstack(all_pred_60d)
    X_vae_cond_8d = np.vstack(all_vae_cond_8d)
    Z_pred_68d = np.concatenate([X_pred_60d, X_vae_cond_8d], axis=1)

    # Build output AnnData
    out_adata = ctrl_adata_test[all_ctrl_indices].copy()
    out_adata.obs['perturbation'] = all_pert_labels
    out_adata.obs['condition'] = all_pert_labels
    out_adata.obsm['Z_pred_68d'] = Z_pred_68d
    out_adata.obsm['mu_cell_60d'] = X_pred_60d
    out_adata.obsm['mu_cond_8d'] = X_vae_cond_8d

    out_path = os.path.join(output_dir, "unseen_pert_predictions.h5ad")
    out_adata.write_h5ad(out_path, compression='gzip')
    qprint(f"  Unseen perturbation predictions saved: {out_path}")

    # Decode and evaluate
    try:
        from decode_eval import evaluate_scprism
        evaluate_scprism(h5ad_path=out_path, vae_ckpt_path=vae_ckpt_path,
                         condition_key='perturbation', n_unique_pert=n_unique_pert,
                         full_data_path=raw_data_path)
    except Exception as e:
        qprint(f"  Evaluation skipped: {e}")


def main():
    parser = argparse.ArgumentParser(description="scPRISM SDEdit Diffusion Pipeline")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "predict", "predict_unseen"])
    parser.add_argument("--pairing_mode", type=str, default="ot")
    parser.add_argument("--test_pairing_mode", type=str, default="random")
    parser.add_argument("--base_output_dir", type=str, default="./output/diffusion_sdedit/")
    parser.add_argument("--rna_dim", type=str, default="1,60")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--microbatch", type=int, default=-1)
    parser.add_argument("--model_channels", type=int, default=64)
    parser.add_argument("--num_res_blocks", type=int, default=2)
    parser.add_argument("--channel_mult", type=str, default="1,2,4")
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--noise_schedule", type=str, default="cosine")
    parser.add_argument("--diffusion_steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--ema_rate", type=str, default="0.9999")
    parser.add_argument("--use_fp16", action="store_true")
    parser.add_argument("--fp16_scale_growth", type=float, default=1e-3)
    parser.add_argument("--resume_checkpoint", type=str, default="")
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--lr_anneal_steps", type=int, default=0)
    parser.add_argument("--gene_embedding_path", type=str, default='./data/GenePT_gene_embedding_ada_text.pickle')
    parser.add_argument("--pert_embed_dim", type=int, default=1536)
    parser.add_argument("--identity_ratio", type=float, default=0,
                        help="ctrl→ctrl identity pairing ratio (0=none, 0.5=50%% identity)")
    parser.add_argument("--noise_ratio", type=float, default=0.95,
                        help="SDEdit noise ratio: T0 = T * noise_ratio (0~1)")
    parser.add_argument("--eta", type=float, default=0.0,
                        help="DDIM eta: 0=deterministic, 1=equivalent to DDPM")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--dataset", type=str, default=None, help="Only process specified dataset")
    parser.add_argument("--vae_dir", type=str, default="")
    parser.add_argument("--gene_vae_dir", type=str, default="./output/maccf_vae")
    parser.add_argument("--drug_vae_dir", type=str, default="")
    parser.add_argument("--unseen_perts", type=str, default="",
                        help="Comma-separated unseen perturbation names for predict_unseen mode")
    parser.add_argument("--unseen_n_samples", type=int, default=100,
                        help="Number of control cells to sample per unseen perturbation")
    args, _ = parser.parse_known_args()

    global _QUIET
    _QUIET = args.quiet

    dist_util.setup_dist()

    DATA_BASE_DIR = "./data/"
    default_gene_vae = "./output/maccf_vae"
    default_drug_vae = "./output/drug/maccf_vae"

    if args.gene_vae_dir:
        gene_vae = args.gene_vae_dir
    elif args.vae_dir:
        gene_vae = os.path.join(args.vae_dir, "gene")
    else:
        gene_vae = default_gene_vae

    if args.drug_vae_dir:
        drug_vae = args.drug_vae_dir
    elif args.vae_dir:
        drug_vae = os.path.join(args.vae_dir, "drug")
    else:
        drug_vae = default_drug_vae

    tasks = [
        (os.path.join(DATA_BASE_DIR, "gene"), gene_vae, "gene"),
        # (os.path.join(DATA_BASE_DIR, "drug"), drug_vae, "drug"),
    ]

    for data_dir, vae_dir, task_type in tasks:
        if not os.path.exists(data_dir):
            continue

        for file in os.listdir(data_dir):
            if not file.endswith(".h5ad"):
                continue
            dataset_name = os.path.splitext(file)[0]
            if args.dataset and dataset_name != args.dataset:
                continue
            raw_data_path = os.path.join(data_dir, file)
            vae_ckpt_path = os.path.join(vae_dir, dataset_name, "best-model.ckpt")

            if not os.path.exists(vae_ckpt_path):
                continue

            output_dir = os.path.join(args.base_output_dir, f"noise{args.noise_ratio}", dataset_name)

            try:
                if args.mode == "train":
                    run_micdiff(raw_data_path, vae_ckpt_path, output_dir, args, task_type)
                elif args.mode == "predict":
                    best_ckpt = os.path.join(output_dir, "model_best.pt")
                    if not os.path.exists(best_ckpt):
                        qprint(f"  Checkpoint not found: {best_ckpt}, skip。")
                        continue

                    (train_ds, val_ds, test_ds, test_adata, scale_factor,
                     num_classes, num_perturbations, n_unique_pert, _,
                     cell_dim, batch_dim) = prepare_data_in_memory(
                        raw_h5ad_path=raw_data_path,
                        vae_ckpt_path=vae_ckpt_path,
                        pairing_mode=args.pairing_mode,
                        gene_embedding_path=args.gene_embedding_path,
                        embed_dim=args.pert_embed_dim,
                        task_type=task_type,
                        test_pairing_mode=args.test_pairing_mode,
                        identity_ratio=args.identity_ratio,
                    )

                    if test_ds is None or test_adata is None:
                        continue

                    args.rna_dim = f"1,{cell_dim}"
                    model, diffusion = create_micdiff(args, num_classes, num_perturbations)
                    model.load_state_dict(torch.load(best_ckpt, map_location=dist_util.dev()))
                    model.to(dist_util.dev())

                    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
                    out_file = os.path.join(output_dir, "test_res_sdedit.h5ad")
                    generate_and_save_test(model, diffusion, test_loader, test_adata, out_file,
                                           scale_factor, noise_ratio=args.noise_ratio, cell_dim=cell_dim, eta=args.eta)

                    from decode_eval import evaluate_scprism
                    evaluate_scprism(h5ad_path=out_file, vae_ckpt_path=vae_ckpt_path,
                                     condition_key='perturbation', n_unique_pert=n_unique_pert,
                                     full_data_path=raw_data_path)

                elif args.mode == "predict_unseen":
                    predict_unseen_perturbations(raw_data_path, vae_ckpt_path, output_dir, args, task_type)

            except Exception as e:
                qprint(f"\n  Processing error: {e}")
                import traceback
                traceback.print_exc()


if __name__ == "__main__":
    main()
