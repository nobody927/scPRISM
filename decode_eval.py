import os
import torch
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import issparse
from scipy.stats import pearsonr
from scipy.spatial.distance import cdist

import torch.nn.functional as F


def load_model(ckpt_path, **kwargs):
    """Attempt to load MacCF-VAE checkpoint"""
    try:
        from maccf_vae import MacCFVAE as Model
        try:
            return Model.load_from_checkpoint(ckpt_path, **kwargs).eval()
        except TypeError:
            # Compatible with old checkpoint (missing new params)
            kwargs.setdefault('w_kl_cell_ratio', 0.05)
            kwargs.setdefault('w_clf_ratio', 0.25)
            kwargs.setdefault('w_supcon_ratio', 0.15)
            kwargs.setdefault('w_align_ratio', 0.10)
            kwargs.setdefault('w_adv_ratio', 0.50)
            return Model.load_from_checkpoint(ckpt_path, **kwargs).eval()
    except Exception:
        from maccf_vae import MacCFVAE as Model
        return Model.load_from_checkpoint(ckpt_path, **kwargs).eval()


def load_gene_names(vae_ckpt_path):
    """Load gene names saved during training"""
    gene_names_path = os.path.join(os.path.dirname(vae_ckpt_path), 'gene_names.txt')
    if not os.path.exists(gene_names_path):
        return None
    with open(gene_names_path, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def perturbation_to_gene_indices(pert_names, gene_names, max_pert_genes=5):
    """Convert perturbation names to gene index sequences (consistent with MacCFVAEDataset)"""
    gene_to_idx = {g: i + 1 for i, g in enumerate(gene_names)}  # 0 = padding
    indices = np.zeros((len(pert_names), max_pert_genes), dtype=np.int64)
    for i, name in enumerate(pert_names):
        sub_genes = [p.strip().lower() for p in str(name).split('+')]
        for j, g in enumerate(sub_genes[:max_pert_genes]):
            if g in gene_to_idx:
                indices[i, j] = gene_to_idx[g]
    return indices

def to_dense(data):
    if issparse(data):
        return data.toarray()
    return data


def get_pseudobulk(adata, condition_key='perturbation', layer=None):
    """Compute pseudobulk (mean expression)"""
    if layer is not None and layer in adata.layers:
        X = to_dense(adata.layers[layer])
    else:
        X = to_dense(adata.X)

    conditions = np.array([str(x) for x in adata.obs[condition_key]])
    unique_conditions = np.unique(conditions)

    pseudobulk_data = []
    valid_conditions = []

    for cond in unique_conditions:
        # Skip explicit control labels
        if str(cond).lower() in ['ctrl', 'control', 'unperturbed', 'nan']:
            continue

        mask = (conditions == cond)
        if np.sum(mask) == 0:
            continue

        mean_expr = np.mean(X[mask], axis=0)
        pseudobulk_data.append(mean_expr)
        valid_conditions.append(cond)

    if not pseudobulk_data:
        raise ValueError("No valid perturbation conditions found in this dataset!")

    df = pd.DataFrame(np.vstack(pseudobulk_data), index=valid_conditions, columns=adata.var_names)
    return df


def calculate_delta_pcc(true_deltas, pred_deltas):
    """Compute delta PCC"""
    pccs = {}
    common_conds = true_deltas.index.intersection(pred_deltas.index)

    for cond in common_conds:
        t_vec = true_deltas.loc[cond].values
        p_vec = pred_deltas.loc[cond].values

        if np.std(t_vec) == 0 or np.std(p_vec) == 0:
            pccs[cond] = 0.0
            continue

        corr, _ = pearsonr(t_vec, p_vec)
        pccs[cond] = corr

    return pd.Series(pccs)


def calculate_systema_score(true_deltas, pred_deltas):
    """Compute Systema Score (Centroid Accuracy / Normalized Rank)"""
    common_conds = true_deltas.index.intersection(pred_deltas.index)
    if len(common_conds) < 2:
        print("   ⚠️ Warning: fewer than 2 common perturbation conditions, cannot compute Rank Score.")
        return 0.0, pd.Series()

    gt_df = true_deltas.loc[common_conds]
    pred_df = pred_deltas.loc[common_conds]

    # Compute Euclidean distance matrix
    distances = cdist(pred_df, gt_df, metric='euclidean')
    dist_df = pd.DataFrame(distances, index=pred_df.index, columns=gt_df.index)

    # Extract self-distances
    self_distances = np.diag(dist_df.values)

    scores = {}
    N = dist_df.shape[1]

    for cond in common_conds:
        dists_to_all_gt = dist_df.loc[cond]
        dist_to_self = self_distances[dist_df.index.get_loc(cond)]
        num_worse = (dists_to_all_gt > dist_to_self).sum()
        score = num_worse / (N - 1)
        scores[cond] = score

    scores_series = pd.Series(scores)
    return scores_series.mean(), scores_series


def calculate_top_deg_pcc(true_deltas, pred_deltas, top_n=20):
    """Compute Top N DEG delta PCC"""
    common_conds = true_deltas.index.intersection(pred_deltas.index)
    pccs = {}
    for cond in common_conds:
        t_delta = true_deltas.loc[cond].values
        p_delta = pred_deltas.loc[cond].values
        # Sort by absolute true delta, take top N
        top_idx = np.argsort(np.abs(t_delta))[-top_n:]
        t_top = t_delta[top_idx]
        p_top = p_delta[top_idx]
        if np.std(t_top) == 0 or np.std(p_top) == 0:
            pccs[cond] = 0.0
        else:
            pccs[cond] = pearsonr(t_top, p_top)[0]
    return pd.Series(pccs)


def calculate_top_deg_raw_pcc(true_raw, pred_raw, true_deltas, top_n=20):
    """Compute Top N DEG raw PCC"""
    common_conds = true_raw.index.intersection(pred_raw.index).intersection(true_deltas.index)
    pccs = {}
    for cond in common_conds:
        t_delta = true_deltas.loc[cond].values
        # Sort by absolute true delta, take top N
        top_idx = np.argsort(np.abs(t_delta))[-top_n:]
        t_top = true_raw.loc[cond].values[top_idx]
        p_top = pred_raw.loc[cond].values[top_idx]
        if np.std(t_top) == 0 or np.std(p_top) == 0:
            pccs[cond] = 0.0
        else:
            pccs[cond] = pearsonr(t_top, p_top)[0]
    return pd.Series(pccs)


@torch.no_grad()
def decode_latent(adata, vae_ckpt_path, n_unique_pert=None, condition_key='perturbation'):
    print(f"\n  Loading VAE model for decoding...")

    # Load MacCF-VAE model
    try:
        model = load_model(vae_ckpt_path, map_location='cuda', target_sum=10000)
    except Exception:
        if n_unique_pert is None:
            raise ValueError("n_unique_pert required to load VAE")
        model = load_model(vae_ckpt_path, map_location='cuda', target_sum=10000, n_unique_pert=n_unique_pert)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    TARGET_LIB = 10000

    # === A. Split diffusion output ===
    z_pred = torch.tensor(adata.obsm['Z_pred_68d'], dtype=torch.float32).to(device)
    cell_dim = model.cell_dim
    z_cell_60d = z_pred[:, :cell_dim]

    # === B. Generate 8D via condition encoder (counterfactual inference) ===
    gene_names = load_gene_names(vae_ckpt_path)
    if gene_names is not None and hasattr(model, 'cond_encoder'):
        # MacCF-VAE: Attention-based perturbation encoder
        print(f"   Using cond_encoder to generate 8D (counterfactual mode)")
        pert_names = adata.obs[condition_key].values
        gene_indices = perturbation_to_gene_indices(pert_names, gene_names)
        gene_indices = torch.tensor(gene_indices, dtype=torch.long).to(device)
        mu_cond_8d = model.cond_encoder(gene_indices).detach()
    elif hasattr(model, 'cond_mu'):
        # vae_multi: Embedding lookup → 8D
        print(f"   Using cond_mu embedding to generate 8D (counterfactual mode)")
        pert_names = adata.obs[condition_key].values
        control_keywords = ['control', 'ctrl', 'unperturbed', 'nan']
        # Build perturbation name → class index mapping (0=control, 1..N=perturbations)
        unique_perts = []
        for p in pert_names:
            s = str(p).lower().strip()
            if s not in control_keywords and s not in unique_perts:
                unique_perts.append(s)
        pert_to_idx = {p: i + 1 for i, p in enumerate(unique_perts)}
        cond_indices = []
        for p in pert_names:
            s = str(p).lower().strip()
            cond_indices.append(0 if s in control_keywords else pert_to_idx.get(s, 0))
        cond_indices = torch.tensor(cond_indices, dtype=torch.long).to(device)
        mu_cond_8d = model.cond_mu(cond_indices).detach()
    else:
        print(f"   ⚠️ Condition encoder not found, falling back to RNA encoder")
        if 'X_ctrl_real' not in adata.layers:
            raise ValueError("Missing 'X_ctrl_real' layer")
        X_ctrl = torch.tensor(to_dense(adata.layers['X_ctrl_real']), dtype=torch.float32).to(device)
        ctrl_lib_size = X_ctrl.sum(1, keepdim=True)
        X_ctrl_norm = torch.log1p(X_ctrl / (ctrl_lib_size + 1e-6) * TARGET_LIB)
        h_rna_ctrl = model.rna_encoder_body(X_ctrl_norm)
        mu_rna_full = model.rna_z_mean(h_rna_ctrl)
        mu_cond_8d = mu_rna_full[:, cell_dim:]

    # === C. Concatenate and decode ===
    z_combined = torch.cat([z_cell_60d, mu_cond_8d], dim=1)

    logits_gene = model.cell_output(model.cell_decoder(z_cell_60d)) + model.pert_decoder(mu_cond_8d)

    # === D. Output count space (softmax * TARGET_LIB) ===
    mu_hat = F.softmax(logits_gene, dim=1) * TARGET_LIB

    adata.layers['pred_expr'] = mu_hat.cpu().numpy()
    print(f"   ✅ Decoding complete! Predicted expression mounted to adata.layers['pred_expr'] (scale: {TARGET_LIB})")

    return adata


def evaluate_scprism(h5ad_path, vae_ckpt_path, condition_key='perturbation', n_unique_pert=None, full_data_path=None):
    print("\n" + "=" * 60)
    print(f"Evaluating dataset: {os.path.basename(h5ad_path)}")
    print("=" * 60)

    TARGET_LIB = 10000

    # 1. Load data
    adata = sc.read_h5ad(h5ad_path)

    # 2. Decode gene expression via VAE decoder (count space, scale=TARGET_LIB)
    adata = decode_latent(adata, vae_ckpt_path, n_unique_pert=n_unique_pert, condition_key=condition_key)

    # 3. Normalize true expression to TARGET_LIB (same scale as pred_expr)
    X_true = to_dense(adata.X).astype(np.float32)
    lib_size = X_true.sum(1, keepdims=True)
    X_true_norm = X_true / (lib_size + 1e-8) * TARGET_LIB
    adata.layers['true_norm'] = X_true_norm

    # 4. Control baseline + true pseudobulk (full dataset)
    if full_data_path is not None and os.path.exists(full_data_path):
        full_adata = sc.read_h5ad(full_data_path)
        full_X = to_dense(full_adata.X).astype(np.float32)
        full_lib = full_X.sum(1, keepdims=True)
        full_X_norm = full_X / (full_lib + 1e-8) * TARGET_LIB
        full_adata.layers['full_norm'] = full_X_norm
        # ctrl mean
        full_conds = np.array([str(x) for x in full_adata.obs[condition_key]])
        ctrl_mask_full = np.array([c.lower() in ['control', 'ctrl', 'unperturbed', 'nan'] for c in full_conds])
        true_ctrl_mean = full_X_norm[ctrl_mask_full].mean(axis=0)
        # pert pseudobulk
        df_true_pert = get_pseudobulk(full_adata, condition_key=condition_key, layer='full_norm')
    else:
        if 'X_ctrl_real' not in adata.layers:
            raise ValueError(f"layers['X_ctrl_real'] not found, check test set.")
        X_ctrl = to_dense(adata.layers['X_ctrl_real']).astype(np.float32)
        ctrl_lib = X_ctrl.sum(1, keepdims=True)
        true_ctrl_mean = np.mean(X_ctrl / (ctrl_lib + 1e-8) * TARGET_LIB, axis=0)
        df_true_pert = get_pseudobulk(adata, condition_key=condition_key, layer='true_norm')

    # 5. Compute predicted pseudobulk (test set)
    df_pred_pert = get_pseudobulk(adata, condition_key=condition_key, layer='pred_expr')

    # 5. Compute deltas (Δ = perturbed mean - control mean)
    common_conds = df_true_pert.index.intersection(df_pred_pert.index)
    if len(common_conds) == 0:
        raise ValueError("No common perturbation conditions found, evaluation failed.")

    delta_true = df_true_pert.loc[common_conds].sub(true_ctrl_mean, axis=1)
    delta_pred = df_pred_pert.loc[common_conds].sub(true_ctrl_mean, axis=1)

    # 6. Computing evaluation metrics
    delta_pccs = calculate_delta_pcc(delta_true, delta_pred)
    mean_systema_score, systema_scores = calculate_systema_score(delta_true, delta_pred)
    raw_pccs = calculate_delta_pcc(df_true_pert.loc[common_conds], df_pred_pert.loc[common_conds])

    # Top20 DEG metrics
    top20_delta_pccs = calculate_top_deg_pcc(delta_true, delta_pred, top_n=20)
    top20_raw_pccs = calculate_top_deg_raw_pcc(
        df_true_pert.loc[common_conds], df_pred_pert.loc[common_conds],
        delta_true, top_n=20
    )

    # 7. Print and save results
    print("\n" + "=" * 50)
    print(f"📊 Final evaluation summary")
    print("=" * 50)
    print(f"Number of perturbations evaluated: {len(common_conds)}")
    print("-" * 40)
    print(f"1. Raw PCC (Pred vs True):            {raw_pccs.mean():.4f}")
    print("-" * 40)
    print(f"2. ΔPCC (Delta_Pred vs Delta_True):    {delta_pccs.mean():.4f}")
    print("-" * 40)
    print(f"3. Top20 DEG Raw PCC:                  {top20_raw_pccs.mean():.4f}")
    print("-" * 40)
    print(f"4. Top20 DEG delta PCC:                     {top20_delta_pccs.mean():.4f}")
    print("-" * 40)
    print(f"5. Systema Score (Normalized Rank):   {mean_systema_score:.4f}")
    print(f"   (1.0 = perfect rank-1 match; 0.5 = random level)")
    print("=" * 50)

    save_path = h5ad_path.replace('.h5ad', '_evaluation_metrics.csv')
    res_df = pd.DataFrame({
        'Delta_PCC': delta_pccs,
        'Systema_Score': systema_scores,
        'Raw_PCC': raw_pccs,
        'Top20_DEG_Raw_PCC': top20_raw_pccs,
        'Top20_DEG_Delta_PCC': top20_delta_pccs
    })
    # res_df.to_csv(save_path)
    # print(f"    📁 Detailed metrics saved to: {save_path}\n")


if __name__ == "__main__":
    # Test set path to evaluate
    TEST_FILE_PATH = "./output/test_res.h5ad"

    # VAE checkpoint path used for this latent space
    VAE_CKPT_PATH = "./output/maccf_vae/best-model.ckpt"

    evaluate_scprism(
        h5ad_path=TEST_FILE_PATH,
        vae_ckpt_path=VAE_CKPT_PATH,
        condition_key='perturbation'
    )

    # AdamsonWeissman2016_GSM2406681_10X010_CRISPR