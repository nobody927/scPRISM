import os
import gc
import re
import pickle
import argparse
import scanpy as sc
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, issparse
from concurrent.futures import ProcessPoolExecutor, as_completed

# ================= Configuration paths =================
INPUT_DIR = './data/raw'
OUTPUT_BASE = './data/processed/'
GENE_EMBEDDING_PATH = './data/GenePT_gene_embedding_ada_text.pickle'

DIR_GENE = os.path.join(OUTPUT_BASE, 'gene')
DIR_DRUG = os.path.join(OUTPUT_BASE, 'drug')
DIR_PROCESSED = os.path.join(OUTPUT_BASE, 'processed')

for d in [DIR_GENE, DIR_DRUG, DIR_PROCESSED]:
    os.makedirs(d, exist_ok=True)


# ================= Utility functions =================
def load_gene2vec_genes(pickle_path):
    with open(pickle_path, 'rb') as f:
        gene_map = pickle.load(f)
    return set(gene_map.keys())


def clean_perturbation_name(val, embedding_genes_upper=None):
    """
    Clean perturbation names, extract gene names.
    Supported formats:
      - Gene name: 'ATF4_pBA576' → 'ATF4'
      - Combination: 'ATF6_IRE1_pMJ152' → 'ATF6+IRE1'
      - TCR library: 'Tcrlibrary_BACH2_3' → 'BACH2'
      - Gene_only: 'ATF6_only_pMJ145' → 'ATF6'
      - Underscore-separated: 'ATP5A1_MAT2A' → 'ATP5A1+MAT2A'
    """
    if pd.isna(val) or val == '*' or str(val).lower() == 'nan':
        return 'control'

    val_str = str(val)
    val_lower = val_str.lower()

    # Control detection
    if '(mod)' in val_lower or 'control' in val_lower or 'ctrl' in val_lower or 'ntc' in val_lower:
        return 'control'

    # If entire value is in embedding, return directly (e.g., 'AAAS')
    if embedding_genes_upper and val_str.upper() in embedding_genes_upper:
        return val_str.upper()

    # Split by '_'
    parts = val_str.split('_')

    # Filter non-gene parts
    gene_parts = []
    plasmid_pattern = re.compile(r'^p[A-Za-z]+\d+(-\d+)?$')

    for p in parts:
        if plasmid_pattern.match(p):        # Plasmid ID: pMJ144, pBA576
            continue
        if p.lower() == 'only':             # "only" suffix
            continue
        if p.isdigit():                     # Pure number (guide ID): 1, 2, 3
            continue
        if re.match(r'^\d+(\.\d+)?$', p):   # Number (with decimal): 1784725.23
            continue
        if p in ['+', '-']:                 # Strand direction
            continue
        if any(c in p for c in ';:/,'):     # Coordinate or other non-gene annotation
            continue
        if p.lower().startswith('chr'):     # Chromosome coordinate
            continue
        if len(p) > 30:                     # Overlong string
            continue
        gene_parts.append(p)

    if not gene_parts:
        return 'control'

    # Check if each gene is in embedding (optional)
    if embedding_genes_upper:
        valid_genes = []
        for g in gene_parts:
            if g.upper() in embedding_genes_upper:
                valid_genes.append(g.upper())
            # Genes not in embedding are discarded
        if not valid_genes:
            return 'control'
        return '+'.join(valid_genes)

    return '+'.join(gene_parts)


def filter_perts_by_gene_embedding(adata, embedding_genes_upper):
    """Remove perturbations without gene embedding. Combination perturbations require all genes in embedding."""
    if 'perturbation' not in adata.obs:
        return adata

    n_before = adata.shape[0]

    def all_genes_have_embedding(pert):
        if pd.isna(pert):
            return True
        pert_str = str(pert).lower()
        if pert_str in ['control', 'nan', '*']:
            return True
        parts = [p.strip() for p in str(pert).split('+')]
        for part in parts:
            if part.upper() not in embedding_genes_upper:
                return False
        return True

    mask = adata.obs['perturbation'].apply(all_genes_have_embedding)
    mask = mask.fillna(False).astype(bool)
    adata = adata[mask].copy()
    n_after = adata.shape[0]
    print(f"  -> Gene embedding filter: {n_before} → {n_after} cells ({n_before - n_after}  cells removed)")
    return adata


def filter_small_perturbations(adata, min_ratio=0.01, min_cells=100):
    """Filter small-sample perturbations: must meet both <1% total cells AND <100 cells"""
    if 'perturbation' not in adata.obs:
        return adata

    n_total = adata.shape[0]
    threshold = max(min_cells, min_ratio * n_total)  # Take larger value to ensure both conditions met

    # Logic: filter perturbations that are both <1% AND <100 cells
    # i.e., keep perturbations with >= 1% OR >= 100 cells
    pert_counts = adata.obs['perturbation'].value_counts()
    small_perts = pert_counts[(pert_counts < min_ratio * n_total) & (pert_counts < min_cells)].index.tolist()

    if not small_perts:
        print(f"  -> Small-sample filter: no filtering needed (threshold: <{min_ratio*100:.0f}% and <{min_cells} cells)")
        return adata

    mask = ~adata.obs['perturbation'].isin(small_perts)
    n_before = adata.shape[0]
    adata = adata[mask].copy()
    n_after = adata.shape[0]
    print(f"  -> Small-sample filter: {n_before} → {n_after} cells (removed {len(small_perts)}  perturbations, {n_before - n_after}  cells)")
    return adata


def optimize_adata(adata):
    if not issparse(adata.X):
        adata.X = csr_matrix(adata.X)
    elif adata.X.format != 'csr':
        adata.X = adata.X.tocsr()

    for col in adata.obs.columns:
        if adata.obs[col].dtype == 'object':
            adata.obs[col] = adata.obs[col].astype('category')

    return adata


def get_gene_embedding_perts():
    """Get perturbation names that exist in gene embedding"""
    if not os.path.exists(GENE_EMBEDDING_PATH):
        print(f"  ⚠️ gene embedding file not found: {GENE_EMBEDDING_PATH}")
        return None
    with open(GENE_EMBEDDING_PATH, 'rb') as f:
        gene_map = pickle.load(f)
    # Return uppercase for case-insensitive matching
    return {g.upper() for g in gene_map.keys()}


def ensure_integer_counts(adata):
    if issparse(adata.X):
        data = adata.X.data
    else:
        data = adata.X

    if not np.all(data == np.floor(data)):
        print("  -> Non-integer expression matrix detected, rounding for seurat_v3...")
        if issparse(adata.X):
            adata.X.data = np.round(data).astype(np.float32)
        else:
            adata.X = np.round(adata.X).astype(np.float32)
    return adata


# ================= Single file processing =================
def process_single_file(file, embedding_genes_upper):
    """Process single h5ad file, return (filename,  success/failed, message)"""
    file_path = os.path.join(INPUT_DIR, file)
    try:
        adata = sc.read_h5ad(file_path)
    except Exception as e:
        return (file, False, f"Read failed: {e}")

    adata.var_names_make_unique()

    # --- 1. Metadata cleaning ---
    if 'perturbation' in adata.obs:
        adata.obs['original_perturbation'] = adata.obs['perturbation']
        adata.obs['perturbation'] = adata.obs['perturbation'].apply(
            lambda v: clean_perturbation_name(v, embedding_genes_upper))

    # --- 2. Quality control ---
    sc.pp.filter_cells(adata, min_counts=200)
    sc.pp.filter_genes(adata, min_cells=3)
    adata.var_names_make_unique()

    # --- 3. Determine perturbation type ---
    # Gene perturbation: CRISPR, CRISPR-cas9, CRISPR-cas13, CRISPRi
    # Drug perturbation: drug, cytokine, cytokines
    GENE_PERT_TYPES = {'CRISPR', 'CRISPR-CAS9', 'CRISPR-CAS13', 'CRISPRI'}
    DRUG_PERT_TYPES = {'DRUG', 'CYTOKINE', 'CYTOKINES'}

    is_gene_perturbation = False
    is_drug_perturbation = False
    if 'perturbation_type' in adata.obs:
        p_types = set(str(t).upper() for t in adata.obs['perturbation_type'].unique())
        if p_types & GENE_PERT_TYPES:
            is_gene_perturbation = True
        if p_types & DRUG_PERT_TYPES:
            is_drug_perturbation = True

    # --- 4. Skip non-gene perturbation datasets ---
    if not is_gene_perturbation:
        return (file, True, "Non-gene perturbation dataset, skipping")

    # --- 4b. Gene perturbation: remove perturbations without embedding ---
    if embedding_genes_upper is not None:
        adata = filter_perts_by_gene_embedding(adata, embedding_genes_upper)

    # --- 4.5. Filter small-sample perturbations (<1% total cell count and <100 cells) ---
    adata = filter_small_perturbations(adata, min_ratio=0.01, min_cells=100)

    # --- 5. HVG selection (ensure perturbation target genes included) ---
    n_top = min(5000, adata.shape[1])

    # Extract all perturbation target genes
    pert_target_genes = set()
    if 'perturbation' in adata.obs:
        for pert in adata.obs['perturbation'].unique():
            pert_str = str(pert).lower()
            if pert_str in ['control', 'nan', '*']:
                continue
            for g in pert_str.split('+'):
                g = g.strip()
                if g and g not in ['control', 'nan']:
                    pert_target_genes.add(g.upper())

    # Select HVGs
    try:
        adata = ensure_integer_counts(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top, flavor='seurat_v3')
        hvg_genes = set(adata.var_names[adata.var.highly_variable])
    except Exception:
        try:
            adata_log = adata.copy()
            sc.pp.normalize_total(adata_log, target_sum=1e4)
            sc.pp.log1p(adata_log)
            sc.pp.highly_variable_genes(adata_log, n_top_genes=n_top, flavor='seurat')
            hvg_genes = set(adata.var_names[adata_log.var.highly_variable])
            del adata_log
        except Exception:
            hvg_genes = set(adata.var_names[:n_top])

    # Ensure perturbation target genes in list (case-insensitive)
    var_names_map = {g.upper(): g for g in adata.var_names}
    hvg_upper = {g.upper() for g in hvg_genes}
    missing_targets = []
    for g in pert_target_genes:
        if g not in hvg_upper:
            if g in var_names_map:
                missing_targets.append(var_names_map[g])
            else:
                print(f"  -> ⚠️ Perturbation gene {g}  not in gene list")

    # Build final list: HVG + missing target genes
    final_genes = list(hvg_genes)
    for g in missing_targets:
        if g.upper() not in {v.upper() for v in final_genes}:
            final_genes.append(g)

    # Truncate: prioritize target genes
    if len(final_genes) > n_top:
        target_in = [g for g in final_genes if g.upper() in pert_target_genes]
        non_target = [g for g in final_genes if g.upper() not in pert_target_genes]
        final_genes = target_in + non_target[:n_top - len(target_in)]

    adata = adata[:, final_genes].copy()
    print(f"  -> HVG: {len(final_genes)}  genes (incl. {len(pert_target_genes)} perturbation target genes)")

    # --- 5b. Filter perturbations with target genes not in gene list ---
    gene_set = set(g.lower() for g in adata.var_names)
    if 'perturbation' in adata.obs:
        keep_mask = np.ones(len(adata), dtype=bool)
        n_removed = 0
        for i, pert in enumerate(adata.obs['perturbation'].values):
            pert_str = str(pert).lower()
            if pert_str in ['control', 'nan', '*']:
                continue
            genes = [g.strip() for g in pert_str.split('+')]
            for g in genes:
                if g.lower() not in gene_set:
                    keep_mask[i] = False
                    n_removed += 1
                    break
        if n_removed > 0:
            adata = adata[keep_mask].copy()
            print(f"  ->  gene filter: removed {n_removed}  cells (target genes not in gene list)")

    adata.var_names_make_unique()

    # --- 6. Normalize library size ---
    sc.pp.normalize_total(adata, target_sum=1e4)

    # --- 7. Data type optimization ---
    adata = optimize_adata(adata)

    # --- 8. Statistics ---
    n_cells, n_genes = adata.shape

    # --- 9. Save ( genesperturbation) ---
    saved_paths = []
    base_name = os.path.splitext(file)[0]
    if 'perturbation_type' in adata.obs:
        for p_type in adata.obs['perturbation_type'].unique():
            p_type_upper = str(p_type).upper()
            if p_type_upper not in GENE_PERT_TYPES:
                continue
            sub_adata = adata[adata.obs['perturbation_type'] == p_type].copy()
            save_path = os.path.join(DIR_GENE, f"{base_name}_{p_type}.h5ad")
            sub_adata.write_h5ad(save_path)
            saved_paths.append(f"{save_path} ({sub_adata.shape[0]} cells)")
            del sub_adata
    else:
        save_path = os.path.join(DIR_GENE, f"{base_name}.h5ad")
        adata.write_h5ad(save_path)
        saved_paths.append(f"{save_path} ({adata.shape[0]} cells)")

    del adata
    gc.collect()

    return (file, True, f"genes={n_genes} | saved: {', '.join(saved_paths)}")


# ================= Main pipeline =================
def process_files(file_filter=None, n_workers=1):
    embedding_genes_upper = get_gene_embedding_perts()

    files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith('.h5ad')])
    if file_filter:
        files = [f for f in files if any(kw in f for kw in file_filter)]

    print(f"Files to process: {len(files)} , parallel: : {n_workers}")

    if n_workers <= 1:
        # Serial
        for i, file in enumerate(files, 1):
            print(f"\n[{i}/{len(files)}] {file}")
            file, ok, msg = process_single_file(file, embedding_genes_upper)
            status = "✅" if ok else "❌"
            print(f"  {status} {msg}")
    else:
        # Parallel
        results = {}
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            future_map = {
                executor.submit(process_single_file, f, embedding_genes_upper): f
                for f in files
            }
            for i, future in enumerate(as_completed(future_map), 1):
                file, ok, msg = future.result()
                status = "✅" if ok else "❌"
                print(f"[{i}/{len(files)}] {status} {file}: {msg}")
                results[file] = ok

        n_ok = sum(results.values())
        print(f"\nDone: {n_ok}/{len(files)}  success")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="scPRISM data preprocessing")
    parser.add_argument("--datasets", type=str, nargs='+', default=None,#['GasperiniShendure2019_lowMOI','DatlingerBock2017','TianKampmann2019_iPSC'],
                        help="Only process datasets with filenames containing these keywords")
    parser.add_argument("--n_workers", type=int, default=5,
                        help="Parallelworkers (default 4)")
    args = parser.parse_args()

    print("Starting data processing...")
    process_files(file_filter=args.datasets, n_workers=args.n_workers)
    print("\nAll files processed！")
