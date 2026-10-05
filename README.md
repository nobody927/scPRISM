# scPRISM: A Macro-to-Micro Counterfactual Diffusion Framework for Sin-gle-Cell Perturbation Prediction

## Overview

scPRISM is a framework for predicting single-cell gene expression responses to perturbations. It uses a two-stage macro-to-micro approach:

1. **Macro-Counterfactual VAE (MacCF-VAE)**: A VAE with an attention-based macro-shift perturbation encoder that disentangles perturbation effects from cell state in a 60D + 8D latent space.
2. **Micro-Diffusion Module (MicDiff)**: An SDEdit-based diffusion model that refines predictions in the decoupled 60D latent space, conditioned on external gene embeddings.


## Requirements

- Python >= 3.8
- PyTorch >= 1.12
- PyTorch Lightning
- scanpy, anndata
- scvi-tools (for NegativeBinomial)
- pot (Python Optimal Transport)
- umap-learn

Install dependencies:
```bash
pip install torch pytorch-lightning scanpy anndata scvi-tools pot umap-learn
```

## Data

Due to anonymity constraints, datasets will be made publicly available upon publication.

Expected data format: `.h5ad` files with:
- `X`: raw count matrix
- `obs['perturbation']`: perturbation labels (e.g., `'GENE_A'`, `'GENE_A+GENE_B'`, `'control'`)
- `var_names`: gene names

## Usage

### Step 1: Data Preprocessing

```bash
python data_process.py \
    --input_dir ./data/raw \
    --output_dir ./data/processed \
    --n_workers 1
```

### Step 2: Train VAE (MacCF-VAE)

```bash
python maccf_vae.py \
    --data ./data/processed/dataset.h5ad \
    --output ./output/maccf_vae
```

MacCF-VAE learns a disentangled 60D (cell state) + 8D (perturbation condition) latent space with:
- Multi-head attention for perturbation gene aggregation
- Adversarial training to enforce disentanglement
- Contrastive learning for perturbation discrimination

### Step 3: Train Diffusion (MicDiff)

```bash
python micdiff.py \
    --mode train \
    --dataset dataset \
    --vae_dir ./output/maccf_vae \
    --base_output_dir ./output/diffusion_sdedit \
    --pairing_mode ot \
    --noise_ratio 0.5 \
    --diffusion_steps 100 \
    --batch_size 64
```

MicDiff operates in the decoupled 60D latent space with SDEdit-style inference: starting from noised control cell latents and denoising toward perturbed states.

### Step 4: Prediction and Evaluation

```bash
python micdiff.py \
    --mode predict \
    --dataset dataset \
    --vae_dir ./output/maccf_vae \
    --base_output_dir ./output/diffusion_sdedit \
    --pairing_mode ot \
    --noise_ratio 0.5
```

### Step 5: Evaluate Results

```bash
python decode_eval.py \
    --h5ad ./output/diffusion_sdedit/noise0.5/dataset/test_res_sdedit.h5ad \
    --vae_ckpt ./output/maccf_vae/dataset/best-model.ckpt \
    --full_data ./data/processed/dataset.h5ad
```

