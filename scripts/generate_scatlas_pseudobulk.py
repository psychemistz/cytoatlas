#!/usr/bin/env python3
"""
Generate scAtlas Pseudobulk Activity Data.

Creates pseudobulk aggregations from scAtlas single-cell data to enable
cross-atlas comparisons with CIMA and Inflammation Atlas.

Pseudobulk aggregation: cell type × sample
- Filter to Immune compartment for comparability with CIMA/Inflammation
- Use cellType1 for cell type grouping (468 types -> mapped to coarse/fine)
- Aggregate expression by sample (706 samples)
- Compute CytoSig and SecAct activity scores

Output format matches CIMA/Inflammation pseudobulk h5ad:
- obs: signatures (index = signature names)
- var: samples (with 'cell_type', 'sample', 'tissue', 'n_cells' columns)
- X: activity matrix (signatures × samples)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse
import warnings
warnings.filterwarnings('ignore')

# Add secactpy to path
sys.path.insert(0, '/vf/users/parks34/projects/1ridgesig/SecActpy')
from secactpy import load_cytosig, load_secact, ridge

# Paths
SCATLAS_NORMAL = Path('/data/Jiang_Lab/Data/Seongyong/scAtlas_2025/igt_s9_fine_counts.h5ad')
SCATLAS_CANCER = Path('/data/Jiang_Lab/Data/Seongyong/scAtlas_2025/PanCancer_igt_s9_fine_counts.h5ad')
OUTPUT_DIR = Path('/data/parks34/projects/2secactpy/results/scatlas')

# Minimum cells per pseudobulk sample
MIN_CELLS = 10


def load_signature_matrix(sig_type: str):
    """Load signature matrix."""
    if sig_type == 'CytoSig':
        return load_cytosig()
    else:
        return load_secact()


def create_pseudobulk(adata, groupby_cols=['cellType1', 'sampleID'], min_cells=MIN_CELLS):
    """
    Create pseudobulk by summing expression within groups.

    Args:
        adata: AnnData object with raw counts
        groupby_cols: Columns to group by
        min_cells: Minimum cells per group

    Returns:
        Pseudobulk AnnData with groups as obs
    """
    print(f"  Creating pseudobulk by {groupby_cols}...")

    # Get grouping
    obs = adata.obs
    groups = obs.groupby(groupby_cols, observed=True)

    # Filter groups with sufficient cells
    group_sizes = groups.size()
    valid_groups = group_sizes[group_sizes >= min_cells].index
    print(f"    Valid groups (>={min_cells} cells): {len(valid_groups)}")

    # Aggregate expression
    pb_data = []
    pb_obs = []

    for i, group_key in enumerate(valid_groups):
        if i % 500 == 0:
            print(f"    Processing group {i}/{len(valid_groups)}...")

        # Get cell indices for this group
        if len(groupby_cols) == 2:
            mask = (obs[groupby_cols[0]] == group_key[0]) & (obs[groupby_cols[1]] == group_key[1])
        else:
            mask = obs[groupby_cols[0]] == group_key

        idx = np.where(mask)[0]
        n_cells = len(idx)

        # Sum expression (mean would lose count information)
        X_subset = adata.X[idx, :]
        if sparse.issparse(X_subset):
            X_sum = np.array(X_subset.sum(axis=0)).flatten()
        else:
            X_sum = X_subset.sum(axis=0)

        # Normalize to CPM-like (sum to 1M per pseudobulk)
        total = X_sum.sum()
        if total > 0:
            X_norm = (X_sum / total) * 1e6
        else:
            X_norm = X_sum

        pb_data.append(X_norm)

        # Store metadata
        sample_obs = obs[mask].iloc[0]
        pb_obs.append({
            'cell_type': group_key[0] if len(groupby_cols) == 2 else group_key,
            'sample': group_key[1] if len(groupby_cols) == 2 else 'all',
            'tissue': sample_obs.get('tissue', 'Unknown'),
            'compartment': sample_obs.get('compartment', 'Unknown'),
            'n_cells': n_cells
        })

    # Create AnnData
    X_pb = np.vstack(pb_data)
    obs_df = pd.DataFrame(pb_obs)
    obs_df.index = [f"{row['cell_type']}_{row['sample']}" for _, row in obs_df.iterrows()]

    pb_adata = ad.AnnData(
        X=X_pb,
        obs=obs_df,
        var=adata.var.copy()
    )

    print(f"    Pseudobulk shape: {pb_adata.shape}")
    return pb_adata


def normalize_and_transform(expr_df: pd.DataFrame) -> pd.DataFrame:
    """TPM normalize and log2 transform expression data (matching CIMA pipeline)."""
    # TPM normalize (sum to 1M per sample)
    col_sums = expr_df.sum(axis=0)
    expr_tpm = expr_df / col_sums * 1e6

    # Log2 transform
    expr_log = np.log2(expr_tpm + 1)

    return expr_log


def compute_differential(expr_df: pd.DataFrame) -> pd.DataFrame:
    """Compute differential expression (subtract row mean) - centers genes across samples."""
    row_means = expr_df.mean(axis=1)
    diff = expr_df.subtract(row_means, axis=0)
    return diff


def compute_activity(pb_adata, sig_matrix, sig_type: str):
    """
    Compute activity scores for pseudobulk data.

    IMPORTANT: This matches the CIMA/Inflammation pipeline:
    1. TPM normalize
    2. Log2 transform
    3. Center genes (subtract row mean)
    4. Z-score normalize before ridge regression

    Args:
        pb_adata: Pseudobulk AnnData (samples × genes)
        sig_matrix: Signature matrix (genes × signatures)
        sig_type: 'CytoSig' or 'SecAct'

    Returns:
        Activity AnnData (signatures × samples) matching CIMA/Inflammation format
    """
    print(f"  Computing {sig_type} activity...")

    # Get overlapping genes (case-insensitive matching)
    pb_genes_upper = {g.upper(): g for g in pb_adata.var_names}
    sig_genes_upper = {g.upper(): g for g in sig_matrix.index}
    common_genes_upper = set(pb_genes_upper.keys()) & set(sig_genes_upper.keys())

    print(f"    Common genes: {len(common_genes_upper)} / {len(sig_matrix.index)} ({100*len(common_genes_upper)/len(sig_matrix.index):.1f}%)")

    if len(common_genes_upper) < 100:
        print(f"    [WARN] Too few common genes, skipping")
        return None

    # Map to original gene names
    pb_common = [pb_genes_upper[g] for g in sorted(common_genes_upper)]
    sig_common = [sig_genes_upper[g] for g in sorted(common_genes_upper)]

    # Subset to common genes
    pb_subset = pb_adata[:, pb_common].copy()
    sig_subset = sig_matrix.loc[sig_common].copy()

    # Make indices consistent (uppercase)
    pb_subset.var_names = [g.upper() for g in pb_subset.var_names]
    sig_subset.index = [g.upper() for g in sig_subset.index]
    common_genes = sorted(common_genes_upper)

    # Get expression matrix (samples × genes)
    X = pb_subset.X
    if sparse.issparse(X):
        X = X.toarray()

    # Create DataFrame for processing (genes × samples for consistency with CIMA pipeline)
    expr_df = pd.DataFrame(X.T, index=common_genes, columns=pb_subset.obs_names)

    # Apply CIMA preprocessing pipeline:
    # 1. TPM normalize and log2 transform
    print(f"    Normalizing (TPM + log2)...")
    expr_log = normalize_and_transform(expr_df)

    # 2. Compute differential (center genes)
    print(f"    Computing differential (centering genes)...")
    expr_diff = compute_differential(expr_log)

    # 3. Z-score columns (samples)
    print(f"    Z-scoring expression...")
    expr_scaled = (expr_diff - expr_diff.mean()) / expr_diff.std(ddof=1)
    expr_scaled = expr_scaled.fillna(0)

    # 4. Z-score signature matrix
    print(f"    Z-scoring signature matrix...")
    sig_aligned = sig_subset.loc[common_genes]
    sig_scaled = (sig_aligned - sig_aligned.mean()) / sig_aligned.std(ddof=1)
    sig_scaled = sig_scaled.fillna(0)

    # Ridge regression expects:
    # X: signature matrix (genes × signatures)
    # Y: expression data (genes × samples)
    # Returns: beta, zscore (signatures × samples)

    print(f"    Running ridge regression with permutation testing...")
    print(f"      Signature matrix: {sig_scaled.shape}")
    print(f"      Expression matrix: {expr_scaled.shape}")

    result = ridge(
        X=sig_scaled.values,      # genes × signatures
        Y=expr_scaled.values,     # genes × samples (already transposed)
        lambda_=5e5,
        n_rand=1000,              # Permutation testing for z-scores (matching CIMA)
        seed=0,
        verbose=True
    )

    # zscore is signatures × samples (this is what CIMA stores in X)
    activity = result['zscore']

    # Debug: check activity range
    print(f"    Activity (zscore) stats: min={activity.min():.3f}, max={activity.max():.3f}, mean={activity.mean():.3f}, std={activity.std():.3f}")

    # Create activity AnnData (matching CIMA format)
    # X = zscore, layers = beta, se, pvalue
    # obs = signatures, var = samples
    act_adata = ad.AnnData(
        X=activity,  # zscore: signatures × samples
        obs=pd.DataFrame(index=sig_scaled.columns),  # Signatures
        var=pb_adata.obs.copy()  # Sample metadata
    )

    # Store additional result matrices in layers (matching CIMA)
    act_adata.layers['beta'] = result['beta']
    act_adata.layers['se'] = result['se']
    act_adata.layers['pvalue'] = result['pvalue']

    # Store metadata
    act_adata.uns['signature'] = sig_type
    act_adata.uns['n_rand'] = 1000
    act_adata.uns['seed'] = 0

    print(f"    Activity shape: {act_adata.shape} (signatures × samples)")
    return act_adata


def process_atlas(input_path: Path, output_prefix: str, compartment_filter: str = 'Immune'):
    """
    Process an atlas to create pseudobulk activity data.

    Args:
        input_path: Path to h5ad file
        output_prefix: Prefix for output files
        compartment_filter: Filter to specific compartment (or 'all')
    """
    print(f"\n{'='*60}")
    print(f"Processing: {input_path.name}")
    print(f"{'='*60}")

    # Load data
    print("Loading data...")
    adata = ad.read_h5ad(input_path, backed='r')
    print(f"  Total cells: {adata.n_obs:,}")

    # Filter to compartment
    if compartment_filter != 'all':
        mask = adata.obs['compartment'] == compartment_filter
        n_cells = mask.sum()
        print(f"  {compartment_filter} cells: {n_cells:,}")

        # Load filtered data into memory
        print("  Loading filtered cells into memory...")
        idx = np.where(mask)[0]

        # Sample if too large
        max_cells = 500000
        if len(idx) > max_cells:
            print(f"  Sampling {max_cells:,} cells for memory efficiency...")
            np.random.seed(42)
            idx = np.random.choice(idx, max_cells, replace=False)
            idx = sorted(idx)

        X = adata.X[idx, :]
        if sparse.issparse(X):
            X = X.toarray()

        obs = adata.obs.iloc[idx].copy()
        var = adata.var.copy()

        adata_filtered = ad.AnnData(X=X, obs=obs, var=var)
        print(f"  Loaded shape: {adata_filtered.shape}")
    else:
        adata_filtered = adata

    # Create pseudobulk
    pb_adata = create_pseudobulk(adata_filtered, groupby_cols=['cellType1', 'sampleID'])

    # Compute activity for both signature types
    for sig_type in ['CytoSig', 'SecAct']:
        print(f"\n  --- {sig_type} ---")
        sig_matrix = load_signature_matrix(sig_type)

        act_adata = compute_activity(pb_adata, sig_matrix, sig_type)

        if act_adata is not None:
            # Save
            output_path = OUTPUT_DIR / f"{output_prefix}_{sig_type}_pseudobulk.h5ad"
            act_adata.write_h5ad(output_path)
            print(f"  Saved: {output_path}")
            print(f"    Shape: {act_adata.shape} (signatures × samples)")
            print(f"    Cell types: {act_adata.var['cell_type'].nunique()}")


def main():
    print("=" * 70)
    print("Generating scAtlas Pseudobulk Activity Data")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Process normal atlas (immune cells only for cross-atlas comparison)
    if SCATLAS_NORMAL.exists():
        process_atlas(SCATLAS_NORMAL, 'scatlas_normal', compartment_filter='Immune')
    else:
        print(f"[WARN] Normal atlas not found: {SCATLAS_NORMAL}")

    # Optionally process cancer atlas
    # if SCATLAS_CANCER.exists():
    #     process_atlas(SCATLAS_CANCER, 'scatlas_cancer', compartment_filter='Immune')

    print("\n" + "=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == '__main__':
    main()
