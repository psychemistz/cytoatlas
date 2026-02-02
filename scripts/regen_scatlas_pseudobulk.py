#!/usr/bin/env python3
"""
Regenerate scAtlas pseudobulk with lower min_cells threshold.
This allows more cell types to have boxplot data.
"""

import sys
sys.path.insert(0, '/vf/users/parks34/projects/2secactpy/scripts')

import numpy as np
import pandas as pd
import anndata as ad
import h5py
from pathlib import Path
from datetime import datetime

# Import from existing script
from secactpy import load_cytosig, load_secact

# Paths
COUNTS_FILE = '/data/Jiang_Lab/Data/Seongyong/scAtlas_2025/igt_s9_fine_counts.h5ad'
OUTPUT_DIR = Path('/data/parks34/projects/2secactpy/results/scatlas')

MIN_CELLS = 10  # Lower threshold to include more cell types


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def aggregate_pseudobulk(adata, tissue_col='tissue', celltype_col='cellType1',
                         sample_col='donorID', min_cells=10):
    """
    Aggregate cells into pseudobulk samples.
    Groups by (tissue, celltype, sample) and sums counts.
    """
    log(f"Aggregating pseudobulk (min_cells={min_cells})...")

    # Create group labels
    obs = adata.obs[[tissue_col, celltype_col, sample_col]].copy()
    obs['group'] = (obs[sample_col].astype(str) + '-' +
                    obs[tissue_col].astype(str))

    # Count cells per (celltype, group)
    obs['ct_group'] = obs[celltype_col].astype(str) + '|' + obs['group'].astype(str)
    group_counts = obs.groupby('ct_group').size()

    # Filter to groups with enough cells
    valid_groups = group_counts[group_counts >= min_cells].index.tolist()
    log(f"  Found {len(valid_groups)} valid (celltype, sample) combinations (>={min_cells} cells)")

    # Aggregate
    results = {}
    meta_rows = []

    for ct_group in valid_groups:
        ct, grp = ct_group.split('|', 1)
        mask = (obs['ct_group'] == ct_group).values

        # Sum counts
        if hasattr(adata.X, 'toarray'):
            counts = np.array(adata.X[mask].sum(axis=0)).flatten()
        else:
            counts = adata.X[mask].sum(axis=0)

        sample_name = f"{ct}|{grp}"
        results[sample_name] = counts

        # Get metadata
        idx = np.where(mask)[0][0]
        tissue = obs[tissue_col].iloc[idx]

        meta_rows.append({
            'sample': sample_name,
            'cell_type': ct,
            'tissue': tissue,
            'n_cells': mask.sum()
        })

    # Create expression DataFrame
    expr_df = pd.DataFrame(results, index=adata.var_names)
    meta_df = pd.DataFrame(meta_rows)

    log(f"  Created pseudobulk: {expr_df.shape[1]} samples x {expr_df.shape[0]} genes")

    return expr_df, meta_df


def compute_and_save_activity(expr_df, meta_df, sig_matrix, sig_type, output_path):
    """Compute activity and save as h5ad."""
    log(f"Computing {sig_type} activity...")

    # Normalize to CPM and log-transform
    cpm = expr_df / expr_df.sum(axis=0) * 1e6
    log_cpm = np.log1p(cpm)

    # Align genes
    common_genes = log_cpm.index.intersection(sig_matrix.index)
    log(f"  Common genes: {len(common_genes)}")

    X = log_cpm.loc[common_genes].values.T  # samples x genes
    S = sig_matrix.loc[common_genes].values  # genes x signatures

    # Ridge regression
    from numpy.linalg import lstsq
    alpha = 1e-4
    n_genes = S.shape[0]
    activity = lstsq(S.T @ S + alpha * np.eye(S.shape[1]), S.T @ X.T, rcond=None)[0].T

    # Z-score normalize
    activity = (activity - activity.mean(axis=0)) / (activity.std(axis=0) + 1e-8)

    # Create AnnData
    adata = ad.AnnData(
        X=activity,
        obs=meta_df.set_index('sample'),
        var=pd.DataFrame(index=sig_matrix.columns)
    )

    # Save
    adata.write_h5ad(output_path)
    log(f"  Saved: {output_path}")

    return adata


def main():
    log("=== Regenerating scAtlas Normal Pseudobulk ===")
    log(f"min_cells threshold: {MIN_CELLS}")

    # Load counts
    log(f"\nLoading counts: {COUNTS_FILE}")
    adata = ad.read_h5ad(COUNTS_FILE, backed='r')
    log(f"  Shape: {adata.shape}")

    # Load into memory for aggregation (subset if needed)
    log("Loading data into memory...")
    adata = adata.to_memory()

    # Aggregate
    expr_df, meta_df = aggregate_pseudobulk(
        adata,
        tissue_col='tissue',
        celltype_col='cellType1',
        sample_col='donorID',
        min_cells=MIN_CELLS
    )

    # Check Progenitor
    prog_count = meta_df[meta_df['cell_type'] == 'Progenitor'].shape[0]
    log(f"\nProgenitor samples after aggregation: {prog_count}")

    del adata

    # Load signature matrices
    log("\nLoading signature matrices...")
    cytosig = load_cytosig()
    secact = load_secact()
    log(f"  CytoSig: {cytosig.shape}")
    log(f"  SecAct: {secact.shape}")

    # Compute and save activities
    output_cytosig = OUTPUT_DIR / 'scatlas_normal_CytoSig_pseudobulk.h5ad'
    output_secact = OUTPUT_DIR / 'scatlas_normal_SecAct_pseudobulk.h5ad'

    compute_and_save_activity(expr_df, meta_df, cytosig, 'CytoSig', output_cytosig)
    compute_and_save_activity(expr_df, meta_df, secact, 'SecAct', output_secact)

    log("\n=== Done ===")
    log(f"Now run 07f_boxplot_data.py to regenerate boxplot data")


if __name__ == '__main__':
    main()
