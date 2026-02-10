#!/usr/bin/env python3
"""
Fix inflammation pseudobulk files by adding gene symbol mapping.

The inflammation atlas uses Ensembl IDs as gene names, but the signature
matrices use gene symbols. This script adds the 'symbol' column to the
var DataFrame so activity inference can match genes properly.

Usage:
    python scripts/11c_fix_inflammation_pseudobulk.py

Author: Claude Code (2026-02-04)
"""

import time
from pathlib import Path

import anndata as ad
import pandas as pd


def log(msg: str) -> None:
    """Print timestamped log message."""
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {msg}", flush=True)


def get_gene_symbol_mapping(h5ad_path: str) -> pd.DataFrame:
    """
    Get gene symbol mapping from original inflammation H5AD file.

    Returns:
        DataFrame with Ensembl ID index and symbol column
    """
    log(f"Loading gene symbol mapping from: {h5ad_path}")
    adata = ad.read_h5ad(h5ad_path, backed='r')
    var_df = adata.var[['symbol']].copy()
    log(f"Loaded mapping for {len(var_df)} genes")
    return var_df


def fix_pseudobulk_file(pb_path: Path, gene_mapping: pd.DataFrame) -> None:
    """
    Add gene symbol mapping to a pseudobulk file.

    Args:
        pb_path: Path to pseudobulk H5AD file
        gene_mapping: DataFrame mapping Ensembl IDs to symbols
    """
    log(f"Processing: {pb_path.name}")

    # Load pseudobulk
    adata = ad.read_h5ad(pb_path)

    # Check if already has symbols
    if 'symbol' in adata.var.columns:
        log(f"  Already has symbol column, skipping")
        return

    # Add symbol column
    matching_genes = adata.var_names.intersection(gene_mapping.index)
    if len(matching_genes) < len(adata.var_names):
        n_missing = len(adata.var_names) - len(matching_genes)
        log(f"  Warning: {n_missing} genes not found in mapping")

    adata.var = gene_mapping.loc[adata.var_names].copy()
    log(f"  Added symbol column with {len(matching_genes)} mapped genes")

    # Save
    adata.write_h5ad(pb_path, compression='gzip')
    log(f"  Saved: {pb_path}")


def main():
    donor_level_dir = Path('/vf/users/parks34/projects/2cytoatlas/results/donor_level')

    # Atlas-specific H5AD paths for gene mapping
    atlas_h5ad = {
        'inflammation_main': '/data/Jiang_Lab/Data/Seongyong/Inflammation_Atlas/INFLAMMATION_ATLAS_main_afterQC.h5ad',
        'inflammation_val': '/data/Jiang_Lab/Data/Seongyong/Inflammation_Atlas/INFLAMMATION_ATLAS_validation_afterQC.h5ad',
        'inflammation_ext': '/data/Jiang_Lab/Data/Seongyong/Inflammation_Atlas/INFLAMMATION_ATLAS_external_afterQC.h5ad',
    }

    for atlas_name, h5ad_path in atlas_h5ad.items():
        log(f"\n{'='*70}")
        log(f"Fixing {atlas_name} pseudobulk files")
        log(f"{'='*70}")

        # Get gene mapping
        gene_mapping = get_gene_symbol_mapping(h5ad_path)

        # Find pseudobulk files
        atlas_dir = donor_level_dir / atlas_name
        if not atlas_dir.exists():
            log(f"Directory not found: {atlas_dir}")
            continue

        pb_files = list(atlas_dir.rglob('*_donor_pseudobulk.h5ad'))
        pb_files.extend(list(atlas_dir.rglob('*_donor_pseudobulk_resampled.h5ad')))

        for pb_path in sorted(pb_files):
            fix_pseudobulk_file(pb_path, gene_mapping)

    log("\n" + "=" * 70)
    log("ALL INFLAMMATION FILES FIXED")
    log("=" * 70)


if __name__ == '__main__':
    main()
