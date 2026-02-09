#!/usr/bin/env python3
"""
Create Symbol-Indexed H5AD Files for Inflammation Atlas
========================================================

Creates new H5AD files with gene symbols as var_names instead of Ensembl IDs.
This allows direct compatibility with signature matrices that use gene symbols.

The original files use Ensembl IDs as var_names but have symbols in var['symbol'].

Usage:
    python create_inflammation_symbol_h5ad.py --cohort main
    python create_inflammation_symbol_h5ad.py --cohort val
    python create_inflammation_symbol_h5ad.py --cohort ext
    python create_inflammation_symbol_h5ad.py --all
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import anndata as ad

# Input paths (original Ensembl-indexed files)
INPUT_PATHS = {
    'main': '/data/Jiang_Lab/Data/Seongyong/Inflammation_Atlas/INFLAMMATION_ATLAS_main_afterQC.h5ad',
    'val': '/data/Jiang_Lab/Data/Seongyong/Inflammation_Atlas/INFLAMMATION_ATLAS_validation_afterQC.h5ad',
    'ext': '/data/Jiang_Lab/Data/Seongyong/Inflammation_Atlas/INFLAMMATION_ATLAS_external_afterQC.h5ad',
}

# Output directory
OUTPUT_DIR = Path('/data/Jiang_Lab/Data/Seongyong/Inflammation_Atlas/symbol_indexed')


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def convert_to_symbol_index(input_path: str, output_path: Path, chunk_size: int = 100000):
    """
    Convert an Ensembl-indexed H5AD to symbol-indexed.

    Reads the file in chunks to manage memory for large files.
    """
    log(f"Input: {input_path}")
    log(f"Output: {output_path}")

    # First, read just the var to build the gene name mapping
    log("Loading gene metadata...")
    adata_backed = ad.read_h5ad(input_path, backed='r')

    n_cells = adata_backed.n_obs
    n_genes = adata_backed.n_vars
    log(f"  Shape: {n_cells:,} cells × {n_genes:,} genes")

    # Get original var DataFrame
    var_df = adata_backed.var.copy()
    original_names = list(adata_backed.var_names)

    # Check if already symbol-indexed
    if not original_names[0].startswith('ENSG'):
        log("  Already has gene symbols as index, skipping")
        adata_backed.file.close()
        return

    # Get symbols
    if 'symbol' not in var_df.columns:
        raise ValueError("No 'symbol' column found in var")

    symbols = var_df['symbol'].tolist()

    # Handle missing/NA symbols - keep Ensembl ID
    new_names = []
    for i, (ensembl, symbol) in enumerate(zip(original_names, symbols)):
        if pd.isna(symbol) or symbol == '' or symbol is None:
            new_names.append(ensembl)
        else:
            new_names.append(str(symbol))

    # Handle duplicates by adding suffix
    name_counts = {}
    unique_names = []
    for name in new_names:
        if name in name_counts:
            name_counts[name] += 1
            unique_names.append(f"{name}_{name_counts[name]}")
        else:
            name_counts[name] = 0
            unique_names.append(name)

    n_duplicates = sum(1 for v in name_counts.values() if v > 0)
    n_unmapped = sum(1 for n in unique_names if n.startswith('ENSG'))
    log(f"  Gene name conversion: {n_genes - n_unmapped} mapped, {n_unmapped} kept as Ensembl, {n_duplicates} duplicates renamed")

    # Store original Ensembl IDs in var
    var_df['ensembl_id'] = original_names
    var_df.index = unique_names

    # Get obs DataFrame
    obs_df = adata_backed.obs.copy()

    # Close backed file before reading full data
    adata_backed.file.close()

    # Now read the full file (this loads into memory)
    log("Loading full expression matrix (this may take a while)...")
    adata = ad.read_h5ad(input_path)

    # Update var_names
    adata.var_names = unique_names
    adata.var = var_df

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save with compression
    log("Saving symbol-indexed H5AD...")
    adata.write_h5ad(output_path, compression='gzip')

    file_size = output_path.stat().st_size / 1e9
    log(f"  Saved: {output_path} ({file_size:.2f} GB)")

    # Verify
    log("Verifying output...")
    adata_check = ad.read_h5ad(output_path, backed='r')
    log(f"  Shape: {adata_check.n_obs:,} cells × {adata_check.n_vars:,} genes")
    log(f"  First 5 genes: {list(adata_check.var_names[:5])}")
    log(f"  Has ensembl_id column: {'ensembl_id' in adata_check.var.columns}")
    adata_check.file.close()

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Create symbol-indexed inflammation H5AD files")
    parser.add_argument('--cohort', choices=['main', 'val', 'ext'], help='Which cohort to process')
    parser.add_argument('--all', action='store_true', help='Process all cohorts')
    parser.add_argument('--output-dir', type=str, default=str(OUTPUT_DIR), help='Output directory')

    args = parser.parse_args()

    if not args.cohort and not args.all:
        parser.print_help()
        sys.exit(1)

    output_dir = Path(args.output_dir)

    cohorts = ['main', 'val', 'ext'] if args.all else [args.cohort]

    log("=" * 70)
    log("Create Symbol-Indexed Inflammation H5AD Files")
    log("=" * 70)
    log(f"Output directory: {output_dir}")
    log(f"Cohorts: {', '.join(cohorts)}")
    log("")

    results = {}
    for cohort in cohorts:
        log("=" * 70)
        log(f"Processing: {cohort}")
        log("=" * 70)

        input_path = INPUT_PATHS[cohort]
        output_path = output_dir / f"INFLAMMATION_ATLAS_{cohort}_symbol.h5ad"

        try:
            convert_to_symbol_index(input_path, output_path)
            results[cohort] = ('success', output_path)
        except Exception as e:
            log(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            results[cohort] = ('failed', str(e))

        log("")

    # Summary
    log("=" * 70)
    log("SUMMARY")
    log("=" * 70)
    for cohort, (status, info) in results.items():
        if status == 'success':
            log(f"  ✓ {cohort}: {info}")
        else:
            log(f"  ✗ {cohort}: {info}")


if __name__ == "__main__":
    main()
