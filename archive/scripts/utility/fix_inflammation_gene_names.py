#!/usr/bin/env python3
"""
Fix Inflammation Pseudobulk Gene Names
======================================

Converts Ensembl IDs to gene symbols in inflammation pseudobulk files.
The original H5AD has a 'symbol' column in .var that maps Ensembl to symbols.
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad

# Paths
INFLAM_MAIN = '/data/Jiang_Lab/Data/Seongyong/Inflammation_Atlas/INFLAMMATION_ATLAS_main_afterQC.h5ad'
INFLAM_VAL = '/data/Jiang_Lab/Data/Seongyong/Inflammation_Atlas/INFLAMMATION_ATLAS_validation_afterQC.h5ad'
INFLAM_EXT = '/data/Jiang_Lab/Data/Seongyong/Inflammation_Atlas/INFLAMMATION_ATLAS_external_afterQC.h5ad'

PSEUDOBULK_ROOT = Path('/vf/users/parks34/projects/2cytoatlas/results/atlas_validation')


def load_gene_mapping(h5ad_path: str) -> dict:
    """Load Ensembl -> Symbol mapping from original H5AD."""
    print(f"Loading gene mapping from: {h5ad_path}")
    adata = ad.read_h5ad(h5ad_path, backed='r')

    # Create mapping from index (Ensembl) to symbol
    mapping = {}
    for ensembl_id, row in adata.var.iterrows():
        symbol = row.get('symbol', ensembl_id)
        if pd.notna(symbol) and symbol:
            mapping[ensembl_id] = symbol
        else:
            mapping[ensembl_id] = ensembl_id

    print(f"  Loaded {len(mapping)} gene mappings")
    return mapping


def convert_pseudobulk(input_path: Path, gene_mapping: dict, output_path: Path = None):
    """Convert pseudobulk gene names from Ensembl to symbols."""
    print(f"\nProcessing: {input_path}")

    if output_path is None:
        output_path = input_path

    # Load pseudobulk
    adata = ad.read_h5ad(input_path)
    print(f"  Shape: {adata.shape}")
    print(f"  Current gene names: {list(adata.var_names[:3])}...")

    # Check if already converted
    if not str(adata.var_names[0]).startswith('ENSG'):
        print("  Already has gene symbols, skipping")
        return

    # Map gene names
    new_names = []
    unmapped = 0
    for gene in adata.var_names:
        if gene in gene_mapping:
            new_names.append(gene_mapping[gene])
        else:
            new_names.append(gene)
            unmapped += 1

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

    # Update var_names
    adata.var_names = unique_names

    # Store original Ensembl IDs
    adata.var['ensembl_id'] = list(adata.var_names) if 'ensembl_id' not in adata.var.columns else adata.var['ensembl_id']

    print(f"  Converted: {len(new_names) - unmapped} mapped, {unmapped} unmapped, {n_duplicates} duplicates handled")
    print(f"  New gene names: {list(adata.var_names[:3])}...")

    # Save
    adata.write_h5ad(output_path, compression='gzip')
    print(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert inflammation pseudobulk gene names")
    parser.add_argument('--dry-run', action='store_true', help="Don't write files")
    args = parser.parse_args()

    # Load gene mappings for each cohort
    print("=" * 60)
    print("Loading Gene Mappings")
    print("=" * 60)

    mappings = {
        'main': load_gene_mapping(INFLAM_MAIN),
        'val': load_gene_mapping(INFLAM_VAL),
        'ext': load_gene_mapping(INFLAM_EXT),
    }

    # Process all inflammation pseudobulk files
    print("\n" + "=" * 60)
    print("Converting Pseudobulk Files")
    print("=" * 60)

    files_to_process = [
        ('main', 'inflammation_main/pseudobulk/inflammation_main_pseudobulk_l1.h5ad'),
        ('main', 'inflammation_main/pseudobulk/inflammation_main_pseudobulk_l2.h5ad'),
        ('val', 'inflammation_val/pseudobulk/inflammation_val_pseudobulk_l1.h5ad'),
        ('val', 'inflammation_val/pseudobulk/inflammation_val_pseudobulk_l2.h5ad'),
        ('ext', 'inflammation_ext/pseudobulk/inflammation_ext_pseudobulk_l1.h5ad'),
        ('ext', 'inflammation_ext/pseudobulk/inflammation_ext_pseudobulk_l2.h5ad'),
    ]

    for cohort, rel_path in files_to_process:
        input_path = PSEUDOBULK_ROOT / rel_path
        if not input_path.exists():
            # Try the longer name version
            alt_name = rel_path.replace('inflammation_val_', 'inflammation_validation_')
            alt_name = alt_name.replace('inflammation_ext_', 'inflammation_external_')
            input_path = PSEUDOBULK_ROOT / alt_name

        if input_path.exists():
            if args.dry_run:
                print(f"\n[DRY RUN] Would process: {input_path}")
            else:
                convert_pseudobulk(input_path, mappings[cohort])
        else:
            print(f"\nWARNING: File not found: {rel_path}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
