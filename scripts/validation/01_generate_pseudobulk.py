#!/usr/bin/env python3
"""
Stage 1: Generate Pseudobulk Expression Data
============================================
Creates cell-type-specific pseudobulk H5AD files from raw atlas data.
This pre-aggregation step allows activity inference to run on smaller datasets.

Usage:
    python 01_generate_pseudobulk.py --atlas cima --level L1
    python 01_generate_pseudobulk.py --atlas cima --level L2
    python 01_generate_pseudobulk.py --atlas inflammation --level L1
    python 01_generate_pseudobulk.py --atlas scatlas_normal --level organ_celltype
    python 01_generate_pseudobulk.py --all
"""

import argparse
import gc
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
import h5py

warnings.filterwarnings('ignore')

# ==============================================================================
# Configuration
# ==============================================================================

# Data paths
ATLAS_PATHS = {
    'cima': '/data/Jiang_Lab/Data/Seongyong/CIMA/Cell_Atlas/CIMA_RNA_6484974cells_36326genes_compressed.h5ad',
    'inflammation_main': '/data/Jiang_Lab/Data/Seongyong/Inflammation_Atlas/INFLAMMATION_ATLAS_main_afterQC.h5ad',
    'inflammation_val': '/data/Jiang_Lab/Data/Seongyong/Inflammation_Atlas/INFLAMMATION_ATLAS_validation_afterQC.h5ad',
    'scatlas_normal': '/data/Jiang_Lab/Data/Seongyong/scAtlas_2025/igt_s9_fine_counts.h5ad',
    'scatlas_cancer': '/data/Jiang_Lab/Data/Seongyong/scAtlas_2025/PanCancer_igt_s9_fine_counts.h5ad',
}

# Cell type column mapping
CELLTYPE_COLUMNS = {
    'cima': {
        'L1': 'cell_type_l1',
        'L2': 'cell_type_l2',
        'L3': 'cell_type_l3',
    },
    'inflammation_main': {
        'L1': 'cell_type_level1',
        'L2': 'cell_type_level2',
    },
    'inflammation_val': {
        'L1': 'cell_type_level1',
        'L2': 'cell_type_level2',
    },
    'scatlas_normal': {
        'organ_celltype': ['organ', 'cell_type'],  # Combined
        'celltype': 'cell_type',
    },
    'scatlas_cancer': {
        'organ_celltype': ['organ', 'cell_type'],
        'celltype': 'cell_type',
    },
}

# Sample columns
SAMPLE_COLUMNS = {
    'cima': 'sample',
    'inflammation_main': 'sample_id',
    'inflammation_val': 'sample_id',
    'scatlas_normal': 'sample',
    'scatlas_cancer': 'sample',
}

OUTPUT_ROOT = Path('/vf/users/parks34/projects/2cytoatlas/results/validation/pseudobulk')

# ==============================================================================
# Logging
# ==============================================================================

def log(msg: str):
    timestamp = time.strftime('%H:%M:%S')
    print(f"[{timestamp}] {msg}", flush=True)

# ==============================================================================
# Pseudobulk Generation
# ==============================================================================

def generate_pseudobulk_streaming(
    h5ad_path: str,
    output_path: Path,
    cell_type_col: str,
    sample_col: str,
    min_cells: int = 10,
    chunk_size: int = 100000,
) -> Dict:
    """
    Generate pseudobulk with streaming output.

    Processes the H5AD in chunks and streams results to output file.

    Args:
        h5ad_path: Path to input H5AD
        output_path: Path for output pseudobulk H5AD
        cell_type_col: Column for cell type annotation
        sample_col: Column for sample ID
        min_cells: Minimum cells per group
        chunk_size: Cells to process per chunk

    Returns:
        Dict with summary statistics
    """
    log(f"Loading {h5ad_path} (backed mode)...")
    adata = ad.read_h5ad(h5ad_path, backed='r')

    n_cells = adata.n_obs
    n_genes = adata.n_vars
    genes = list(adata.var_names)

    log(f"  Shape: {adata.shape}")
    log(f"  Cell type column: {cell_type_col}")
    log(f"  Sample column: {sample_col}")

    # Get metadata
    obs = adata.obs[[cell_type_col, sample_col]].copy()
    obs['group'] = obs[cell_type_col].astype(str) + '__' + obs[sample_col].astype(str)

    # Count cells per group
    group_counts = obs['group'].value_counts()
    valid_groups = group_counts[group_counts >= min_cells].index.tolist()
    log(f"  {len(valid_groups)} valid groups (>= {min_cells} cells)")

    if len(valid_groups) == 0:
        raise ValueError("No valid groups found!")

    # Create group to indices mapping
    log("Building group index mapping...")
    idx_to_pos = {idx: pos for pos, idx in enumerate(adata.obs_names)}

    group_indices = {}
    for group in valid_groups:
        mask = obs['group'] == group
        indices = obs.index[mask].tolist()
        group_indices[group] = np.array([idx_to_pos[idx] for idx in indices])

    # Initialize output arrays
    n_groups = len(valid_groups)
    log(f"Initializing output matrix ({n_genes} genes × {n_groups} pseudobulk samples)...")

    # Use float32 to save memory
    pseudobulk_matrix = np.zeros((n_genes, n_groups), dtype=np.float32)

    # Process in chunks for memory efficiency
    log(f"Aggregating expression (chunk_size={chunk_size})...")

    X = adata.X
    n_chunks = (n_cells + chunk_size - 1) // chunk_size

    for chunk_idx in range(n_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, n_cells)

        if chunk_idx % 10 == 0:
            log(f"  Processing chunk {chunk_idx + 1}/{n_chunks} (cells {start_idx}-{end_idx})...")

        # Load chunk
        chunk_data = X[start_idx:end_idx]
        if sp.issparse(chunk_data):
            chunk_data = chunk_data.toarray()

        # Accumulate to appropriate groups
        for group_idx, group_name in enumerate(valid_groups):
            group_cell_indices = group_indices[group_name]

            # Find which cells in this chunk belong to this group
            chunk_mask = (group_cell_indices >= start_idx) & (group_cell_indices < end_idx)
            if not chunk_mask.any():
                continue

            local_indices = group_cell_indices[chunk_mask] - start_idx
            pseudobulk_matrix[:, group_idx] += chunk_data[local_indices].sum(axis=0)

    # Close backed file
    adata.file.close()
    del adata
    gc.collect()

    # CPM normalize
    log("Applying CPM normalization...")
    col_sums = pseudobulk_matrix.sum(axis=0, keepdims=True)
    col_sums[col_sums == 0] = 1
    pseudobulk_cpm = (pseudobulk_matrix / col_sums) * 1e6

    del pseudobulk_matrix
    gc.collect()

    # Create metadata
    meta_records = []
    for group_name in valid_groups:
        parts = group_name.split('__')
        meta_records.append({
            'pseudobulk_id': group_name,
            'cell_type': parts[0],
            'sample': parts[1] if len(parts) > 1 else parts[0],
            'n_cells': len(group_indices[group_name]),
        })

    meta_df = pd.DataFrame(meta_records)
    meta_df.index = meta_df['pseudobulk_id']

    # Save as AnnData H5AD
    log(f"Saving to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create AnnData object
    adata_out = ad.AnnData(
        X=pseudobulk_cpm.T,  # (samples × genes)
        obs=meta_df,
        var=pd.DataFrame(index=genes),
    )
    adata_out.write_h5ad(output_path, compression='gzip')

    log(f"  Saved: {adata_out.shape}")

    # Summary
    summary = {
        'n_genes': n_genes,
        'n_pseudobulk_samples': n_groups,
        'n_cell_types': meta_df['cell_type'].nunique(),
        'n_samples': meta_df['sample'].nunique(),
        'total_cells': sum(len(v) for v in group_indices.values()),
        'output_path': str(output_path),
    }

    return summary


def generate_pseudobulk_combined_cols(
    h5ad_path: str,
    output_path: Path,
    cell_type_cols: List[str],
    sample_col: str,
    min_cells: int = 10,
    chunk_size: int = 100000,
) -> Dict:
    """
    Generate pseudobulk with combined cell type columns (e.g., organ × cell_type).
    """
    log(f"Loading {h5ad_path} (backed mode)...")
    adata = ad.read_h5ad(h5ad_path, backed='r')

    n_cells = adata.n_obs
    n_genes = adata.n_vars
    genes = list(adata.var_names)

    log(f"  Shape: {adata.shape}")
    log(f"  Cell type columns: {cell_type_cols}")
    log(f"  Sample column: {sample_col}")

    # Get metadata and create combined cell type
    cols_needed = cell_type_cols + [sample_col]
    obs = adata.obs[cols_needed].copy()

    # Combine cell type columns
    combined_celltype = obs[cell_type_cols[0]].astype(str)
    for col in cell_type_cols[1:]:
        combined_celltype = combined_celltype + '_' + obs[col].astype(str)

    obs['cell_type_combined'] = combined_celltype
    obs['group'] = obs['cell_type_combined'] + '__' + obs[sample_col].astype(str)

    # Count cells per group
    group_counts = obs['group'].value_counts()
    valid_groups = group_counts[group_counts >= min_cells].index.tolist()
    log(f"  {len(valid_groups)} valid groups (>= {min_cells} cells)")

    if len(valid_groups) == 0:
        raise ValueError("No valid groups found!")

    # Create group to indices mapping
    log("Building group index mapping...")
    idx_to_pos = {idx: pos for pos, idx in enumerate(adata.obs_names)}

    group_indices = {}
    for group in valid_groups:
        mask = obs['group'] == group
        indices = obs.index[mask].tolist()
        group_indices[group] = np.array([idx_to_pos[idx] for idx in indices])

    # Initialize output arrays
    n_groups = len(valid_groups)
    log(f"Initializing output matrix ({n_genes} genes × {n_groups} pseudobulk samples)...")

    pseudobulk_matrix = np.zeros((n_genes, n_groups), dtype=np.float32)

    # Process in chunks
    log(f"Aggregating expression (chunk_size={chunk_size})...")

    X = adata.X
    n_chunks = (n_cells + chunk_size - 1) // chunk_size

    for chunk_idx in range(n_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, n_cells)

        if chunk_idx % 10 == 0:
            log(f"  Processing chunk {chunk_idx + 1}/{n_chunks} (cells {start_idx}-{end_idx})...")

        chunk_data = X[start_idx:end_idx]
        if sp.issparse(chunk_data):
            chunk_data = chunk_data.toarray()

        for group_idx, group_name in enumerate(valid_groups):
            group_cell_indices = group_indices[group_name]
            chunk_mask = (group_cell_indices >= start_idx) & (group_cell_indices < end_idx)
            if not chunk_mask.any():
                continue

            local_indices = group_cell_indices[chunk_mask] - start_idx
            pseudobulk_matrix[:, group_idx] += chunk_data[local_indices].sum(axis=0)

    # Close backed file
    adata.file.close()
    del adata
    gc.collect()

    # CPM normalize
    log("Applying CPM normalization...")
    col_sums = pseudobulk_matrix.sum(axis=0, keepdims=True)
    col_sums[col_sums == 0] = 1
    pseudobulk_cpm = (pseudobulk_matrix / col_sums) * 1e6

    del pseudobulk_matrix
    gc.collect()

    # Create metadata
    meta_records = []
    for group_name in valid_groups:
        parts = group_name.split('__')
        cell_type_parts = parts[0].split('_') if len(cell_type_cols) > 1 else [parts[0]]

        record = {
            'pseudobulk_id': group_name,
            'cell_type': parts[0],
            'sample': parts[1] if len(parts) > 1 else parts[0],
            'n_cells': len(group_indices[group_name]),
        }

        # Add individual cell type columns
        for i, col in enumerate(cell_type_cols):
            if i < len(cell_type_parts):
                record[col] = cell_type_parts[i]

        meta_records.append(record)

    meta_df = pd.DataFrame(meta_records)
    meta_df.index = meta_df['pseudobulk_id']

    # Save as AnnData H5AD
    log(f"Saving to {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    adata_out = ad.AnnData(
        X=pseudobulk_cpm.T,
        obs=meta_df,
        var=pd.DataFrame(index=genes),
    )
    adata_out.write_h5ad(output_path, compression='gzip')

    log(f"  Saved: {adata_out.shape}")

    summary = {
        'n_genes': n_genes,
        'n_pseudobulk_samples': n_groups,
        'n_cell_types': meta_df['cell_type'].nunique(),
        'n_samples': meta_df['sample'].nunique(),
        'total_cells': sum(len(v) for v in group_indices.values()),
        'output_path': str(output_path),
    }

    return summary


# ==============================================================================
# Main
# ==============================================================================

def run_pseudobulk_generation(atlas: str, level: str) -> Dict:
    """Run pseudobulk generation for a specific atlas and level."""

    h5ad_path = ATLAS_PATHS[atlas]
    cell_type_config = CELLTYPE_COLUMNS[atlas][level]
    sample_col = SAMPLE_COLUMNS[atlas]

    output_path = OUTPUT_ROOT / f"{atlas}_{level}_pseudobulk.h5ad"

    log("=" * 60)
    log(f"Generating Pseudobulk: {atlas} / {level}")
    log("=" * 60)

    start_time = time.time()

    if isinstance(cell_type_config, list):
        # Combined columns (e.g., organ × cell_type)
        summary = generate_pseudobulk_combined_cols(
            h5ad_path=h5ad_path,
            output_path=output_path,
            cell_type_cols=cell_type_config,
            sample_col=sample_col,
        )
    else:
        # Single column
        summary = generate_pseudobulk_streaming(
            h5ad_path=h5ad_path,
            output_path=output_path,
            cell_type_col=cell_type_config,
            sample_col=sample_col,
        )

    elapsed = time.time() - start_time
    summary['elapsed_seconds'] = elapsed

    log(f"Completed in {elapsed:.1f}s")
    log(f"Summary: {summary}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Generate pseudobulk expression data")
    parser.add_argument('--atlas', type=str, help='Atlas name')
    parser.add_argument('--level', type=str, help='Cell type level')
    parser.add_argument('--all', action='store_true', help='Run all combinations')
    parser.add_argument('--config-index', type=int, help='Config index for SLURM array')

    args = parser.parse_args()

    # Define all combinations
    ALL_CONFIGS = [
        # CIMA
        ('cima', 'L1'),
        ('cima', 'L2'),
        ('cima', 'L3'),
        # Inflammation
        ('inflammation_main', 'L1'),
        ('inflammation_main', 'L2'),
        ('inflammation_val', 'L1'),
        ('inflammation_val', 'L2'),
        # scAtlas Normal
        ('scatlas_normal', 'organ_celltype'),
        ('scatlas_normal', 'celltype'),
        # scAtlas Cancer
        ('scatlas_cancer', 'organ_celltype'),
        ('scatlas_cancer', 'celltype'),
    ]

    if args.config_index is not None:
        # Run specific config by index
        if args.config_index >= len(ALL_CONFIGS):
            print(f"Config index {args.config_index} out of range (max {len(ALL_CONFIGS) - 1})")
            sys.exit(1)
        atlas, level = ALL_CONFIGS[args.config_index]
        run_pseudobulk_generation(atlas, level)
    elif args.all:
        # Run all
        for atlas, level in ALL_CONFIGS:
            try:
                run_pseudobulk_generation(atlas, level)
            except Exception as e:
                log(f"ERROR: {atlas}/{level}: {e}")
    elif args.atlas and args.level:
        # Run specific
        run_pseudobulk_generation(args.atlas, args.level)
    else:
        parser.print_help()
        print("\nAvailable configs:")
        for i, (atlas, level) in enumerate(ALL_CONFIGS):
            print(f"  {i}: {atlas} / {level}")


if __name__ == "__main__":
    main()
