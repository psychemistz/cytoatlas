#!/usr/bin/env python3
"""
Single-Cell Batch Processing with Streaming Output
===================================================
Process millions of cells with memory-efficient batch processing and streaming output.

This script handles single-cell level activity inference for:
- CIMA: 6.5M cells
- Inflammation Atlas: 6.3M cells (main + validation + external)
- scAtlas: 6.4M cells (normal + cancer)

Key features:
- Read cells in chunks from backed h5ad files
- Process each chunk through normalization and activity inference
- Stream results incrementally to output h5ad files
- Memory-efficient: only holds one batch in memory at a time

Output format:
- H5AD files with zscore, beta, se, pvalue as layers
- Cells as observations, signatures as variables
"""

import os
import sys
import gc
import time
import warnings
from pathlib import Path
from typing import Optional, List, Tuple, Union
import argparse

import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
import h5py

# Add SecActpy to path
sys.path.insert(0, '/vf/users/parks34/projects/1ridgesig/SecActpy')
from secactpy import (
    load_cytosig, load_secact,
    ridge, ridge_batch,
    estimate_batch_size,
    CUPY_AVAILABLE
)

# ==============================================================================
# Configuration
# ==============================================================================

# Data paths
CIMA_H5AD = Path('/data/Jiang_Lab/Data/Seongyong/CIMA/Cell_Atlas/CIMA_RNA_6484974cells_36326genes_compressed.h5ad')
INFLAM_MAIN_H5AD = Path('/data/Jiang_Lab/Data/Seongyong/Inflammation_Atlas/INFLAMMATION_ATLAS_main_afterQC.h5ad')
INFLAM_VAL_H5AD = Path('/data/Jiang_Lab/Data/Seongyong/Inflammation_Atlas/INFLAMMATION_ATLAS_validation_afterQC.h5ad')
INFLAM_EXT_H5AD = Path('/data/Jiang_Lab/Data/Seongyong/Inflammation_Atlas/INFLAMMATION_ATLAS_external_afterQC.h5ad')
SCATLAS_NORMAL_H5AD = Path('/data/Jiang_Lab/Data/Seongyong/scAtlas_2025/igt_s9_fine_counts.h5ad')
SCATLAS_CANCER_H5AD = Path('/data/Jiang_Lab/Data/Seongyong/scAtlas_2025/PanCancer_igt_s9_fine_counts.h5ad')

# Output directory
OUTPUT_DIR = Path('/vf/users/parks34/projects/2cytoatlas/results')

# Processing parameters
CELL_BATCH_SIZE = 50000  # Process 50K cells at a time
N_RAND = 1000
SEED = 0
LAMBDA = 5e5

# GPU settings
BACKEND = 'cupy' if CUPY_AVAILABLE else 'numpy'


# ==============================================================================
# Utility Functions
# ==============================================================================

def log(msg: str):
    """Print timestamped log message."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def get_gene_names(adata) -> List[str]:
    """Get gene names from AnnData, preferring symbol column."""
    if 'symbol' in adata.var.columns:
        return list(adata.var['symbol'].values)
    else:
        return list(adata.var_names)


def align_genes_with_signature(
    data_genes: List[str],
    signature: pd.DataFrame
) -> Tuple[List[int], pd.DataFrame]:
    """
    Align data genes with signature genes.

    Returns:
        common_idx: Indices into data_genes for common genes
        sig_aligned: Signature matrix aligned and scaled
    """
    # Uppercase for matching
    data_genes_upper = [g.upper() for g in data_genes]
    sig_genes = set(signature.index.str.upper())

    # Find common genes
    common_mask = [g in sig_genes for g in data_genes_upper]
    common_genes = [g for g, m in zip(data_genes_upper, common_mask) if m]
    common_idx = [i for i, m in enumerate(common_mask) if m]

    # Align and scale signature
    sig_aligned = signature.copy()
    sig_aligned.index = sig_aligned.index.str.upper()
    sig_aligned = sig_aligned[~sig_aligned.index.duplicated(keep='first')]
    sig_aligned = sig_aligned.loc[common_genes]

    # Z-score normalize
    sig_scaled = (sig_aligned - sig_aligned.mean()) / sig_aligned.std(ddof=1)
    sig_scaled = sig_scaled.fillna(0)

    return common_idx, sig_scaled


def normalize_batch(
    X_batch: np.ndarray,
    gene_idx: List[int]
) -> np.ndarray:
    """
    Normalize a batch of cells: TPM + log2 + differential.

    Args:
        X_batch: (n_cells, n_genes) expression matrix
        gene_idx: Indices of genes to keep (for aligned genes)

    Returns:
        (n_genes_aligned, n_cells) normalized and transposed matrix
    """
    # Convert sparse to dense if needed
    if sp.issparse(X_batch):
        X_batch = np.asarray(X_batch.todense())

    # Subset to aligned genes
    X_subset = X_batch[:, gene_idx]  # (n_cells, n_genes_aligned)

    # TPM normalize per cell
    cell_sums = X_subset.sum(axis=1, keepdims=True)
    cell_sums = np.where(cell_sums == 0, 1, cell_sums)  # Avoid division by zero
    X_tpm = X_subset / cell_sums * 1e6

    # Log2 transform
    X_log = np.log2(X_tpm + 1)

    # Differential (subtract gene mean across this batch)
    gene_means = X_log.mean(axis=0, keepdims=True)
    X_diff = X_log - gene_means

    # Transpose to (n_genes, n_cells) and z-score normalize genes
    X_T = X_diff.T.astype(np.float64)
    gene_std = X_T.std(axis=1, keepdims=True, ddof=1)
    gene_std = np.where(gene_std == 0, 1, gene_std)
    X_scaled = (X_T - X_T.mean(axis=1, keepdims=True)) / gene_std

    return np.ascontiguousarray(X_scaled)


class StreamingH5ADWriter:
    """
    Stream activity results to h5ad file incrementally.

    Writes results as:
    - X: zscore matrix (n_cells, n_signatures)
    - layers/beta, layers/se, layers/pvalue
    - obs: cell metadata
    - var: signature metadata
    """

    def __init__(
        self,
        path: Path,
        n_cells: int,
        signature_names: List[str],
        compression: str = "gzip"
    ):
        self.path = path
        self.n_cells = n_cells
        self.n_signatures = len(signature_names)
        self.signature_names = signature_names
        self.compression = compression
        self.cells_written = 0

        # Create h5py file with pre-allocated datasets
        self.file = h5py.File(path, 'w')

        # Create main matrix (zscore)
        self.file.create_dataset(
            'X',
            shape=(n_cells, self.n_signatures),
            dtype='float32',
            compression=compression,
            chunks=(min(10000, n_cells), self.n_signatures)
        )

        # Create layers
        for layer_name in ['beta', 'se', 'pvalue']:
            self.file.create_dataset(
                f'layers/{layer_name}',
                shape=(n_cells, self.n_signatures),
                dtype='float32',
                compression=compression,
                chunks=(min(10000, n_cells), self.n_signatures)
            )

        # Store signature names as var
        self.file.create_dataset(
            'var/_index',
            data=np.array(signature_names, dtype='S')
        )

        # Create obs group for cell names (will be written at end)
        self.obs_names = []

        log(f"Created streaming output: {path}")
        log(f"  Shape: ({n_cells}, {self.n_signatures})")

    def write_batch(
        self,
        result: dict,
        cell_names: List[str]
    ):
        """Write a batch of results."""
        batch_size = result['zscore'].shape[1]
        start_idx = self.cells_written
        end_idx = start_idx + batch_size

        # Transpose results from (signatures, cells) to (cells, signatures)
        self.file['X'][start_idx:end_idx, :] = result['zscore'].T.astype(np.float32)
        self.file['layers/beta'][start_idx:end_idx, :] = result['beta'].T.astype(np.float32)
        self.file['layers/se'][start_idx:end_idx, :] = result['se'].T.astype(np.float32)
        self.file['layers/pvalue'][start_idx:end_idx, :] = result['pvalue'].T.astype(np.float32)

        # Store cell names
        self.obs_names.extend(cell_names)
        self.cells_written = end_idx

        # Flush periodically
        if self.cells_written % 100000 == 0:
            self.file.flush()

    def finalize(self, obs_df: Optional[pd.DataFrame] = None):
        """Finalize the file with obs metadata."""
        # Write cell names
        self.file.create_dataset(
            'obs/_index',
            data=np.array(self.obs_names, dtype='S')
        )

        # Write obs columns if provided
        if obs_df is not None:
            for col in obs_df.columns:
                try:
                    if obs_df[col].dtype == object:
                        self.file.create_dataset(
                            f'obs/{col}',
                            data=np.array(obs_df[col].values, dtype='S')
                        )
                    else:
                        self.file.create_dataset(
                            f'obs/{col}',
                            data=obs_df[col].values
                        )
                except Exception as e:
                    log(f"  Warning: Could not write obs/{col}: {e}")

        self.file.flush()
        self.file.close()
        log(f"Finalized output: {self.path} ({self.cells_written:,} cells)")


def process_single_cell_streaming(
    h5ad_path: Path,
    output_path: Path,
    signature: pd.DataFrame,
    sig_name: str,
    cell_batch_size: int = CELL_BATCH_SIZE,
    layer: Optional[str] = None,
    obs_cols: Optional[List[str]] = None
):
    """
    Process single-cell data with streaming input/output.

    Args:
        h5ad_path: Path to input h5ad file
        output_path: Path to output h5ad file
        signature: Signature matrix (genes x proteins)
        sig_name: Signature name for logging
        cell_batch_size: Number of cells per batch
        layer: Layer to use for expression (None = .X)
        obs_cols: obs columns to include in output
    """
    log(f"\n{'='*60}")
    log(f"Processing {sig_name}: {h5ad_path.name}")
    log(f"{'='*60}")

    # Open h5ad in backed mode
    log("Loading h5ad in backed mode...")
    adata = ad.read_h5ad(h5ad_path, backed='r')
    n_cells = adata.shape[0]
    n_genes = adata.shape[1]
    log(f"  Shape: ({n_cells:,}, {n_genes})")

    # Get gene names
    gene_names = get_gene_names(adata)

    # Align genes with signature
    common_idx, sig_scaled = align_genes_with_signature(gene_names, signature)
    n_common = len(common_idx)
    log(f"  Common genes: {n_common} / {len(signature)} signature genes")

    if n_common < 100:
        raise ValueError(f"Too few common genes: {n_common}")

    # Get expression matrix reference
    if layer and layer in adata.layers:
        X = adata.layers[layer]
        log(f"  Using layer: {layer}")
    else:
        X = adata.X
        log("  Using .X")

    # Get obs columns if specified
    if obs_cols:
        obs_cols = [c for c in obs_cols if c in adata.obs.columns]

    # Create streaming writer
    writer = StreamingH5ADWriter(
        output_path,
        n_cells=n_cells,
        signature_names=list(sig_scaled.columns)
    )

    # Prepare signature for ridge regression
    sig_matrix = sig_scaled.values.astype(np.float64)

    # Process cells in batches
    n_batches = (n_cells + cell_batch_size - 1) // cell_batch_size
    log(f"Processing {n_cells:,} cells in {n_batches} batches of {cell_batch_size}...")

    start_time = time.time()

    for batch_idx in range(n_batches):
        batch_start = batch_idx * cell_batch_size
        batch_end = min(batch_start + cell_batch_size, n_cells)
        batch_size = batch_end - batch_start

        # Read batch of cells
        X_batch = X[batch_start:batch_end, :]

        # Normalize batch
        Y_batch = normalize_batch(X_batch, common_idx)  # (n_genes, batch_size)

        # Run ridge regression
        if CUPY_AVAILABLE and batch_size > 100:
            result = ridge(
                X=sig_matrix,
                Y=Y_batch,
                lambda_=LAMBDA,
                n_rand=N_RAND,
                seed=SEED,
                backend='cupy',
                verbose=False
            )
        else:
            result = ridge(
                X=sig_matrix,
                Y=Y_batch,
                lambda_=LAMBDA,
                n_rand=N_RAND,
                seed=SEED,
                backend='numpy',
                verbose=False
            )

        # Get cell names for this batch
        cell_names = list(adata.obs_names[batch_start:batch_end])

        # Write batch results
        writer.write_batch(result, cell_names)

        # Progress logging
        elapsed = time.time() - start_time
        cells_per_sec = (batch_end) / elapsed
        eta = (n_cells - batch_end) / cells_per_sec if cells_per_sec > 0 else 0

        if (batch_idx + 1) % 10 == 0 or batch_idx == n_batches - 1:
            log(f"  Batch {batch_idx + 1}/{n_batches}: {batch_end:,}/{n_cells:,} cells "
                f"({cells_per_sec:.0f} cells/s, ETA: {eta/60:.1f} min)")

        # Memory cleanup
        del X_batch, Y_batch, result
        if batch_idx % 50 == 0:
            gc.collect()

    # Finalize with obs metadata
    obs_df = None
    if obs_cols:
        obs_df = adata.obs[obs_cols].copy()

    writer.finalize(obs_df)

    total_time = time.time() - start_time
    log(f"Completed {sig_name} in {total_time/60:.1f} minutes")
    log(f"  Output: {output_path}")


# ==============================================================================
# Dataset-Specific Processing Functions
# ==============================================================================

def process_cima(signatures: str = 'both'):
    """Process CIMA single-cell data."""
    log("\n" + "=" * 60)
    log("CIMA SINGLE-CELL ANALYSIS")
    log("=" * 60)

    output_dir = OUTPUT_DIR / 'cima'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load signatures
    log("Loading signatures...")
    cytosig = load_cytosig()
    secact = load_secact()
    log(f"  CytoSig: {cytosig.shape}")
    log(f"  SecAct: {secact.shape}")

    # Process CytoSig
    if signatures in ['both', 'cytosig']:
        process_single_cell_streaming(
            h5ad_path=CIMA_H5AD,
            output_path=output_dir / 'CIMA_CytoSig_singlecell.h5ad',
            signature=cytosig,
            sig_name='CytoSig',
            layer='counts',
            obs_cols=['sample', 'cell_type_l2']
        )

    # Process SecAct
    if signatures in ['both', 'secact']:
        process_single_cell_streaming(
            h5ad_path=CIMA_H5AD,
            output_path=output_dir / 'CIMA_SecAct_singlecell.h5ad',
            signature=secact,
            sig_name='SecAct',
            layer='counts',
            obs_cols=['sample', 'cell_type_l2']
        )


def process_inflammation(
    dataset: str = 'main',
    signatures: str = 'both'
):
    """Process Inflammation Atlas single-cell data."""
    log("\n" + "=" * 60)
    log(f"INFLAMMATION ATLAS SINGLE-CELL ANALYSIS ({dataset})")
    log("=" * 60)

    output_dir = OUTPUT_DIR / 'inflammation'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Select dataset
    if dataset == 'main':
        h5ad_path = INFLAM_MAIN_H5AD
    elif dataset == 'validation':
        h5ad_path = INFLAM_VAL_H5AD
    elif dataset == 'external':
        h5ad_path = INFLAM_EXT_H5AD
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # Load signatures
    log("Loading signatures...")
    cytosig = load_cytosig()
    secact = load_secact()
    log(f"  CytoSig: {cytosig.shape}")
    log(f"  SecAct: {secact.shape}")

    # Process CytoSig
    if signatures in ['both', 'cytosig']:
        process_single_cell_streaming(
            h5ad_path=h5ad_path,
            output_path=output_dir / f'{dataset}_CytoSig_singlecell.h5ad',
            signature=cytosig,
            sig_name='CytoSig',
            layer=None,  # Inflammation Atlas uses .X
            obs_cols=['sampleID', 'Level2', 'disease', 'diseaseGroup']
        )

    # Process SecAct
    if signatures in ['both', 'secact']:
        process_single_cell_streaming(
            h5ad_path=h5ad_path,
            output_path=output_dir / f'{dataset}_SecAct_singlecell.h5ad',
            signature=secact,
            sig_name='SecAct',
            layer=None,
            obs_cols=['sampleID', 'Level2', 'disease', 'diseaseGroup']
        )


def process_scatlas(
    dataset: str = 'normal',
    signatures: str = 'both'
):
    """Process scAtlas single-cell data."""
    log("\n" + "=" * 60)
    log(f"SCATLAS SINGLE-CELL ANALYSIS ({dataset})")
    log("=" * 60)

    output_dir = OUTPUT_DIR / 'scatlas'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Select dataset
    if dataset == 'normal':
        h5ad_path = SCATLAS_NORMAL_H5AD
        obs_cols = ['tissue', 'cellType1', 'cellType2']
    elif dataset == 'cancer':
        h5ad_path = SCATLAS_CANCER_H5AD
        obs_cols = ['tissue', 'cancerType', 'cellType1', 'cellType2']
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    # Load signatures
    log("Loading signatures...")
    cytosig = load_cytosig()
    secact = load_secact()
    log(f"  CytoSig: {cytosig.shape}")
    log(f"  SecAct: {secact.shape}")

    # Process CytoSig
    if signatures in ['both', 'cytosig']:
        process_single_cell_streaming(
            h5ad_path=h5ad_path,
            output_path=output_dir / f'scatlas_{dataset}_CytoSig_singlecell.h5ad',
            signature=cytosig,
            sig_name='CytoSig',
            layer=None,  # scAtlas uses .X
            obs_cols=obs_cols
        )

    # Process SecAct
    if signatures in ['both', 'secact']:
        process_single_cell_streaming(
            h5ad_path=h5ad_path,
            output_path=output_dir / f'scatlas_{dataset}_SecAct_singlecell.h5ad',
            signature=secact,
            sig_name='SecAct',
            layer=None,
            obs_cols=obs_cols
        )


# ==============================================================================
# Main Entry Point
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Single-cell batch processing with streaming output'
    )
    parser.add_argument(
        '--dataset',
        choices=['cima', 'inflam_main', 'inflam_validation', 'inflam_external',
                 'scatlas_normal', 'scatlas_cancer', 'all'],
        default='all',
        help='Dataset to process'
    )
    parser.add_argument(
        '--signature',
        choices=['cytosig', 'secact', 'both'],
        default='both',
        help='Signature(s) to compute'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=CELL_BATCH_SIZE,
        help='Number of cells per batch'
    )

    args = parser.parse_args()

    # Use command-line batch size
    batch_size = args.batch_size

    log("=" * 60)
    log("SINGLE-CELL BATCH PROCESSING")
    log("=" * 60)
    log(f"Dataset: {args.dataset}")
    log(f"Signature: {args.signature}")
    log(f"Batch size: {batch_size:,}")
    log(f"Backend: {BACKEND}")
    log(f"Output: {OUTPUT_DIR}")

    start_time = time.time()

    # Process requested datasets
    if args.dataset == 'cima' or args.dataset == 'all':
        process_cima(args.signature)

    if args.dataset == 'inflam_main' or args.dataset == 'all':
        process_inflammation('main', args.signature)

    if args.dataset == 'inflam_validation' or args.dataset == 'all':
        process_inflammation('validation', args.signature)

    if args.dataset == 'inflam_external' or args.dataset == 'all':
        process_inflammation('external', args.signature)

    if args.dataset == 'scatlas_normal' or args.dataset == 'all':
        process_scatlas('normal', args.signature)

    if args.dataset == 'scatlas_cancer' or args.dataset == 'all':
        process_scatlas('cancer', args.signature)

    total_time = time.time() - start_time
    log(f"\nTotal time: {total_time/60:.1f} minutes")
    log("Done!")


if __name__ == '__main__':
    main()
