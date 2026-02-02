#!/usr/bin/env python3
"""
Cell-Type-Aware Cytokine Activity Inference

Computes cytokine activity using cell-type-specific signatures across all
single-cell atlases (CIMA, Inflammation, scAtlas) at both:
- Pseudo-bulk level: Aggregated by sample × cell type
- Single-cell level: Per-cell activity scores (batch processed)

Uses the cell-type-specific signatures generated from CytoSig database,
with atlas cell types mapped to appropriate signature matrices.

Author: Seongyong Park
Date: 2026-02-02
"""

import os
import sys
import gc
import json
import warnings
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from scipy import sparse
from sklearn.linear_model import Ridge
from datetime import datetime

warnings.filterwarnings('ignore')

# ==============================================================================
# Configuration
# ==============================================================================

# Data paths
CIMA_H5AD = Path('/data/Jiang_Lab/Data/Seongyong/CIMA/Cell_Atlas/CIMA_RNA_6484974cells_36326genes_compressed.h5ad')
INFLAMMATION_MAIN = Path('/data/Jiang_Lab/Data/Seongyong/Inflammation_Atlas/INFLAMMATION_ATLAS_main_afterQC.h5ad')
INFLAMMATION_VAL = Path('/data/Jiang_Lab/Data/Seongyong/Inflammation_Atlas/INFLAMMATION_ATLAS_validation_afterQC.h5ad')
SCATLAS_NORMAL = Path('/data/Jiang_Lab/Data/Seongyong/scAtlas_2025/igt_s9_fine_counts.h5ad')
SCATLAS_CANCER = Path('/data/Jiang_Lab/Data/Seongyong/scAtlas_2025/PanCancer_igt_s9_fine_counts.h5ad')

# Signature paths
SIGNATURE_DIR = Path('/data/parks34/projects/2secactpy/results/celltype_signatures')
MAPPING_FILE = SIGNATURE_DIR / 'atlas_signature_mapping.json'

# Output paths
OUTPUT_DIR = Path('/data/parks34/projects/2secactpy/results/celltype_aware_activity')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Processing parameters
BATCH_SIZE = 10000  # Cells per batch for single-cell inference
RIDGE_ALPHA = 1.0   # Ridge regression regularization
MIN_GENES_OVERLAP = 50  # Minimum genes overlap required


def log(msg: str):
    """Print timestamped log message."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ==============================================================================
# Signature Loading
# ==============================================================================

def load_atlas_mapping() -> Dict:
    """Load atlas-to-signature mapping."""
    with open(MAPPING_FILE) as f:
        return json.load(f)


# Import the dynamic matching function
sys.path.insert(0, str(Path(__file__).parent))
from atlas_signature_mapping import get_best_signature_match, get_available_signatures as get_sig_list


def load_signature_matrix(celltype: str) -> Optional[pd.DataFrame]:
    """Load signature matrix for a cell type."""
    sig_path = SIGNATURE_DIR / 'signatures' / f'{celltype}_signatures.csv'
    if sig_path.exists():
        return pd.read_csv(sig_path, index_col=0)
    return None


def get_available_signatures() -> Dict[str, pd.DataFrame]:
    """Load all available signature matrices."""
    signatures = {}
    sig_dir = SIGNATURE_DIR / 'signatures'

    for sig_file in sig_dir.glob('*_signatures.csv'):
        celltype = sig_file.stem.replace('_signatures', '')
        signatures[celltype] = pd.read_csv(sig_file, index_col=0)

    return signatures


# ==============================================================================
# Activity Inference
# ==============================================================================

def infer_activity_ridge(
    expression: np.ndarray,
    signature: np.ndarray,
    alpha: float = RIDGE_ALPHA
) -> np.ndarray:
    """
    Infer cytokine activity using ridge regression.

    Args:
        expression: Gene expression matrix (cells/samples × genes)
        signature: Signature matrix (genes × cytokines)
        alpha: Ridge regularization parameter

    Returns:
        Activity matrix (cells/samples × cytokines)
    """
    model = Ridge(alpha=alpha, fit_intercept=True)

    # Fit: expression ~ signature @ activity
    # We want to find activity such that expression ≈ signature @ activity
    # Ridge regression: activity = (signature.T @ signature + alpha*I)^-1 @ signature.T @ expression.T

    # For each sample, regress expression onto signature
    model.fit(signature, expression.T)
    activity = model.coef_

    return activity


def compute_activity_for_celltype(
    adata: ad.AnnData,
    signature_df: pd.DataFrame,
    cell_mask: np.ndarray,
    mode: str = 'pseudobulk',
    sample_col: str = 'sample_id',
    batch_size: int = BATCH_SIZE
) -> Tuple[pd.DataFrame, Dict]:
    """
    Compute cytokine activity for cells of a specific type.

    Args:
        adata: AnnData object
        signature_df: Signature matrix (genes × cytokines)
        cell_mask: Boolean mask for cells of this type
        mode: 'pseudobulk' or 'singlecell'
        sample_col: Column for sample grouping (pseudobulk)
        batch_size: Batch size for single-cell processing

    Returns:
        Activity DataFrame and metadata dict
    """
    # Find overlapping genes - handle Ensembl ID vs gene symbol mismatch
    sig_genes = set(signature_df.index)
    data_genes = set(adata.var_names)

    # Try direct match first
    overlap_genes = list(sig_genes & data_genes)

    # If no direct overlap, check for symbol column (Ensembl ID -> gene symbol mapping)
    gene_symbol_map = None
    if len(overlap_genes) < MIN_GENES_OVERLAP and 'symbol' in adata.var.columns:
        log(f"      [DEBUG] Direct overlap: {len(overlap_genes)}, trying symbol column...")
        # Build symbol -> ensembl mapping using vectorized pandas (fast)
        symbol_series = adata.var['symbol']
        valid_mask = symbol_series.notna() & (symbol_series != '')
        symbol_to_ensembl = dict(zip(symbol_series[valid_mask], adata.var_names[valid_mask]))

        # Find overlap using symbols
        overlap_symbols = list(sig_genes & set(symbol_to_ensembl.keys()))
        if len(overlap_symbols) >= MIN_GENES_OVERLAP:
            overlap_genes = overlap_symbols
            gene_symbol_map = symbol_to_ensembl
            log(f"      [DEBUG] Symbol-based overlap: {len(overlap_genes)} genes")

    log(f"      [DEBUG] Signature genes: {len(sig_genes)}, Data genes: {len(data_genes)}, Overlap: {len(overlap_genes)}")

    if len(overlap_genes) < MIN_GENES_OVERLAP:
        log(f"      [DEBUG] SKIP: Insufficient gene overlap ({len(overlap_genes)} < {MIN_GENES_OVERLAP})")
        return None, {'error': f'Insufficient gene overlap: {len(overlap_genes)}'}

    # Subset to overlapping genes
    sig_subset = signature_df.loc[overlap_genes].values

    # Handle NaN values in signature matrix
    if np.any(np.isnan(sig_subset)):
        sig_subset = np.nan_to_num(sig_subset, nan=0.0)

    cytokines = signature_df.columns.tolist()

    # Get gene indices in adata (handle symbol mapping if needed)
    if gene_symbol_map:
        # Map symbols to Ensembl IDs, then to indices
        ensembl_ids = [gene_symbol_map[g] for g in overlap_genes]
        gene_idx = [list(adata.var_names).index(e) for e in ensembl_ids]
    else:
        gene_idx = [list(adata.var_names).index(g) for g in overlap_genes]

    if mode == 'pseudobulk':
        # Aggregate by sample
        cells_idx = np.where(cell_mask)[0]
        log(f"      [DEBUG] Total cells for this type: {len(cells_idx)}")

        if sample_col in adata.obs.columns:
            samples = adata.obs.iloc[cells_idx][sample_col].values
            log(f"      [DEBUG] Sample column '{sample_col}' found")
        else:
            samples = np.array(['all'] * len(cells_idx))
            log(f"      [DEBUG] Sample column '{sample_col}' NOT FOUND - using 'all'")

        unique_samples = np.unique(samples)
        log(f"      [DEBUG] Unique samples: {len(unique_samples)}")

        # Compute mean expression per sample
        sample_expr = []
        sample_ids = []
        skipped_small = 0

        for sample in unique_samples:
            sample_mask = samples == sample
            sample_cells = cells_idx[sample_mask]

            if len(sample_cells) < 10:  # Skip samples with too few cells
                skipped_small += 1
                continue

            # Get expression data - handle backed mode, sparse, and dense
            X_sample = adata.X[sample_cells][:, gene_idx]
            if sparse.issparse(X_sample):
                expr = np.asarray(X_sample.mean(axis=0)).flatten()
            elif hasattr(X_sample, 'toarray'):
                expr = np.asarray(X_sample.toarray().mean(axis=0)).flatten()
            else:
                expr = np.asarray(X_sample).mean(axis=0).flatten()

            # Handle NaN in expression (can occur with sparse/backed data)
            if np.any(np.isnan(expr)):
                expr = np.nan_to_num(expr, nan=0.0)

            sample_expr.append(expr)
            sample_ids.append(sample)

        log(f"      [DEBUG] Samples processed: {len(sample_ids)}, Skipped (small): {skipped_small}")

        if len(sample_expr) == 0:
            log(f"      [DEBUG] SKIP: No samples with sufficient cells (all {len(unique_samples)} samples had <10 cells)")
            return None, {'error': 'No samples with sufficient cells'}

        expr_matrix = np.array(sample_expr)

        # Handle NaN values
        if np.any(np.isnan(expr_matrix)):
            expr_matrix = np.nan_to_num(expr_matrix, nan=0.0)

        # Z-score normalize
        expr_matrix = (expr_matrix - expr_matrix.mean(axis=0)) / (expr_matrix.std(axis=0) + 1e-6)

        # Handle any NaN from z-scoring
        if np.any(np.isnan(expr_matrix)):
            expr_matrix = np.nan_to_num(expr_matrix, nan=0.0)

        # Compute activity
        activity = infer_activity_ridge(expr_matrix, sig_subset)

        # Ensure 2D array (sklearn returns 1D when only 1 sample/target)
        # Shape should be (n_samples, n_cytokines)
        if activity.ndim == 1:
            # When only 1 sample, sklearn returns (n_cytokines,)
            # Reshape to (1, n_cytokines)
            activity = activity.reshape(1, -1)

        # Z-score activity
        activity = (activity - activity.mean(axis=0)) / (activity.std(axis=0) + 1e-6)

        result_df = pd.DataFrame(
            activity,
            index=sample_ids,
            columns=cytokines
        )

        metadata = {
            'n_samples': len(sample_ids),
            'n_cells': cell_mask.sum(),
            'n_genes': len(overlap_genes),
            'n_cytokines': len(cytokines)
        }

    else:  # singlecell
        cells_idx = np.where(cell_mask)[0]
        n_cells = len(cells_idx)

        # Process in batches
        all_activity = []

        for batch_start in range(0, n_cells, batch_size):
            batch_end = min(batch_start + batch_size, n_cells)
            batch_cells = cells_idx[batch_start:batch_end]

            # Get expression data - handle backed mode, sparse, and dense
            X_batch = adata.X[batch_cells][:, gene_idx]
            if sparse.issparse(X_batch):
                expr = np.asarray(X_batch.toarray())
            elif hasattr(X_batch, 'todense'):
                expr = np.asarray(X_batch.todense())
            else:
                expr = np.asarray(X_batch)

            # Handle NaN values - replace with 0
            if np.any(np.isnan(expr)):
                expr = np.nan_to_num(expr, nan=0.0)

            # Z-score normalize per cell
            row_mean = expr.mean(axis=1, keepdims=True)
            row_std = expr.std(axis=1, keepdims=True) + 1e-6
            expr = (expr - row_mean) / row_std

            # Handle any NaN from z-scoring (e.g., zero variance rows)
            if np.any(np.isnan(expr)):
                expr = np.nan_to_num(expr, nan=0.0)

            # Compute activity
            activity = infer_activity_ridge(expr, sig_subset)
            # Ensure 2D array (sklearn returns 1D when only 1 cell in batch)
            # Shape should be (n_cells, n_cytokines)
            if activity.ndim == 1:
                activity = activity.reshape(1, -1)
            all_activity.append(activity)

        activity_matrix = np.vstack(all_activity)

        # Z-score activity across cells
        activity_matrix = (activity_matrix - activity_matrix.mean(axis=0)) / (activity_matrix.std(axis=0) + 1e-6)

        result_df = pd.DataFrame(
            activity_matrix,
            index=adata.obs_names[cells_idx],
            columns=cytokines
        )

        metadata = {
            'n_cells': n_cells,
            'n_genes': len(overlap_genes),
            'n_cytokines': len(cytokines)
        }

    return result_df, metadata


# ==============================================================================
# Atlas Processing
# ==============================================================================

def process_atlas(
    atlas_name: str,
    h5ad_path: Path,
    celltype_col: str,
    sample_col: str,
    atlas_mapping: Dict,
    signatures: Dict[str, pd.DataFrame],
    mode: str = 'pseudobulk'
) -> Dict:
    """
    Process a single atlas for cell-type-aware activity inference.

    Args:
        atlas_name: Name of the atlas
        h5ad_path: Path to h5ad file
        celltype_col: Column containing cell type annotations
        sample_col: Column containing sample IDs
        atlas_mapping: Mapping from atlas cell types to signature cell types
        signatures: Dict of signature matrices
        mode: 'pseudobulk' or 'singlecell'

    Returns:
        Dict with results for each cell type
    """
    log(f"Processing {atlas_name} ({mode} mode)...")

    # Load data - always use backed mode for memory efficiency
    log(f"  Loading {h5ad_path.name}...")
    adata = sc.read_h5ad(h5ad_path, backed='r')
    log(f"  Loaded: {adata.n_obs:,} cells, {adata.n_vars:,} genes")

    # Get cell types
    if celltype_col not in adata.obs.columns:
        log(f"  ERROR: Column '{celltype_col}' not found")
        return {}

    celltypes = adata.obs[celltype_col].unique()
    log(f"  Found {len(celltypes)} cell types")

    results = {}

    for ct in celltypes:
        # Get signature mapping - try pre-defined first, then dynamic matching
        if ct in atlas_mapping:
            sig_celltype = atlas_mapping[ct].get('signature')
        else:
            # Try dynamic matching
            sig_celltype = get_best_signature_match(ct, atlas_name.split('_')[0], list(signatures.keys()))

        if sig_celltype is None or sig_celltype not in signatures:
            log(f"    {ct}: No signature available, skipping")
            continue

        signature_df = signatures[sig_celltype]

        # Get cell mask
        cell_mask = (adata.obs[celltype_col] == ct).values
        n_cells = cell_mask.sum()

        if n_cells < 100:  # Skip very small populations
            log(f"    {ct}: Only {n_cells} cells, skipping")
            continue

        log(f"    {ct} → {sig_celltype}: {n_cells:,} cells, {len(signature_df.columns)} cytokines")

        # Compute activity
        activity_df, metadata = compute_activity_for_celltype(
            adata=adata,
            signature_df=signature_df,
            cell_mask=cell_mask,
            mode=mode,
            sample_col=sample_col,
            batch_size=BATCH_SIZE
        )

        if activity_df is not None:
            results[ct] = {
                'activity': activity_df,
                'signature_celltype': sig_celltype,
                'metadata': metadata
            }

            if mode == 'pseudobulk':
                log(f"      → {metadata['n_samples']} samples")
            else:
                log(f"      → {metadata['n_cells']:,} cells processed")
        else:
            log(f"      [DEBUG] FAILED: {metadata.get('error', 'Unknown error')}")

        # Free memory after each cell type
        gc.collect()

    # Close backed file
    if hasattr(adata, 'file') and adata.file is not None:
        adata.file.close()

    log(f"  [DEBUG] Total results: {len(results)} cell types with valid activity")
    if len(results) == 0:
        log(f"  [DEBUG] WARNING: No valid results for any cell type!")

    return results


def save_results(results: Dict, atlas_name: str, mode: str):
    """Save activity results."""
    output_subdir = OUTPUT_DIR / atlas_name / mode
    output_subdir.mkdir(parents=True, exist_ok=True)

    # Save per-celltype results
    for celltype, data in results.items():
        safe_name = celltype.replace('/', '_').replace(' ', '_')

        # Save activity matrix
        data['activity'].to_csv(output_subdir / f'{safe_name}_activity.csv')

        # Save metadata
        meta = data['metadata'].copy()
        meta['atlas_celltype'] = celltype
        meta['signature_celltype'] = data['signature_celltype']

        with open(output_subdir / f'{safe_name}_metadata.json', 'w') as f:
            json.dump(meta, f, indent=2)

    # Save combined summary
    summary = []
    for celltype, data in results.items():
        row = {
            'atlas_celltype': celltype,
            'signature_celltype': data['signature_celltype'],
            **data['metadata']
        }
        summary.append(row)

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(output_subdir / 'summary.csv', index=False)

    log(f"  Saved results to {output_subdir}")


# ==============================================================================
# Main Processing Functions
# ==============================================================================

def process_cima(mode: str = 'pseudobulk'):
    """Process CIMA atlas."""
    atlas_mapping_full = load_atlas_mapping()
    atlas_mapping = {k: v for k, v in atlas_mapping_full.get('cima', {}).items()}
    signatures = get_available_signatures()

    results = process_atlas(
        atlas_name='cima',
        h5ad_path=CIMA_H5AD,
        celltype_col='cell_type_l2',  # Fine-level cell types in CIMA
        sample_col='sample',
        atlas_mapping=atlas_mapping,
        signatures=signatures,
        mode=mode
    )

    if results:
        save_results(results, 'cima', mode)

    return results


def process_inflammation(mode: str = 'pseudobulk'):
    """Process Inflammation Atlas (main cohort)."""
    atlas_mapping_full = load_atlas_mapping()
    atlas_mapping = {k: v for k, v in atlas_mapping_full.get('inflammation', {}).items()}
    signatures = get_available_signatures()

    results = process_atlas(
        atlas_name='inflammation',
        h5ad_path=INFLAMMATION_MAIN,
        celltype_col='Level2',  # Fine-level cell types in Inflammation Atlas
        sample_col='sampleID',
        atlas_mapping=atlas_mapping,
        signatures=signatures,
        mode=mode
    )

    if results:
        save_results(results, 'inflammation', mode)

    return results


def process_scatlas_normal(mode: str = 'pseudobulk'):
    """Process scAtlas normal tissues."""
    atlas_mapping_full = load_atlas_mapping()
    atlas_mapping = {k: v for k, v in atlas_mapping_full.get('scatlas', {}).items()}
    signatures = get_available_signatures()

    results = process_atlas(
        atlas_name='scatlas_normal',
        h5ad_path=SCATLAS_NORMAL,
        celltype_col='cellType1',  # Coarse cell types (consistent annotation)
        sample_col='sampleID',
        atlas_mapping=atlas_mapping,
        signatures=signatures,
        mode=mode
    )

    if results:
        save_results(results, 'scatlas_normal', mode)

    return results


def process_scatlas_cancer(mode: str = 'pseudobulk'):
    """Process scAtlas cancer."""
    atlas_mapping_full = load_atlas_mapping()
    atlas_mapping = {k: v for k, v in atlas_mapping_full.get('scatlas', {}).items()}
    signatures = get_available_signatures()

    results = process_atlas(
        atlas_name='scatlas_cancer',
        h5ad_path=SCATLAS_CANCER,
        celltype_col='cellType1',  # Coarse cell types (consistent annotation)
        sample_col='sampleID',
        atlas_mapping=atlas_mapping,
        signatures=signatures,
        mode=mode
    )

    if results:
        save_results(results, 'scatlas_cancer', mode)

    return results


# ==============================================================================
# Main
# ==============================================================================

def main():
    """Main execution."""
    import argparse

    parser = argparse.ArgumentParser(description='Cell-type-aware cytokine activity inference')
    parser.add_argument('--atlas', type=str, default='all',
                       choices=['all', 'cima', 'inflammation', 'scatlas', 'scatlas_normal', 'scatlas_cancer'],
                       help='Atlas to process (scatlas = both normal and cancer)')
    parser.add_argument('--mode', type=str, default='pseudobulk',
                       choices=['pseudobulk', 'singlecell', 'both'],
                       help='Processing mode')

    args = parser.parse_args()

    log("=" * 60)
    log("Cell-Type-Aware Cytokine Activity Inference")
    log("=" * 60)

    # Load mapping info
    atlas_mapping = load_atlas_mapping()
    signatures = get_available_signatures()
    log(f"Loaded {len(signatures)} signature matrices")

    modes = ['pseudobulk', 'singlecell'] if args.mode == 'both' else [args.mode]

    for mode in modes:
        log(f"\n{'='*60}")
        log(f"Mode: {mode.upper()}")
        log(f"{'='*60}")

        if args.atlas in ['all', 'cima']:
            process_cima(mode)

        if args.atlas in ['all', 'inflammation']:
            process_inflammation(mode)

        if args.atlas in ['all', 'scatlas', 'scatlas_normal']:
            process_scatlas_normal(mode)

        if args.atlas in ['all', 'scatlas', 'scatlas_cancer']:
            process_scatlas_cancer(mode)

    log("\n" + "=" * 60)
    log("Complete!")
    log(f"Output directory: {OUTPUT_DIR}")
    log("=" * 60)


if __name__ == '__main__':
    main()
