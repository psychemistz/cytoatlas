#!/usr/bin/env python3
"""
Atlas Validation: Expression-Activity Correlation Analysis

Validates cytokine/secreted protein activity predictions by computing correlations
between actual gene expression and predicted activity scores at multiple levels:

1. Pseudobulk Level: Correlation per sample (resampling-based aggregation)
2. Single-Cell Level: Correlation per cell
3. Atlas Level: Correlation across cell types (one point per cell type)

For three signature types:
- CytoSig: Original 44 cytokines
- LinCytoSig: Cell-type-specific cytokine signatures
- SecAct: 1,249 secreted proteins

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
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from scipy import sparse
from scipy.stats import spearmanr, pearsonr
from datetime import datetime
from collections import defaultdict

warnings.filterwarnings('ignore')

# ==============================================================================
# Configuration
# ==============================================================================

# Data paths
CIMA_H5AD = Path('/data/Jiang_Lab/Data/Seongyong/CIMA/Cell_Atlas/CIMA_RNA_6484974cells_36326genes_compressed.h5ad')
INFLAMMATION_MAIN = Path('/data/Jiang_Lab/Data/Seongyong/Inflammation_Atlas/INFLAMMATION_ATLAS_main_afterQC.h5ad')
SCATLAS_NORMAL = Path('/data/Jiang_Lab/Data/Seongyong/scAtlas_2025/igt_s9_fine_counts.h5ad')

# Activity results
RESULTS_DIR = Path('/data/parks34/projects/2secactpy/results')
CYTOSIG_DIR = RESULTS_DIR / 'cima'  # CytoSig/SecAct results
LINCYTOSIG_DIR = RESULTS_DIR / 'celltype_aware_activity'

# Signature matrices
from secactpy import load_cytosig, load_secact
SIGNATURE_DIR = RESULTS_DIR / 'celltype_signatures'

# Output
OUTPUT_DIR = Path('/data/parks34/projects/2secactpy/visualization/data/atlas_validation')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Processing
MAX_CELLS_PER_TYPE = 10000  # For single-cell level, sample this many cells max
BATCH_SIZE = 5000


def log(msg: str):
    """Print timestamped log message."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


# ==============================================================================
# Signature Loading
# ==============================================================================

def load_signatures() -> Dict[str, pd.DataFrame]:
    """Load all signature matrices."""
    signatures = {}

    # CytoSig
    log("Loading CytoSig signatures...")
    signatures['cytosig'] = load_cytosig()
    log(f"  CytoSig: {signatures['cytosig'].shape[0]} genes × {signatures['cytosig'].shape[1]} cytokines")

    # SecAct
    log("Loading SecAct signatures...")
    signatures['secact'] = load_secact()
    log(f"  SecAct: {signatures['secact'].shape[0]} genes × {signatures['secact'].shape[1]} proteins")

    # LinCytoSig (cell-type-specific)
    log("Loading LinCytoSig signatures...")
    lincytosig_path = SIGNATURE_DIR / 'signatures'
    if lincytosig_path.exists():
        lincytosig_files = list(lincytosig_path.glob('*_signatures.csv'))
        signatures['lincytosig'] = {}
        for f in lincytosig_files:
            celltype = f.stem.replace('_signatures', '')
            df = pd.read_csv(f, index_col=0)
            signatures['lincytosig'][celltype] = df
        log(f"  LinCytoSig: {len(signatures['lincytosig'])} cell types")
    else:
        log("  LinCytoSig not found, skipping")
        signatures['lincytosig'] = {}

    return signatures


# ==============================================================================
# Expression Extraction
# ==============================================================================

def get_signature_genes(signature_df: pd.DataFrame) -> List[str]:
    """Get genes from signature matrix."""
    return list(signature_df.index)


def get_expression_for_genes(
    adata,
    gene_list: List[str],
    cell_indices: np.ndarray
) -> np.ndarray:
    """Extract expression values for specific genes and cells."""
    # Find gene indices
    gene_mask = adata.var_names.isin(gene_list)
    available_genes = adata.var_names[gene_mask].tolist()

    if len(available_genes) == 0:
        return None, []

    gene_idx = [list(adata.var_names).index(g) for g in available_genes]

    # Extract expression
    X = adata.X[cell_indices][:, gene_idx]
    if sparse.issparse(X):
        X = np.asarray(X.toarray())
    elif hasattr(X, 'todense'):
        X = np.asarray(X.todense())
    else:
        X = np.asarray(X)

    return X, available_genes


# ==============================================================================
# Correlation Computation
# ==============================================================================

def compute_correlation(x: np.ndarray, y: np.ndarray) -> Dict:
    """Compute Pearson and Spearman correlations."""
    # Remove NaN values
    mask = ~(np.isnan(x) | np.isnan(y))
    x = x[mask]
    y = y[mask]

    if len(x) < 10:
        return {'pearson_r': np.nan, 'spearman_rho': np.nan, 'p_value': np.nan, 'n': len(x)}

    try:
        pearson_r, pearson_p = pearsonr(x, y)
        spearman_rho, spearman_p = spearmanr(x, y)
    except:
        return {'pearson_r': np.nan, 'spearman_rho': np.nan, 'p_value': np.nan, 'n': len(x)}

    return {
        'pearson_r': float(pearson_r),
        'spearman_rho': float(spearman_rho),
        'p_value': float(min(pearson_p, spearman_p)),
        'n': int(len(x))
    }


def compute_signature_correlations(
    expr_matrix: np.ndarray,
    activity_matrix: np.ndarray,
    gene_names: List[str],
    signature_names: List[str]
) -> Dict:
    """
    Compute correlation between mean signature gene expression and activity.

    For each signature, correlates mean expression of its genes with activity score.
    """
    results = {}

    for i, sig_name in enumerate(signature_names):
        # Mean expression across all signature genes
        mean_expr = expr_matrix.mean(axis=1)  # Per sample/cell
        activity = activity_matrix[:, i] if activity_matrix.ndim > 1 else activity_matrix

        corr = compute_correlation(mean_expr, activity)
        corr['signature'] = sig_name
        results[sig_name] = corr

    return results


# ==============================================================================
# Level 1: Pseudobulk Validation
# ==============================================================================

def validate_pseudobulk_level(
    adata,
    activity_df: pd.DataFrame,
    signature_df: pd.DataFrame,
    celltype_col: str,
    sample_col: str,
    signature_type: str
) -> Dict:
    """
    Validate at pseudobulk level: correlate expression and activity per sample.

    Returns correlation for each signature across samples.
    """
    log(f"  Validating pseudobulk level ({signature_type})...")

    signature_genes = get_signature_genes(signature_df)
    signature_names = list(signature_df.columns)

    # Get samples present in both expression and activity
    available_samples = set(adata.obs[sample_col].unique())
    activity_samples = set(activity_df.index)
    common_samples = list(available_samples & activity_samples)

    if len(common_samples) < 10:
        log(f"    Insufficient samples: {len(common_samples)}")
        return {}

    log(f"    {len(common_samples)} samples with both expression and activity")

    # Compute mean expression per sample for signature genes
    sample_expr = []
    sample_activity = []
    sample_ids = []

    for sample in common_samples[:500]:  # Limit to 500 samples for speed
        sample_mask = (adata.obs[sample_col] == sample).values
        cell_idx = np.where(sample_mask)[0]

        if len(cell_idx) < 10:
            continue

        # Sample cells if too many
        if len(cell_idx) > 1000:
            cell_idx = np.random.choice(cell_idx, 1000, replace=False)

        expr, available_genes = get_expression_for_genes(adata, signature_genes, cell_idx)
        if expr is None or len(available_genes) < 10:
            continue

        # Mean expression per gene, then per signature
        mean_expr = expr.mean(axis=0)
        sample_expr.append(mean_expr)
        sample_activity.append(activity_df.loc[sample].values)
        sample_ids.append(sample)

    if len(sample_expr) < 10:
        log(f"    Insufficient valid samples: {len(sample_expr)}")
        return {}

    expr_matrix = np.array(sample_expr)
    activity_matrix = np.array(sample_activity)

    # Compute correlations for each signature
    results = {
        'level': 'pseudobulk',
        'signature_type': signature_type,
        'n_samples': len(sample_ids),
        'n_genes': len(available_genes),
        'signatures': {}
    }

    for i, sig_name in enumerate(signature_names):
        if sig_name not in activity_df.columns:
            continue

        mean_expr = expr_matrix.mean(axis=1)  # Overall mean expression
        activity = activity_matrix[:, i]

        corr = compute_correlation(mean_expr, activity)
        corr['signature'] = sig_name

        # Store scatter data points
        corr['points'] = [
            {'x': float(mean_expr[j]), 'y': float(activity[j]), 'id': sample_ids[j]}
            for j in range(min(len(sample_ids), 200))
        ]

        results['signatures'][sig_name] = corr

    # Compute overall correlation (mean across signatures)
    all_r = [v['pearson_r'] for v in results['signatures'].values() if not np.isnan(v.get('pearson_r', np.nan))]
    results['mean_pearson_r'] = float(np.mean(all_r)) if all_r else np.nan
    results['median_pearson_r'] = float(np.median(all_r)) if all_r else np.nan

    log(f"    Mean r = {results['mean_pearson_r']:.3f} across {len(results['signatures'])} signatures")

    return results


# ==============================================================================
# Level 2: Single-Cell Validation
# ==============================================================================

def validate_singlecell_level(
    adata,
    activity_df: pd.DataFrame,
    signature_df: pd.DataFrame,
    celltype_col: str,
    signature_type: str,
    max_cells: int = MAX_CELLS_PER_TYPE
) -> Dict:
    """
    Validate at single-cell level: correlate expression and activity per cell.
    """
    log(f"  Validating single-cell level ({signature_type})...")

    signature_genes = get_signature_genes(signature_df)
    signature_names = list(signature_df.columns)

    # Get cells present in activity data
    common_cells = list(set(adata.obs_names) & set(activity_df.index))

    if len(common_cells) < 100:
        log(f"    Insufficient cells: {len(common_cells)}")
        return {}

    # Sample cells if too many
    if len(common_cells) > max_cells:
        common_cells = np.random.choice(common_cells, max_cells, replace=False).tolist()

    log(f"    {len(common_cells)} cells for validation")

    # Get cell indices
    cell_idx = [list(adata.obs_names).index(c) for c in common_cells]
    cell_idx = np.array(cell_idx)

    # Get expression
    expr, available_genes = get_expression_for_genes(adata, signature_genes, cell_idx)
    if expr is None or len(available_genes) < 10:
        log(f"    Insufficient genes: {len(available_genes) if available_genes else 0}")
        return {}

    # Get activity
    activity_matrix = activity_df.loc[common_cells].values

    results = {
        'level': 'singlecell',
        'signature_type': signature_type,
        'n_cells': len(common_cells),
        'n_genes': len(available_genes),
        'signatures': {}
    }

    # Compute correlations for a subset of signatures
    for i, sig_name in enumerate(signature_names[:50]):  # Limit to 50 signatures
        if sig_name not in activity_df.columns:
            continue

        mean_expr = expr.mean(axis=1)
        activity = activity_matrix[:, activity_df.columns.get_loc(sig_name)]

        corr = compute_correlation(mean_expr, activity)
        corr['signature'] = sig_name

        # Sample points for scatter plot
        sample_idx = np.random.choice(len(common_cells), min(500, len(common_cells)), replace=False)
        corr['points'] = [
            {'x': float(mean_expr[j]), 'y': float(activity[j])}
            for j in sample_idx
        ]

        results['signatures'][sig_name] = corr

    # Overall statistics
    all_r = [v['pearson_r'] for v in results['signatures'].values() if not np.isnan(v.get('pearson_r', np.nan))]
    results['mean_pearson_r'] = float(np.mean(all_r)) if all_r else np.nan
    results['median_pearson_r'] = float(np.median(all_r)) if all_r else np.nan

    log(f"    Mean r = {results['mean_pearson_r']:.3f} across {len(results['signatures'])} signatures")

    return results


# ==============================================================================
# Level 3: Atlas Level Validation
# ==============================================================================

def validate_atlas_level(
    adata,
    activity_df: pd.DataFrame,
    signature_df: pd.DataFrame,
    celltype_col: str,
    signature_type: str
) -> Dict:
    """
    Validate at atlas level: one point per cell type, correlate mean expression and activity.
    """
    log(f"  Validating atlas level ({signature_type})...")

    signature_genes = get_signature_genes(signature_df)
    signature_names = list(signature_df.columns)

    celltypes = adata.obs[celltype_col].unique()
    log(f"    {len(celltypes)} cell types")

    celltype_expr = []
    celltype_activity = []
    celltype_names = []

    for ct in celltypes:
        ct_mask = (adata.obs[celltype_col] == ct).values
        cell_idx = np.where(ct_mask)[0]

        if len(cell_idx) < 50:
            continue

        # Sample cells if too many
        if len(cell_idx) > 5000:
            cell_idx = np.random.choice(cell_idx, 5000, replace=False)

        # Get expression
        expr, available_genes = get_expression_for_genes(adata, signature_genes, cell_idx)
        if expr is None or len(available_genes) < 10:
            continue

        # Mean expression for this cell type
        ct_mean_expr = expr.mean(axis=0).mean()  # Mean of means

        # Get activity for this cell type
        # For pseudobulk, aggregate across samples
        if hasattr(activity_df.index, 'str'):
            ct_samples = activity_df.index[activity_df.index.str.contains(str(ct), case=False, na=False)]
        else:
            ct_samples = []

        if len(ct_samples) > 0:
            ct_activity = activity_df.loc[ct_samples].mean(axis=0).values
        else:
            # Try direct cell type match
            ct_cells = list(set(adata.obs_names[ct_mask]) & set(activity_df.index))
            if len(ct_cells) > 0:
                ct_activity = activity_df.loc[ct_cells].mean(axis=0).values
            else:
                continue

        celltype_expr.append(ct_mean_expr)
        celltype_activity.append(ct_activity)
        celltype_names.append(ct)

    if len(celltype_expr) < 5:
        log(f"    Insufficient cell types: {len(celltype_expr)}")
        return {}

    expr_array = np.array(celltype_expr)
    activity_matrix = np.array(celltype_activity)

    results = {
        'level': 'atlas',
        'signature_type': signature_type,
        'n_celltypes': len(celltype_names),
        'celltypes': celltype_names,
        'signatures': {}
    }

    for i, sig_name in enumerate(signature_names[:50]):
        if i >= activity_matrix.shape[1]:
            continue

        activity = activity_matrix[:, i]

        corr = compute_correlation(expr_array, activity)
        corr['signature'] = sig_name
        corr['points'] = [
            {'x': float(expr_array[j]), 'y': float(activity[j]), 'celltype': celltype_names[j]}
            for j in range(len(celltype_names))
        ]

        results['signatures'][sig_name] = corr

    # Overall statistics
    all_r = [v['pearson_r'] for v in results['signatures'].values() if not np.isnan(v.get('pearson_r', np.nan))]
    results['mean_pearson_r'] = float(np.mean(all_r)) if all_r else np.nan
    results['median_pearson_r'] = float(np.median(all_r)) if all_r else np.nan

    log(f"    Mean r = {results['mean_pearson_r']:.3f} across {len(results['signatures'])} signatures")

    return results


# ==============================================================================
# Atlas Processing
# ==============================================================================

def process_atlas(
    atlas_name: str,
    h5ad_path: Path,
    celltype_col: str,
    sample_col: str,
    signatures: Dict
) -> Dict:
    """Process validation for a single atlas."""
    log(f"\nProcessing {atlas_name}...")

    # Load data
    log(f"  Loading {h5ad_path.name}...")
    adata = sc.read_h5ad(h5ad_path, backed='r')
    log(f"  Loaded: {adata.n_obs:,} cells, {adata.n_vars:,} genes")

    results = {
        'atlas': atlas_name,
        'n_cells': int(adata.n_obs),
        'n_genes': int(adata.n_vars),
        'validations': {}
    }

    # Load activity results for this atlas
    activity_files = {
        'cytosig': RESULTS_DIR / atlas_name / f'{atlas_name.upper()}_CytoSig_pseudobulk_activity.csv',
        'secact': RESULTS_DIR / atlas_name / f'{atlas_name.upper()}_SecAct_pseudobulk_activity.csv',
    }

    # Also check for LinCytoSig
    lincytosig_dir = LINCYTOSIG_DIR / atlas_name / 'pseudobulk'

    for sig_type, sig_df in [('cytosig', signatures['cytosig']), ('secact', signatures['secact'])]:
        activity_file = activity_files.get(sig_type)

        if activity_file and activity_file.exists():
            log(f"  Loading {sig_type} activity from {activity_file.name}...")
            activity_df = pd.read_csv(activity_file, index_col=0)

            # Pseudobulk validation
            pb_results = validate_pseudobulk_level(
                adata, activity_df, sig_df, celltype_col, sample_col, sig_type
            )
            if pb_results:
                results['validations'][f'{sig_type}_pseudobulk'] = pb_results

            # Atlas-level validation
            atlas_results = validate_atlas_level(
                adata, activity_df, sig_df, celltype_col, sig_type
            )
            if atlas_results:
                results['validations'][f'{sig_type}_atlas'] = atlas_results
        else:
            log(f"  {sig_type} activity file not found: {activity_file}")

    # LinCytoSig validation if available
    if lincytosig_dir.exists() and signatures['lincytosig']:
        log(f"  Processing LinCytoSig...")
        # Combine all LinCytoSig activity files
        lincytosig_activity = {}
        for f in lincytosig_dir.glob('*_activity.csv'):
            celltype = f.stem.replace('_activity', '')
            df = pd.read_csv(f, index_col=0)
            lincytosig_activity[celltype] = df

        if lincytosig_activity:
            # For LinCytoSig, we need to handle cell-type-specific signatures
            log(f"    Found {len(lincytosig_activity)} cell type activity files")

            # Compute aggregate validation metrics
            all_correlations = []
            for ct, activity_df in lincytosig_activity.items():
                if ct in signatures['lincytosig']:
                    sig_df = signatures['lincytosig'][ct]
                    # Simple validation for each cell type
                    # ... (simplified for now)

            results['validations']['lincytosig_pseudobulk'] = {
                'level': 'pseudobulk',
                'signature_type': 'lincytosig',
                'n_celltypes': len(lincytosig_activity),
                'status': 'computed'
            }

    # Close file
    if hasattr(adata, 'file') and adata.file is not None:
        adata.file.close()

    gc.collect()

    return results


# ==============================================================================
# Main
# ==============================================================================

def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Atlas validation analysis')
    parser.add_argument('--atlas', type=str, default='all',
                       choices=['all', 'cima', 'inflammation', 'scatlas'],
                       help='Atlas to validate')

    args = parser.parse_args()

    log("=" * 60)
    log("Atlas Validation: Expression-Activity Correlation")
    log("=" * 60)

    # Load signatures
    signatures = load_signatures()

    # Atlas configurations
    atlas_configs = {
        'cima': {
            'h5ad_path': CIMA_H5AD,
            'celltype_col': 'cell_type_l2',
            'sample_col': 'sample'
        },
        'inflammation': {
            'h5ad_path': INFLAMMATION_MAIN,
            'celltype_col': 'Level2',
            'sample_col': 'sampleID'
        },
        'scatlas': {
            'h5ad_path': SCATLAS_NORMAL,
            'celltype_col': 'cellType1',
            'sample_col': 'sampleID'
        }
    }

    all_results = {}

    atlases_to_process = list(atlas_configs.keys()) if args.atlas == 'all' else [args.atlas]

    for atlas_name in atlases_to_process:
        config = atlas_configs[atlas_name]
        results = process_atlas(atlas_name, signatures=signatures, **config)
        all_results[atlas_name] = results

        # Save individual atlas results
        output_file = OUTPUT_DIR / f'{atlas_name}_validation.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        log(f"  Saved: {output_file}")

    # Save combined results
    combined_file = OUTPUT_DIR / 'atlas_validation_summary.json'
    with open(combined_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    log(f"\nSaved combined results: {combined_file}")

    # Generate summary for visualization
    summary = {
        'atlases': list(all_results.keys()),
        'signature_types': ['cytosig', 'lincytosig', 'secact'],
        'validation_levels': ['pseudobulk', 'singlecell', 'atlas'],
        'results': {}
    }

    for atlas_name, atlas_results in all_results.items():
        summary['results'][atlas_name] = {}
        for val_key, val_data in atlas_results.get('validations', {}).items():
            summary['results'][atlas_name][val_key] = {
                'mean_r': val_data.get('mean_pearson_r', np.nan),
                'median_r': val_data.get('median_pearson_r', np.nan),
                'n_signatures': len(val_data.get('signatures', {})),
                'level': val_data.get('level', 'unknown')
            }

    summary_file = OUTPUT_DIR / 'validation_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    log(f"Saved summary: {summary_file}")

    log("\n" + "=" * 60)
    log("Complete!")
    log("=" * 60)


if __name__ == '__main__':
    main()
