#!/usr/bin/env python3
"""
GPU Validation Runner
====================
Runs activity inference and validation for specified atlas/signature/level combinations.
Outputs validation results and scatter plot data for CytoAtlas service.

Usage:
    python run_validation.py --atlas cima --signature cytosig --aggregation pseudobulk --level L2
    python run_validation.py --config-index 0  # Run first test from matrix
    python run_validation.py --all  # Run all tests (for SLURM array)
"""

import argparse
import gc
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
from scipy import stats

# Add paths
sys.path.insert(0, '/vf/users/parks34/projects/1ridgesig/SecActpy')
sys.path.insert(0, '/vf/users/parks34/projects/2secactpy/cytoatlas-pipeline/src')

from secactpy import (
    load_cytosig, load_secact, load_lincytosig,
    ridge_batch, estimate_batch_size,
    CUPY_AVAILABLE
)

from config import (
    ATLAS_CONFIG, SIGNATURE_CONFIG, AGGREGATION_CONFIG,
    VALIDATION_CONFIG, OUTPUT_ROOT, generate_test_matrix,
    get_output_path, get_scatter_data_path
)

warnings.filterwarnings('ignore')

# ==============================================================================
# Logging
# ==============================================================================

def log(msg: str):
    """Print timestamped log message."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ==============================================================================
# Signature Loading
# ==============================================================================

def load_signature(signature_type: str) -> pd.DataFrame:
    """Load signature matrix."""
    if signature_type == 'cytosig':
        return load_cytosig()
    elif signature_type == 'lincytosig':
        return load_lincytosig()
    elif signature_type == 'secact':
        return load_secact()
    else:
        raise ValueError(f"Unknown signature type: {signature_type}")


# ==============================================================================
# Aggregation Methods
# ==============================================================================

def aggregate_pseudobulk(
    adata: ad.AnnData,
    cell_type_col: str,
    sample_col: str,
    min_cells: int = 10,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Standard pseudobulk aggregation.

    Returns:
        expression_df: (genes × pseudobulk_samples)
        metadata_df: metadata for each pseudobulk sample
    """
    log(f"Aggregating pseudobulk by {cell_type_col} × {sample_col}...")

    obs = adata.obs[[cell_type_col, sample_col]].copy()
    obs['group'] = obs[cell_type_col].astype(str) + '__' + obs[sample_col].astype(str)

    groups = obs.groupby('group', observed=True).groups

    valid_groups = {k: v for k, v in groups.items() if len(v) >= min_cells}
    log(f"  {len(valid_groups)} groups with >= {min_cells} cells")

    # Aggregate
    expr_data = {}
    meta_records = []

    X = adata.X
    genes = list(adata.var_names)

    for group_name, indices in valid_groups.items():
        idx_list = list(indices)

        # Sum counts
        if sp.issparse(X):
            group_sum = np.asarray(X[idx_list].sum(axis=0)).flatten()
        else:
            group_sum = X[idx_list].sum(axis=0)

        # CPM normalize
        total = group_sum.sum()
        if total > 0:
            cpm = (group_sum / total) * 1e6
        else:
            cpm = group_sum

        expr_data[group_name] = cpm

        # Metadata
        parts = group_name.split('__')
        meta_records.append({
            'pseudobulk_id': group_name,
            'cell_type': parts[0],
            'sample': parts[1] if len(parts) > 1 else parts[0],
            'n_cells': len(idx_list),
        })

    expr_df = pd.DataFrame(expr_data, index=genes)
    meta_df = pd.DataFrame(meta_records).set_index('pseudobulk_id')

    log(f"  Expression matrix: {expr_df.shape}")
    return expr_df, meta_df


def aggregate_resampled(
    adata: ad.AnnData,
    cell_type_col: str,
    sample_col: str,
    n_cells_per_group: int = 100,
    n_replicates: int = 10,
    min_cells: int = 50,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Resampled pseudobulk aggregation to normalize cell counts.

    Returns:
        expression_df: (genes × pseudobulk_samples)
        metadata_df: metadata for each pseudobulk sample
    """
    log(f"Aggregating resampled pseudobulk (n={n_cells_per_group}, reps={n_replicates})...")

    np.random.seed(seed)

    obs = adata.obs[[cell_type_col, sample_col]].copy()
    obs['group'] = obs[cell_type_col].astype(str) + '__' + obs[sample_col].astype(str)

    groups = obs.groupby('group', observed=True).groups
    valid_groups = {k: v for k, v in groups.items() if len(v) >= min_cells}
    log(f"  {len(valid_groups)} groups with >= {min_cells} cells")

    expr_data = {}
    meta_records = []

    X = adata.X
    genes = list(adata.var_names)

    for group_name, indices in valid_groups.items():
        idx_list = list(indices)
        n_available = len(idx_list)
        n_sample = min(n_cells_per_group, n_available)

        for rep in range(n_replicates):
            # Random sample with replacement if needed
            sampled_idx = np.random.choice(idx_list, size=n_sample, replace=(n_sample > n_available))

            if sp.issparse(X):
                group_sum = np.asarray(X[sampled_idx].sum(axis=0)).flatten()
            else:
                group_sum = X[sampled_idx].sum(axis=0)

            # CPM normalize
            total = group_sum.sum()
            cpm = (group_sum / total) * 1e6 if total > 0 else group_sum

            rep_name = f"{group_name}__rep{rep}"
            expr_data[rep_name] = cpm

            parts = group_name.split('__')
            meta_records.append({
                'pseudobulk_id': rep_name,
                'cell_type': parts[0],
                'sample': parts[1] if len(parts) > 1 else parts[0],
                'replicate': rep,
                'n_cells': n_sample,
            })

    expr_df = pd.DataFrame(expr_data, index=genes)
    meta_df = pd.DataFrame(meta_records).set_index('pseudobulk_id')

    log(f"  Expression matrix: {expr_df.shape}")
    return expr_df, meta_df


def aggregate_singlecell(
    adata: ad.AnnData,
    cell_type_col: str,
    max_cells: int = 100000,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Single-cell level (subsampled if too large).

    Returns:
        expression_df: (genes × cells)
        metadata_df: cell metadata
    """
    log(f"Preparing single-cell data (max {max_cells} cells)...")

    n_cells = adata.n_obs

    if n_cells > max_cells:
        np.random.seed(seed)
        idx = np.random.choice(n_cells, size=max_cells, replace=False)
        idx = np.sort(idx)
        log(f"  Subsampled {max_cells} from {n_cells} cells")
    else:
        idx = np.arange(n_cells)

    X = adata.X[idx]
    obs = adata.obs.iloc[idx].copy()

    # CPM normalize per cell
    if sp.issparse(X):
        row_sums = np.asarray(X.sum(axis=1)).flatten()
        # Normalize in chunks to save memory
        X_dense = X.toarray()
    else:
        row_sums = X.sum(axis=1)
        X_dense = X

    row_sums[row_sums == 0] = 1
    X_cpm = (X_dense.T / row_sums * 1e6).T

    cell_ids = obs.index.tolist()
    genes = list(adata.var_names)

    expr_df = pd.DataFrame(X_cpm.T, index=genes, columns=cell_ids)
    meta_df = obs[[cell_type_col]].copy()
    meta_df.columns = ['cell_type']

    log(f"  Expression matrix: {expr_df.shape}")
    return expr_df, meta_df


# ==============================================================================
# Activity Inference
# ==============================================================================

def run_activity_inference(
    expression: pd.DataFrame,
    signature: pd.DataFrame,
    n_rand: int = 1000,
    seed: int = 42,
    batch_size: int = 10000,
) -> pd.DataFrame:
    """
    Run ridge regression activity inference.

    Args:
        expression: (genes × samples) expression matrix
        signature: (genes × signatures) signature matrix

    Returns:
        activity: (signatures × samples) activity z-scores
    """
    log(f"Running activity inference ({expression.shape[1]} samples, {signature.shape[1]} signatures)...")

    # Align genes
    common_genes = list(set(expression.index) & set(signature.index))
    log(f"  {len(common_genes)} common genes")

    if len(common_genes) < 100:
        raise ValueError(f"Too few common genes: {len(common_genes)}")

    expr_aligned = expression.loc[common_genes]
    sig_aligned = signature.loc[common_genes]

    # Convert to numpy
    X = expr_aligned.values.T  # (samples × genes)
    Y = sig_aligned.values     # (genes × signatures)

    # Run ridge regression
    n_samples = X.shape[0]

    if n_samples <= batch_size:
        # Single batch
        from secactpy import ridge
        activity, _, pvalues = ridge(X, Y, nrand=n_rand, seed=seed)
    else:
        # Batch processing
        log(f"  Using batch processing ({n_samples} samples)")
        results = ridge_batch(
            X, Y,
            batch_size=batch_size,
            nrand=n_rand,
            seed=seed,
            verbose=True
        )
        activity = results['zscore']
        pvalues = results['pvalue']

    # Create DataFrame
    activity_df = pd.DataFrame(
        activity.T,
        index=sig_aligned.columns,
        columns=expr_aligned.columns
    )

    log(f"  Activity matrix: {activity_df.shape}")
    return activity_df


# ==============================================================================
# Validation
# ==============================================================================

def validate_expression_vs_activity(
    expression: pd.DataFrame,
    activity: pd.DataFrame,
    signature_type: str,
    metadata: pd.DataFrame,
    min_samples: int = 10,
) -> pd.DataFrame:
    """
    Validate activity by correlating with target gene expression.

    For CytoSig/SecAct: signature name should match gene name
    For LinCytoSig: CellType__Cytokine format, validate within matching cell type

    Returns:
        DataFrame with validation results per signature
    """
    log("Validating expression vs activity...")

    # Map CytoSig names to HGNC
    CYTOSIG_TO_HGNC = {
        'TNFA': 'TNF', 'IFNA': 'IFNA1', 'IFNB': 'IFNB1', 'IFNL': 'IFNL1',
        'GMCSF': 'CSF2', 'GCSF': 'CSF3', 'MCSF': 'CSF1',
        'IL12': 'IL12A', 'Activin A': 'INHBA', 'TWEAK': 'TNFSF12',
        'CD40L': 'CD40LG', 'PDL1': 'CD274',
    }

    results = []
    expr_genes_upper = {g.upper(): g for g in expression.index}

    for sig_name in activity.index:
        # Determine target gene
        if signature_type == 'lincytosig' and '__' in sig_name:
            # LinCytoSig: CellType__Cytokine
            parts = sig_name.split('__')
            cell_type = parts[0]
            cytokine = parts[1]
            gene_name = CYTOSIG_TO_HGNC.get(cytokine, cytokine)

            # Filter to matching cell type
            if 'cell_type' in metadata.columns:
                mask = metadata['cell_type'] == cell_type
                valid_samples = metadata.index[mask].tolist()
                valid_samples = [s for s in valid_samples if s in activity.columns]
            else:
                valid_samples = list(activity.columns)
        else:
            # Standard signature
            gene_name = CYTOSIG_TO_HGNC.get(sig_name, sig_name)
            valid_samples = list(activity.columns)
            cell_type = None
            cytokine = sig_name

        # Find gene in expression
        if gene_name.upper() not in expr_genes_upper:
            continue
        actual_gene = expr_genes_upper[gene_name.upper()]

        # Get valid samples
        common_samples = [s for s in valid_samples if s in expression.columns]
        if len(common_samples) < min_samples:
            continue

        # Get values
        expr_vals = expression.loc[actual_gene, common_samples].values
        act_vals = activity.loc[sig_name, common_samples].values

        # Remove NaN
        mask = ~(np.isnan(expr_vals) | np.isnan(act_vals))
        if mask.sum() < min_samples:
            continue

        expr_vals = expr_vals[mask]
        act_vals = act_vals[mask]

        # Correlations
        r_pearson, p_pearson = stats.pearsonr(expr_vals, act_vals)
        r_spearman, p_spearman = stats.spearmanr(expr_vals, act_vals)

        results.append({
            'signature': sig_name,
            'gene': actual_gene,
            'cell_type': cell_type,
            'cytokine': cytokine,
            'pearson_r': r_pearson,
            'pearson_p': p_pearson,
            'spearman_r': r_spearman,
            'spearman_p': p_spearman,
            'r2': r_pearson ** 2,
            'n_samples': len(expr_vals),
            'mean_expression': np.mean(expr_vals),
            'mean_activity': np.mean(act_vals),
        })

    results_df = pd.DataFrame(results)

    if len(results_df) > 0:
        # FDR correction
        from statsmodels.stats.multitest import multipletests
        _, results_df['pearson_q'], _, _ = multipletests(
            results_df['pearson_p'], method='fdr_bh'
        )
        _, results_df['spearman_q'], _, _ = multipletests(
            results_df['spearman_p'], method='fdr_bh'
        )

    log(f"  Validated {len(results_df)} signatures")
    return results_df


def generate_scatter_data(
    expression: pd.DataFrame,
    activity: pd.DataFrame,
    validation_results: pd.DataFrame,
    metadata: pd.DataFrame,
    atlas: str,
    signature_type: str,
    level: str,
    max_points_per_sig: int = 500,
) -> Dict:
    """
    Generate scatter plot data for CytoAtlas service.
    """
    log("Generating scatter plot data...")

    CYTOSIG_TO_HGNC = {
        'TNFA': 'TNF', 'IFNA': 'IFNA1', 'IFNB': 'IFNB1', 'IFNL': 'IFNL1',
        'GMCSF': 'CSF2', 'GCSF': 'CSF3', 'MCSF': 'CSF1',
        'IL12': 'IL12A', 'Activin A': 'INHBA', 'TWEAK': 'TNFSF12',
        'CD40L': 'CD40LG', 'PDL1': 'CD274',
    }

    expr_genes_upper = {g.upper(): g for g in expression.index}

    signatures_data = []

    for _, row in validation_results.iterrows():
        sig_name = row['signature']
        gene = row['gene']

        if gene not in expression.index:
            continue

        # Get common samples
        common = [c for c in activity.columns if c in expression.columns]

        # Subsample if needed
        if len(common) > max_points_per_sig:
            np.random.seed(42)
            common = list(np.random.choice(common, max_points_per_sig, replace=False))

        # Get values
        expr_vals = expression.loc[gene, common].values
        act_vals = activity.loc[sig_name, common].values

        # Build data points
        points = []
        for i, sample in enumerate(common):
            ct = metadata.loc[sample, 'cell_type'] if sample in metadata.index else 'unknown'
            points.append({
                'expression': float(expr_vals[i]) if not np.isnan(expr_vals[i]) else None,
                'activity': float(act_vals[i]) if not np.isnan(act_vals[i]) else None,
                'celltype': str(ct),
                'sample': str(sample),
            })

        # Filter out None values
        points = [p for p in points if p['expression'] is not None and p['activity'] is not None]

        signatures_data.append({
            'name': sig_name,
            'gene': gene,
            'data': points,
            'correlation': float(row['pearson_r']),
            'pvalue': float(row['pearson_p']),
            'n': int(row['n_samples']),
        })

    return {
        'atlas': atlas,
        'signature_type': signature_type,
        'level': level,
        'n_signatures': len(signatures_data),
        'signatures': signatures_data,
    }


# ==============================================================================
# Main Runner
# ==============================================================================

def run_validation_test(
    atlas: str,
    signature: str,
    aggregation: str,
    level: str,
    save_results: bool = True,
) -> Dict:
    """
    Run a single validation test.

    Args:
        atlas: Atlas key (cima, inflammation, etc.)
        signature: Signature type (cytosig, lincytosig, secact)
        aggregation: Aggregation method (pseudobulk, resampled, singlecell)
        level: Cell type level (L1, L2, L3)
        save_results: Whether to save outputs to disk

    Returns:
        Dictionary with validation summary
    """
    start_time = time.time()

    log("=" * 60)
    log(f"Validation Test: {atlas} / {signature} / {aggregation} / {level}")
    log("=" * 60)

    # Get configuration
    atlas_cfg = ATLAS_CONFIG[atlas]
    sig_cfg = SIGNATURE_CONFIG[signature]
    agg_cfg = AGGREGATION_CONFIG[aggregation]

    cell_type_col = atlas_cfg['cell_type_columns'][level]
    sample_col = atlas_cfg['sample_col']
    h5ad_path = atlas_cfg['h5ad_path']

    # Load data
    log(f"Loading {h5ad_path}...")
    adata = ad.read_h5ad(h5ad_path, backed='r')
    log(f"  Shape: {adata.shape}")

    # Load signature
    sig_matrix = load_signature(signature)
    log(f"  Signature: {sig_matrix.shape}")

    # Aggregate expression
    if aggregation == 'pseudobulk':
        expr_df, meta_df = aggregate_pseudobulk(
            adata, cell_type_col, sample_col,
            min_cells=agg_cfg['min_cells']
        )
    elif aggregation == 'resampled':
        expr_df, meta_df = aggregate_resampled(
            adata, cell_type_col, sample_col,
            n_cells_per_group=agg_cfg['n_cells_per_group'],
            n_replicates=agg_cfg['n_replicates'],
            min_cells=agg_cfg['min_cells'],
            seed=VALIDATION_CONFIG['seed']
        )
    elif aggregation == 'singlecell':
        expr_df, meta_df = aggregate_singlecell(
            adata, cell_type_col,
            max_cells=agg_cfg['max_cells'],
            seed=VALIDATION_CONFIG['seed']
        )
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")

    # Close backed file
    adata.file.close()
    del adata
    gc.collect()

    # Run activity inference
    activity_df = run_activity_inference(
        expr_df, sig_matrix,
        n_rand=VALIDATION_CONFIG['n_rand'],
        seed=VALIDATION_CONFIG['seed'],
        batch_size=agg_cfg.get('batch_size', 10000)
    )

    # Validate
    validation_df = validate_expression_vs_activity(
        expr_df, activity_df, signature, meta_df,
        min_samples=VALIDATION_CONFIG['min_samples']
    )

    # Generate scatter data
    scatter_data = generate_scatter_data(
        expr_df, activity_df, validation_df, meta_df,
        atlas_cfg['name'], sig_cfg['name'], level
    )

    # Summary stats
    n_sig = len(validation_df)
    n_pos = (validation_df['pearson_r'] > 0).sum() if n_sig > 0 else 0
    n_significant = (validation_df['pearson_q'] < 0.05).sum() if n_sig > 0 else 0
    mean_r = validation_df['pearson_r'].mean() if n_sig > 0 else 0
    mean_r2 = validation_df['r2'].mean() if n_sig > 0 else 0

    elapsed = time.time() - start_time

    summary = {
        'atlas': atlas,
        'signature': signature,
        'aggregation': aggregation,
        'level': level,
        'n_signatures_validated': n_sig,
        'n_positive_correlation': n_pos,
        'n_significant_fdr05': n_significant,
        'mean_pearson_r': mean_r,
        'mean_r2': mean_r2,
        'elapsed_seconds': elapsed,
    }

    log(f"\nSummary:")
    log(f"  Signatures validated: {n_sig}")
    log(f"  Positive correlation: {n_pos} ({100*n_pos/max(1,n_sig):.1f}%)")
    log(f"  Significant (FDR<0.05): {n_significant}")
    log(f"  Mean r: {mean_r:.3f}")
    log(f"  Mean R²: {mean_r2:.3f}")
    log(f"  Elapsed: {elapsed:.1f}s")

    # Save results
    if save_results:
        # Validation results
        output_path = get_output_path(atlas, signature, aggregation, level)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        validation_df.to_csv(output_path, index=False)
        log(f"  Saved: {output_path}")

        # Scatter data
        scatter_path = get_scatter_data_path(atlas, signature, f"{aggregation}_{level}")
        scatter_path.parent.mkdir(parents=True, exist_ok=True)
        with open(scatter_path, 'w') as f:
            json.dump(scatter_data, f, indent=2)
        log(f"  Saved: {scatter_path}")

    log("=" * 60)
    return summary


# ==============================================================================
# CLI
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description='Run GPU validation tests')
    parser.add_argument('--atlas', type=str, help='Atlas key')
    parser.add_argument('--signature', type=str, help='Signature type')
    parser.add_argument('--aggregation', type=str, help='Aggregation method')
    parser.add_argument('--level', type=str, help='Cell type level')
    parser.add_argument('--config-index', type=int, help='Run test by index from matrix')
    parser.add_argument('--all', action='store_true', help='Run all tests')
    parser.add_argument('--dry-run', action='store_true', help='Show tests without running')

    args = parser.parse_args()

    tests = generate_test_matrix()

    if args.dry_run:
        print(f"Total tests: {len(tests)}")
        for i, t in enumerate(tests):
            print(f"  [{i}] {t['atlas']} / {t['signature']} / {t['aggregation']} / {t['level']}")
        return

    if args.config_index is not None:
        test = tests[args.config_index]
        run_validation_test(
            test['atlas'], test['signature'],
            test['aggregation'], test['level']
        )
    elif args.all:
        for i, test in enumerate(tests):
            log(f"\n\n{'#'*60}")
            log(f"Test {i+1}/{len(tests)}")
            log(f"{'#'*60}\n")
            try:
                run_validation_test(
                    test['atlas'], test['signature'],
                    test['aggregation'], test['level']
                )
            except Exception as e:
                log(f"ERROR: {e}")
                continue
    elif args.atlas and args.signature and args.aggregation and args.level:
        run_validation_test(args.atlas, args.signature, args.aggregation, args.level)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
