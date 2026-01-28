#!/usr/bin/env python3
"""
scAtlas Analysis
================
Compute cytokine (CytoSig, 44 signatures) and secreted protein (SecAct, 1,249
signatures) activities from raw counts in the scAtlas datasets using SecActpy.

Datasets:
- Normal organs: 2,293,951 cells, 35 organs
- PanCancer: 4,146,975 cells, multiple cancer types

Analysis focus:
- Compute activities from raw counts (pseudo-bulk by tissue/cell type)
- Organ-specific cytokine/secreted protein signatures
- Cell type-specific activities across organs
- Tumor vs adjacent tissue comparison
- Normal vs cancer comparison for matched organs
"""

import os
import sys
import gc
import warnings
import time
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
from scipy import stats

# Add SecActpy to path
sys.path.insert(0, '/vf/users/parks34/projects/1ridgesig/SecActpy')
from secactpy import (
    load_cytosig, load_secact,
    ridge_batch, estimate_batch_size,
    CUPY_AVAILABLE
)

# ==============================================================================
# Configuration
# ==============================================================================

# Data paths
DATA_DIR = Path('/data/Jiang_Lab/Data/Seongyong/scAtlas_2025')

# Count data (raw counts for activity computation)
NORMAL_COUNTS = DATA_DIR / 'igt_s9_fine_counts.h5ad'
CANCER_COUNTS = DATA_DIR / 'PanCancer_igt_s9_fine_counts.h5ad'

# Output paths
OUTPUT_DIR = Path('/vf/users/parks34/projects/2secactpy/results/scatlas')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Analysis parameters
TISSUE_COL = 'tissue'
CELLTYPE_COARSE = 'cellType1'
CELLTYPE_FINE = 'cellType2'

# Activity computation parameters
BATCH_SIZE = 10000
N_RAND = 1000
SEED = 0
BACKEND = 'cupy' if CUPY_AVAILABLE else 'numpy'

# ==============================================================================
# Utility Functions
# ==============================================================================

def log(msg: str):
    """Print timestamped log message."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def aggregate_by_tissue_celltype(
    adata: ad.AnnData,
    tissue_col: str,
    celltype_col: str,
    min_cells: int = 50,
    extra_cols: List[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aggregate expression by tissue and cell type for pseudo-bulk analysis.

    Args:
        adata: AnnData with raw counts
        tissue_col: Column for tissue/organ
        celltype_col: Column for cell type
        min_cells: Minimum cells per group
        extra_cols: Additional columns to include in metadata (e.g., ['cancerType', 'donorID'])

    Returns:
        expr_df: DataFrame (genes x (tissue_celltype combinations))
        meta_df: DataFrame with tissue and cell type info for each column
    """
    log(f"Aggregating by {tissue_col} and {celltype_col}...")

    # Determine columns to extract
    cols_to_use = [tissue_col, celltype_col]
    if extra_cols:
        cols_to_use.extend([c for c in extra_cols if c in adata.obs.columns])

    # Get unique combinations
    obs = adata.obs[cols_to_use].copy()
    obs = obs.reset_index(drop=True)
    groups = obs.groupby([tissue_col, celltype_col], observed=True).groups

    log(f"  Found {len(groups)} tissue-celltype combinations")

    # Get raw counts
    if 'counts' in adata.layers:
        X = adata.layers['counts']
        log("  Using raw counts from layers['counts']")
    else:
        X = adata.X
        log("  Using .X (may be log-normalized)")

    gene_names = list(adata.var_names)

    aggregated = {}
    meta_dict = {}  # Use dict instead of list to avoid duplicates

    for i, ((tissue, celltype), indices) in enumerate(groups.items()):
        if len(indices) < min_cells:
            continue

        col_name = f"{tissue}_{celltype}"
        idx_array = np.array(indices, dtype=np.int64)

        # Sum counts for this group
        if sp.issparse(X):
            group_sum = np.asarray(X[idx_array, :].sum(axis=0)).ravel()
        else:
            group_sum = np.asarray(X[idx_array, :].sum(axis=0)).ravel()

        # If duplicate, merge counts and cells
        if col_name in aggregated:
            aggregated[col_name] = aggregated[col_name] + group_sum
            meta_dict[col_name]['n_cells'] += len(idx_array)
        else:
            aggregated[col_name] = group_sum
            meta_entry = {
                'tissue': tissue,
                'cell_type': celltype,
                'n_cells': len(idx_array)
            }
            # Add extra columns (use most common value for categorical)
            if extra_cols:
                for ec in extra_cols:
                    if ec in obs.columns:
                        ec_vals = obs.loc[idx_array, ec].dropna()
                        if len(ec_vals) > 0:
                            meta_entry[ec] = ec_vals.mode().iloc[0] if len(ec_vals.mode()) > 0 else ec_vals.iloc[0]
            meta_dict[col_name] = meta_entry

        if (i + 1) % 500 == 0:
            log(f"    Processed {i + 1}/{len(groups)} groups...")

    expr_df = pd.DataFrame(aggregated, index=gene_names)
    # Create metadata from the same keys as aggregated dict to ensure consistency
    meta_rows = [{'column': k, **v} for k, v in meta_dict.items()]
    meta_df = pd.DataFrame(meta_rows).set_index('column')
    # Ensure same order as expr_df columns
    meta_df = meta_df.loc[expr_df.columns]

    log(f"  Aggregated expression: {expr_df.shape}")
    log(f"  Total cells: {meta_df['n_cells'].sum():,}")

    return expr_df, meta_df


def normalize_and_transform(expr_df: pd.DataFrame) -> pd.DataFrame:
    """TPM normalize and log2 transform expression data."""
    col_sums = expr_df.sum(axis=0)
    expr_tpm = expr_df / col_sums * 1e6
    expr_log = np.log2(expr_tpm + 1)
    return expr_log


def compute_differential(expr_df: pd.DataFrame) -> pd.DataFrame:
    """Compute differential expression (subtract row mean)."""
    row_means = expr_df.mean(axis=1)
    diff = expr_df.subtract(row_means, axis=0)
    return diff


def run_activity_inference(
    expr_df: pd.DataFrame,
    signature: pd.DataFrame,
    sig_name: str,
    output_prefix: str = None
) -> Dict[str, pd.DataFrame]:
    """
    Run SecActpy activity inference.

    Args:
        expr_df: Differential expression (genes x samples)
        signature: Signature matrix (genes x proteins)
        sig_name: Name for logging
        output_prefix: Prefix for output files (optional)

    Returns:
        dict with beta, se, zscore, pvalue DataFrames
    """
    log(f"Running {sig_name} activity inference...")

    # Find overlapping genes
    expr_genes = set(expr_df.index.str.upper())
    sig_genes = set(signature.index.str.upper())
    common_genes = list(expr_genes & sig_genes)

    log(f"  Common genes: {len(common_genes)} / {len(sig_genes)} signature genes")

    if len(common_genes) < 10:
        raise ValueError(f"Too few common genes: {len(common_genes)}")

    # Align data (handle duplicate gene symbols by keeping first occurrence)
    expr_aligned = expr_df.copy()
    expr_aligned.index = expr_aligned.index.str.upper()
    expr_aligned = expr_aligned[~expr_aligned.index.duplicated(keep='first')]
    expr_aligned = expr_aligned.loc[expr_aligned.index.isin(common_genes)]

    sig_aligned = signature.copy()
    sig_aligned.index = sig_aligned.index.str.upper()
    sig_aligned = sig_aligned[~sig_aligned.index.duplicated(keep='first')]

    # Re-compute common genes after dedup
    common_genes = list(set(expr_aligned.index) & set(sig_aligned.index))
    expr_aligned = expr_aligned.loc[common_genes]
    sig_aligned = sig_aligned.loc[common_genes]

    # Z-score normalize columns
    expr_scaled = (expr_aligned - expr_aligned.mean()) / expr_aligned.std(ddof=1)
    expr_scaled = expr_scaled.fillna(0)

    sig_scaled = (sig_aligned - sig_aligned.mean()) / sig_aligned.std(ddof=1)
    sig_scaled = sig_scaled.fillna(0)

    # Run ridge regression
    n_samples = expr_scaled.shape[1]

    if n_samples > 1000:
        batch_size = min(BATCH_SIZE, n_samples)
        result = ridge_batch(
            X=sig_scaled.values,
            Y=expr_scaled.values,
            lambda_=5e5,
            n_rand=N_RAND,
            seed=SEED,
            batch_size=batch_size,
            backend=BACKEND,
            verbose=True
        )
    else:
        from secactpy import ridge
        result = ridge(
            X=sig_scaled.values,
            Y=expr_scaled.values,
            lambda_=5e5,
            n_rand=N_RAND,
            seed=SEED,
            backend=BACKEND,
            verbose=True
        )

    # Convert to DataFrames
    feature_names = list(sig_scaled.columns)
    sample_names = list(expr_scaled.columns)

    result_df = {
        'beta': pd.DataFrame(result['beta'], index=feature_names, columns=sample_names),
        'se': pd.DataFrame(result['se'], index=feature_names, columns=sample_names),
        'zscore': pd.DataFrame(result['zscore'], index=feature_names, columns=sample_names),
        'pvalue': pd.DataFrame(result['pvalue'], index=feature_names, columns=sample_names),
    }

    log(f"  Activity matrix: {result_df['zscore'].shape}")
    log(f"  Time: {result['time']:.2f}s")

    return result_df


def compute_activities_from_counts(
    counts_path: Path,
    tissue_col: str,
    celltype_col: str,
    dataset_name: str,
    extra_cols: List[str] = None
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Complete pipeline: load counts, aggregate, compute activities.

    Args:
        counts_path: Path to h5ad with raw counts
        tissue_col: Column for tissue grouping
        celltype_col: Column for cell type
        dataset_name: Name for logging and output files
        extra_cols: Additional columns to include in metadata

    Returns:
        cytosig_results: Dict with CytoSig activity matrices
        secact_results: Dict with SecAct activity matrices
        agg_meta: Aggregation metadata
    """
    log(f"\n{'='*60}")
    log(f"COMPUTING ACTIVITIES: {dataset_name}")
    log(f"{'='*60}")

    # Load counts data
    log(f"Loading counts: {counts_path}")
    adata = ad.read_h5ad(counts_path, backed='r')
    log(f"  Shape: {adata.shape}")
    log(f"  Columns: {list(adata.obs.columns)}")

    # Aggregate by tissue and cell type
    expr_df, agg_meta = aggregate_by_tissue_celltype(
        adata, tissue_col, celltype_col, extra_cols=extra_cols
    )

    # Release memory
    del adata
    gc.collect()

    # Normalize and compute differential
    log("Normalizing (TPM + log2)...")
    expr_log = normalize_and_transform(expr_df)
    log("Computing differential expression...")
    expr_diff = compute_differential(expr_log)

    del expr_df, expr_log
    gc.collect()

    # Load signatures
    log("Loading CytoSig signatures...")
    cytosig = load_cytosig()
    log(f"  CytoSig: {cytosig.shape}")

    log("Loading SecAct signatures...")
    secact = load_secact()
    log(f"  SecAct: {secact.shape}")

    # Compute activities
    cytosig_results = run_activity_inference(
        expr_diff, cytosig, 'CytoSig', f'{dataset_name}_cytosig'
    )
    secact_results = run_activity_inference(
        expr_diff, secact, 'SecAct', f'{dataset_name}_secact'
    )

    # Save activity results
    log(f"Saving activity results to {OUTPUT_DIR}...")

    for name, df in cytosig_results.items():
        df.to_csv(OUTPUT_DIR / f'{dataset_name}_cytosig_{name}.csv')
    for name, df in secact_results.items():
        df.to_csv(OUTPUT_DIR / f'{dataset_name}_secact_{name}.csv')

    agg_meta.to_csv(OUTPUT_DIR / f'{dataset_name}_aggregation_meta.csv')

    log(f"Saved activity results for {dataset_name}")

    return cytosig_results, secact_results, agg_meta


def activity_to_adata(
    activity_results: Dict[str, pd.DataFrame],
    agg_meta: pd.DataFrame,
    sig_name: str
) -> ad.AnnData:
    """
    Convert activity results to AnnData for downstream analysis.

    Args:
        activity_results: Dict with zscore, pvalue, etc.
        agg_meta: Aggregation metadata
        sig_name: Signature name (CytoSig or SecAct)

    Returns:
        AnnData with activity z-scores and metadata
    """
    zscore_df = activity_results['zscore']

    # Filter to columns present in BOTH results and metadata
    # This handles any mismatches from aggregation or activity inference
    result_cols = set(zscore_df.columns)
    meta_cols = set(agg_meta.index)
    valid_cols = list(result_cols & meta_cols)

    if len(valid_cols) < len(result_cols):
        log(f"  Note: Filtered {len(result_cols) - len(valid_cols)} columns not in metadata")
    if len(valid_cols) < len(meta_cols):
        log(f"  Note: Filtered {len(meta_cols) - len(valid_cols)} metadata rows not in results")

    # Maintain order from results
    valid_cols = [c for c in zscore_df.columns if c in valid_cols]

    zscore_filtered = zscore_df[valid_cols]
    beta_filtered = activity_results['beta'][valid_cols]
    pvalue_filtered = activity_results['pvalue'][valid_cols]
    obs_df = agg_meta.loc[valid_cols].copy()

    # Final dimension check
    assert zscore_filtered.shape[1] == len(obs_df), \
        f"Dimension mismatch: X has {zscore_filtered.shape[1]} cols, obs has {len(obs_df)} rows"

    # Create AnnData (samples/columns x features/proteins)
    adata = ad.AnnData(
        X=zscore_filtered.T.values,
        obs=obs_df,
        var=pd.DataFrame(index=zscore_filtered.index)
    )

    # Store other results in layers
    adata.layers['beta'] = beta_filtered.T.values
    adata.layers['pvalue'] = pvalue_filtered.T.values

    adata.uns['signature_type'] = sig_name

    return adata


def compute_organ_signatures(
    adata: ad.AnnData,
    tissue_col: str = 'tissue'
) -> pd.DataFrame:
    """
    Compute organ-specific activity signatures.

    For each organ, compute mean activity per signature and identify
    significantly elevated signatures.

    Returns:
        DataFrame with organ, signature, mean_activity, specificity_score
    """
    log("Computing organ-specific signatures...")

    if tissue_col not in adata.obs.columns:
        raise ValueError(f"Column '{tissue_col}' not found in obs")

    organs = adata.obs[tissue_col].unique()
    log(f"  Found {len(organs)} organs")

    # Get activity matrix
    if sp.issparse(adata.X):
        X = adata.X.toarray()
    else:
        X = np.array(adata.X)

    signatures = list(adata.var_names)

    # Compute mean activity per organ
    results = []

    for organ in organs:
        mask = (adata.obs[tissue_col] == organ).values
        organ_activity = X[mask, :]
        mean_activity = organ_activity.mean(axis=0)

        # Compute other organs mean for comparison
        other_activity = X[~mask, :]
        other_mean = other_activity.mean(axis=0)

        # Specificity score: (organ_mean - global_mean) / global_std
        global_mean = X.mean(axis=0)
        global_std = X.std(axis=0)
        global_std[global_std < 1e-6] = 1  # Avoid division by zero

        specificity = (mean_activity - global_mean) / global_std

        for i, sig in enumerate(signatures):
            results.append({
                'organ': organ,
                'signature': sig,
                'mean_activity': mean_activity[i],
                'other_mean': other_mean[i],
                'log2fc': np.log2(mean_activity[i] + 0.01) - np.log2(other_mean[i] + 0.01),
                'specificity_score': specificity[i],
                'n_cells': mask.sum()
            })

    result_df = pd.DataFrame(results)
    log(f"  Computed {len(result_df)} organ-signature combinations")

    return result_df


def compute_celltype_signatures(
    adata: ad.AnnData,
    celltype_col: str = 'cellType1',
    tissue_col: str = 'tissue',
    min_cells: int = 100
) -> pd.DataFrame:
    """
    Compute cell type-specific activity signatures across organs.

    Returns:
        DataFrame with cell_type, organ, signature, mean_activity
    """
    log(f"Computing cell type signatures ({celltype_col})...")

    if celltype_col not in adata.obs.columns:
        raise ValueError(f"Column '{celltype_col}' not found in obs")

    # Get unique cell type-organ combinations
    obs = adata.obs[[celltype_col, tissue_col]].copy()
    obs['group'] = obs[celltype_col].astype(str) + '_' + obs[tissue_col].astype(str)
    groups = obs.groupby('group').size()
    valid_groups = groups[groups >= min_cells].index.tolist()

    log(f"  Found {len(valid_groups)} valid cell type-organ groups (>={min_cells} cells)")

    # Get activity matrix
    if sp.issparse(adata.X):
        X = adata.X.toarray()
    else:
        X = np.array(adata.X)

    signatures = list(adata.var_names)

    results = []

    for group in valid_groups:
        mask = (obs['group'] == group).values
        group_activity = X[mask, :].mean(axis=0)

        # Parse group name
        parts = group.rsplit('_', 1)
        celltype = parts[0]
        organ = parts[1] if len(parts) > 1 else 'unknown'

        for i, sig in enumerate(signatures):
            results.append({
                'cell_type': celltype,
                'organ': organ,
                'signature': sig,
                'mean_activity': group_activity[i],
                'n_cells': mask.sum()
            })

    result_df = pd.DataFrame(results)
    log(f"  Computed {len(result_df)} cell type-organ-signature combinations")

    return result_df


def compute_tumor_vs_adjacent(
    adata: ad.AnnData,
    tissue_col: str = 'tissue'
) -> pd.DataFrame:
    """
    Compare activity between tumor and adjacent normal tissue.

    The cancer dataset has tissue column with values:
    Tumor, Adjacent, Blood, Metastasis, PreLesion, PleuralFluids

    Returns:
        DataFrame with differential statistics
    """
    log("Computing tumor vs adjacent tissue comparison...")

    if tissue_col not in adata.obs.columns:
        raise ValueError(f"Column '{tissue_col}' not found in obs")

    # Get activity matrix
    if sp.issparse(adata.X):
        X = adata.X.toarray()
    else:
        X = np.array(adata.X)

    signatures = list(adata.var_names)

    # Get tissue types
    tissues = adata.obs[tissue_col].unique()
    log(f"  Tissue types: {tissues}")

    # Check for Tumor and Adjacent
    tumor_mask = (adata.obs[tissue_col] == 'Tumor').values
    adjacent_mask = (adata.obs[tissue_col] == 'Adjacent').values

    n_tumor = tumor_mask.sum()
    n_adjacent = adjacent_mask.sum()
    log(f"  Tumor cells: {n_tumor:,}")
    log(f"  Adjacent cells: {n_adjacent:,}")

    if n_tumor < 100 or n_adjacent < 100:
        log("  Warning: Too few cells for comparison")
        return pd.DataFrame()

    # Compute differential
    results = []

    tumor_activity = X[tumor_mask, :]
    adjacent_activity = X[adjacent_mask, :]

    for i, sig in enumerate(signatures):
        tumor_vals = tumor_activity[:, i]
        adjacent_vals = adjacent_activity[:, i]

        # Mann-Whitney U test
        try:
            stat, pval = stats.mannwhitneyu(tumor_vals, adjacent_vals, alternative='two-sided')
        except Exception:
            stat, pval = np.nan, np.nan

        results.append({
            'signature': sig,
            'mean_tumor': tumor_vals.mean(),
            'mean_adjacent': adjacent_vals.mean(),
            'median_tumor': np.median(tumor_vals),
            'median_adjacent': np.median(adjacent_vals),
            'log2fc': np.log2(tumor_vals.mean() + 0.01) - np.log2(adjacent_vals.mean() + 0.01),
            'statistic': stat,
            'pvalue': pval,
            'n_tumor': n_tumor,
            'n_adjacent': n_adjacent
        })

    result_df = pd.DataFrame(results)

    # FDR correction
    if len(result_df) > 0:
        try:
            from statsmodels.stats.multitest import multipletests
            valid_pvals = result_df['pvalue'].dropna()
            if len(valid_pvals) > 0:
                _, pvals_corrected, _, _ = multipletests(valid_pvals.values, method='fdr_bh')
                result_df.loc[valid_pvals.index, 'qvalue'] = pvals_corrected
        except ImportError:
            result_df['qvalue'] = result_df['pvalue']

    log(f"  Computed {len(result_df)} signature comparisons")

    return result_df


def compute_normal_vs_cancer(
    adata_normal: ad.AnnData,
    adata_cancer: ad.AnnData,
    tissue_col_normal: str = 'tissue',
    cancertype_col: str = 'cancerType'
) -> pd.DataFrame:
    """
    Compare normal organ activity with matched cancer type activity.

    Returns:
        DataFrame with matched organ-cancer comparisons
    """
    log("Computing normal vs cancer comparison...")

    # Get signatures (should be the same)
    signatures_normal = set(adata_normal.var_names)
    signatures_cancer = set(adata_cancer.var_names)
    common_sigs = list(signatures_normal & signatures_cancer)

    if len(common_sigs) == 0:
        log("  Warning: No common signatures between datasets")
        return pd.DataFrame()

    log(f"  Common signatures: {len(common_sigs)}")

    # Get activity matrices
    if sp.issparse(adata_normal.X):
        X_normal = adata_normal[:, common_sigs].X.toarray()
    else:
        X_normal = np.array(adata_normal[:, common_sigs].X)

    if sp.issparse(adata_cancer.X):
        X_cancer = adata_cancer[:, common_sigs].X.toarray()
    else:
        X_cancer = np.array(adata_cancer[:, common_sigs].X)

    # Get normal organs and cancer types
    normal_organs = adata_normal.obs[tissue_col_normal].unique()
    cancer_types = adata_cancer.obs[cancertype_col].unique() if cancertype_col in adata_cancer.obs.columns else []

    log(f"  Normal organs: {len(normal_organs)}")
    log(f"  Cancer types: {len(cancer_types)}")

    # Define organ-cancer mappings (manual matching)
    organ_cancer_map = {
        'Liver': ['HCC', 'LIHC', 'Liver'],
        'Lung': ['LUAD', 'LUSC', 'NSCLC', 'SCLC', 'Lung'],
        'Breast': ['BRCA', 'Breast'],
        'Colon': ['CRC', 'COAD', 'Colon', 'Colorectal'],
        'Kidney': ['KIRC', 'Kidney'],
        'Pancreas': ['PAAD', 'Pancreas', 'Pancreatic'],
        'Stomach': ['STAD', 'Stomach', 'Gastric'],
        'Prostate': ['PRAD', 'Prostate'],
        'Bladder': ['BLCA', 'Bladder'],
        'Thyroid': ['THCA', 'Thyroid'],
        'Skin': ['SKCM', 'Melanoma', 'Skin'],
        'Brain': ['GBM', 'Brain', 'Glioma'],
        'Ovary': ['OV', 'Ovary', 'Ovarian'],
    }

    results = []

    for organ, cancer_patterns in organ_cancer_map.items():
        # Find matching normal organ
        normal_mask = None
        for o in normal_organs:
            if organ.lower() in o.lower():
                normal_mask = (adata_normal.obs[tissue_col_normal] == o).values
                break

        if normal_mask is None or normal_mask.sum() < 100:
            continue

        # Find matching cancer type
        cancer_mask = None
        matched_cancer = None
        for ct in cancer_types:
            for pattern in cancer_patterns:
                if pattern.lower() in ct.lower():
                    cancer_mask = (adata_cancer.obs[cancertype_col] == ct).values
                    matched_cancer = ct
                    break
            if cancer_mask is not None:
                break

        if cancer_mask is None or cancer_mask.sum() < 100:
            continue

        # Compute comparison
        normal_activity = X_normal[normal_mask, :]
        cancer_activity = X_cancer[cancer_mask, :]

        for i, sig in enumerate(common_sigs):
            normal_vals = normal_activity[:, i]
            cancer_vals = cancer_activity[:, i]

            try:
                stat, pval = stats.mannwhitneyu(normal_vals, cancer_vals, alternative='two-sided')
            except Exception:
                stat, pval = np.nan, np.nan

            results.append({
                'organ': organ,
                'cancer_type': matched_cancer,
                'signature': sig,
                'mean_normal': normal_vals.mean(),
                'mean_cancer': cancer_vals.mean(),
                'log2fc': np.log2(cancer_vals.mean() + 0.01) - np.log2(normal_vals.mean() + 0.01),
                'statistic': stat,
                'pvalue': pval,
                'n_normal': normal_mask.sum(),
                'n_cancer': cancer_mask.sum()
            })

    result_df = pd.DataFrame(results)

    # FDR correction
    if len(result_df) > 0:
        try:
            from statsmodels.stats.multitest import multipletests
            valid_pvals = result_df['pvalue'].dropna()
            if len(valid_pvals) > 0:
                _, pvals_corrected, _, _ = multipletests(valid_pvals.values, method='fdr_bh')
                result_df.loc[valid_pvals.index, 'qvalue'] = pvals_corrected
        except ImportError:
            result_df['qvalue'] = result_df['pvalue']

    log(f"  Computed {len(result_df)} organ-cancer-signature comparisons")

    return result_df


def identify_pan_signatures(
    result_df: pd.DataFrame,
    qvalue_threshold: float = 0.05,
    log2fc_threshold: float = 0.5
) -> pd.DataFrame:
    """
    Identify signatures that are consistently altered across multiple organs/cancers.

    Returns:
        DataFrame with pan-signatures and their statistics
    """
    log("Identifying pan-signatures...")

    if 'signature' not in result_df.columns:
        return pd.DataFrame()

    # Count significant changes per signature
    sig_stats = []

    for sig in result_df['signature'].unique():
        sig_data = result_df[result_df['signature'] == sig]

        # Count significant up/down regulated
        if 'qvalue' in sig_data.columns and 'log2fc' in sig_data.columns:
            sig_up = sig_data[(sig_data['qvalue'] < qvalue_threshold) &
                              (sig_data['log2fc'] > log2fc_threshold)]
            sig_down = sig_data[(sig_data['qvalue'] < qvalue_threshold) &
                                (sig_data['log2fc'] < -log2fc_threshold)]

            sig_stats.append({
                'signature': sig,
                'n_comparisons': len(sig_data),
                'n_significant_up': len(sig_up),
                'n_significant_down': len(sig_down),
                'mean_log2fc': sig_data['log2fc'].mean(),
                'consistency_score': max(len(sig_up), len(sig_down)) / len(sig_data) if len(sig_data) > 0 else 0
            })

    pan_df = pd.DataFrame(sig_stats)

    # Sort by consistency
    if len(pan_df) > 0:
        pan_df = pan_df.sort_values('consistency_score', ascending=False)

    log(f"  Identified {len(pan_df)} signatures with pan-analysis")

    return pan_df


def compute_immune_infiltration(
    adata: ad.AnnData,
    tissue_col: str = 'tissue',
    celltype_col: str = 'cell_type',
    cancertype_col: str = 'cancerType'
) -> pd.DataFrame:
    """
    Compute immune cell infiltration proportions per cancer type.

    Returns:
        DataFrame with cancer_type, immune_proportion, immune_celltypes, and activities
    """
    log("Computing immune infiltration analysis...")

    # Define immune cell type patterns
    immune_patterns = [
        'T', 'B', 'NK', 'Mono', 'Macro', 'DC', 'Neutro', 'Mast',
        'Plasma', 'Treg', 'CD4', 'CD8', 'gdT', 'MAIT', 'ILC'
    ]

    # Get activity matrix
    if sp.issparse(adata.X):
        X = adata.X.toarray()
    else:
        X = np.array(adata.X)

    signatures = list(adata.var_names)
    obs = adata.obs.copy()

    # Classify cells as immune or non-immune
    def is_immune(celltype):
        celltype_str = str(celltype).lower()
        return any(p.lower() in celltype_str for p in immune_patterns)

    obs['is_immune'] = obs[celltype_col].apply(is_immune)

    # Only proceed if we have cancerType column
    if cancertype_col not in obs.columns:
        log("  Warning: No cancerType column found")
        return pd.DataFrame()

    results = []

    # Group by cancer type
    cancer_types = obs[cancertype_col].dropna().unique()
    log(f"  Found {len(cancer_types)} cancer types")

    for ct in cancer_types:
        ct_mask = (obs[cancertype_col] == ct).values
        tumor_mask = (obs[tissue_col] == 'Tumor').values if tissue_col in obs.columns else np.ones(len(obs), dtype=bool)
        combined_mask = ct_mask & tumor_mask

        if combined_mask.sum() < 100:
            continue

        ct_obs = obs[combined_mask]
        n_immune = ct_obs['is_immune'].sum()
        n_total = len(ct_obs)
        immune_prop = n_immune / n_total

        # Get mean activity for immune vs non-immune
        ct_X = X[combined_mask, :]
        immune_activity = ct_X[ct_obs['is_immune'].values, :].mean(axis=0)
        nonimmune_activity = ct_X[~ct_obs['is_immune'].values, :].mean(axis=0) if (~ct_obs['is_immune'].values).sum() > 0 else np.zeros(len(signatures))

        for i, sig in enumerate(signatures):
            results.append({
                'cancer_type': ct,
                'signature': sig,
                'immune_proportion': immune_prop,
                'n_immune': n_immune,
                'n_total': n_total,
                'mean_immune_activity': immune_activity[i],
                'mean_nonimmune_activity': nonimmune_activity[i],
                'immune_enrichment': immune_activity[i] - nonimmune_activity[i]
            })

    result_df = pd.DataFrame(results)
    log(f"  Computed {len(result_df)} cancer-signature infiltration records")

    return result_df


def compute_tcell_exhaustion(
    adata: ad.AnnData,
    celltype_col: str = 'cell_type'
) -> pd.DataFrame:
    """
    Compute T cell exhaustion analysis comparing exhausted vs non-exhausted T cells.

    Uses cell type annotations containing 'Tex' (exhausted) patterns.

    Returns:
        DataFrame with exhaustion comparison statistics
    """
    log("Computing T cell exhaustion analysis...")

    # Patterns for exhausted T cells
    exhausted_patterns = ['Tex', 'exhausted', 'PDCD1', 'CTLA4', 'LAG3', 'HAVCR2', 'TIGIT']

    # Get activity matrix
    if sp.issparse(adata.X):
        X = adata.X.toarray()
    else:
        X = np.array(adata.X)

    signatures = list(adata.var_names)
    obs = adata.obs.copy()

    # Identify T cells
    def is_tcell(celltype):
        celltype_str = str(celltype).lower()
        return any(p in celltype_str for p in ['cd4t', 'cd8t', 't_cd4', 't_cd8', 'tcell', 't cell', 't_cell'])

    # Identify exhausted T cells
    def is_exhausted(celltype):
        celltype_str = str(celltype)
        return any(p.lower() in celltype_str.lower() for p in exhausted_patterns)

    obs['is_tcell'] = obs[celltype_col].apply(is_tcell)
    obs['is_exhausted'] = obs[celltype_col].apply(is_exhausted)

    # Filter to T cells
    tcell_mask = obs['is_tcell'].values
    n_tcells = tcell_mask.sum()
    log(f"  Found {n_tcells:,} T cells")

    if n_tcells < 100:
        log("  Warning: Too few T cells for analysis")
        return pd.DataFrame()

    # Split into exhausted and non-exhausted
    exhausted_mask = obs['is_exhausted'].values & tcell_mask
    nonexhausted_mask = ~obs['is_exhausted'].values & tcell_mask

    n_exhausted = exhausted_mask.sum()
    n_nonexhausted = nonexhausted_mask.sum()
    log(f"  Exhausted T cells: {n_exhausted:,}")
    log(f"  Non-exhausted T cells: {n_nonexhausted:,}")

    if n_exhausted < 50 or n_nonexhausted < 50:
        log("  Warning: Too few cells in one group")
        return pd.DataFrame()

    results = []

    exhausted_X = X[exhausted_mask, :]
    nonexhausted_X = X[nonexhausted_mask, :]

    for i, sig in enumerate(signatures):
        exhausted_vals = exhausted_X[:, i]
        nonexhausted_vals = nonexhausted_X[:, i]

        # Statistical comparison
        try:
            stat, pval = stats.mannwhitneyu(exhausted_vals, nonexhausted_vals, alternative='two-sided')
        except Exception:
            stat, pval = np.nan, np.nan

        results.append({
            'signature': sig,
            'mean_exhausted': exhausted_vals.mean(),
            'mean_nonexhausted': nonexhausted_vals.mean(),
            'median_exhausted': np.median(exhausted_vals),
            'median_nonexhausted': np.median(nonexhausted_vals),
            'log2fc': np.log2(exhausted_vals.mean() + 0.01) - np.log2(nonexhausted_vals.mean() + 0.01),
            'statistic': stat,
            'pvalue': pval,
            'n_exhausted': n_exhausted,
            'n_nonexhausted': n_nonexhausted
        })

    result_df = pd.DataFrame(results)

    # FDR correction
    if len(result_df) > 0:
        try:
            from statsmodels.stats.multitest import multipletests
            valid_pvals = result_df['pvalue'].dropna()
            if len(valid_pvals) > 0:
                _, pvals_corrected, _, _ = multipletests(valid_pvals.values, method='fdr_bh')
                result_df.loc[valid_pvals.index, 'qvalue'] = pvals_corrected
        except ImportError:
            result_df['qvalue'] = result_df['pvalue']

    log(f"  Computed {len(result_df)} exhaustion comparisons")

    return result_df


def compute_caf_signatures(
    adata: ad.AnnData,
    celltype_col: str = 'cell_type',
    cancertype_col: str = 'cancerType'
) -> pd.DataFrame:
    """
    Compute cancer-associated fibroblast (CAF) signatures.

    Identifies fibroblast subtypes and computes activity patterns.

    Returns:
        DataFrame with CAF subtype activities per cancer type
    """
    log("Computing CAF (cancer-associated fibroblast) signatures...")

    # Fibroblast patterns
    fibroblast_patterns = ['fb', 'fibroblast', 'caf', 'stromal']

    # Get activity matrix
    if sp.issparse(adata.X):
        X = adata.X.toarray()
    else:
        X = np.array(adata.X)

    signatures = list(adata.var_names)
    obs = adata.obs.copy()

    # Identify fibroblasts
    def is_fibroblast(celltype):
        celltype_str = str(celltype).lower()
        return any(p in celltype_str for p in fibroblast_patterns)

    obs['is_fibroblast'] = obs[celltype_col].apply(is_fibroblast)

    # Filter to fibroblasts
    fb_mask = obs['is_fibroblast'].values
    n_fibroblasts = fb_mask.sum()
    log(f"  Found {n_fibroblasts:,} fibroblasts")

    if n_fibroblasts < 100:
        log("  Warning: Too few fibroblasts for analysis")
        return pd.DataFrame()

    results = []

    # Get unique fibroblast subtypes
    fb_celltypes = obs.loc[fb_mask, celltype_col].unique()
    log(f"  Fibroblast subtypes: {len(fb_celltypes)}")

    # Analyze by cancer type if available
    if cancertype_col in obs.columns:
        cancer_types = obs.loc[fb_mask, cancertype_col].dropna().unique()

        for ct in cancer_types:
            ct_fb_mask = fb_mask & (obs[cancertype_col] == ct).values
            n_ct_fb = ct_fb_mask.sum()

            if n_ct_fb < 50:
                continue

            ct_fb_X = X[ct_fb_mask, :]
            mean_activity = ct_fb_X.mean(axis=0)

            for i, sig in enumerate(signatures):
                results.append({
                    'cancer_type': ct,
                    'cell_type': 'All_Fibroblasts',
                    'signature': sig,
                    'mean_activity': mean_activity[i],
                    'n_cells': n_ct_fb
                })

            # Also compute per fibroblast subtype within this cancer type
            for fb_ct in fb_celltypes:
                subtype_mask = ct_fb_mask & (obs[celltype_col] == fb_ct).values
                n_subtype = subtype_mask.sum()

                if n_subtype < 20:
                    continue

                subtype_X = X[subtype_mask, :]
                subtype_mean = subtype_X.mean(axis=0)

                for i, sig in enumerate(signatures):
                    results.append({
                        'cancer_type': ct,
                        'cell_type': fb_ct,
                        'signature': sig,
                        'mean_activity': subtype_mean[i],
                        'n_cells': n_subtype
                    })
    else:
        # Just analyze overall fibroblast subtypes
        for fb_ct in fb_celltypes:
            subtype_mask = (obs[celltype_col] == fb_ct).values
            n_subtype = subtype_mask.sum()

            if n_subtype < 20:
                continue

            subtype_X = X[subtype_mask, :]
            subtype_mean = subtype_X.mean(axis=0)

            for i, sig in enumerate(signatures):
                results.append({
                    'cancer_type': 'All',
                    'cell_type': fb_ct,
                    'signature': sig,
                    'mean_activity': subtype_mean[i],
                    'n_cells': n_subtype
                })

    result_df = pd.DataFrame(results)
    log(f"  Computed {len(result_df)} CAF signature records")

    return result_df


def compute_adjacent_signatures(
    adata_cancer: ad.AnnData,
    adata_normal: ad.AnnData = None,
    tissue_col: str = 'tissue',
    cancertype_col: str = 'cancerType'
) -> pd.DataFrame:
    """
    Analyze field effect in adjacent tissue.

    Compares Adjacent tissue to Tumor tissue within cancer data,
    and optionally to Normal tissue from normal organ atlas.

    Returns:
        DataFrame with adjacent tissue signature differences
    """
    log("Computing adjacent tissue signatures (field effect)...")

    # Get activity matrix from cancer data
    if sp.issparse(adata_cancer.X):
        X = adata_cancer.X.toarray()
    else:
        X = np.array(adata_cancer.X)

    signatures = list(adata_cancer.var_names)
    obs = adata_cancer.obs.copy()

    # Get tissue types
    adjacent_mask = (obs[tissue_col] == 'Adjacent').values if tissue_col in obs.columns else None
    tumor_mask = (obs[tissue_col] == 'Tumor').values if tissue_col in obs.columns else None

    if adjacent_mask is None or tumor_mask is None:
        log("  Warning: No tissue column found")
        return pd.DataFrame()

    n_adjacent = adjacent_mask.sum()
    n_tumor = tumor_mask.sum()
    log(f"  Adjacent cells: {n_adjacent:,}")
    log(f"  Tumor cells: {n_tumor:,}")

    if n_adjacent < 100 or n_tumor < 100:
        log("  Warning: Too few cells for comparison")
        return pd.DataFrame()

    results = []

    # Overall comparison
    adjacent_X = X[adjacent_mask, :]
    tumor_X = X[tumor_mask, :]

    for i, sig in enumerate(signatures):
        adjacent_vals = adjacent_X[:, i]
        tumor_vals = tumor_X[:, i]

        try:
            stat, pval = stats.mannwhitneyu(adjacent_vals, tumor_vals, alternative='two-sided')
        except Exception:
            stat, pval = np.nan, np.nan

        results.append({
            'cancer_type': 'All',
            'signature': sig,
            'mean_adjacent': adjacent_vals.mean(),
            'mean_tumor': tumor_vals.mean(),
            'log2fc_adj_vs_tumor': np.log2(adjacent_vals.mean() + 0.01) - np.log2(tumor_vals.mean() + 0.01),
            'statistic': stat,
            'pvalue': pval,
            'n_adjacent': n_adjacent,
            'n_tumor': n_tumor
        })

    # Per cancer type comparison
    if cancertype_col in obs.columns:
        cancer_types = obs[cancertype_col].dropna().unique()

        for ct in cancer_types:
            ct_adj_mask = adjacent_mask & (obs[cancertype_col] == ct).values
            ct_tumor_mask = tumor_mask & (obs[cancertype_col] == ct).values

            n_ct_adj = ct_adj_mask.sum()
            n_ct_tumor = ct_tumor_mask.sum()

            if n_ct_adj < 50 or n_ct_tumor < 50:
                continue

            ct_adj_X = X[ct_adj_mask, :]
            ct_tumor_X = X[ct_tumor_mask, :]

            for i, sig in enumerate(signatures):
                adj_vals = ct_adj_X[:, i]
                tumor_vals = ct_tumor_X[:, i]

                try:
                    stat, pval = stats.mannwhitneyu(adj_vals, tumor_vals, alternative='two-sided')
                except Exception:
                    stat, pval = np.nan, np.nan

                results.append({
                    'cancer_type': ct,
                    'signature': sig,
                    'mean_adjacent': adj_vals.mean(),
                    'mean_tumor': tumor_vals.mean(),
                    'log2fc_adj_vs_tumor': np.log2(adj_vals.mean() + 0.01) - np.log2(tumor_vals.mean() + 0.01),
                    'statistic': stat,
                    'pvalue': pval,
                    'n_adjacent': n_ct_adj,
                    'n_tumor': n_ct_tumor
                })

    result_df = pd.DataFrame(results)

    # FDR correction
    if len(result_df) > 0:
        try:
            from statsmodels.stats.multitest import multipletests
            valid_pvals = result_df['pvalue'].dropna()
            if len(valid_pvals) > 0:
                _, pvals_corrected, _, _ = multipletests(valid_pvals.values, method='fdr_bh')
                result_df.loc[valid_pvals.index, 'qvalue'] = pvals_corrected
        except ImportError:
            result_df['qvalue'] = result_df['pvalue']

    log(f"  Computed {len(result_df)} adjacent tissue comparisons")

    return result_df


# ==============================================================================
# Main Analysis Pipeline
# ==============================================================================

def analyze_normal_atlas():
    """Analyze normal organ atlas by computing activities from raw counts."""
    log("\n" + "=" * 60)
    log("NORMAL ORGAN ATLAS ANALYSIS")
    log("=" * 60)

    # Check if counts file exists
    if not NORMAL_COUNTS.exists():
        raise FileNotFoundError(f"Normal counts file not found: {NORMAL_COUNTS}")

    # Compute activities from raw counts
    cytosig_results, secact_results, agg_meta = compute_activities_from_counts(
        NORMAL_COUNTS,
        TISSUE_COL,
        CELLTYPE_COARSE,
        'normal'
    )

    results = {}

    # Analyze CytoSig activities
    log("\nAnalyzing CytoSig organ and cell type signatures...")
    adata_cytosig = activity_to_adata(cytosig_results, agg_meta, 'CytoSig')

    organ_sigs_cytosig = compute_organ_signatures(adata_cytosig, 'tissue')
    organ_sigs_cytosig['signature_type'] = 'CytoSig'
    results['CytoSig_organ'] = organ_sigs_cytosig

    celltype_sigs_cytosig = compute_celltype_signatures(
        adata_cytosig, 'cell_type', 'tissue', min_cells=1
    )
    celltype_sigs_cytosig['signature_type'] = 'CytoSig'
    results['CytoSig_celltype'] = celltype_sigs_cytosig

    del adata_cytosig
    gc.collect()

    # Analyze SecAct activities
    log("\nAnalyzing SecAct organ and cell type signatures...")
    adata_secact = activity_to_adata(secact_results, agg_meta, 'SecAct')

    organ_sigs_secact = compute_organ_signatures(adata_secact, 'tissue')
    organ_sigs_secact['signature_type'] = 'SecAct'
    results['SecAct_organ'] = organ_sigs_secact

    celltype_sigs_secact = compute_celltype_signatures(
        adata_secact, 'cell_type', 'tissue', min_cells=1
    )
    celltype_sigs_secact['signature_type'] = 'SecAct'
    results['SecAct_celltype'] = celltype_sigs_secact

    del adata_secact
    gc.collect()

    # Combine and save results
    organ_df = pd.concat([results['CytoSig_organ'], results['SecAct_organ']])
    organ_df.to_csv(OUTPUT_DIR / 'normal_organ_signatures.csv', index=False)
    log(f"Saved organ signatures: {len(organ_df)} rows")

    celltype_df = pd.concat([results['CytoSig_celltype'], results['SecAct_celltype']])
    celltype_df.to_csv(OUTPUT_DIR / 'normal_celltype_signatures.csv', index=False)
    log(f"Saved cell type signatures: {len(celltype_df)} rows")

    # Identify organ-specific top signatures
    top_organ_sigs = organ_df.groupby(['organ', 'signature_type']).apply(
        lambda x: x.nlargest(10, 'specificity_score')
    ).reset_index(drop=True)
    top_organ_sigs.to_csv(OUTPUT_DIR / 'normal_top_organ_signatures.csv', index=False)
    log(f"Saved top organ signatures: {len(top_organ_sigs)} rows")

    return results, cytosig_results, secact_results, agg_meta


def analyze_cancer_atlas():
    """Analyze cancer atlas by computing activities from raw counts."""
    log("\n" + "=" * 60)
    log("CANCER ATLAS ANALYSIS")
    log("=" * 60)

    # Check if counts file exists
    if not CANCER_COUNTS.exists():
        raise FileNotFoundError(f"Cancer counts file not found: {CANCER_COUNTS}")

    # Compute activities from raw counts
    # For cancer data, aggregate by tissue type (Tumor/Adjacent) and cell type
    # Include cancerType and donorID in metadata for downstream analysis
    cytosig_results, secact_results, agg_meta = compute_activities_from_counts(
        CANCER_COUNTS,
        TISSUE_COL,
        CELLTYPE_COARSE,
        'cancer',
        extra_cols=['cancerType', 'donorID', 'subCluster']
    )

    results = {}

    # Analyze CytoSig activities
    log("\nAnalyzing CytoSig tumor signatures...")
    adata_cytosig = activity_to_adata(cytosig_results, agg_meta, 'CytoSig')

    tumor_adj_cytosig = compute_tumor_vs_adjacent(adata_cytosig, 'tissue')
    tumor_adj_cytosig['signature_type'] = 'CytoSig'
    results['CytoSig_tumor_adj'] = tumor_adj_cytosig

    # Cancer type signatures if we have cancerType in metadata
    if 'cancerType' in agg_meta.columns:
        cancer_sigs_cytosig = compute_organ_signatures(adata_cytosig, 'cancerType')
        cancer_sigs_cytosig['signature_type'] = 'CytoSig'
        results['CytoSig_cancer'] = cancer_sigs_cytosig

    del adata_cytosig
    gc.collect()

    # Analyze SecAct activities
    log("\nAnalyzing SecAct tumor signatures...")
    adata_secact = activity_to_adata(secact_results, agg_meta, 'SecAct')

    tumor_adj_secact = compute_tumor_vs_adjacent(adata_secact, 'tissue')
    tumor_adj_secact['signature_type'] = 'SecAct'
    results['SecAct_tumor_adj'] = tumor_adj_secact

    if 'cancerType' in agg_meta.columns:
        cancer_sigs_secact = compute_organ_signatures(adata_secact, 'cancerType')
        cancer_sigs_secact['signature_type'] = 'SecAct'
        results['SecAct_cancer'] = cancer_sigs_secact

    del adata_secact
    gc.collect()

    # Combine and save results
    tumor_adj_df = pd.concat([results.get('CytoSig_tumor_adj', pd.DataFrame()),
                              results.get('SecAct_tumor_adj', pd.DataFrame())])
    if len(tumor_adj_df) > 0:
        tumor_adj_df.to_csv(OUTPUT_DIR / 'cancer_tumor_vs_adjacent.csv', index=False)
        log(f"Saved tumor vs adjacent: {len(tumor_adj_df)} rows")

    if 'CytoSig_cancer' in results or 'SecAct_cancer' in results:
        cancer_df = pd.concat([results.get('CytoSig_cancer', pd.DataFrame()),
                               results.get('SecAct_cancer', pd.DataFrame())])
        cancer_df.to_csv(OUTPUT_DIR / 'cancer_type_signatures.csv', index=False)
        log(f"Saved cancer type signatures: {len(cancer_df)} rows")

    # Additional analyses using CytoSig results
    log("\nRunning additional cancer analyses...")

    # Immune infiltration analysis
    adata_cytosig = activity_to_adata(cytosig_results, agg_meta, 'CytoSig')
    if 'cancerType' in adata_cytosig.obs.columns:
        immune_df = compute_immune_infiltration(
            adata_cytosig, 'tissue', 'cell_type', 'cancerType'
        )
        if len(immune_df) > 0:
            immune_df['signature_type'] = 'CytoSig'
            immune_df.to_csv(OUTPUT_DIR / 'cancer_immune_infiltration.csv', index=False)
            log(f"Saved immune infiltration: {len(immune_df)} rows")
            results['immune_infiltration'] = immune_df

    # T cell exhaustion analysis
    exhaustion_df = compute_tcell_exhaustion(adata_cytosig, 'cell_type')
    if len(exhaustion_df) > 0:
        exhaustion_df['signature_type'] = 'CytoSig'
        exhaustion_df.to_csv(OUTPUT_DIR / 'cancer_tcell_exhaustion.csv', index=False)
        log(f"Saved T cell exhaustion: {len(exhaustion_df)} rows")
        results['tcell_exhaustion'] = exhaustion_df

    # CAF analysis
    caf_df = compute_caf_signatures(
        adata_cytosig, 'cell_type',
        'cancerType' if 'cancerType' in adata_cytosig.obs.columns else None
    )
    if len(caf_df) > 0:
        caf_df['signature_type'] = 'CytoSig'
        caf_df.to_csv(OUTPUT_DIR / 'cancer_caf_signatures.csv', index=False)
        log(f"Saved CAF signatures: {len(caf_df)} rows")
        results['caf_signatures'] = caf_df

    # Adjacent tissue analysis
    adjacent_df = compute_adjacent_signatures(
        adata_cytosig, None, 'tissue',
        'cancerType' if 'cancerType' in adata_cytosig.obs.columns else None
    )
    if len(adjacent_df) > 0:
        adjacent_df['signature_type'] = 'CytoSig'
        adjacent_df.to_csv(OUTPUT_DIR / 'cancer_adjacent_signatures.csv', index=False)
        log(f"Saved adjacent signatures: {len(adjacent_df)} rows")
        results['adjacent_signatures'] = adjacent_df

    del adata_cytosig
    gc.collect()

    return results, cytosig_results, secact_results, agg_meta


def analyze_normal_vs_cancer(
    normal_cytosig: Dict[str, pd.DataFrame] = None,
    normal_secact: Dict[str, pd.DataFrame] = None,
    normal_meta: pd.DataFrame = None,
    cancer_cytosig: Dict[str, pd.DataFrame] = None,
    cancer_secact: Dict[str, pd.DataFrame] = None,
    cancer_meta: pd.DataFrame = None
):
    """
    Compare normal and cancer atlases.

    If activity results are not provided, they will be loaded from saved files.
    """
    log("\n" + "=" * 60)
    log("NORMAL VS CANCER COMPARISON")
    log("=" * 60)

    results = {}

    # Try to load from saved files if not provided
    if normal_cytosig is None:
        normal_cytosig_path = OUTPUT_DIR / 'normal_cytosig_zscore.csv'
        if normal_cytosig_path.exists():
            log(f"Loading normal CytoSig from {normal_cytosig_path}")
            normal_cytosig = {'zscore': pd.read_csv(normal_cytosig_path, index_col=0)}
        else:
            log("Warning: Normal CytoSig not available")
            return results

    if cancer_cytosig is None:
        cancer_cytosig_path = OUTPUT_DIR / 'cancer_cytosig_zscore.csv'
        if cancer_cytosig_path.exists():
            log(f"Loading cancer CytoSig from {cancer_cytosig_path}")
            cancer_cytosig = {'zscore': pd.read_csv(cancer_cytosig_path, index_col=0)}
        else:
            log("Warning: Cancer CytoSig not available")
            return results

    if normal_secact is None:
        normal_secact_path = OUTPUT_DIR / 'normal_secact_zscore.csv'
        if normal_secact_path.exists():
            log(f"Loading normal SecAct from {normal_secact_path}")
            normal_secact = {'zscore': pd.read_csv(normal_secact_path, index_col=0)}

    if cancer_secact is None:
        cancer_secact_path = OUTPUT_DIR / 'cancer_secact_zscore.csv'
        if cancer_secact_path.exists():
            log(f"Loading cancer SecAct from {cancer_secact_path}")
            cancer_secact = {'zscore': pd.read_csv(cancer_secact_path, index_col=0)}

    # Load metadata
    if normal_meta is None:
        normal_meta_path = OUTPUT_DIR / 'normal_aggregation_meta.csv'
        if normal_meta_path.exists():
            normal_meta = pd.read_csv(normal_meta_path, index_col=0)

    if cancer_meta is None:
        cancer_meta_path = OUTPUT_DIR / 'cancer_aggregation_meta.csv'
        if cancer_meta_path.exists():
            cancer_meta = pd.read_csv(cancer_meta_path, index_col=0)

    # Compare CytoSig
    if normal_cytosig and cancer_cytosig:
        log("\nComparing CytoSig activities...")

        # Convert to AnnData for comparison functions
        adata_normal = activity_to_adata(normal_cytosig, normal_meta, 'CytoSig')
        adata_cancer = activity_to_adata(cancer_cytosig, cancer_meta, 'CytoSig')

        comparison = compute_normal_vs_cancer(
            adata_normal, adata_cancer,
            tissue_col_normal='tissue',
            cancertype_col='cancerType' if 'cancerType' in adata_cancer.obs.columns else 'tissue'
        )
        comparison['signature_type'] = 'CytoSig'
        results['CytoSig'] = comparison

        del adata_normal, adata_cancer
        gc.collect()

    # Compare SecAct
    if normal_secact and cancer_secact:
        log("\nComparing SecAct activities...")

        adata_normal = activity_to_adata(normal_secact, normal_meta, 'SecAct')
        adata_cancer = activity_to_adata(cancer_secact, cancer_meta, 'SecAct')

        comparison = compute_normal_vs_cancer(
            adata_normal, adata_cancer,
            tissue_col_normal='tissue',
            cancertype_col='cancerType' if 'cancerType' in adata_cancer.obs.columns else 'tissue'
        )
        comparison['signature_type'] = 'SecAct'
        results['SecAct'] = comparison

        del adata_normal, adata_cancer
        gc.collect()

    # Combine and save
    comparison_df = pd.concat([results.get('CytoSig', pd.DataFrame()),
                               results.get('SecAct', pd.DataFrame())])

    if len(comparison_df) > 0:
        comparison_df.to_csv(OUTPUT_DIR / 'normal_vs_cancer_comparison.csv', index=False)
        log(f"Saved normal vs cancer comparison: {len(comparison_df)} rows")

        # Pan-signatures
        pan_sigs = identify_pan_signatures(comparison_df)
        pan_sigs.to_csv(OUTPUT_DIR / 'pan_cancer_signatures.csv', index=False)
        log(f"Saved pan-cancer signatures: {len(pan_sigs)} rows")

    return results


# ==============================================================================
# Entry Point
# ==============================================================================

def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='scAtlas Activity Analysis')
    parser.add_argument('--mode', choices=['normal', 'cancer', 'comparison', 'all'],
                       default='all', help='Analysis mode')
    parser.add_argument('--test', action='store_true',
                       help='Run on subset for testing')
    args = parser.parse_args()

    log("=" * 60)
    log("scATLAS ACTIVITY ANALYSIS (SecActpy)")
    log("=" * 60)
    log(f"Mode: {args.mode}")
    log(f"Output: {OUTPUT_DIR}")
    log(f"Backend: {BACKEND}")

    # Check for raw count files
    log("\nChecking for raw count files...")
    for path, name in [
        (NORMAL_COUNTS, "Normal counts"),
        (CANCER_COUNTS, "Cancer counts"),
    ]:
        status = "Found" if path.exists() else "NOT FOUND"
        log(f"  {name}: {status}")

    start_time = time.time()

    # Store results for cross-analysis
    normal_cytosig = None
    normal_secact = None
    normal_meta = None
    cancer_cytosig = None
    cancer_secact = None
    cancer_meta = None

    if args.mode in ['normal', 'all']:
        results, normal_cytosig, normal_secact, normal_meta = analyze_normal_atlas()

    if args.mode in ['cancer', 'all']:
        results, cancer_cytosig, cancer_secact, cancer_meta = analyze_cancer_atlas()

    if args.mode in ['comparison', 'all']:
        analyze_normal_vs_cancer(
            normal_cytosig=normal_cytosig,
            normal_secact=normal_secact,
            normal_meta=normal_meta,
            cancer_cytosig=cancer_cytosig,
            cancer_secact=cancer_secact,
            cancer_meta=cancer_meta
        )

    total_time = time.time() - start_time
    log(f"\nTotal time: {total_time/60:.1f} minutes")
    log("Done!")


if __name__ == '__main__':
    main()
