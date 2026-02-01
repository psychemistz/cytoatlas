#!/usr/bin/env python3
"""
Inflammation Atlas Activity Analysis
=====================================
Compute cytokine (CytoSig, 44 signatures) and secreted protein (SecAct, 1,249 signatures)
activities across the Inflammation Atlas (6.3M cells) and associate with patient metadata.

Datasets:
- Main dataset: 4,918,140 cells, 22,826 genes, 817 samples
- Validation dataset: 849,922 cells, 22,826 genes, 144 samples
- External dataset: 572,872 cells, 37,124 genes, 86 samples
- Sample metadata: 1,047 samples with disease, treatment, response info

Analysis levels:
1. Pseudo-bulk (aggregate by cell type and sample)
2. Single-cell level (per-cell activities)

Statistical analyses:
- Differential: disease groups (20 diseases, 6 groups)
- Differential: treatment response (R vs NR)
- Correlation: activity vs age, BMI
- Treatment response PREDICTION (ML models with cross-validation)
- Cross-cohort validation (main → validation)
"""

import os
import sys
import gc
import warnings
import time
from pathlib import Path
from typing import Optional, List, Tuple

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
DATA_DIR = Path('/data/Jiang_Lab/Data/Seongyong/Inflammation_Atlas')
MAIN_H5AD = DATA_DIR / 'INFLAMMATION_ATLAS_main_afterQC.h5ad'
VAL_H5AD = DATA_DIR / 'INFLAMMATION_ATLAS_validation_afterQC.h5ad'
EXT_H5AD = DATA_DIR / 'INFLAMMATION_ATLAS_external_afterQC.h5ad'
SAMPLE_META_PATH = DATA_DIR / 'INFLAMMATION_ATLAS_afterQC_sampleMetadata.csv'

# Output paths
OUTPUT_DIR = Path('/vf/users/parks34/projects/2secactpy/results/inflammation')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Analysis parameters
CELL_TYPE_COL = 'Level2'  # 66 fine cell types
SAMPLE_COL = 'sampleID'
BATCH_SIZE = 10000
N_RAND = 1000
SEED = 0

# GPU settings
BACKEND = 'cupy' if CUPY_AVAILABLE else 'numpy'

# ==============================================================================
# Utility Functions
# ==============================================================================

def log(msg: str):
    """Print timestamped log message."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_sample_metadata() -> pd.DataFrame:
    """Load sample metadata."""
    log("Loading sample metadata...")
    meta = pd.read_csv(SAMPLE_META_PATH)
    log(f"  Samples: {len(meta)}")
    log(f"  Columns: {list(meta.columns)}")
    log(f"  Diseases: {meta['disease'].nunique()}")
    log(f"  Disease groups: {meta['diseaseGroup'].nunique()}")

    # Treatment response stats
    response_counts = meta['therapyResponse'].value_counts()
    log(f"  Treatment response: {dict(response_counts)}")

    return meta


def aggregate_by_sample_celltype(
    adata,
    cell_type_col: str,
    sample_col: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aggregate expression by sample and cell type for pseudo-bulk analysis.

    Returns:
        expr_df: DataFrame (genes x (sample_celltype combinations))
        meta_df: DataFrame with sample and cell type info for each column
    """
    log(f"Aggregating by {sample_col} and {cell_type_col}...")

    # Get unique combinations
    obs = adata.obs[[sample_col, cell_type_col]].copy()
    obs = obs.reset_index(drop=True)  # Reset to integer index for positional access
    groups = obs.groupby([sample_col, cell_type_col], observed=True).groups

    log(f"  Found {len(groups)} sample-celltype combinations")

    # Expression matrix (raw counts in .X for this dataset)
    X = adata.X

    # Use gene symbols if available (Inflammation Atlas uses Ensembl IDs as index)
    if 'symbol' in adata.var.columns:
        gene_names = list(adata.var['symbol'].values)
        log("  Using gene symbols from var['symbol']")
    else:
        gene_names = list(adata.var_names)
    n_genes = len(gene_names)

    # Aggregate by summing
    aggregated = {}
    meta_rows = []

    for i, ((sample, celltype), indices) in enumerate(groups.items()):
        col_name = f"{sample}_{celltype}"

        # Get indices as integer array (these are now positional after reset_index)
        idx_array = np.array(indices, dtype=np.int64)

        # Sum counts for this group (ensure 1D array)
        if sp.issparse(X):
            group_sum = np.asarray(X[idx_array, :].sum(axis=0)).flatten()
        else:
            group_sum = np.asarray(X[idx_array, :].sum(axis=0)).flatten()
        # Ensure truly 1D (handles edge cases with matrix types)
        if group_sum.ndim > 1:
            group_sum = group_sum.ravel()

        aggregated[col_name] = group_sum
        meta_rows.append({
            'column': col_name,
            'sample': sample,
            'cell_type': celltype,
            'n_cells': len(idx_array)
        })

        # Progress logging
        if (i + 1) % 1000 == 0:
            log(f"    Processed {i + 1}/{len(groups)} groups...")

    # Create DataFrames
    expr_df = pd.DataFrame(aggregated, index=gene_names)
    meta_df = pd.DataFrame(meta_rows).set_index('column')

    log(f"  Aggregated expression: {expr_df.shape}")

    return expr_df, meta_df


def normalize_and_transform(expr_df: pd.DataFrame) -> pd.DataFrame:
    """TPM normalize and log2 transform expression data."""
    col_sums = expr_df.sum(axis=0)
    col_sums = col_sums.replace(0, 1)  # Avoid division by zero
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
    sig_name: str
) -> dict:
    """
    Run SecActpy activity inference.

    Args:
        expr_df: Differential expression (genes x samples)
        signature: Signature matrix (genes x proteins)
        sig_name: Name for logging

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


def disease_differential_analysis(
    activity_df: pd.DataFrame,
    sample_meta: pd.DataFrame,
    agg_meta_df: pd.DataFrame,
    group_col: str = 'disease'
) -> pd.DataFrame:
    """
    Compute differential activity between disease groups and healthy.

    Args:
        activity_df: Activity z-scores (proteins x sample_celltype)
        sample_meta: Sample-level metadata
        agg_meta_df: Aggregation metadata (sample, cell_type per column)
        group_col: Column for disease grouping

    Returns:
        DataFrame with differential statistics
    """
    log(f"Computing disease differential analysis ({group_col})...")

    # Map sample_celltype columns to disease
    col_to_sample = agg_meta_df['sample'].to_dict()

    # Merge sample metadata
    sample_disease = sample_meta.set_index('sampleID')[group_col].to_dict()

    # Get healthy samples
    healthy_samples = sample_meta[sample_meta['disease'] == 'healthy']['sampleID'].tolist()

    # Compute differential for each disease vs healthy
    results = []
    diseases = sample_meta[sample_meta['disease'] != 'healthy'][group_col].unique()

    for disease in diseases:
        disease_samples = sample_meta[sample_meta[group_col] == disease]['sampleID'].tolist()

        # Get columns for this disease and healthy
        disease_cols = [c for c in activity_df.columns
                       if col_to_sample.get(c) in disease_samples]
        healthy_cols = [c for c in activity_df.columns
                       if col_to_sample.get(c) in healthy_samples]

        if len(disease_cols) < 3 or len(healthy_cols) < 3:
            continue

        for protein in activity_df.index:
            vals_disease = activity_df.loc[protein, disease_cols].dropna()
            vals_healthy = activity_df.loc[protein, healthy_cols].dropna()

            if len(vals_disease) < 3 or len(vals_healthy) < 3:
                continue

            stat, pval = stats.mannwhitneyu(vals_disease, vals_healthy, alternative='two-sided')

            results.append({
                'protein': protein,
                'disease': disease,
                'median_disease': vals_disease.median(),
                'median_healthy': vals_healthy.median(),
                'activity_diff': vals_disease.median() - vals_healthy.median(),  # Activity difference for z-scores
                'n_disease': len(vals_disease),
                'n_healthy': len(vals_healthy),
                'statistic': stat,
                'pvalue': pval
            })

    result_df = pd.DataFrame(results)

    # FDR correction
    if len(result_df) > 0:
        try:
            from statsmodels.stats.multitest import multipletests
            _, pvals_corrected, _, _ = multipletests(result_df['pvalue'].values, method='fdr_bh')
            result_df['qvalue'] = pvals_corrected
        except ImportError:
            result_df['qvalue'] = result_df['pvalue']

    log(f"  Computed {len(result_df)} disease-protein comparisons")

    return result_df


def treatment_response_analysis(
    activity_df: pd.DataFrame,
    sample_meta: pd.DataFrame,
    agg_meta_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute differential activity between responders and non-responders.

    Args:
        activity_df: Activity z-scores (proteins x sample_celltype)
        sample_meta: Sample-level metadata
        agg_meta_df: Aggregation metadata

    Returns:
        DataFrame with response differential statistics
    """
    log("Computing treatment response analysis...")

    # Filter to samples with response data
    response_meta = sample_meta[sample_meta['therapyResponse'].isin(['R', 'NR'])]
    log(f"  Samples with response data: {len(response_meta)}")

    if len(response_meta) < 10:
        log("  Warning: Too few samples with response data")
        return pd.DataFrame()

    col_to_sample = agg_meta_df['sample'].to_dict()

    # Get responder and non-responder columns
    responder_samples = response_meta[response_meta['therapyResponse'] == 'R']['sampleID'].tolist()
    nonresponder_samples = response_meta[response_meta['therapyResponse'] == 'NR']['sampleID'].tolist()

    r_cols = [c for c in activity_df.columns if col_to_sample.get(c) in responder_samples]
    nr_cols = [c for c in activity_df.columns if col_to_sample.get(c) in nonresponder_samples]

    log(f"  Responder columns: {len(r_cols)}")
    log(f"  Non-responder columns: {len(nr_cols)}")

    # Compute differential per disease
    results = []
    diseases = response_meta['disease'].unique()

    for disease in diseases:
        disease_meta = response_meta[response_meta['disease'] == disease]
        r_samples = disease_meta[disease_meta['therapyResponse'] == 'R']['sampleID'].tolist()
        nr_samples = disease_meta[disease_meta['therapyResponse'] == 'NR']['sampleID'].tolist()

        r_cols_d = [c for c in activity_df.columns if col_to_sample.get(c) in r_samples]
        nr_cols_d = [c for c in activity_df.columns if col_to_sample.get(c) in nr_samples]

        if len(r_cols_d) < 3 or len(nr_cols_d) < 3:
            continue

        for protein in activity_df.index:
            vals_r = activity_df.loc[protein, r_cols_d].dropna()
            vals_nr = activity_df.loc[protein, nr_cols_d].dropna()

            if len(vals_r) < 3 or len(vals_nr) < 3:
                continue

            stat, pval = stats.mannwhitneyu(vals_r, vals_nr, alternative='two-sided')

            results.append({
                'protein': protein,
                'disease': disease,
                'median_responder': vals_r.median(),
                'median_nonresponder': vals_nr.median(),
                'activity_diff': vals_r.median() - vals_nr.median(),  # Activity difference for z-scores
                'n_responder': len(vals_r),
                'n_nonresponder': len(vals_nr),
                'statistic': stat,
                'pvalue': pval
            })

    result_df = pd.DataFrame(results)

    # FDR correction
    if len(result_df) > 0:
        try:
            from statsmodels.stats.multitest import multipletests
            _, pvals_corrected, _, _ = multipletests(result_df['pvalue'].values, method='fdr_bh')
            result_df['qvalue'] = pvals_corrected
        except ImportError:
            result_df['qvalue'] = result_df['pvalue']

    log(f"  Computed {len(result_df)} response comparisons")

    return result_df


def correlation_analysis(
    activity_df: pd.DataFrame,
    sample_meta: pd.DataFrame,
    agg_meta_df: pd.DataFrame,
    feature_cols: List[str]
) -> pd.DataFrame:
    """
    Compute Spearman correlations between activities and continuous variables.

    Args:
        activity_df: Activity z-scores (proteins x sample_celltype)
        sample_meta: Sample-level metadata
        agg_meta_df: Aggregation metadata
        feature_cols: Columns to correlate with

    Returns:
        DataFrame with correlations and p-values
    """
    log(f"Computing correlations with {feature_cols}...")

    col_to_sample = agg_meta_df['sample'].to_dict()

    # Aggregate activity by sample (mean across cell types)
    activity_T = activity_df.T.copy()
    activity_T['sample'] = activity_T.index.map(col_to_sample)
    activity_by_sample = activity_T.groupby('sample').mean()

    # Merge with metadata
    merged = activity_by_sample.merge(
        sample_meta[['sampleID'] + feature_cols].drop_duplicates(),
        left_index=True,
        right_on='sampleID',
        how='inner'
    )

    log(f"  Merged samples: {len(merged)}")

    # Compute correlations
    results = []
    proteins = activity_df.index.tolist()

    for protein in proteins:
        if protein not in merged.columns:
            continue
        for feature in feature_cols:
            if feature not in merged.columns:
                continue

            valid = merged[[protein, feature]].dropna()
            if len(valid) < 10:
                continue

            # Convert to numeric
            try:
                valid[feature] = pd.to_numeric(valid[feature], errors='coerce')
                valid = valid.dropna()
            except Exception:
                continue

            if len(valid) < 10:
                continue

            rho, pval = stats.spearmanr(valid[protein], valid[feature])
            results.append({
                'protein': protein,
                'feature': feature,
                'rho': rho,
                'pvalue': pval,
                'n': len(valid)
            })

    result_df = pd.DataFrame(results)

    # FDR correction
    if len(result_df) > 0:
        try:
            from statsmodels.stats.multitest import multipletests
            _, pvals_corrected, _, _ = multipletests(result_df['pvalue'].values, method='fdr_bh')
            result_df['qvalue'] = pvals_corrected
        except ImportError:
            result_df['qvalue'] = result_df['pvalue']

    log(f"  Computed {len(result_df)} correlations")

    return result_df


def save_activity_to_h5ad(
    result: dict,
    meta_df: pd.DataFrame,
    output_path: Path,
    sig_name: str
):
    """Save activity results to h5ad format."""
    log(f"Saving {sig_name} results to {output_path}...")

    adata = ad.AnnData(
        X=result['zscore'].values,
        obs=pd.DataFrame(index=result['zscore'].index),
        var=meta_df
    )

    adata.layers['beta'] = result['beta'].values
    adata.layers['se'] = result['se'].values
    adata.layers['pvalue'] = result['pvalue'].values

    adata.uns['signature'] = sig_name
    adata.uns['n_rand'] = N_RAND
    adata.uns['seed'] = SEED

    adata.write_h5ad(output_path, compression='gzip')
    log(f"  Saved: {output_path}")


# ==============================================================================
# Dataset Processing
# ==============================================================================

def process_dataset(
    h5ad_path: Path,
    dataset_name: str,
    sample_meta: pd.DataFrame,
    cytosig: pd.DataFrame,
    secact: pd.DataFrame
) -> Tuple[dict, dict, pd.DataFrame]:
    """
    Process a single dataset (main, validation, or external).

    Returns:
        cytosig_result, secact_result, agg_meta_df
    """
    log(f"\n{'='*60}")
    log(f"Processing {dataset_name}")
    log(f"{'='*60}")

    # Load h5ad
    log(f"Loading: {h5ad_path}")
    adata = ad.read_h5ad(h5ad_path, backed='r')
    log(f"  Shape: {adata.shape}")

    # Check available columns
    log(f"  obs columns: {list(adata.obs.columns)}")

    # Detect cell type column (main has 'Level2', validation/external have 'Level2pred')
    if CELL_TYPE_COL in adata.obs.columns:
        cell_type_col = CELL_TYPE_COL
    elif 'Level2pred' in adata.obs.columns:
        cell_type_col = 'Level2pred'
        log(f"  Using 'Level2pred' instead of 'Level2'")
    else:
        raise KeyError(f"Neither '{CELL_TYPE_COL}' nor 'Level2pred' found in obs columns")

    # Aggregate by sample and cell type
    expr_df, agg_meta_df = aggregate_by_sample_celltype(adata, cell_type_col, SAMPLE_COL)

    # Normalize and transform
    expr_log = normalize_and_transform(expr_df)

    # Compute differential
    expr_diff = compute_differential(expr_log)

    # Free memory
    del adata
    gc.collect()

    # Run activity inference
    cytosig_result = run_activity_inference(expr_diff, cytosig, 'CytoSig')
    secact_result = run_activity_inference(expr_diff, secact, 'SecAct')

    # Save results
    save_activity_to_h5ad(
        cytosig_result, agg_meta_df,
        OUTPUT_DIR / f'{dataset_name}_CytoSig_pseudobulk.h5ad', 'CytoSig'
    )
    save_activity_to_h5ad(
        secact_result, agg_meta_df,
        OUTPUT_DIR / f'{dataset_name}_SecAct_pseudobulk.h5ad', 'SecAct'
    )

    return cytosig_result, secact_result, agg_meta_df


# ==============================================================================
# Main Analysis Pipeline
# ==============================================================================

def run_pseudobulk_analysis():
    """Run pseudo-bulk level analysis for all datasets."""
    log("=" * 60)
    log("PSEUDO-BULK ANALYSIS - INFLAMMATION ATLAS")
    log("=" * 60)

    # Load metadata
    sample_meta = load_sample_metadata()

    # Load signatures
    log("\nLoading signatures...")
    cytosig = load_cytosig()
    secact = load_secact()
    log(f"  CytoSig: {cytosig.shape}")
    log(f"  SecAct: {secact.shape}")

    # Process main dataset
    main_cyto, main_secact, main_meta = process_dataset(
        MAIN_H5AD, 'main', sample_meta, cytosig, secact
    )

    # Process validation dataset
    val_cyto, val_secact, val_meta = process_dataset(
        VAL_H5AD, 'validation', sample_meta, cytosig, secact
    )

    # Process external dataset
    ext_cyto, ext_secact, ext_meta = process_dataset(
        EXT_H5AD, 'external', sample_meta, cytosig, secact
    )

    # Statistical analyses (on main dataset)
    log("\n" + "=" * 60)
    log("STATISTICAL ANALYSIS")
    log("=" * 60)

    # Disease differential analysis
    disease_diff_cyto = disease_differential_analysis(
        main_cyto['zscore'], sample_meta, main_meta, 'disease'
    )
    disease_diff_cyto['signature'] = 'CytoSig'

    disease_diff_secact = disease_differential_analysis(
        main_secact['zscore'], sample_meta, main_meta, 'disease'
    )
    disease_diff_secact['signature'] = 'SecAct'

    disease_diff = pd.concat([disease_diff_cyto, disease_diff_secact])
    disease_diff.to_csv(OUTPUT_DIR / 'disease_differential.csv', index=False)
    log(f"  Saved disease differential: {len(disease_diff)} rows")

    # Disease group differential
    group_diff_cyto = disease_differential_analysis(
        main_cyto['zscore'], sample_meta, main_meta, 'diseaseGroup'
    )
    group_diff_cyto['signature'] = 'CytoSig'

    group_diff_secact = disease_differential_analysis(
        main_secact['zscore'], sample_meta, main_meta, 'diseaseGroup'
    )
    group_diff_secact['signature'] = 'SecAct'

    group_diff = pd.concat([group_diff_cyto, group_diff_secact])
    group_diff.to_csv(OUTPUT_DIR / 'diseaseGroup_differential.csv', index=False)
    log(f"  Saved disease group differential: {len(group_diff)} rows")

    # Treatment response analysis
    response_cyto = treatment_response_analysis(
        main_cyto['zscore'], sample_meta, main_meta
    )
    response_cyto['signature'] = 'CytoSig'

    response_secact = treatment_response_analysis(
        main_secact['zscore'], sample_meta, main_meta
    )
    response_secact['signature'] = 'SecAct'

    response_diff = pd.concat([response_cyto, response_secact])
    response_diff.to_csv(OUTPUT_DIR / 'treatment_response.csv', index=False)
    log(f"  Saved treatment response: {len(response_diff)} rows")

    # Treatment response PREDICTION (ML models)
    log("\n" + "=" * 60)
    log("TREATMENT RESPONSE PREDICTION")
    log("=" * 60)

    # CytoSig prediction
    pred_summary_cyto, pred_details_cyto = run_treatment_response_prediction(
        main_cyto['zscore'], sample_meta, main_meta, 'CytoSig'
    )
    pred_summary_cyto.to_csv(OUTPUT_DIR / 'treatment_prediction_cytosig.csv', index=False)

    # SecAct prediction
    pred_summary_secact, pred_details_secact = run_treatment_response_prediction(
        main_secact['zscore'], sample_meta, main_meta, 'SecAct'
    )
    pred_summary_secact.to_csv(OUTPUT_DIR / 'treatment_prediction_secact.csv', index=False)

    # Combined prediction summary
    pred_summary = pd.concat([pred_summary_cyto, pred_summary_secact])
    pred_summary.to_csv(OUTPUT_DIR / 'treatment_prediction_summary.csv', index=False)
    log(f"  Saved prediction summary: {len(pred_summary)} rows")

    # Save detailed results (feature importance, predictions)
    import json
    pred_details = {
        'cytosig': pred_details_cyto,
        'secact': pred_details_secact
    }
    with open(OUTPUT_DIR / 'treatment_prediction_details.json', 'w') as f:
        json.dump(pred_details, f, indent=2, default=str)
    log(f"  Saved prediction details")

    # Cross-cohort validation (if validation data available)
    if val_meta is not None:
        log("\n" + "=" * 60)
        log("CROSS-COHORT PREDICTION VALIDATION")
        log("=" * 60)

        # Validation with CytoSig
        validation_cyto = cross_cohort_response_validation(
            main_cyto['zscore'], main_meta,
            val_cyto['zscore'], val_meta,
            sample_meta
        )
        if len(validation_cyto) > 0:
            validation_cyto['signature'] = 'CytoSig'

        # Validation with SecAct
        validation_secact = cross_cohort_response_validation(
            main_secact['zscore'], main_meta,
            val_secact['zscore'], val_meta,
            sample_meta
        )
        if len(validation_secact) > 0:
            validation_secact['signature'] = 'SecAct'

        validation_all = pd.concat([validation_cyto, validation_secact])
        if len(validation_all) > 0:
            validation_all.to_csv(OUTPUT_DIR / 'cross_cohort_prediction_validation.csv', index=False)
            log(f"  Saved cross-cohort validation: {len(validation_all)} rows")

    # Cell-type stratified disease analysis
    log("\n" + "=" * 60)
    log("CELL-TYPE STRATIFIED ANALYSIS")
    log("=" * 60)

    # CytoSig cell-type stratified
    celltype_diff_cyto = celltype_stratified_disease_analysis(
        main_cyto['zscore'], sample_meta, main_meta
    )
    celltype_diff_cyto['signature'] = 'CytoSig'

    # SecAct cell-type stratified
    celltype_diff_secact = celltype_stratified_disease_analysis(
        main_secact['zscore'], sample_meta, main_meta
    )
    celltype_diff_secact['signature'] = 'SecAct'

    celltype_diff_all = pd.concat([celltype_diff_cyto, celltype_diff_secact])
    celltype_diff_all.to_csv(OUTPUT_DIR / 'celltype_stratified_differential.csv', index=False)
    log(f"  Saved cell-type stratified results: {len(celltype_diff_all)} rows")

    # Identify driving cell populations
    drivers_cyto = identify_driving_cell_populations(celltype_diff_cyto)
    drivers_secact = identify_driving_cell_populations(celltype_diff_secact)
    drivers_all = pd.concat([
        drivers_cyto.assign(signature='CytoSig') if len(drivers_cyto) > 0 else pd.DataFrame(),
        drivers_secact.assign(signature='SecAct') if len(drivers_secact) > 0 else pd.DataFrame()
    ])
    if len(drivers_all) > 0:
        drivers_all.to_csv(OUTPUT_DIR / 'driving_cell_populations.csv', index=False)
        log(f"  Saved driving cell populations: {len(drivers_all)} rows")

    # Identify conserved programs
    conserved_cyto = identify_conserved_programs(celltype_diff_cyto)
    conserved_secact = identify_conserved_programs(celltype_diff_secact)
    conserved_all = pd.concat([
        conserved_cyto.assign(signature='CytoSig') if len(conserved_cyto) > 0 else pd.DataFrame(),
        conserved_secact.assign(signature='SecAct') if len(conserved_secact) > 0 else pd.DataFrame()
    ])
    if len(conserved_all) > 0:
        conserved_all.to_csv(OUTPUT_DIR / 'conserved_cytokine_programs.csv', index=False)
        log(f"  Saved conserved programs: {len(conserved_all)} rows")

    # Correlation with age
    log("\n" + "=" * 60)
    log("AGE CORRELATION ANALYSIS")
    log("=" * 60)

    corr_age_cyto = correlation_analysis(
        main_cyto['zscore'], sample_meta, main_meta, ['age']
    )
    corr_age_cyto['signature'] = 'CytoSig'

    corr_age_secact = correlation_analysis(
        main_secact['zscore'], sample_meta, main_meta, ['age']
    )
    corr_age_secact['signature'] = 'SecAct'

    corr_age = pd.concat([corr_age_cyto, corr_age_secact])
    corr_age.to_csv(OUTPUT_DIR / 'correlation_age.csv', index=False)
    log(f"  Saved age correlations: {len(corr_age)} rows")

    # Correlation with BMI
    corr_bmi_cyto = correlation_analysis(
        main_cyto['zscore'], sample_meta, main_meta, ['BMI']
    )
    corr_bmi_cyto['signature'] = 'CytoSig'

    corr_bmi_secact = correlation_analysis(
        main_secact['zscore'], sample_meta, main_meta, ['BMI']
    )
    corr_bmi_secact['signature'] = 'SecAct'

    corr_bmi = pd.concat([corr_bmi_cyto, corr_bmi_secact])
    corr_bmi.to_csv(OUTPUT_DIR / 'correlation_bmi.csv', index=False)
    log(f"  Saved BMI correlations: {len(corr_bmi)} rows")

    # Cross-cohort validation
    log("\n" + "=" * 60)
    log("CROSS-COHORT VALIDATION")
    log("=" * 60)

    # Compare disease signatures across cohorts
    validation_results = []

    for disease in disease_diff['disease'].unique():
        # Get top signatures from main cohort
        main_sigs = disease_diff[
            (disease_diff['disease'] == disease) &
            (disease_diff['qvalue'] < 0.05)
        ]['protein'].unique()

        if len(main_sigs) == 0:
            continue

        # Check replication in validation
        if val_meta is not None:
            val_disease_diff = disease_differential_analysis(
                val_cyto['zscore'], sample_meta, val_meta, 'disease'
            )
            val_disease_diff = pd.concat([
                val_disease_diff,
                disease_differential_analysis(
                    val_secact['zscore'], sample_meta, val_meta, 'disease'
                )
            ])

            val_sigs = val_disease_diff[
                (val_disease_diff['disease'] == disease) &
                (val_disease_diff['qvalue'] < 0.05)
            ]['protein'].unique()

            replicated = len(set(main_sigs) & set(val_sigs))

            validation_results.append({
                'disease': disease,
                'main_significant': len(main_sigs),
                'validation_significant': len(val_sigs),
                'replicated': replicated,
                'replication_rate': replicated / len(main_sigs) if len(main_sigs) > 0 else 0
            })

    if validation_results:
        val_df = pd.DataFrame(validation_results)
        val_df.to_csv(OUTPUT_DIR / 'cross_cohort_validation.csv', index=False)
        log(f"  Saved validation results: {len(val_df)} diseases")

    log("\nPseudo-bulk analysis complete!")


def run_singlecell_analysis():
    """Run single-cell level analysis for main dataset."""
    log("=" * 60)
    log("SINGLE-CELL ANALYSIS - MAIN DATASET")
    log("=" * 60)

    # Load signatures
    log("Loading signatures...")
    cytosig = load_cytosig()
    secact = load_secact()

    # Load main h5ad
    log(f"Loading: {MAIN_H5AD}")
    adata = ad.read_h5ad(MAIN_H5AD, backed='r')
    n_cells = adata.shape[0]
    log(f"  Shape: {adata.shape} ({n_cells:,} cells)")

    X = adata.X

    # Use gene symbols if available (Inflammation Atlas uses Ensembl IDs as index)
    if 'symbol' in adata.var.columns:
        gene_names = list(adata.var['symbol'].values)
        log("  Using gene symbols from var['symbol']")
    else:
        gene_names = list(adata.var_names)

    # Estimate batch size
    available_gb = 32 if CUPY_AVAILABLE else 16
    batch_size = estimate_batch_size(
        n_genes=len(gene_names),
        n_features=max(cytosig.shape[1], secact.shape[1]),
        available_gb=available_gb
    )
    log(f"  Estimated batch size: {batch_size}")
    batch_size = min(batch_size, BATCH_SIZE)

    # Process signatures
    for sig, sig_name in [(cytosig, 'CytoSig'), (secact, 'SecAct')]:
        log(f"\nProcessing {sig_name}...")

        output_path = OUTPUT_DIR / f'main_{sig_name}_singlecell.h5ad'

        # Find overlapping genes
        sig_genes = set(sig.index.str.upper())
        data_genes = [g.upper() for g in gene_names]
        common_mask = [g in sig_genes for g in data_genes]
        common_genes = [g for g, m in zip(data_genes, common_mask) if m]
        common_idx = [i for i, m in enumerate(common_mask) if m]

        log(f"  Common genes: {len(common_genes)}")

        # Align signature
        sig_aligned = sig.copy()
        sig_aligned.index = sig_aligned.index.str.upper()
        sig_aligned = sig_aligned.loc[common_genes]
        sig_scaled = (sig_aligned - sig_aligned.mean()) / sig_aligned.std(ddof=1)
        sig_scaled = sig_scaled.fillna(0)

        # Process cells in batches
        log(f"  Processing {n_cells:,} cells in batches of {batch_size}...")

        ridge_batch(
            X=sig_scaled.values,
            Y=X[:, common_idx].T,
            lambda_=5e5,
            n_rand=N_RAND,
            seed=SEED,
            batch_size=batch_size,
            backend=BACKEND,
            output_path=str(output_path),
            feature_names=list(sig_scaled.columns),
            sample_names=list(adata.obs_names),
            verbose=True
        )

        log(f"  Saved: {output_path}")
        gc.collect()

    log("\nSingle-cell analysis complete!")


# ==============================================================================
# Treatment Response Prediction
# ==============================================================================

def build_response_predictor(
    activity_df: pd.DataFrame,
    sample_meta: pd.DataFrame,
    agg_meta_df: pd.DataFrame,
    disease: str = None,
    n_folds: int = 5
) -> dict:
    """
    Build treatment response predictor using cytokine activities.

    Args:
        activity_df: Activity z-scores (proteins x sample_celltype)
        sample_meta: Sample-level metadata with therapyResponse column
        agg_meta_df: Aggregation metadata (sample, cell_type per column)
        disease: Specific disease to analyze (None = all diseases)
        n_folds: Number of cross-validation folds

    Returns:
        dict with model performance metrics and feature importance
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
    from sklearn.preprocessing import StandardScaler

    log(f"Building treatment response predictor{' for ' + disease if disease else ''}...")

    # Filter to samples with response data
    response_meta = sample_meta[sample_meta['therapyResponse'].isin(['R', 'NR'])].copy()

    if disease:
        response_meta = response_meta[response_meta['disease'] == disease]

    if len(response_meta) < 20:
        log(f"  Warning: Too few samples ({len(response_meta)}) for prediction")
        return None

    log(f"  Samples with response data: {len(response_meta)}")

    # Map columns to samples
    col_to_sample = agg_meta_df['sample'].to_dict()

    # Aggregate activity by sample (mean across cell types)
    activity_T = activity_df.T.copy()
    activity_T['sample'] = activity_T.index.map(col_to_sample)
    activity_by_sample = activity_T.groupby('sample').mean()

    # Merge with response labels
    merged = activity_by_sample.merge(
        response_meta[['sampleID', 'therapyResponse', 'disease']],
        left_index=True,
        right_on='sampleID',
        how='inner'
    )

    if len(merged) < 20:
        log(f"  Warning: Too few matched samples ({len(merged)})")
        return None

    # Prepare features and labels
    feature_cols = [c for c in activity_df.index if c in merged.columns]
    X = merged[feature_cols].values
    y = (merged['therapyResponse'] == 'R').astype(int).values

    log(f"  Features: {len(feature_cols)}, Samples: {len(y)}")
    log(f"  Class balance: R={y.sum()}, NR={len(y) - y.sum()}")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Cross-validation
    cv = StratifiedKFold(n_splits=min(n_folds, min(y.sum(), len(y) - y.sum())), shuffle=True, random_state=SEED)

    results = {
        'disease': disease if disease else 'all',
        'n_samples': len(y),
        'n_responders': int(y.sum()),
        'n_nonresponders': int(len(y) - y.sum()),
        'n_features': len(feature_cols),
    }

    # Logistic Regression
    try:
        lr = LogisticRegression(max_iter=1000, random_state=SEED, class_weight='balanced')
        y_pred_lr = cross_val_predict(lr, X_scaled, y, cv=cv, method='predict_proba')[:, 1]
        auc_lr = roc_auc_score(y, y_pred_lr)
        results['lr_auc'] = auc_lr
        log(f"  Logistic Regression AUC: {auc_lr:.3f}")

        # Fit final model for coefficients
        lr.fit(X_scaled, y)
        coef_df = pd.DataFrame({
            'feature': feature_cols,
            'coefficient': lr.coef_[0]
        }).sort_values('coefficient', key=abs, ascending=False)
        results['lr_top_features'] = coef_df.head(20).to_dict('records')
    except Exception as e:
        log(f"  Logistic Regression failed: {e}")
        results['lr_auc'] = None

    # Random Forest
    try:
        rf = RandomForestClassifier(n_estimators=100, random_state=SEED, class_weight='balanced')
        y_pred_rf = cross_val_predict(rf, X_scaled, y, cv=cv, method='predict_proba')[:, 1]
        auc_rf = roc_auc_score(y, y_pred_rf)
        results['rf_auc'] = auc_rf
        log(f"  Random Forest AUC: {auc_rf:.3f}")

        # Fit final model for feature importance
        rf.fit(X_scaled, y)
        imp_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        results['rf_top_features'] = imp_df.head(20).to_dict('records')
    except Exception as e:
        log(f"  Random Forest failed: {e}")
        results['rf_auc'] = None

    # Store predictions for further analysis
    results['predictions'] = {
        'sample_id': merged['sampleID'].tolist(),
        'true_label': y.tolist(),
        'lr_prob': y_pred_lr.tolist() if 'lr_auc' in results and results['lr_auc'] else None,
        'rf_prob': y_pred_rf.tolist() if 'rf_auc' in results and results['rf_auc'] else None,
    }

    return results


def run_treatment_response_prediction(
    activity_df: pd.DataFrame,
    sample_meta: pd.DataFrame,
    agg_meta_df: pd.DataFrame,
    sig_name: str
) -> pd.DataFrame:
    """
    Run treatment response prediction across all diseases with sufficient data.

    Args:
        activity_df: Activity z-scores (proteins x sample_celltype)
        sample_meta: Sample-level metadata
        agg_meta_df: Aggregation metadata
        sig_name: Signature name for output

    Returns:
        DataFrame with prediction results per disease
    """
    log(f"\n{'='*60}")
    log(f"TREATMENT RESPONSE PREDICTION - {sig_name}")
    log(f"{'='*60}")

    # Get diseases with response data
    response_meta = sample_meta[sample_meta['therapyResponse'].isin(['R', 'NR'])]
    disease_counts = response_meta.groupby('disease').agg({
        'therapyResponse': [
            lambda x: (x == 'R').sum(),
            lambda x: (x == 'NR').sum()
        ]
    })
    disease_counts.columns = ['n_R', 'n_NR']
    disease_counts['n_total'] = disease_counts['n_R'] + disease_counts['n_NR']

    # Filter diseases with enough samples in both classes
    min_per_class = 5
    eligible_diseases = disease_counts[
        (disease_counts['n_R'] >= min_per_class) &
        (disease_counts['n_NR'] >= min_per_class)
    ].index.tolist()

    log(f"Diseases with sufficient response data: {eligible_diseases}")

    all_results = []

    # Run per-disease prediction
    for disease in eligible_diseases:
        result = build_response_predictor(
            activity_df, sample_meta, agg_meta_df, disease=disease
        )
        if result:
            result['signature'] = sig_name
            all_results.append(result)

    # Run pan-disease prediction (all diseases combined)
    log("\nRunning pan-disease prediction...")
    pan_result = build_response_predictor(
        activity_df, sample_meta, agg_meta_df, disease=None
    )
    if pan_result:
        pan_result['signature'] = sig_name
        all_results.append(pan_result)

    # Create summary DataFrame
    summary_rows = []
    for r in all_results:
        summary_rows.append({
            'disease': r['disease'],
            'signature': r['signature'],
            'n_samples': r['n_samples'],
            'n_responders': r['n_responders'],
            'n_nonresponders': r['n_nonresponders'],
            'lr_auc': r.get('lr_auc'),
            'rf_auc': r.get('rf_auc'),
            'best_auc': max(r.get('lr_auc', 0) or 0, r.get('rf_auc', 0) or 0)
        })

    result_df = pd.DataFrame(summary_rows)

    if len(result_df) > 0:
        result_df = result_df.sort_values('best_auc', ascending=False)
        log(f"\nPrediction Summary:")
        for _, row in result_df.iterrows():
            log(f"  {row['disease']}: AUC={row['best_auc']:.3f} (n={row['n_samples']})")

    return result_df, all_results


def cross_cohort_response_validation(
    main_activity: pd.DataFrame,
    main_meta: pd.DataFrame,
    val_activity: pd.DataFrame,
    val_meta: pd.DataFrame,
    sample_meta: pd.DataFrame
) -> pd.DataFrame:
    """
    Validate treatment response signatures across cohorts.

    Train on main cohort, test on validation cohort.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    log("\nCross-cohort response validation...")

    results = []

    # Get response data
    response_meta = sample_meta[sample_meta['therapyResponse'].isin(['R', 'NR'])]

    # Map main columns to samples
    main_col_to_sample = main_meta['sample'].to_dict()
    main_activity_T = main_activity.T.copy()
    main_activity_T['sample'] = main_activity_T.index.map(main_col_to_sample)
    main_by_sample = main_activity_T.groupby('sample').mean()

    # Map validation columns to samples
    val_col_to_sample = val_meta['sample'].to_dict()
    val_activity_T = val_activity.T.copy()
    val_activity_T['sample'] = val_activity_T.index.map(val_col_to_sample)
    val_by_sample = val_activity_T.groupby('sample').mean()

    # Get common features
    common_features = list(set(main_activity.index) & set(val_activity.index))

    for disease in response_meta['disease'].unique():
        disease_response = response_meta[response_meta['disease'] == disease]

        # Main cohort data
        main_samples = disease_response[disease_response['sampleID'].isin(main_by_sample.index)]
        if len(main_samples) < 10:
            continue

        # Validation cohort data
        val_samples = disease_response[disease_response['sampleID'].isin(val_by_sample.index)]
        if len(val_samples) < 5:
            continue

        # Prepare training data (main)
        X_train = main_by_sample.loc[main_samples['sampleID'], common_features].values
        y_train = (main_samples['therapyResponse'] == 'R').astype(int).values

        # Prepare test data (validation)
        X_test = val_by_sample.loc[val_samples['sampleID'], common_features].values
        y_test = (val_samples['therapyResponse'] == 'R').astype(int).values

        # Check class balance
        if y_train.sum() < 3 or (len(y_train) - y_train.sum()) < 3:
            continue
        if y_test.sum() < 2 or (len(y_test) - y_test.sum()) < 2:
            continue

        try:
            # Scale and train
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            lr = LogisticRegression(max_iter=1000, random_state=SEED, class_weight='balanced')
            lr.fit(X_train_scaled, y_train)

            # Predict on validation
            y_pred = lr.predict_proba(X_test_scaled)[:, 1]
            auc = roc_auc_score(y_test, y_pred)

            results.append({
                'disease': disease,
                'train_n': len(y_train),
                'train_R': int(y_train.sum()),
                'test_n': len(y_test),
                'test_R': int(y_test.sum()),
                'validation_auc': auc
            })

            log(f"  {disease}: Train n={len(y_train)}, Test n={len(y_test)}, AUC={auc:.3f}")

        except Exception as e:
            log(f"  {disease}: Failed - {e}")

    return pd.DataFrame(results)


# ==============================================================================
# Cell-Type Stratified Analysis
# ==============================================================================

def celltype_stratified_disease_analysis(
    activity_df: pd.DataFrame,
    sample_meta: pd.DataFrame,
    agg_meta_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute disease differential within each cell type to identify
    which cell populations drive disease-specific signatures.

    Args:
        activity_df: Activity z-scores (proteins x sample_celltype)
        sample_meta: Sample-level metadata
        agg_meta_df: Aggregation metadata (sample, cell_type per column)

    Returns:
        DataFrame with cell type-stratified disease associations
    """
    log("Computing cell-type stratified disease analysis...")

    # Map columns to samples and cell types
    col_to_sample = agg_meta_df['sample'].to_dict()
    col_to_celltype = agg_meta_df['cell_type'].to_dict()

    # Add disease info to columns
    sample_disease = sample_meta.set_index('sampleID')['disease'].to_dict()

    # Get healthy columns
    healthy_samples = sample_meta[sample_meta['disease'] == 'healthy']['sampleID'].tolist()

    results = []
    celltypes = agg_meta_df['cell_type'].unique()
    diseases = sample_meta[sample_meta['disease'] != 'healthy']['disease'].unique()

    log(f"  Cell types: {len(celltypes)}")
    log(f"  Diseases: {len(diseases)}")

    for celltype in celltypes:
        # Get columns for this cell type
        celltype_cols = [c for c in activity_df.columns if col_to_celltype.get(c) == celltype]

        if len(celltype_cols) < 10:
            continue

        # Healthy columns for this cell type
        healthy_cols = [c for c in celltype_cols if col_to_sample.get(c) in healthy_samples]

        if len(healthy_cols) < 3:
            continue

        for disease in diseases:
            disease_samples = sample_meta[sample_meta['disease'] == disease]['sampleID'].tolist()
            disease_cols = [c for c in celltype_cols if col_to_sample.get(c) in disease_samples]

            if len(disease_cols) < 3:
                continue

            # Compute differential per protein
            for protein in activity_df.index:
                vals_disease = activity_df.loc[protein, disease_cols].dropna()
                vals_healthy = activity_df.loc[protein, healthy_cols].dropna()

                if len(vals_disease) < 3 or len(vals_healthy) < 3:
                    continue

                try:
                    stat, pval = stats.mannwhitneyu(vals_disease, vals_healthy, alternative='two-sided')
                    activity_diff = vals_disease.median() - vals_healthy.median()  # Activity difference for z-scores

                    results.append({
                        'cell_type': celltype,
                        'disease': disease,
                        'protein': protein,
                        'median_disease': vals_disease.median(),
                        'median_healthy': vals_healthy.median(),
                        'activity_diff': activity_diff,
                        'n_disease': len(vals_disease),
                        'n_healthy': len(vals_healthy),
                        'statistic': stat,
                        'pvalue': pval
                    })
                except Exception:
                    continue

    result_df = pd.DataFrame(results)

    # FDR correction
    if len(result_df) > 0:
        try:
            from statsmodels.stats.multitest import multipletests
            _, pvals_corrected, _, _ = multipletests(result_df['pvalue'].values, method='fdr_bh')
            result_df['qvalue'] = pvals_corrected
        except ImportError:
            result_df['qvalue'] = result_df['pvalue']

    log(f"  Computed {len(result_df)} cell type-disease-protein combinations")

    return result_df


def identify_driving_cell_populations(
    celltype_diff: pd.DataFrame,
    qvalue_threshold: float = 0.05,
    activity_diff_threshold: float = 0.5
) -> pd.DataFrame:
    """
    Identify which cell populations drive disease-specific cytokine signatures.

    Args:
        celltype_diff: Results from celltype_stratified_disease_analysis
        qvalue_threshold: FDR threshold
        activity_diff_threshold: Minimum effect size

    Returns:
        DataFrame summarizing driving cell populations per disease
    """
    log("Identifying driving cell populations...")

    if len(celltype_diff) == 0:
        return pd.DataFrame()

    # Filter significant results
    sig_df = celltype_diff[
        (celltype_diff['qvalue'] < qvalue_threshold) &
        (celltype_diff['activity_diff'].abs() > activity_diff_threshold)
    ].copy()

    if len(sig_df) == 0:
        log("  No significant cell type-specific associations found")
        return pd.DataFrame()

    # Summarize per disease-celltype
    summary = sig_df.groupby(['disease', 'cell_type']).agg({
        'protein': 'count',
        'activity_diff': ['mean', 'std'],
        'qvalue': 'min'
    }).reset_index()

    summary.columns = ['disease', 'cell_type', 'n_significant_proteins',
                       'mean_activity_diff', 'std_activity_diff', 'min_qvalue']

    # Sort by number of significant proteins
    summary = summary.sort_values(['disease', 'n_significant_proteins'], ascending=[True, False])

    # Identify top driving cell types per disease
    top_drivers = summary.groupby('disease').head(5)

    log(f"  Top driving cell populations:")
    for disease in top_drivers['disease'].unique():
        disease_drivers = top_drivers[top_drivers['disease'] == disease]
        drivers = disease_drivers['cell_type'].tolist()[:3]
        log(f"    {disease}: {', '.join(drivers)}")

    return summary


def identify_conserved_programs(
    celltype_diff: pd.DataFrame,
    min_diseases: int = 3,
    qvalue_threshold: float = 0.05
) -> pd.DataFrame:
    """
    Identify cytokine programs conserved across multiple diseases and cell types.

    Args:
        celltype_diff: Results from celltype_stratified_disease_analysis
        min_diseases: Minimum number of diseases showing the signature
        qvalue_threshold: FDR threshold

    Returns:
        DataFrame with conserved program identification
    """
    log(f"Identifying conserved programs (≥{min_diseases} diseases)...")

    if len(celltype_diff) == 0:
        return pd.DataFrame()

    # Filter significant results
    sig_df = celltype_diff[celltype_diff['qvalue'] < qvalue_threshold].copy()

    if len(sig_df) == 0:
        return pd.DataFrame()

    # Count diseases per protein-celltype combination
    protein_celltype_diseases = sig_df.groupby(['protein', 'cell_type']).agg({
        'disease': lambda x: list(x.unique()),
        'activity_diff': 'mean',
        'qvalue': 'min'
    }).reset_index()

    protein_celltype_diseases['n_diseases'] = protein_celltype_diseases['disease'].apply(len)

    # Filter to conserved
    conserved = protein_celltype_diseases[
        protein_celltype_diseases['n_diseases'] >= min_diseases
    ].copy()

    # Classify direction
    conserved['direction'] = conserved['activity_diff'].apply(
        lambda x: 'up' if x > 0 else 'down'
    )

    # Sort by conservation
    conserved = conserved.sort_values('n_diseases', ascending=False)

    if len(conserved) > 0:
        log(f"  Found {len(conserved)} conserved protein-celltype programs")
        log(f"  Top conserved:")
        for _, row in conserved.head(10).iterrows():
            log(f"    {row['protein']} in {row['cell_type']}: {row['n_diseases']} diseases ({row['direction']})")

    return conserved


# ==============================================================================
# Entry Point
# ==============================================================================

def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Inflammation Atlas Activity Analysis')
    parser.add_argument('--mode', choices=['pseudobulk', 'singlecell', 'both'],
                       default='pseudobulk', help='Analysis mode')
    parser.add_argument('--test', action='store_true',
                       help='Run on subset for testing')
    args = parser.parse_args()

    log("=" * 60)
    log("INFLAMMATION ATLAS ACTIVITY ANALYSIS")
    log("=" * 60)
    log(f"Mode: {args.mode}")
    log(f"Backend: {BACKEND}")
    log(f"Output: {OUTPUT_DIR}")

    if args.test:
        log("TEST MODE: Processing subset only")
        global BATCH_SIZE, N_RAND
        BATCH_SIZE = 1000
        N_RAND = 100

    start_time = time.time()

    if args.mode in ['pseudobulk', 'both']:
        run_pseudobulk_analysis()

    if args.mode in ['singlecell', 'both']:
        run_singlecell_analysis()

    total_time = time.time() - start_time
    log(f"\nTotal time: {total_time/60:.1f} minutes")
    log("Done!")


if __name__ == '__main__':
    main()
