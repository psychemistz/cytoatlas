#!/usr/bin/env python3
"""
CIMA Atlas Activity Analysis
=============================
Compute cytokine (CytoSig, 44 signatures) and secreted protein (SecAct, 1,249 signatures)
activities across the CIMA single-cell atlas (6.5M cells) and associate with patient metadata.

Datasets:
- CIMA Cell Atlas: 6,484,974 cells, 36,326 genes, 428 samples
- Blood biochemistry: 399 samples, 19 markers
- Plasma metabolites: 390 samples, 1,549 features

Analysis levels:
1. Pseudo-bulk (aggregate by cell type and sample)
2. Single-cell level (per-cell activities)

Statistical analyses:
- Correlation: activity vs biochemistry/metabolites (Spearman)
- Differential: activity by sex, smoking, blood type (Wilcoxon)
"""

import os
import sys
import gc
import warnings
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
from scipy import stats

# Add SecActpy to path
sys.path.insert(0, '/data/parks34/projects/1ridgesig/SecActpy')
from secactpy import (
    load_cytosig, load_secact,
    ridge_batch, estimate_batch_size,
    CUPY_AVAILABLE
)

# ==============================================================================
# Configuration
# ==============================================================================

# Data paths
DATA_DIR = Path('/data/Jiang_Lab/Data/Seongyong/CIMA')
H5AD_PATH = DATA_DIR / 'Cell_Atlas/CIMA_RNA_6484974cells_36326genes_compressed.h5ad'
BIOCHEMISTRY_PATH = DATA_DIR / 'Cell_Atlas/CIMA_Sample_Blood_Biochemistry_Results.csv'
METABOLITES_PATH = DATA_DIR / 'Cell_Atlas/CIMA_Sample_Plasma_Metabolites_and_Lipids_Results.csv'
SAMPLE_META_PATH = DATA_DIR / 'Metadata/CIMA_Sample_Information_Metadata.csv'

# Output paths
OUTPUT_DIR = Path('/data/parks34/projects/2cytoatlas/results/cima')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Analysis parameters
CELL_TYPE_COL = 'cell_type_l2'  # 27 intermediate cell types
SAMPLE_COL = 'sample'
BATCH_SIZE = 10000  # For single-cell analysis
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


def load_metadata():
    """Load all CIMA metadata files."""
    log("Loading metadata files...")

    # Sample information
    sample_meta = pd.read_csv(SAMPLE_META_PATH)
    sample_meta = sample_meta.rename(columns={'Sample_name': 'sample'})
    log(f"  Sample metadata: {len(sample_meta)} samples")

    # Blood biochemistry
    biochem = pd.read_csv(BIOCHEMISTRY_PATH)
    biochem = biochem.rename(columns={'Sample': 'sample'})
    log(f"  Biochemistry: {len(biochem)} samples, {len(biochem.columns)-1} markers")

    # Plasma metabolites
    metab = pd.read_csv(METABOLITES_PATH)
    # Column might be 'sample' (lowercase)
    if 'sample' not in metab.columns and 'Sample' in metab.columns:
        metab = metab.rename(columns={'Sample': 'sample'})
    log(f"  Metabolites: {len(metab)} samples, {len(metab.columns)-1} features")

    return sample_meta, biochem, metab


def aggregate_by_sample_celltype(adata, cell_type_col: str, sample_col: str):
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

    # Get raw counts from layers
    if 'counts' in adata.layers:
        X = adata.layers['counts']
        log("  Using raw counts from layers['counts']")
    else:
        X = adata.X
        log("  Using .X (may be log-normalized)")

    # Aggregate by summing
    gene_names = list(adata.var_names)
    n_genes = len(gene_names)

    aggregated = {}
    meta_rows = []

    for i, ((sample, celltype), indices) in enumerate(groups.items()):
        col_name = f"{sample}_{celltype}"

        # Get indices as integer array (these are now positional after reset_index)
        idx_array = np.array(indices, dtype=np.int64)

        # Sum counts for this group (ensure 1D array)
        if sp.issparse(X):
            group_sum = np.asarray(X[idx_array, :].sum(axis=0)).ravel()
        else:
            group_sum = np.asarray(X[idx_array, :].sum(axis=0)).ravel()
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
    # TPM normalize
    col_sums = expr_df.sum(axis=0)
    expr_tpm = expr_df / col_sums * 1e6

    # Log2 transform
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
    output_prefix: str
) -> dict:
    """
    Run SecActpy activity inference.

    Args:
        expr_df: Differential expression (genes x samples), z-scored
        signature: Signature matrix (genes x proteins)
        sig_name: Name for logging
        output_prefix: Prefix for output files

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
    n_features = sig_scaled.shape[1]

    if n_samples > 1000:
        # Use batch processing
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
        # Use standard ridge
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


def correlation_analysis(
    activity_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    feature_cols: list,
    sample_col: str = 'sample',
    agg_meta_df: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Compute Spearman correlations between activities and continuous variables.

    Args:
        activity_df: Activity z-scores (proteins x samples)
        metadata_df: Metadata with continuous variables
        feature_cols: Columns to correlate with
        sample_col: Column to merge on
        agg_meta_df: Aggregation metadata with sample info per column

    Returns:
        DataFrame with correlations and p-values
    """
    log(f"Computing correlations with {len(feature_cols)} features...")

    # Transpose activity to (samples x proteins)
    activity_T = activity_df.T.copy()

    # Extract sample names from activity columns
    if agg_meta_df is not None and 'sample' in agg_meta_df.columns:
        # Use the aggregation metadata which has correct sample mapping
        col_to_sample = agg_meta_df['sample'].to_dict()
        activity_T['sample'] = activity_T.index.map(col_to_sample)
    else:
        # Fallback: parse column names (format: CIMA_H001_celltype)
        # Sample ID has format CIMA_HXXX (first two underscore parts)
        def extract_sample(col_name):
            parts = col_name.split('_')
            if len(parts) >= 2:
                return f"{parts[0]}_{parts[1]}"
            return col_name
        activity_T['sample'] = activity_T.index.map(extract_sample)

    # Aggregate activity by sample (mean across cell types)
    activity_by_sample = activity_T.groupby('sample').mean()

    # Merge with metadata
    merged = activity_by_sample.merge(
        metadata_df[[sample_col] + feature_cols].drop_duplicates(),
        left_index=True,
        right_on=sample_col,
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

            # Remove NaN values
            valid = merged[[protein, feature]].dropna()
            if len(valid) < 5:
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

    # Multiple testing correction (FDR)
    if len(result_df) > 0:
        try:
            from statsmodels.stats.multitest import multipletests
            _, pvals_corrected, _, _ = multipletests(result_df['pvalue'].values, method='fdr_bh')
            result_df['qvalue'] = pvals_corrected
        except ImportError:
            # Fallback to scipy if statsmodels not available
            result_df['qvalue'] = result_df['pvalue']

    log(f"  Computed {len(result_df)} correlations")

    return result_df


def differential_analysis(
    activity_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    group_col: str,
    sample_col: str = 'sample',
    agg_meta_df: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Compute Wilcoxon rank-sum tests for activities between groups.

    Args:
        activity_df: Activity z-scores (proteins x samples)
        metadata_df: Metadata with grouping variable
        group_col: Column for grouping
        sample_col: Column to merge on
        agg_meta_df: Aggregation metadata with sample info per column

    Returns:
        DataFrame with test statistics and p-values
    """
    log(f"Computing differential analysis for {group_col}...")

    # Transpose and add sample info
    activity_T = activity_df.T.copy()

    if agg_meta_df is not None and 'sample' in agg_meta_df.columns:
        col_to_sample = agg_meta_df['sample'].to_dict()
        activity_T['sample'] = activity_T.index.map(col_to_sample)
    else:
        # Fallback: parse column names (format: CIMA_H001_celltype)
        def extract_sample(col_name):
            parts = col_name.split('_')
            if len(parts) >= 2:
                return f"{parts[0]}_{parts[1]}"
            return col_name
        activity_T['sample'] = activity_T.index.map(extract_sample)

    # Aggregate by sample
    activity_by_sample = activity_T.groupby('sample').mean()

    # Merge with metadata
    merged = activity_by_sample.merge(
        metadata_df[[sample_col, group_col]].drop_duplicates(),
        left_index=True,
        right_on=sample_col,
        how='inner'
    )

    # Get unique groups
    groups = merged[group_col].dropna().unique()
    log(f"  Groups: {groups}")

    if len(groups) < 2:
        log("  Warning: Less than 2 groups, skipping")
        return pd.DataFrame()

    # Compute pairwise tests
    results = []
    proteins = activity_df.index.tolist()

    for protein in proteins:
        if protein not in merged.columns:
            continue

        for i, g1 in enumerate(groups):
            for g2 in groups[i+1:]:
                vals1 = merged[merged[group_col] == g1][protein].dropna()
                vals2 = merged[merged[group_col] == g2][protein].dropna()

                if len(vals1) < 3 or len(vals2) < 3:
                    continue

                stat, pval = stats.mannwhitneyu(vals1, vals2, alternative='two-sided')

                results.append({
                    'protein': protein,
                    'group1': g1,
                    'group2': g2,
                    'median_g1': vals1.median(),
                    'median_g2': vals2.median(),
                    'n_g1': len(vals1),
                    'n_g2': len(vals2),
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
            result_df['qvalue'] = result_df['pvalue']  # Fallback

    log(f"  Computed {len(result_df)} comparisons")

    return result_df


def save_activity_to_h5ad(
    result: dict,
    meta_df: pd.DataFrame,
    output_path: Path,
    sig_name: str
):
    """Save activity results to h5ad format."""
    log(f"Saving {sig_name} results to {output_path}...")

    # Create AnnData with activities as the main matrix
    # Proteins are observations, samples are variables (transposed convention)
    adata = ad.AnnData(
        X=result['zscore'].values,
        obs=pd.DataFrame(index=result['zscore'].index),
        var=meta_df
    )

    # Store other matrices in layers
    adata.layers['beta'] = result['beta'].values
    adata.layers['se'] = result['se'].values
    adata.layers['pvalue'] = result['pvalue'].values

    # Store metadata
    adata.uns['signature'] = sig_name
    adata.uns['n_rand'] = N_RAND
    adata.uns['seed'] = SEED

    adata.write_h5ad(output_path, compression='gzip')
    log(f"  Saved: {output_path}")


# ==============================================================================
# Main Analysis Pipeline
# ==============================================================================

def run_pseudobulk_analysis():
    """Run pseudo-bulk level analysis (aggregated by cell type and sample)."""
    log("=" * 60)
    log("PSEUDO-BULK ANALYSIS")
    log("=" * 60)

    # Load metadata
    sample_meta, biochem, metab = load_metadata()

    # Load signatures
    log("Loading signatures...")
    cytosig = load_cytosig()
    secact = load_secact()
    log(f"  CytoSig: {cytosig.shape}")
    log(f"  SecAct: {secact.shape}")

    # Load h5ad in backed mode
    log(f"Loading h5ad: {H5AD_PATH}")
    adata = ad.read_h5ad(H5AD_PATH, backed='r')
    log(f"  Shape: {adata.shape}")

    # Aggregate by sample and cell type
    expr_df, meta_df = aggregate_by_sample_celltype(adata, CELL_TYPE_COL, SAMPLE_COL)

    # Normalize and transform
    expr_log = normalize_and_transform(expr_df)

    # Compute differential expression
    expr_diff = compute_differential(expr_log)

    # Free memory
    del adata
    gc.collect()

    # Run CytoSig analysis
    cytosig_result = run_activity_inference(
        expr_diff, cytosig, 'CytoSig',
        str(OUTPUT_DIR / 'CIMA_CytoSig_pseudobulk')
    )

    # Run SecAct analysis
    secact_result = run_activity_inference(
        expr_diff, secact, 'SecAct',
        str(OUTPUT_DIR / 'CIMA_SecAct_pseudobulk')
    )

    # Save activity results
    save_activity_to_h5ad(
        cytosig_result, meta_df,
        OUTPUT_DIR / 'CIMA_CytoSig_pseudobulk.h5ad', 'CytoSig'
    )
    save_activity_to_h5ad(
        secact_result, meta_df,
        OUTPUT_DIR / 'CIMA_SecAct_pseudobulk.h5ad', 'SecAct'
    )

    # Statistical analyses
    log("\n" + "=" * 60)
    log("STATISTICAL ANALYSIS")
    log("=" * 60)

    # Biochemistry correlations
    biochem_cols = [c for c in biochem.columns if c != 'sample']
    corr_biochem_cytosig = correlation_analysis(
        cytosig_result['zscore'], biochem, biochem_cols, agg_meta_df=meta_df
    )
    corr_biochem_cytosig['signature'] = 'CytoSig'

    corr_biochem_secact = correlation_analysis(
        secact_result['zscore'], biochem, biochem_cols, agg_meta_df=meta_df
    )
    corr_biochem_secact['signature'] = 'SecAct'

    corr_biochem = pd.concat([corr_biochem_cytosig, corr_biochem_secact])
    corr_biochem.to_csv(OUTPUT_DIR / 'CIMA_correlation_biochemistry.csv', index=False)
    log(f"  Saved biochemistry correlations: {len(corr_biochem)} rows")

    # Age correlations
    log("Computing age correlations...")
    corr_age_cytosig = correlation_analysis(
        cytosig_result['zscore'], sample_meta, ['Age'], agg_meta_df=meta_df
    )
    corr_age_cytosig['signature'] = 'CytoSig'

    corr_age_secact = correlation_analysis(
        secact_result['zscore'], sample_meta, ['Age'], agg_meta_df=meta_df
    )
    corr_age_secact['signature'] = 'SecAct'

    corr_age = pd.concat([corr_age_cytosig, corr_age_secact])
    corr_age.to_csv(OUTPUT_DIR / 'CIMA_correlation_age.csv', index=False)
    log(f"  Saved age correlations: {len(corr_age)} rows")

    # BMI correlations
    log("Computing BMI correlations...")
    corr_bmi_cytosig = correlation_analysis(
        cytosig_result['zscore'], sample_meta, ['BMI'], agg_meta_df=meta_df
    )
    corr_bmi_cytosig['signature'] = 'CytoSig'

    corr_bmi_secact = correlation_analysis(
        secact_result['zscore'], sample_meta, ['BMI'], agg_meta_df=meta_df
    )
    corr_bmi_secact['signature'] = 'SecAct'

    corr_bmi = pd.concat([corr_bmi_cytosig, corr_bmi_secact])
    corr_bmi.to_csv(OUTPUT_DIR / 'CIMA_correlation_bmi.csv', index=False)
    log(f"  Saved BMI correlations: {len(corr_bmi)} rows")

    # Metabolite correlations (top features by variance)
    metab_cols = [c for c in metab.columns if c != 'sample']
    # Subsample to top 500 most variable metabolites for efficiency
    metab_numeric = metab[metab_cols].select_dtypes(include=[np.number])
    top_metab_cols = metab_numeric.var().nlargest(500).index.tolist()

    corr_metab_cytosig = correlation_analysis(
        cytosig_result['zscore'], metab, top_metab_cols, agg_meta_df=meta_df
    )
    corr_metab_cytosig['signature'] = 'CytoSig'

    corr_metab_secact = correlation_analysis(
        secact_result['zscore'], metab, top_metab_cols, agg_meta_df=meta_df
    )
    corr_metab_secact['signature'] = 'SecAct'

    corr_metab = pd.concat([corr_metab_cytosig, corr_metab_secact])
    corr_metab.to_csv(OUTPUT_DIR / 'CIMA_correlation_metabolites.csv', index=False)
    log(f"  Saved metabolite correlations: {len(corr_metab)} rows")

    # Differential analysis by demographics
    # Sex (column is lowercase 'sex')
    diff_sex_cytosig = differential_analysis(
        cytosig_result['zscore'], sample_meta, 'sex', agg_meta_df=meta_df
    )
    diff_sex_cytosig['signature'] = 'CytoSig'

    diff_sex_secact = differential_analysis(
        secact_result['zscore'], sample_meta, 'sex', agg_meta_df=meta_df
    )
    diff_sex_secact['signature'] = 'SecAct'

    diff_sex = pd.concat([diff_sex_cytosig, diff_sex_secact])
    diff_sex['comparison'] = 'sex'

    # Smoking (column is 'smoking_ststus' - note the typo in original data)
    diff_smoke_cytosig = differential_analysis(
        cytosig_result['zscore'], sample_meta, 'smoking_ststus', agg_meta_df=meta_df
    )
    diff_smoke_cytosig['signature'] = 'CytoSig'

    diff_smoke_secact = differential_analysis(
        secact_result['zscore'], sample_meta, 'smoking_ststus', agg_meta_df=meta_df
    )
    diff_smoke_secact['signature'] = 'SecAct'

    diff_smoke = pd.concat([diff_smoke_cytosig, diff_smoke_secact])
    diff_smoke['comparison'] = 'smoking_status'

    # Blood type (column is lowercase 'blood_type')
    diff_blood_cytosig = differential_analysis(
        cytosig_result['zscore'], sample_meta, 'blood_type', agg_meta_df=meta_df
    )
    diff_blood_cytosig['signature'] = 'CytoSig'

    diff_blood_secact = differential_analysis(
        secact_result['zscore'], sample_meta, 'blood_type', agg_meta_df=meta_df
    )
    diff_blood_secact['signature'] = 'SecAct'

    diff_blood = pd.concat([diff_blood_cytosig, diff_blood_secact])
    diff_blood['comparison'] = 'blood_type'

    # Combine all differential analyses
    diff_all = pd.concat([diff_sex, diff_smoke, diff_blood])
    diff_all.to_csv(OUTPUT_DIR / 'CIMA_differential_demographics.csv', index=False)
    log(f"  Saved differential analysis: {len(diff_all)} rows")

    log("\nPseudo-bulk analysis complete!")

    return cytosig_result, secact_result


def run_singlecell_analysis():
    """Run single-cell level analysis (per-cell activities)."""
    log("=" * 60)
    log("SINGLE-CELL ANALYSIS")
    log("=" * 60)

    # Load signatures
    log("Loading signatures...")
    cytosig = load_cytosig()
    secact = load_secact()

    # Load h5ad
    log(f"Loading h5ad: {H5AD_PATH}")
    adata = ad.read_h5ad(H5AD_PATH, backed='r')
    n_cells = adata.shape[0]
    log(f"  Shape: {adata.shape} ({n_cells:,} cells)")

    # Get raw counts
    if 'counts' in adata.layers:
        log("  Using raw counts from layers['counts']")
        X = adata.layers['counts']
    else:
        log("  Using .X")
        X = adata.X

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

    # Process in batches and stream to disk
    for sig, sig_name in [(cytosig, 'CytoSig'), (secact, 'SecAct')]:
        log(f"\nProcessing {sig_name}...")

        output_path = OUTPUT_DIR / f'CIMA_{sig_name}_singlecell.h5ad'

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

        # For very large datasets, use streaming output
        ridge_batch(
            X=sig_scaled.values,
            Y=X[:, common_idx].T,  # genes x cells
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
# Entry Point
# ==============================================================================

def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='CIMA Atlas Activity Analysis')
    parser.add_argument('--mode', choices=['pseudobulk', 'singlecell', 'both'],
                       default='pseudobulk', help='Analysis mode')
    parser.add_argument('--test', action='store_true',
                       help='Run on subset for testing')
    args = parser.parse_args()

    log("=" * 60)
    log("CIMA ATLAS ACTIVITY ANALYSIS")
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
