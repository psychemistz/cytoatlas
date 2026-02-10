#!/usr/bin/env python3
"""
parse_10M Cytokine Perturbation Activity Analysis
===================================================
Compute cytokine (CytoSig, 44 signatures) and secreted protein (SecAct, 1,249 signatures)
activities from the parse_10M PBMC cytokine perturbation dataset (9.7M cells) and quantify
treatment effects relative to PBS control.

Datasets:
- Parse_10M_PBMC_cytokines.h5ad: 9,700,000 cells, 40,352 genes, 212 GB
- Metadata: cytokine_origin_parse10M.csv (12 donors x 91 conditions x 18 PBMC cell types)
- Conditions: 90 cytokine perturbations + PBS control

Analysis levels:
1. Pseudo-bulk (aggregate by donor x cytokine x cell_type)
2. Single-cell level (per-cell activities)

Statistical analyses:
- Treatment vs control: cytokine-treated - PBS control per donor per cell type
- Response matrix: cytokine x signature activity matrix across cell types
- FDR correction (Benjamini-Hochberg)
"""

import os
import sys
import gc
import warnings
import time
from pathlib import Path
from typing import Optional, Dict, List, Tuple

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
DATA_DIR = Path('/data/Jiang_Lab/Data/Seongyong/parse_10M')
H5AD_PATH = DATA_DIR / 'Parse_10M_PBMC_cytokines.h5ad'
METADATA_PATH = DATA_DIR / 'cytokine_origin_parse10M.csv'

# Output paths
OUTPUT_DIR = Path('/vf/users/parks34/projects/2secactpy/results/parse10m')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Analysis parameters
DONOR_COL = 'donor'
CYTOKINE_COL = 'cytokine'
CELL_TYPE_COL = 'cell_type'
CONTROL_CONDITION = 'PBS'
BATCH_SIZE = 10000
N_RAND = 1000
SEED = 0
CHUNK_SIZE = 50000  # For backed mode reading

# GPU settings
BACKEND = 'cupy' if CUPY_AVAILABLE else 'numpy'

# ==============================================================================
# Utility Functions
# ==============================================================================

def log(msg: str):
    """Print timestamped log message."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_metadata() -> pd.DataFrame:
    """Load cytokine perturbation metadata."""
    log("Loading metadata...")
    meta = pd.read_csv(METADATA_PATH)
    log(f"  Shape: {meta.shape}")
    log(f"  Columns: {list(meta.columns)}")
    log(f"  Donors: {meta[DONOR_COL].nunique() if DONOR_COL in meta.columns else 'N/A'}")
    log(f"  Cytokines: {meta[CYTOKINE_COL].nunique() if CYTOKINE_COL in meta.columns else 'N/A'}")
    log(f"  Cell types: {meta[CELL_TYPE_COL].nunique() if CELL_TYPE_COL in meta.columns else 'N/A'}")
    return meta


def aggregate_pseudobulk(
    adata,
    donor_col: str,
    cytokine_col: str,
    celltype_col: str,
    min_cells: int = 10,
    chunk_size: int = 50000
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aggregate expression by donor x cytokine x cell_type for pseudo-bulk analysis.

    Uses chunked reading for efficiency with backed mode (reads file sequentially).

    Args:
        adata: AnnData with raw counts (can be backed)
        donor_col: Column for donor ID
        cytokine_col: Column for cytokine condition
        celltype_col: Column for cell type
        min_cells: Minimum cells per group
        chunk_size: Number of cells to read at a time

    Returns:
        expr_df: DataFrame (genes x pseudobulk groups)
        meta_df: DataFrame with donor, cytokine, cell_type info for each column
    """
    grouping_cols = [donor_col, cytokine_col, celltype_col]
    log(f"Aggregating by {grouping_cols}...")

    # Get obs data (fast even for backed mode)
    obs = adata.obs[grouping_cols].copy()
    obs = obs.reset_index(drop=True)

    # Create group labels
    obs['_group'] = (
        obs[donor_col].astype(str) + '|' +
        obs[cytokine_col].astype(str) + '|' +
        obs[celltype_col].astype(str)
    )
    unique_groups = obs['_group'].unique()
    group_to_idx = {g: i for i, g in enumerate(unique_groups)}
    obs['_group_idx'] = obs['_group'].map(group_to_idx)

    log(f"  Found {len(unique_groups)} donor-cytokine-celltype combinations")

    # Get raw counts source
    if 'counts' in adata.layers:
        X = adata.layers['counts']
        log("  Using raw counts from layers['counts']")
    else:
        X = adata.X
        log("  Using .X (may be log-normalized)")

    gene_names = list(adata.var_names)
    n_genes = len(gene_names)
    n_cells = adata.n_obs
    n_groups = len(unique_groups)

    # Initialize accumulators
    group_sums = np.zeros((n_groups, n_genes), dtype=np.float64)
    group_counts = np.zeros(n_groups, dtype=np.int64)

    # Process in chunks (efficient for backed mode)
    log(f"  Processing {n_cells:,} cells in chunks of {chunk_size:,}...")
    n_chunks = (n_cells + chunk_size - 1) // chunk_size

    for chunk_i in range(n_chunks):
        start_idx = chunk_i * chunk_size
        end_idx = min((chunk_i + 1) * chunk_size, n_cells)

        # Read chunk
        chunk_X = X[start_idx:end_idx, :]
        if sp.issparse(chunk_X):
            chunk_X = chunk_X.toarray()
        chunk_X = np.asarray(chunk_X, dtype=np.float64)

        # Get group indices for this chunk
        chunk_groups = obs['_group_idx'].iloc[start_idx:end_idx].values

        # Accumulate sums for each group
        for local_i in range(chunk_X.shape[0]):
            g_idx = chunk_groups[local_i]
            group_sums[g_idx, :] += chunk_X[local_i, :]
            group_counts[g_idx] += 1

        if (chunk_i + 1) % 10 == 0 or chunk_i == n_chunks - 1:
            log(f"    Processed chunk {chunk_i + 1}/{n_chunks} ({end_idx:,} cells)")

    # Build results, filtering by min_cells
    aggregated = {}
    meta_rows = []

    for g_idx, group_name in enumerate(unique_groups):
        if group_counts[g_idx] < min_cells:
            continue

        aggregated[group_name] = group_sums[g_idx, :]

        # Parse group name
        parts = group_name.split('|')
        meta_rows.append({
            'column': group_name,
            'donor': parts[0],
            'cytokine': parts[1],
            'cell_type': parts[2],
            'n_cells': int(group_counts[g_idx])
        })

    expr_df = pd.DataFrame(aggregated, index=gene_names)
    meta_df = pd.DataFrame(meta_rows).set_index('column')
    meta_df = meta_df.loc[expr_df.columns]

    log(f"  Aggregated expression: {expr_df.shape}")
    log(f"  Total cells: {meta_df['n_cells'].sum():,}")
    log(f"  Groups passing min_cells filter: {len(meta_df)}")

    return expr_df, meta_df


def normalize_and_transform(expr_df: pd.DataFrame) -> pd.DataFrame:
    """TPM normalize and log1p transform expression data."""
    # TPM normalize
    col_sums = expr_df.sum(axis=0)
    col_sums = col_sums.replace(0, 1)  # Avoid division by zero
    expr_tpm = expr_df / col_sums * 1e6

    # log1p transform (natural log)
    expr_log = np.log1p(expr_tpm)

    return expr_log


def subtract_pbs_control(
    expr_log: pd.DataFrame,
    meta_df: pd.DataFrame,
    control_condition: str = 'PBS'
) -> pd.DataFrame:
    """
    Subtract PBS control per donor per cell_type.

    For each donor x cell_type combination, subtract the PBS control profile
    from each cytokine-treated profile.

    Args:
        expr_log: Log-transformed expression (genes x pseudobulk groups)
        meta_df: Metadata with donor, cytokine, cell_type columns
        control_condition: Name of the control condition

    Returns:
        expr_diff: Control-subtracted expression (genes x pseudobulk groups)
    """
    log(f"Subtracting {control_condition} control per donor per cell_type...")

    expr_diff = expr_log.copy()
    n_subtracted = 0
    n_missing_control = 0

    # Group by donor x cell_type
    for (donor, celltype), group in meta_df.groupby(['donor', 'cell_type']):
        # Find the control column for this donor x cell_type
        control_mask = group['cytokine'] == control_condition
        control_cols = group.index[control_mask].tolist()

        if len(control_cols) == 0:
            n_missing_control += 1
            # Drop treated columns that lack a control
            treated_cols = group.index[~control_mask].tolist()
            for col in treated_cols:
                if col in expr_diff.columns:
                    expr_diff = expr_diff.drop(columns=[col])
            continue

        # Get control profile (average if multiple)
        control_profile = expr_log[control_cols].mean(axis=1)

        # Subtract control from each treated column in this group
        treated_cols = group.index[~control_mask].tolist()
        for col in treated_cols:
            if col in expr_diff.columns:
                expr_diff[col] = expr_diff[col] - control_profile
                n_subtracted += 1

        # Also subtract control from itself (should be ~0)
        for col in control_cols:
            if col in expr_diff.columns:
                expr_diff[col] = expr_diff[col] - control_profile

    log(f"  Subtracted control from {n_subtracted} treated profiles")
    log(f"  Missing control for {n_missing_control} donor-celltype groups")
    log(f"  Remaining columns: {expr_diff.shape[1]}")

    return expr_diff


def run_activity_inference(
    expr_df: pd.DataFrame,
    signature: pd.DataFrame,
    sig_name: str,
    output_prefix: str = None
) -> Dict[str, pd.DataFrame]:
    """
    Run SecActpy activity inference.

    Args:
        expr_df: Differential expression (genes x samples), control-subtracted
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


def save_activity_to_h5ad(
    result: Dict[str, pd.DataFrame],
    meta_df: pd.DataFrame,
    output_path: Path,
    sig_name: str
):
    """Save activity results to h5ad format."""
    log(f"Saving {sig_name} results to {output_path}...")

    # Filter to columns present in both results and metadata
    result_cols = set(result['zscore'].columns)
    meta_cols = set(meta_df.index)
    valid_cols = [c for c in result['zscore'].columns if c in meta_cols]

    if len(valid_cols) < len(result_cols):
        log(f"  Note: Filtered {len(result_cols) - len(valid_cols)} columns not in metadata")

    zscore_filtered = result['zscore'][valid_cols]
    obs_df = meta_df.loc[valid_cols].copy()

    # Create AnnData (samples x features)
    adata = ad.AnnData(
        X=zscore_filtered.T.values,
        obs=obs_df,
        var=pd.DataFrame(index=zscore_filtered.index)
    )

    # Store other matrices in layers
    adata.layers['beta'] = result['beta'][valid_cols].T.values
    adata.layers['se'] = result['se'][valid_cols].T.values
    adata.layers['pvalue'] = result['pvalue'][valid_cols].T.values

    # Store metadata
    adata.uns['signature'] = sig_name
    adata.uns['n_rand'] = N_RAND
    adata.uns['seed'] = SEED
    adata.uns['control_condition'] = CONTROL_CONDITION

    adata.write_h5ad(output_path, compression='gzip')
    log(f"  Saved: {output_path}")


def compute_treatment_vs_control(
    activity_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    control_condition: str = 'PBS'
) -> pd.DataFrame:
    """
    Compute treatment vs control differential (treated - PBS) per signature.

    Uses simple activity difference (not log2FC) since activities are z-scores.

    Args:
        activity_df: Activity z-scores (signatures x pseudobulk groups)
        meta_df: Metadata with donor, cytokine, cell_type columns
        control_condition: Name of the control condition

    Returns:
        DataFrame with treatment effect statistics per cytokine x signature x cell_type
    """
    log("Computing treatment vs control differential...")

    results = []

    # For each cell_type
    celltypes = meta_df['cell_type'].unique()
    cytokines = meta_df[meta_df['cytokine'] != control_condition]['cytokine'].unique()

    log(f"  Cell types: {len(celltypes)}")
    log(f"  Cytokines: {len(cytokines)}")

    for celltype in celltypes:
        celltype_meta = meta_df[meta_df['cell_type'] == celltype]

        # Get control columns for this cell type
        control_cols = celltype_meta[celltype_meta['cytokine'] == control_condition].index.tolist()
        control_cols = [c for c in control_cols if c in activity_df.columns]

        if len(control_cols) == 0:
            continue

        for cytokine in cytokines:
            # Get treated columns for this cell type and cytokine
            treated_mask = (celltype_meta['cytokine'] == cytokine)
            treated_cols = celltype_meta[treated_mask].index.tolist()
            treated_cols = [c for c in treated_cols if c in activity_df.columns]

            if len(treated_cols) == 0:
                continue

            # Compute differential per signature
            for signature in activity_df.index:
                treated_vals = activity_df.loc[signature, treated_cols].values
                control_vals = activity_df.loc[signature, control_cols].values

                # Remove NaN
                treated_vals = treated_vals[~np.isnan(treated_vals)]
                control_vals = control_vals[~np.isnan(control_vals)]

                if len(treated_vals) < 2 or len(control_vals) < 2:
                    continue

                # Activity difference (simple subtraction, not log2FC)
                activity_diff = np.mean(treated_vals) - np.mean(control_vals)

                # Wilcoxon rank-sum test
                try:
                    stat, pval = stats.mannwhitneyu(
                        treated_vals, control_vals, alternative='two-sided'
                    )
                except Exception:
                    stat, pval = np.nan, np.nan

                results.append({
                    'cytokine': cytokine,
                    'signature': signature,
                    'cell_type': celltype,
                    'mean_treated': np.mean(treated_vals),
                    'mean_control': np.mean(control_vals),
                    'activity_diff': activity_diff,
                    'n_treated': len(treated_vals),
                    'n_control': len(control_vals),
                    'statistic': stat,
                    'pvalue': pval
                })

    result_df = pd.DataFrame(results)

    # FDR correction
    if len(result_df) > 0:
        try:
            from statsmodels.stats.multitest import multipletests
            valid_mask = result_df['pvalue'].notna()
            if valid_mask.sum() > 0:
                _, pvals_corrected, _, _ = multipletests(
                    result_df.loc[valid_mask, 'pvalue'].values, method='fdr_bh'
                )
                result_df.loc[valid_mask, 'qvalue'] = pvals_corrected
        except ImportError:
            result_df['qvalue'] = result_df['pvalue']

    log(f"  Computed {len(result_df)} treatment-signature-celltype comparisons")

    return result_df


def build_cytokine_response_matrix(
    treatment_df: pd.DataFrame,
    sig_type: str = 'CytoSig'
) -> pd.DataFrame:
    """
    Build a cytokine x signature response matrix showing mean activity difference
    across donors and cell types.

    Args:
        treatment_df: Output from compute_treatment_vs_control
        sig_type: Signature type label for output

    Returns:
        DataFrame: cytokine (rows) x signature (columns) with mean activity_diff
    """
    log(f"Building cytokine response matrix ({sig_type})...")

    if len(treatment_df) == 0:
        return pd.DataFrame()

    # Pivot: cytokine x signature, averaging across cell types and donors
    response_matrix = treatment_df.pivot_table(
        index='cytokine',
        columns='signature',
        values='activity_diff',
        aggfunc='mean'
    )

    log(f"  Response matrix: {response_matrix.shape}")

    return response_matrix


# ==============================================================================
# Main Analysis Pipeline
# ==============================================================================

def run_pseudobulk_analysis():
    """Run pseudo-bulk level analysis (aggregated by donor x cytokine x cell_type)."""
    log("=" * 60)
    log("PSEUDO-BULK ANALYSIS")
    log("=" * 60)

    # Load metadata
    meta = load_metadata()

    # Detect column names from metadata
    meta_cols = list(meta.columns)
    log(f"  Available metadata columns: {meta_cols}")

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
    log(f"  Obs columns: {list(adata.obs.columns)}")

    # Aggregate by donor x cytokine x cell_type
    expr_df, agg_meta = aggregate_pseudobulk(
        adata, DONOR_COL, CYTOKINE_COL, CELL_TYPE_COL,
        min_cells=10, chunk_size=CHUNK_SIZE
    )

    # Normalize and transform
    log("Normalizing (TPM + log1p)...")
    expr_log = normalize_and_transform(expr_df)

    # Subtract PBS control per donor per cell_type
    expr_diff = subtract_pbs_control(expr_log, agg_meta, CONTROL_CONDITION)

    # Update metadata to match remaining columns
    agg_meta = agg_meta.loc[agg_meta.index.isin(expr_diff.columns)]

    # Free memory
    del adata, expr_df, expr_log
    gc.collect()

    # Run CytoSig activity inference
    cytosig_result = run_activity_inference(
        expr_diff, cytosig, 'CytoSig',
        str(OUTPUT_DIR / 'parse10m_CytoSig_pseudobulk')
    )

    # Run SecAct activity inference
    secact_result = run_activity_inference(
        expr_diff, secact, 'SecAct',
        str(OUTPUT_DIR / 'parse10m_SecAct_pseudobulk')
    )

    # Save activity results as h5ad
    save_activity_to_h5ad(
        cytosig_result, agg_meta,
        OUTPUT_DIR / 'parse10m_pseudobulk_activity_cytosig.h5ad', 'CytoSig'
    )
    save_activity_to_h5ad(
        secact_result, agg_meta,
        OUTPUT_DIR / 'parse10m_pseudobulk_activity_secact.h5ad', 'SecAct'
    )

    # Also save combined activity h5ad
    log("Saving combined pseudobulk activity h5ad...")
    combined_zscore = pd.concat([cytosig_result['zscore'], secact_result['zscore']])
    combined_result = {
        'zscore': combined_zscore,
        'beta': pd.concat([cytosig_result['beta'], secact_result['beta']]),
        'se': pd.concat([cytosig_result['se'], secact_result['se']]),
        'pvalue': pd.concat([cytosig_result['pvalue'], secact_result['pvalue']]),
    }
    save_activity_to_h5ad(
        combined_result, agg_meta,
        OUTPUT_DIR / 'parse10m_pseudobulk_activity.h5ad', 'CytoSig+SecAct'
    )

    del combined_result, combined_zscore
    gc.collect()

    # Treatment vs control differential
    log("\n" + "=" * 60)
    log("TREATMENT VS CONTROL ANALYSIS")
    log("=" * 60)

    # CytoSig treatment effects
    treat_cytosig = compute_treatment_vs_control(
        cytosig_result['zscore'], agg_meta, CONTROL_CONDITION
    )
    treat_cytosig['signature_type'] = 'CytoSig'

    # SecAct treatment effects
    treat_secact = compute_treatment_vs_control(
        secact_result['zscore'], agg_meta, CONTROL_CONDITION
    )
    treat_secact['signature_type'] = 'SecAct'

    # Combine and save
    treat_all = pd.concat([treat_cytosig, treat_secact])
    treat_all.to_csv(OUTPUT_DIR / 'parse10m_treatment_vs_control.csv', index=False)
    log(f"  Saved treatment vs control: {len(treat_all)} rows")

    # Build cytokine response matrices
    log("\n" + "=" * 60)
    log("CYTOKINE RESPONSE MATRICES")
    log("=" * 60)

    response_cytosig = build_cytokine_response_matrix(treat_cytosig, 'CytoSig')
    if len(response_cytosig) > 0:
        response_cytosig.to_csv(OUTPUT_DIR / 'parse10m_cytokine_response_matrix_cytosig.csv')
        log(f"  CytoSig response matrix: {response_cytosig.shape}")

    response_secact = build_cytokine_response_matrix(treat_secact, 'SecAct')
    if len(response_secact) > 0:
        response_secact.to_csv(OUTPUT_DIR / 'parse10m_cytokine_response_matrix_secact.csv')
        log(f"  SecAct response matrix: {response_secact.shape}")

    # Combined response matrix
    response_combined = pd.concat([response_cytosig, response_secact], axis=1)
    if len(response_combined) > 0:
        response_combined.to_csv(OUTPUT_DIR / 'parse10m_cytokine_response_matrix.csv')
        log(f"  Combined response matrix: {response_combined.shape}")

    # Summary statistics
    log("\n" + "=" * 60)
    log("SUMMARY")
    log("=" * 60)

    if len(treat_all) > 0:
        sig_hits = treat_all[treat_all.get('qvalue', treat_all['pvalue']) < 0.05]
        log(f"  Total comparisons: {len(treat_all)}")
        log(f"  Significant (FDR < 0.05): {len(sig_hits)}")
        log(f"  Cytokines with significant effects: {sig_hits['cytokine'].nunique()}")
        log(f"  Cell types with significant effects: {sig_hits['cell_type'].nunique()}")

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

    # Process signatures
    for sig, sig_name in [(cytosig, 'CytoSig'), (secact, 'SecAct')]:
        log(f"\nProcessing {sig_name}...")

        output_path = OUTPUT_DIR / f'parse10m_{sig_name}_singlecell.h5ad'

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
# Entry Point
# ==============================================================================

def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='parse_10M Cytokine Perturbation Activity Analysis'
    )
    parser.add_argument('--mode', choices=['pseudobulk', 'singlecell', 'both'],
                       default='pseudobulk', help='Analysis mode')
    parser.add_argument('--test', action='store_true',
                       help='Run on subset for testing')
    parser.add_argument('--backend', choices=['numpy', 'cupy', 'auto'],
                       default='auto', help='Computation backend')
    args = parser.parse_args()

    # Set backend
    global BACKEND
    if args.backend == 'auto':
        BACKEND = 'cupy' if CUPY_AVAILABLE else 'numpy'
    else:
        BACKEND = args.backend

    log("=" * 60)
    log("parse_10M CYTOKINE PERTURBATION ACTIVITY ANALYSIS")
    log("=" * 60)
    log(f"Mode: {args.mode}")
    log(f"Backend: {BACKEND}")
    log(f"Output: {OUTPUT_DIR}")
    log(f"Data: {H5AD_PATH}")

    if args.test:
        log("TEST MODE: Processing subset only")
        global BATCH_SIZE, N_RAND, CHUNK_SIZE
        BATCH_SIZE = 1000
        N_RAND = 100
        CHUNK_SIZE = 10000

    # Verify data exists
    if not H5AD_PATH.exists():
        log(f"ERROR: H5AD file not found: {H5AD_PATH}")
        sys.exit(1)
    if not METADATA_PATH.exists():
        log(f"WARNING: Metadata file not found: {METADATA_PATH}")

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
