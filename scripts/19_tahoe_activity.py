#!/usr/bin/env python3
"""
Tahoe Drug-Response Activity Analysis
=======================================
Compute cytokine (CytoSig, 44 signatures) and secreted protein (SecAct, 1,249 signatures)
activities from the Tahoe 100M drug-response dataset and quantify drug effects on cytokine
signaling pathways.

Datasets:
- 14 plate H5AD files: plate{1-14}_filt_Vevo_Tahoe100M_WServicesFrom_ParseGigalab.h5ad
- Location: /data/Jiang_Lab/Data/Seongyong/tahoe/
- Each plate: 4.7-35 GB, ~50 cell lines x 95 drugs
- Control condition: DMSO_TF
- Plate 13: dose-response (3 doses x 25 drugs)

Analysis:
1. Process plate-by-plate (backed='r' for memory efficiency)
2. Pseudobulk aggregation: plate x drug x cell_line
3. TPM normalize -> log1p -> subtract DMSO_TF control per cell_line per plate
4. Ridge regression activity inference (CytoSig + SecAct)
5. Drug vs control differential (treated - DMSO_TF)
6. Drug sensitivity matrix across cell lines
7. Dose-response analysis (plate 13)
8. Cytokine pathway activation summary

Output files:
- tahoe_pseudobulk_activity.h5ad
- tahoe_drug_vs_control.csv
- tahoe_drug_sensitivity_matrix.csv
- tahoe_dose_response.csv
- tahoe_cytokine_pathway_activation.csv
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
DATA_DIR = Path('/data/Jiang_Lab/Data/Seongyong/tahoe')
PLATE_TEMPLATE = 'plate{}_filt_Vevo_Tahoe100M_WServicesFrom_ParseGigalab.h5ad'
N_PLATES = 14
DOSE_RESPONSE_PLATE = 13  # Plate 13 has dose-response data

# Output paths
OUTPUT_DIR = Path('/data/parks34/projects/2cytoatlas/results/tahoe')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Analysis parameters
DRUG_COL = 'drug'
CELL_LINE_COL = 'cell_line'
PLATE_COL = 'plate'
DOSE_COL = 'dose'
CONTROL_CONDITION = 'DMSO_TF'
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


def get_plate_path(plate_num: int) -> Path:
    """Get the H5AD path for a given plate number."""
    return DATA_DIR / PLATE_TEMPLATE.format(plate_num)


def get_available_plates() -> List[int]:
    """Return list of plate numbers with available H5AD files."""
    available = []
    for i in range(1, N_PLATES + 1):
        if get_plate_path(i).exists():
            available.append(i)
    return available


def aggregate_plate_pseudobulk(
    adata,
    drug_col: str,
    cellline_col: str,
    plate_num: int,
    min_cells: int = 10,
    chunk_size: int = 50000
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aggregate expression by drug x cell_line for a single plate.

    Uses chunked reading for efficiency with backed mode.

    Args:
        adata: AnnData with raw counts (can be backed)
        drug_col: Column for drug condition
        cellline_col: Column for cell line
        plate_num: Plate number (for metadata)
        min_cells: Minimum cells per group
        chunk_size: Number of cells to read at a time

    Returns:
        expr_df: DataFrame (genes x pseudobulk groups)
        meta_df: DataFrame with drug, cell_line, plate info for each column
    """
    grouping_cols = [drug_col, cellline_col]
    available_cols = [c for c in grouping_cols if c in adata.obs.columns]

    if len(available_cols) < 2:
        log(f"  WARNING: Missing columns. Available: {list(adata.obs.columns)}")
        log(f"  Expected: {grouping_cols}")
        # Try to find alternative column names
        obs_cols = list(adata.obs.columns)
        log(f"  Obs columns: {obs_cols}")

    log(f"Aggregating plate {plate_num} by {available_cols}...")

    # Get obs data
    obs = adata.obs[available_cols].copy()
    obs = obs.reset_index(drop=True)

    # Create group labels
    if len(available_cols) == 2:
        obs['_group'] = (
            obs[available_cols[0]].astype(str) + '|' +
            obs[available_cols[1]].astype(str)
        )
    else:
        obs['_group'] = obs[available_cols[0]].astype(str)

    unique_groups = obs['_group'].unique()
    group_to_idx = {g: i for i, g in enumerate(unique_groups)}
    obs['_group_idx'] = obs['_group'].map(group_to_idx)

    log(f"  Found {len(unique_groups)} drug-cellline combinations")

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

    # Process in chunks
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

        # Accumulate sums
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

        # Add plate prefix to make column names unique across plates
        col_name = f"p{plate_num}|{group_name}"
        aggregated[col_name] = group_sums[g_idx, :]

        # Parse group name
        parts = group_name.split('|')
        meta_entry = {
            'column': col_name,
            'plate': plate_num,
            'n_cells': int(group_counts[g_idx])
        }
        if len(parts) >= 2:
            meta_entry['drug'] = parts[0]
            meta_entry['cell_line'] = parts[1]
        elif len(parts) == 1:
            meta_entry['drug'] = parts[0]
            meta_entry['cell_line'] = 'unknown'

        meta_rows.append(meta_entry)

    expr_df = pd.DataFrame(aggregated, index=gene_names)
    meta_df = pd.DataFrame(meta_rows).set_index('column')
    if len(meta_df) > 0:
        meta_df = meta_df.loc[expr_df.columns]

    log(f"  Aggregated expression: {expr_df.shape}")
    log(f"  Total cells: {meta_df['n_cells'].sum():,}" if len(meta_df) > 0 else "  No groups passed filter")

    return expr_df, meta_df


def normalize_and_transform(expr_df: pd.DataFrame) -> pd.DataFrame:
    """TPM normalize and log1p transform expression data."""
    col_sums = expr_df.sum(axis=0)
    col_sums = col_sums.replace(0, 1)  # Avoid division by zero
    expr_tpm = expr_df / col_sums * 1e6

    # log1p transform (natural log)
    expr_log = np.log1p(expr_tpm)

    return expr_log


def subtract_dmso_control(
    expr_log: pd.DataFrame,
    meta_df: pd.DataFrame,
    control_condition: str = 'DMSO_TF'
) -> pd.DataFrame:
    """
    Subtract DMSO_TF control per cell_line per plate.

    For each cell_line within a plate, subtract the DMSO_TF control profile
    from each drug-treated profile.

    Args:
        expr_log: Log-transformed expression (genes x pseudobulk groups)
        meta_df: Metadata with drug, cell_line, plate columns
        control_condition: Name of the control condition

    Returns:
        expr_diff: Control-subtracted expression (genes x pseudobulk groups)
    """
    log(f"Subtracting {control_condition} control per cell_line per plate...")

    expr_diff = expr_log.copy()
    n_subtracted = 0
    n_missing_control = 0

    # Group by plate x cell_line
    for (plate, cellline), group in meta_df.groupby(['plate', 'cell_line']):
        # Find control column
        control_mask = group['drug'] == control_condition
        control_cols = group.index[control_mask].tolist()
        control_cols = [c for c in control_cols if c in expr_diff.columns]

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

        # Subtract control from each treated column
        treated_cols = group.index[~control_mask].tolist()
        for col in treated_cols:
            if col in expr_diff.columns:
                expr_diff[col] = expr_diff[col] - control_profile
                n_subtracted += 1

        # Also subtract control from itself
        for col in control_cols:
            if col in expr_diff.columns:
                expr_diff[col] = expr_diff[col] - control_profile

    log(f"  Subtracted control from {n_subtracted} treated profiles")
    log(f"  Missing control for {n_missing_control} plate-cellline groups")
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
    # Avoid full copy: slice to common genes directly, then uppercase in-place
    expr_aligned = expr_df.loc[expr_df.index.str.upper().isin(common_genes)].copy()
    expr_aligned.index = expr_aligned.index.str.upper()
    expr_aligned = expr_aligned[~expr_aligned.index.duplicated(keep='first')]

    sig_aligned = signature.copy()
    sig_aligned.index = sig_aligned.index.str.upper()
    sig_aligned = sig_aligned[~sig_aligned.index.duplicated(keep='first')]

    # Re-compute common genes after dedup
    common_genes = list(set(expr_aligned.index) & set(sig_aligned.index))
    expr_aligned = expr_aligned.loc[common_genes]
    sig_aligned = sig_aligned.loc[common_genes]

    # Z-score normalize columns (in-place to reduce memory)
    expr_mean = expr_aligned.mean()
    expr_std = expr_aligned.std(ddof=1)
    expr_std = expr_std.replace(0, 1)
    expr_scaled = (expr_aligned - expr_mean) / expr_std
    del expr_aligned, expr_mean, expr_std
    expr_scaled = expr_scaled.fillna(0)

    sig_scaled = (sig_aligned - sig_aligned.mean()) / sig_aligned.std(ddof=1)
    sig_scaled = sig_scaled.fillna(0)
    del sig_aligned
    gc.collect()

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


def compute_drug_vs_control(
    activity_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    control_condition: str = 'DMSO_TF'
) -> pd.DataFrame:
    """
    Compute drug vs control differential (treated - DMSO_TF) per signature.

    Uses simple activity difference (not log2FC) since activities are z-scores.

    Args:
        activity_df: Activity z-scores (signatures x pseudobulk groups)
        meta_df: Metadata with drug, cell_line, plate columns
        control_condition: Name of the control condition

    Returns:
        DataFrame with drug effect statistics per drug x signature x cell_line
    """
    log("Computing drug vs control differential...")

    results = []

    # For each cell_line
    celllines = meta_df['cell_line'].unique()
    drugs = meta_df[meta_df['drug'] != control_condition]['drug'].unique()

    log(f"  Cell lines: {len(celllines)}")
    log(f"  Drugs: {len(drugs)}")

    for cellline in celllines:
        cellline_meta = meta_df[meta_df['cell_line'] == cellline]

        # Get control columns for this cell line (across plates)
        control_cols = cellline_meta[cellline_meta['drug'] == control_condition].index.tolist()
        control_cols = [c for c in control_cols if c in activity_df.columns]

        if len(control_cols) == 0:
            continue

        for drug in drugs:
            # Get treated columns for this cell line and drug
            treated_mask = (cellline_meta['drug'] == drug)
            treated_cols = cellline_meta[treated_mask].index.tolist()
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

                if len(treated_vals) < 1 or len(control_vals) < 1:
                    continue

                # Activity difference (simple subtraction, not log2FC)
                activity_diff = np.mean(treated_vals) - np.mean(control_vals)

                # Statistical test (if enough replicates)
                pval = np.nan
                stat = np.nan
                if len(treated_vals) >= 2 and len(control_vals) >= 2:
                    try:
                        stat, pval = stats.mannwhitneyu(
                            treated_vals, control_vals, alternative='two-sided'
                        )
                    except Exception:
                        pass

                results.append({
                    'drug': drug,
                    'signature': signature,
                    'cell_line': cellline,
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

    log(f"  Computed {len(result_df)} drug-signature-cellline comparisons")

    return result_df


def build_drug_sensitivity_matrix(
    drug_diff_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Build drug x signature sensitivity matrix averaged across cell lines.

    Args:
        drug_diff_df: Output from compute_drug_vs_control

    Returns:
        DataFrame: drug (rows) x signature (columns) with mean activity_diff
    """
    log("Building drug sensitivity matrix...")

    if len(drug_diff_df) == 0:
        return pd.DataFrame()

    # Pivot: drug x signature, averaging across cell lines
    sensitivity_matrix = drug_diff_df.pivot_table(
        index='drug',
        columns='signature',
        values='activity_diff',
        aggfunc='mean'
    )

    log(f"  Drug sensitivity matrix: {sensitivity_matrix.shape}")

    return sensitivity_matrix


def analyze_dose_response(
    activity_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    control_condition: str = 'DMSO_TF'
) -> pd.DataFrame:
    """
    Analyze dose-response relationships from plate 13.

    Plate 13 contains 3 doses x 25 drugs. Compute activity across doses
    to identify dose-dependent cytokine pathway modulation.

    Args:
        activity_df: Activity z-scores (signatures x pseudobulk groups)
        meta_df: Metadata with drug, cell_line, dose, plate columns
        control_condition: Name of the control condition

    Returns:
        DataFrame with dose-response statistics per drug x signature x cell_line
    """
    log("Analyzing dose-response relationships (plate 13)...")

    # Filter to plate 13 data
    plate13_meta = meta_df[meta_df['plate'] == DOSE_RESPONSE_PLATE]

    if len(plate13_meta) == 0:
        log("  WARNING: No plate 13 data found")
        return pd.DataFrame()

    # Check for dose column
    if DOSE_COL not in plate13_meta.columns:
        log(f"  WARNING: Dose column '{DOSE_COL}' not found in metadata")
        log(f"  Available columns: {list(plate13_meta.columns)}")
        # Try to extract dose from drug name (e.g., "DrugA_1uM")
        log("  Attempting to extract dose information from drug names...")
        return pd.DataFrame()

    doses = sorted(plate13_meta[DOSE_COL].dropna().unique())
    drugs = plate13_meta[plate13_meta['drug'] != control_condition]['drug'].unique()

    log(f"  Doses: {doses}")
    log(f"  Drugs: {len(drugs)}")

    results = []

    for drug in drugs:
        drug_meta = plate13_meta[plate13_meta['drug'] == drug]

        for cellline in drug_meta['cell_line'].unique():
            cellline_drug_meta = drug_meta[drug_meta['cell_line'] == cellline]

            # Get control columns for this cell line in plate 13
            control_meta = plate13_meta[
                (plate13_meta['drug'] == control_condition) &
                (plate13_meta['cell_line'] == cellline)
            ]
            control_cols = [c for c in control_meta.index if c in activity_df.columns]

            if len(control_cols) == 0:
                continue

            for dose in doses:
                dose_meta = cellline_drug_meta[cellline_drug_meta[DOSE_COL] == dose]
                dose_cols = [c for c in dose_meta.index if c in activity_df.columns]

                if len(dose_cols) == 0:
                    continue

                for signature in activity_df.index:
                    treated_vals = activity_df.loc[signature, dose_cols].values
                    control_vals = activity_df.loc[signature, control_cols].values

                    treated_vals = treated_vals[~np.isnan(treated_vals)]
                    control_vals = control_vals[~np.isnan(control_vals)]

                    if len(treated_vals) < 1 or len(control_vals) < 1:
                        continue

                    activity_diff = np.mean(treated_vals) - np.mean(control_vals)

                    results.append({
                        'drug': drug,
                        'dose': dose,
                        'signature': signature,
                        'cell_line': cellline,
                        'mean_treated': np.mean(treated_vals),
                        'mean_control': np.mean(control_vals),
                        'activity_diff': activity_diff,
                        'n_treated': len(treated_vals),
                        'n_control': len(control_vals)
                    })

    result_df = pd.DataFrame(results)

    # Compute dose-response correlation (Spearman) per drug x signature x cell_line
    if len(result_df) > 0 and len(doses) >= 3:
        log("  Computing dose-response correlations...")
        dose_corr_rows = []

        for (drug, sig, cellline), group in result_df.groupby(['drug', 'signature', 'cell_line']):
            if len(group) < 3:
                continue

            try:
                rho, pval = stats.spearmanr(group['dose'].values, group['activity_diff'].values)
                dose_corr_rows.append({
                    'drug': drug,
                    'signature': sig,
                    'cell_line': cellline,
                    'dose_response_rho': rho,
                    'dose_response_pval': pval,
                    'n_doses': len(group)
                })
            except Exception:
                continue

        if dose_corr_rows:
            dose_corr_df = pd.DataFrame(dose_corr_rows)
            result_df = result_df.merge(
                dose_corr_df, on=['drug', 'signature', 'cell_line'], how='left'
            )

    log(f"  Computed {len(result_df)} dose-response records")

    return result_df


def compute_cytokine_pathway_activation(
    drug_diff_df: pd.DataFrame,
    sig_type: str = 'CytoSig'
) -> pd.DataFrame:
    """
    Summarize cytokine pathway activation patterns across drugs.

    Identifies which drugs activate/suppress specific cytokine pathways
    and computes pathway-level summary statistics.

    Args:
        drug_diff_df: Output from compute_drug_vs_control
        sig_type: Signature type label

    Returns:
        DataFrame with pathway activation summary per drug
    """
    log(f"Computing cytokine pathway activation summary ({sig_type})...")

    if len(drug_diff_df) == 0:
        return pd.DataFrame()

    results = []

    for drug in drug_diff_df['drug'].unique():
        drug_data = drug_diff_df[drug_diff_df['drug'] == drug]

        # Average across cell lines per signature
        sig_summary = drug_data.groupby('signature').agg({
            'activity_diff': ['mean', 'std', 'count'],
            'pvalue': 'min'
        }).reset_index()
        sig_summary.columns = ['signature', 'mean_activity_diff', 'std_activity_diff',
                                'n_celllines', 'min_pvalue']

        # Count significantly activated and suppressed signatures
        qval_col = 'qvalue' if 'qvalue' in drug_data.columns else 'pvalue'
        sig_activated = drug_data[
            (drug_data[qval_col] < 0.05) & (drug_data['activity_diff'] > 0)
        ]['signature'].nunique()
        sig_suppressed = drug_data[
            (drug_data[qval_col] < 0.05) & (drug_data['activity_diff'] < 0)
        ]['signature'].nunique()

        # Top activated signatures
        top_activated = sig_summary.nlargest(5, 'mean_activity_diff')
        # Top suppressed signatures
        top_suppressed = sig_summary.nsmallest(5, 'mean_activity_diff')

        results.append({
            'drug': drug,
            'signature_type': sig_type,
            'n_signatures_tested': len(sig_summary),
            'n_activated': sig_activated,
            'n_suppressed': sig_suppressed,
            'mean_abs_activity_diff': sig_summary['mean_activity_diff'].abs().mean(),
            'top_activated': ';'.join(top_activated['signature'].tolist()),
            'top_activated_diff': ';'.join(
                [f"{v:.3f}" for v in top_activated['mean_activity_diff'].values]
            ),
            'top_suppressed': ';'.join(top_suppressed['signature'].tolist()),
            'top_suppressed_diff': ';'.join(
                [f"{v:.3f}" for v in top_suppressed['mean_activity_diff'].values]
            ),
            'n_celllines': drug_data['cell_line'].nunique()
        })

    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values('mean_abs_activity_diff', ascending=False)

    log(f"  Computed pathway activation for {len(result_df)} drugs")

    return result_df


# ==============================================================================
# Main Analysis Pipeline
# ==============================================================================

def process_single_plate(
    plate_num: int,
    cytosig: pd.DataFrame,
    secact: pd.DataFrame
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Process a single plate: aggregate, normalize, compute activities.

    Args:
        plate_num: Plate number (1-14)
        cytosig: CytoSig signature matrix
        secact: SecAct signature matrix

    Returns:
        Tuple of (expr_diff, agg_meta, plate activity results) or (None, None, None) on failure
    """
    plate_path = get_plate_path(plate_num)

    if not plate_path.exists():
        log(f"  Plate {plate_num}: file not found, skipping")
        return None, None, None

    log(f"\n{'='*60}")
    log(f"PROCESSING PLATE {plate_num}")
    log(f"{'='*60}")
    log(f"  Path: {plate_path}")

    # Load plate in backed mode
    adata = ad.read_h5ad(plate_path, backed='r')
    log(f"  Shape: {adata.shape}")
    log(f"  Obs columns: {list(adata.obs.columns)}")

    # Aggregate by drug x cell_line
    expr_df, agg_meta = aggregate_plate_pseudobulk(
        adata, DRUG_COL, CELL_LINE_COL, plate_num,
        min_cells=10, chunk_size=CHUNK_SIZE
    )

    # Release adata
    del adata
    gc.collect()

    if expr_df.shape[1] == 0:
        log(f"  Plate {plate_num}: no groups passed filter, skipping")
        return None, None, None

    # Normalize
    log("  Normalizing (TPM + log1p)...")
    expr_log = normalize_and_transform(expr_df)

    # Subtract DMSO control
    expr_diff = subtract_dmso_control(expr_log, agg_meta, CONTROL_CONDITION)

    # Update metadata to match remaining columns
    agg_meta = agg_meta.loc[agg_meta.index.isin(expr_diff.columns)]

    del expr_df, expr_log
    gc.collect()

    return expr_diff, agg_meta, None


def run_all_plates_analysis():
    """Run pseudo-bulk analysis across all plates."""
    log("=" * 60)
    log("PSEUDO-BULK ANALYSIS - ALL PLATES")
    log("=" * 60)

    # Check available plates
    available = get_available_plates()
    log(f"Available plates: {available}")

    if len(available) == 0:
        log("ERROR: No plate files found")
        return

    # Load signatures
    log("Loading signatures...")
    cytosig = load_cytosig()
    secact = load_secact()
    log(f"  CytoSig: {cytosig.shape}")
    log(f"  SecAct: {secact.shape}")

    # Process all plates and collect expression data
    all_expr_dfs = []
    all_meta_dfs = []

    for plate_num in available:
        expr_diff, agg_meta, _ = process_single_plate(plate_num, cytosig, secact)

        if expr_diff is not None:
            all_expr_dfs.append(expr_diff)
            all_meta_dfs.append(agg_meta)

        gc.collect()

    if len(all_expr_dfs) == 0:
        log("ERROR: No plates produced valid data")
        return

    # Combine all plates
    log(f"\nCombining {len(all_expr_dfs)} plates...")

    # Find common genes across plates
    common_genes = set(all_expr_dfs[0].index)
    for df in all_expr_dfs[1:]:
        common_genes = common_genes & set(df.index)
    common_genes = sorted(common_genes)
    log(f"  Common genes across plates: {len(common_genes)}")

    # Concatenate
    combined_expr = pd.concat([df.loc[common_genes] for df in all_expr_dfs], axis=1)
    combined_meta = pd.concat(all_meta_dfs)
    log(f"  Combined expression: {combined_expr.shape}")
    log(f"  Combined metadata: {len(combined_meta)} groups")

    del all_expr_dfs, all_meta_dfs
    gc.collect()

    # Run activity inference on combined data
    log("\n" + "=" * 60)
    log("ACTIVITY INFERENCE")
    log("=" * 60)

    # CytoSig
    cytosig_result = run_activity_inference(
        combined_expr, cytosig, 'CytoSig'
    )

    # SecAct
    secact_result = run_activity_inference(
        combined_expr, secact, 'SecAct'
    )

    del combined_expr
    gc.collect()

    # Save activity results
    save_activity_to_h5ad(
        cytosig_result, combined_meta,
        OUTPUT_DIR / 'tahoe_pseudobulk_activity_cytosig.h5ad', 'CytoSig'
    )
    save_activity_to_h5ad(
        secact_result, combined_meta,
        OUTPUT_DIR / 'tahoe_pseudobulk_activity_secact.h5ad', 'SecAct'
    )

    # Combined h5ad
    combined_zscore = pd.concat([cytosig_result['zscore'], secact_result['zscore']])
    combined_result = {
        'zscore': combined_zscore,
        'beta': pd.concat([cytosig_result['beta'], secact_result['beta']]),
        'se': pd.concat([cytosig_result['se'], secact_result['se']]),
        'pvalue': pd.concat([cytosig_result['pvalue'], secact_result['pvalue']]),
    }
    save_activity_to_h5ad(
        combined_result, combined_meta,
        OUTPUT_DIR / 'tahoe_pseudobulk_activity.h5ad', 'CytoSig+SecAct'
    )

    del combined_result, combined_zscore
    gc.collect()

    # Drug vs control differential
    log("\n" + "=" * 60)
    log("DRUG VS CONTROL ANALYSIS")
    log("=" * 60)

    drug_diff_cytosig = compute_drug_vs_control(
        cytosig_result['zscore'], combined_meta, CONTROL_CONDITION
    )
    drug_diff_cytosig['signature_type'] = 'CytoSig'

    drug_diff_secact = compute_drug_vs_control(
        secact_result['zscore'], combined_meta, CONTROL_CONDITION
    )
    drug_diff_secact['signature_type'] = 'SecAct'

    drug_diff_all = pd.concat([drug_diff_cytosig, drug_diff_secact])
    drug_diff_all.to_csv(OUTPUT_DIR / 'tahoe_drug_vs_control.csv', index=False)
    log(f"  Saved drug vs control: {len(drug_diff_all)} rows")

    # Drug sensitivity matrix
    log("\n" + "=" * 60)
    log("DRUG SENSITIVITY MATRICES")
    log("=" * 60)

    sensitivity_cytosig = build_drug_sensitivity_matrix(drug_diff_cytosig)
    if len(sensitivity_cytosig) > 0:
        sensitivity_cytosig.to_csv(OUTPUT_DIR / 'tahoe_drug_sensitivity_matrix_cytosig.csv')
        log(f"  CytoSig sensitivity matrix: {sensitivity_cytosig.shape}")

    sensitivity_secact = build_drug_sensitivity_matrix(drug_diff_secact)
    if len(sensitivity_secact) > 0:
        sensitivity_secact.to_csv(OUTPUT_DIR / 'tahoe_drug_sensitivity_matrix_secact.csv')
        log(f"  SecAct sensitivity matrix: {sensitivity_secact.shape}")

    # Combined sensitivity matrix
    sensitivity_combined = pd.concat([sensitivity_cytosig, sensitivity_secact], axis=1)
    if len(sensitivity_combined) > 0:
        sensitivity_combined.to_csv(OUTPUT_DIR / 'tahoe_drug_sensitivity_matrix.csv')
        log(f"  Combined sensitivity matrix: {sensitivity_combined.shape}")

    # Dose-response analysis (plate 13)
    log("\n" + "=" * 60)
    log("DOSE-RESPONSE ANALYSIS (PLATE 13)")
    log("=" * 60)

    # Filter to plate 13 activity data
    plate13_cols = [c for c in combined_meta.index
                    if combined_meta.loc[c, 'plate'] == DOSE_RESPONSE_PLATE
                    and c in cytosig_result['zscore'].columns]

    if len(plate13_cols) > 0:
        plate13_meta = combined_meta.loc[plate13_cols]

        dose_resp_cytosig = analyze_dose_response(
            cytosig_result['zscore'], plate13_meta, CONTROL_CONDITION
        )
        if len(dose_resp_cytosig) > 0:
            dose_resp_cytosig['signature_type'] = 'CytoSig'

        dose_resp_secact = analyze_dose_response(
            secact_result['zscore'], plate13_meta, CONTROL_CONDITION
        )
        if len(dose_resp_secact) > 0:
            dose_resp_secact['signature_type'] = 'SecAct'

        dose_resp_all = pd.concat([
            dose_resp_cytosig if len(dose_resp_cytosig) > 0 else pd.DataFrame(),
            dose_resp_secact if len(dose_resp_secact) > 0 else pd.DataFrame()
        ])

        if len(dose_resp_all) > 0:
            dose_resp_all.to_csv(OUTPUT_DIR / 'tahoe_dose_response.csv', index=False)
            log(f"  Saved dose-response: {len(dose_resp_all)} rows")
    else:
        log("  No plate 13 data available for dose-response analysis")

    # Cytokine pathway activation summary
    log("\n" + "=" * 60)
    log("CYTOKINE PATHWAY ACTIVATION SUMMARY")
    log("=" * 60)

    pathway_cytosig = compute_cytokine_pathway_activation(drug_diff_cytosig, 'CytoSig')
    pathway_secact = compute_cytokine_pathway_activation(drug_diff_secact, 'SecAct')

    pathway_all = pd.concat([pathway_cytosig, pathway_secact])
    if len(pathway_all) > 0:
        pathway_all.to_csv(OUTPUT_DIR / 'tahoe_cytokine_pathway_activation.csv', index=False)
        log(f"  Saved pathway activation: {len(pathway_all)} rows")

    # Summary statistics
    log("\n" + "=" * 60)
    log("SUMMARY")
    log("=" * 60)

    if len(drug_diff_all) > 0:
        qval_col = 'qvalue' if 'qvalue' in drug_diff_all.columns else 'pvalue'
        sig_hits = drug_diff_all[drug_diff_all[qval_col] < 0.05]
        log(f"  Total plates processed: {len(available)}")
        log(f"  Total pseudobulk groups: {len(combined_meta)}")
        log(f"  Total comparisons: {len(drug_diff_all)}")
        log(f"  Significant (FDR < 0.05): {len(sig_hits)}")
        log(f"  Drugs with significant effects: {sig_hits['drug'].nunique()}")
        log(f"  Cell lines with significant effects: {sig_hits['cell_line'].nunique()}")

    log("\nAll-plates analysis complete!")


def run_single_plate_analysis(plate_num: int):
    """Run analysis for a single plate."""
    log("=" * 60)
    log(f"SINGLE PLATE ANALYSIS - PLATE {plate_num}")
    log("=" * 60)

    plate_path = get_plate_path(plate_num)
    if not plate_path.exists():
        log(f"ERROR: Plate {plate_num} file not found: {plate_path}")
        return

    # Load signatures
    log("Loading signatures...")
    cytosig = load_cytosig()
    secact = load_secact()
    log(f"  CytoSig: {cytosig.shape}")
    log(f"  SecAct: {secact.shape}")

    # Process plate
    expr_diff, agg_meta, _ = process_single_plate(plate_num, cytosig, secact)

    if expr_diff is None:
        log("ERROR: Plate processing failed")
        return

    # Run activity inference
    cytosig_result = run_activity_inference(expr_diff, cytosig, 'CytoSig')
    secact_result = run_activity_inference(expr_diff, secact, 'SecAct')

    del expr_diff
    gc.collect()

    # Save plate-specific results
    plate_prefix = f'tahoe_plate{plate_num}'

    save_activity_to_h5ad(
        cytosig_result, agg_meta,
        OUTPUT_DIR / f'{plate_prefix}_activity_cytosig.h5ad', 'CytoSig'
    )
    save_activity_to_h5ad(
        secact_result, agg_meta,
        OUTPUT_DIR / f'{plate_prefix}_activity_secact.h5ad', 'SecAct'
    )

    # Drug vs control for this plate
    drug_diff_cytosig = compute_drug_vs_control(
        cytosig_result['zscore'], agg_meta, CONTROL_CONDITION
    )
    drug_diff_cytosig['signature_type'] = 'CytoSig'
    drug_diff_cytosig['plate'] = plate_num

    drug_diff_secact = compute_drug_vs_control(
        secact_result['zscore'], agg_meta, CONTROL_CONDITION
    )
    drug_diff_secact['signature_type'] = 'SecAct'
    drug_diff_secact['plate'] = plate_num

    drug_diff_all = pd.concat([drug_diff_cytosig, drug_diff_secact])
    drug_diff_all.to_csv(OUTPUT_DIR / f'{plate_prefix}_drug_vs_control.csv', index=False)
    log(f"  Saved plate {plate_num} drug vs control: {len(drug_diff_all)} rows")

    # Dose-response if plate 13
    if plate_num == DOSE_RESPONSE_PLATE:
        log("\nRunning dose-response analysis for plate 13...")
        dose_resp_cytosig = analyze_dose_response(
            cytosig_result['zscore'], agg_meta, CONTROL_CONDITION
        )
        dose_resp_secact = analyze_dose_response(
            secact_result['zscore'], agg_meta, CONTROL_CONDITION
        )
        dose_resp_all = pd.concat([
            dose_resp_cytosig if len(dose_resp_cytosig) > 0 else pd.DataFrame(),
            dose_resp_secact if len(dose_resp_secact) > 0 else pd.DataFrame()
        ])
        if len(dose_resp_all) > 0:
            dose_resp_all.to_csv(OUTPUT_DIR / f'{plate_prefix}_dose_response.csv', index=False)
            log(f"  Saved dose-response: {len(dose_resp_all)} rows")

    log(f"\nPlate {plate_num} analysis complete!")


# ==============================================================================
# Entry Point
# ==============================================================================

def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Tahoe Drug-Response Activity Analysis'
    )
    parser.add_argument('--mode', choices=['all', 'plate'],
                       default='all', help='Analysis mode: all plates or single plate')
    parser.add_argument('--plate', type=int, choices=range(1, N_PLATES + 1),
                       metavar='{1-14}',
                       help='Plate number (required if --mode plate)')
    parser.add_argument('--test', action='store_true',
                       help='Run on subset for testing')
    parser.add_argument('--backend', choices=['numpy', 'cupy', 'auto'],
                       default='auto', help='Computation backend')
    args = parser.parse_args()

    # Validate arguments
    if args.mode == 'plate' and args.plate is None:
        parser.error("--plate is required when --mode is 'plate'")

    # Set backend
    global BACKEND
    if args.backend == 'auto':
        BACKEND = 'cupy' if CUPY_AVAILABLE else 'numpy'
    else:
        BACKEND = args.backend

    log("=" * 60)
    log("TAHOE DRUG-RESPONSE ACTIVITY ANALYSIS")
    log("=" * 60)
    log(f"Mode: {args.mode}")
    log(f"Backend: {BACKEND}")
    log(f"Output: {OUTPUT_DIR}")
    log(f"Data directory: {DATA_DIR}")

    if args.test:
        log("TEST MODE: Processing subset only")
        global BATCH_SIZE, N_RAND, CHUNK_SIZE
        BATCH_SIZE = 1000
        N_RAND = 100
        CHUNK_SIZE = 10000

    # Check available plates
    available = get_available_plates()
    log(f"Available plate files: {available}")

    if len(available) == 0:
        log(f"ERROR: No plate H5AD files found in {DATA_DIR}")
        sys.exit(1)

    start_time = time.time()

    if args.mode == 'all':
        run_all_plates_analysis()
    elif args.mode == 'plate':
        run_single_plate_analysis(args.plate)

    total_time = time.time() - start_time
    log(f"\nTotal time: {total_time/60:.1f} minutes")
    log("Done!")


if __name__ == '__main__':
    main()
