#!/usr/bin/env python3
"""
CytoSig Ground-Truth Validation Using parse_10M
=================================================
Validates CytoSig cytokine activity inference against ground-truth cytokine
stimulation data from the parse_10M perturbation experiment.

The parse_10M dataset contains PBMC samples treated with 90 different cytokines
across 12 donors. For each of the 44 CytoSig cytokines that overlap with the
90 tested, we evaluate:

1. Self-signature test: AUC-ROC for detecting the treated cytokine from its
   own signature activity (self vs non-self).
2. Cell type specificity: Which cell types show strongest response.
3. Cross-donor consistency: Spearman correlation of activity across 12 donors.

Output:
    - parse10m_ground_truth_validation.csv   (full validation metrics)
    - parse10m_self_signature_auc.csv        (AUC-ROC per cytokine)
    - parse10m_donor_consistency.csv         (cross-donor Spearman)

Usage:
    # Full validation
    python scripts/21_parse10m_ground_truth.py

    # Specific signature type
    python scripts/21_parse10m_ground_truth.py --signature-type cytosig

    # Force overwrite
    python scripts/21_parse10m_ground_truth.py --force
"""

import os
import sys
import gc
import warnings
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import anndata as ad
from scipy import stats
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings('ignore')

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

RESULTS_DIR = Path('/data/parks34/projects/2cytoatlas/results/parse10m')
OUTPUT_DIR = Path('/data/parks34/projects/2cytoatlas/results/parse10m')

# Signature parameters
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


def safe_auc_roc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute AUC-ROC without sklearn dependency.

    Uses the Mann-Whitney U statistic: AUC = U / (n_pos * n_neg).

    Args:
        labels: Binary labels (0/1).
        scores: Continuous scores.

    Returns:
        AUC-ROC value, or NaN if computation fails.
    """
    pos = scores[labels == 1]
    neg = scores[labels == 0]

    if len(pos) == 0 or len(neg) == 0:
        return np.nan

    try:
        u_stat, _ = stats.mannwhitneyu(pos, neg, alternative='greater')
        auc = u_stat / (len(pos) * len(neg))
    except Exception:
        auc = np.nan

    return auc


def load_activity_results() -> Dict[str, ad.AnnData]:
    """Load pre-computed activity results from results/parse10m/.

    Expects H5AD files named like:
      parse10m_cytosig.h5ad, parse10m_secact.h5ad

    Returns:
        Dict mapping signature type -> AnnData with activity z-scores.
    """
    log("Loading parse10m activity results...")
    results = {}

    for sig_type in ['cytosig', 'secact']:
        # Try multiple naming conventions
        candidates = [
            RESULTS_DIR / f'parse10m_pseudobulk_activity_{sig_type}.h5ad',
            RESULTS_DIR / f'parse10m_{sig_type}.h5ad',
            RESULTS_DIR / f'parse10m_{sig_type}_zscore.h5ad',
            RESULTS_DIR / f'parse10m_activity_{sig_type}.h5ad',
            RESULTS_DIR / f'{sig_type}_activity.h5ad',
        ]

        for path in candidates:
            if path.exists():
                log(f"  Loading {sig_type}: {path.name}")
                adata = ad.read_h5ad(path)
                log(f"    Shape: {adata.shape}")
                results[sig_type] = adata
                break
        else:
            log(f"  WARNING: No {sig_type} activity file found in {RESULTS_DIR}")

    return results


def load_metadata() -> Optional[pd.DataFrame]:
    """Load parse10m metadata (treatment, donor, cell type).

    Looks for a metadata CSV or extracts from the activity H5AD obs.

    Returns:
        DataFrame with columns: treatment, donor, cell_type (at minimum).
    """
    # Try dedicated metadata file
    meta_candidates = [
        RESULTS_DIR / 'parse10m_metadata.csv',
        RESULTS_DIR / 'metadata.csv',
        RESULTS_DIR / 'parse10m_obs.csv',
    ]

    for path in meta_candidates:
        if path.exists():
            log(f"  Loading metadata: {path.name}")
            meta = pd.read_csv(path, index_col=0)
            log(f"    Shape: {meta.shape}")
            return meta

    log("  No dedicated metadata file found; will extract from activity H5AD obs")
    return None


def detect_treatment_column(obs: pd.DataFrame) -> Optional[str]:
    """Detect the column containing cytokine treatment labels."""
    for col in ['treatment', 'cytokine', 'perturbation', 'condition',
                'stim', 'stimulus', 'Treatment', 'Cytokine']:
        if col in obs.columns:
            return col
    return None


def detect_donor_column(obs: pd.DataFrame) -> Optional[str]:
    """Detect the column containing donor/subject IDs."""
    for col in ['donor', 'donor_id', 'donorID', 'subject', 'patient',
                'sample', 'Donor', 'donor_ID']:
        if col in obs.columns:
            return col
    return None


def detect_celltype_column(obs: pd.DataFrame) -> Optional[str]:
    """Detect the column containing cell type labels."""
    for col in ['cell_type', 'celltype', 'cell_type_l1', 'cell_type_l2',
                'cellType1', 'subCluster', 'cluster', 'CellType']:
        if col in obs.columns:
            return col
    return None


# ==============================================================================
# Self-Signature AUC-ROC Test
# ==============================================================================

def compute_self_signature_auc(
    activity_adata: ad.AnnData,
    treatment_col: str,
    sig_type: str,
) -> pd.DataFrame:
    """Compute AUC-ROC for self-signature detection.

    For each CytoSig cytokine present in the treatment panel, compute
    AUC-ROC: can the cytokine's own activity score distinguish treated
    from untreated samples?

    Args:
        activity_adata: AnnData with activity z-scores (obs x signatures).
        treatment_col: Column in obs with treatment labels.
        sig_type: 'cytosig' or 'secact'.

    Returns:
        DataFrame with columns: cytokine, auc_roc, n_treated, n_control,
        mean_self_activity, mean_nonself_activity, activity_diff.
    """
    log(f"  Computing self-signature AUC-ROC ({sig_type})...")

    obs = activity_adata.obs
    treatments = obs[treatment_col].unique()
    signatures = list(activity_adata.var_names)

    # Get activity matrix
    act_matrix = activity_adata.X
    if hasattr(act_matrix, 'toarray'):
        act_matrix = act_matrix.toarray()
    act_matrix = act_matrix.astype(np.float64)

    rows = []

    for sig in signatures:
        sig_idx = signatures.index(sig)

        # Find matching treatment
        # Try exact match, then case-insensitive, then partial match
        matched_treatment = None
        for t in treatments:
            t_str = str(t).strip()
            if t_str == sig:
                matched_treatment = t
                break
        if matched_treatment is None:
            for t in treatments:
                if str(t).strip().lower() == sig.lower():
                    matched_treatment = t
                    break
        if matched_treatment is None:
            for t in treatments:
                t_lower = str(t).strip().lower()
                sig_lower = sig.lower()
                if sig_lower in t_lower or t_lower in sig_lower:
                    matched_treatment = t
                    break

        if matched_treatment is None:
            continue

        # Binary labels: 1 = treated with this cytokine, 0 = other treatment
        treated_mask = obs[treatment_col] == matched_treatment
        n_treated = treated_mask.sum()
        n_control = (~treated_mask).sum()

        if n_treated < 3 or n_control < 3:
            continue

        labels = treated_mask.astype(int).values
        scores = act_matrix[:, sig_idx]

        # Remove NaN
        valid = np.isfinite(scores)
        labels = labels[valid]
        scores = scores[valid]

        if labels.sum() < 3:
            continue

        auc = safe_auc_roc(labels, scores)
        mean_self = float(np.mean(scores[labels == 1]))
        mean_nonself = float(np.mean(scores[labels == 0]))
        activity_diff = mean_self - mean_nonself

        # Wilcoxon rank-sum test
        try:
            stat, pval = stats.mannwhitneyu(
                scores[labels == 1], scores[labels == 0],
                alternative='greater'
            )
        except Exception:
            pval = np.nan

        rows.append({
            'cytokine': sig,
            'matched_treatment': str(matched_treatment),
            'signature_type': sig_type,
            'auc_roc': round(auc, 4) if np.isfinite(auc) else None,
            'n_treated': int(n_treated),
            'n_control': int(n_control),
            'mean_self_activity': round(mean_self, 4),
            'mean_nonself_activity': round(mean_nonself, 4),
            'activity_diff': round(activity_diff, 4),
            'pvalue': pval,
        })

    df = pd.DataFrame(rows)

    # FDR correction
    if len(df) > 0 and 'pvalue' in df.columns:
        valid_pvals = df['pvalue'].dropna()
        if len(valid_pvals) > 0:
            _, fdr, _, _ = multipletests(valid_pvals.values, method='fdr_bh')
            df.loc[valid_pvals.index, 'fdr'] = fdr

    log(f"    {len(df)} cytokines with matched treatments")
    if len(df) > 0:
        median_auc = df['auc_roc'].dropna().median()
        log(f"    Median AUC-ROC: {median_auc:.3f}")

    return df


# ==============================================================================
# Cell Type Specificity Analysis
# ==============================================================================

def compute_celltype_specificity(
    activity_adata: ad.AnnData,
    treatment_col: str,
    celltype_col: str,
    sig_type: str,
) -> pd.DataFrame:
    """Analyze cell-type-specific responses to cytokine treatment.

    For each cytokine-celltype pair, compute:
    - Activity difference (treated - control)
    - Wilcoxon p-value
    - Effect size

    Args:
        activity_adata: AnnData with activity z-scores.
        treatment_col: Column with treatment labels.
        celltype_col: Column with cell type labels.
        sig_type: 'cytosig' or 'secact'.

    Returns:
        DataFrame with celltype-specific response metrics.
    """
    log(f"  Computing cell-type specificity ({sig_type})...")

    obs = activity_adata.obs
    treatments = obs[treatment_col].unique()
    cell_types = obs[celltype_col].unique()
    signatures = list(activity_adata.var_names)

    act_matrix = activity_adata.X
    if hasattr(act_matrix, 'toarray'):
        act_matrix = act_matrix.toarray()
    act_matrix = act_matrix.astype(np.float64)

    rows = []

    for sig in signatures:
        sig_idx = signatures.index(sig)

        # Find matching treatment (same logic as above)
        matched_treatment = None
        for t in treatments:
            t_str = str(t).strip()
            if t_str == sig or t_str.lower() == sig.lower():
                matched_treatment = t
                break
        if matched_treatment is None:
            for t in treatments:
                if sig.lower() in str(t).lower() or str(t).lower() in sig.lower():
                    matched_treatment = t
                    break
        if matched_treatment is None:
            continue

        treated_mask = obs[treatment_col] == matched_treatment

        for ct in cell_types:
            ct_mask = obs[celltype_col] == ct
            ct_treated = treated_mask & ct_mask
            ct_control = (~treated_mask) & ct_mask

            n_treated = ct_treated.sum()
            n_control = ct_control.sum()

            if n_treated < 3 or n_control < 3:
                continue

            treated_vals = act_matrix[ct_treated.values, sig_idx]
            control_vals = act_matrix[ct_control.values, sig_idx]

            # Remove NaN
            treated_vals = treated_vals[np.isfinite(treated_vals)]
            control_vals = control_vals[np.isfinite(control_vals)]

            if len(treated_vals) < 3 or len(control_vals) < 3:
                continue

            mean_treated = float(np.mean(treated_vals))
            mean_control = float(np.mean(control_vals))
            activity_diff = mean_treated - mean_control

            try:
                stat, pval = stats.mannwhitneyu(
                    treated_vals, control_vals, alternative='two-sided'
                )
            except Exception:
                pval = np.nan

            rows.append({
                'cytokine': sig,
                'cell_type': ct,
                'signature_type': sig_type,
                'mean_treated': round(mean_treated, 4),
                'mean_control': round(mean_control, 4),
                'activity_diff': round(activity_diff, 4),
                'n_treated': int(n_treated),
                'n_control': int(n_control),
                'pvalue': pval,
            })

    df = pd.DataFrame(rows)

    # FDR correction
    if len(df) > 0 and 'pvalue' in df.columns:
        valid_pvals = df['pvalue'].dropna()
        if len(valid_pvals) > 0:
            _, fdr, _, _ = multipletests(valid_pvals.values, method='fdr_bh')
            df.loc[valid_pvals.index, 'fdr'] = fdr

    log(f"    {len(df)} cytokine-celltype pairs")
    return df


# ==============================================================================
# Cross-Donor Consistency
# ==============================================================================

def compute_donor_consistency(
    activity_adata: ad.AnnData,
    treatment_col: str,
    donor_col: str,
    sig_type: str,
) -> pd.DataFrame:
    """Compute cross-donor consistency of cytokine activity.

    For each cytokine signature, compute Spearman correlation of activity
    differences across all donor pairs.

    Args:
        activity_adata: AnnData with activity z-scores.
        treatment_col: Column with treatment labels.
        donor_col: Column with donor IDs.
        sig_type: 'cytosig' or 'secact'.

    Returns:
        DataFrame with cross-donor consistency metrics.
    """
    log(f"  Computing cross-donor consistency ({sig_type})...")

    obs = activity_adata.obs
    donors = sorted(obs[donor_col].unique())
    treatments = obs[treatment_col].unique()
    signatures = list(activity_adata.var_names)

    act_matrix = activity_adata.X
    if hasattr(act_matrix, 'toarray'):
        act_matrix = act_matrix.toarray()
    act_matrix = act_matrix.astype(np.float64)

    n_donors = len(donors)
    log(f"    {n_donors} donors, {len(treatments)} treatments, "
        f"{len(signatures)} signatures")

    if n_donors < 3:
        log(f"    WARNING: Too few donors ({n_donors}), skipping consistency")
        return pd.DataFrame()

    rows = []

    for sig_idx, sig in enumerate(signatures):
        # Build donor x treatment activity matrix
        # For each donor, compute mean activity for this signature per treatment
        donor_profiles = {}

        for donor in donors:
            donor_mask = obs[donor_col] == donor
            profile = {}

            for treatment in treatments:
                treat_mask = obs[treatment_col] == treatment
                combined = donor_mask & treat_mask
                if combined.sum() < 1:
                    continue

                vals = act_matrix[combined.values, sig_idx]
                vals = vals[np.isfinite(vals)]
                if len(vals) > 0:
                    profile[treatment] = float(np.mean(vals))

            if len(profile) > 5:
                donor_profiles[donor] = profile

        if len(donor_profiles) < 3:
            continue

        # Compute pairwise Spearman correlations between donors
        donor_list = sorted(donor_profiles.keys())
        pair_rhos = []

        for i in range(len(donor_list)):
            for j in range(i + 1, len(donor_list)):
                d1, d2 = donor_list[i], donor_list[j]
                common_treats = set(donor_profiles[d1].keys()) & set(donor_profiles[d2].keys())

                if len(common_treats) < 5:
                    continue

                vals1 = [donor_profiles[d1][t] for t in sorted(common_treats)]
                vals2 = [donor_profiles[d2][t] for t in sorted(common_treats)]

                rho, pval = stats.spearmanr(vals1, vals2)
                if np.isfinite(rho):
                    pair_rhos.append(rho)

        if len(pair_rhos) < 3:
            continue

        mean_rho = float(np.mean(pair_rhos))
        median_rho = float(np.median(pair_rhos))
        std_rho = float(np.std(pair_rhos))
        min_rho = float(np.min(pair_rhos))
        max_rho = float(np.max(pair_rhos))

        rows.append({
            'signature': sig,
            'signature_type': sig_type,
            'mean_spearman': round(mean_rho, 4),
            'median_spearman': round(median_rho, 4),
            'std_spearman': round(std_rho, 4),
            'min_spearman': round(min_rho, 4),
            'max_spearman': round(max_rho, 4),
            'n_donor_pairs': len(pair_rhos),
            'n_donors': len(donor_profiles),
        })

    df = pd.DataFrame(rows)
    log(f"    {len(df)} signatures with donor consistency")
    if len(df) > 0:
        log(f"    Median cross-donor Spearman: {df['median_spearman'].median():.3f}")

    return df


# ==============================================================================
# Full Validation Summary
# ==============================================================================

def build_validation_summary(
    auc_df: pd.DataFrame,
    celltype_df: pd.DataFrame,
    donor_df: pd.DataFrame,
) -> pd.DataFrame:
    """Combine all validation metrics into a single summary.

    Returns:
        DataFrame with one row per cytokine, combining AUC, best cell type
        response, and cross-donor consistency.
    """
    log("  Building validation summary...")

    if auc_df.empty:
        return pd.DataFrame()

    summary = auc_df.copy()

    # Add best cell type response per cytokine
    if not celltype_df.empty:
        best_ct = (celltype_df
                   .sort_values('activity_diff', ascending=False)
                   .drop_duplicates(subset=['cytokine', 'signature_type'], keep='first'))
        best_ct = best_ct.rename(columns={
            'cell_type': 'best_celltype',
            'activity_diff': 'best_celltype_diff',
            'pvalue': 'best_celltype_pval',
        })[['cytokine', 'signature_type', 'best_celltype',
            'best_celltype_diff', 'best_celltype_pval']]

        summary = summary.merge(best_ct, on=['cytokine', 'signature_type'], how='left')

    # Add donor consistency per cytokine
    if not donor_df.empty:
        donor_summary = donor_df.rename(columns={
            'signature': 'cytokine',
            'mean_spearman': 'donor_consistency',
            'n_donors': 'n_donors_consistency',
        })[['cytokine', 'signature_type', 'donor_consistency', 'n_donors_consistency']]

        summary = summary.merge(donor_summary, on=['cytokine', 'signature_type'], how='left')

    return summary


# ==============================================================================
# Main Pipeline
# ==============================================================================

def run_pipeline(
    signature_type: str = 'all',
    force: bool = False,
) -> None:
    """Run the parse10m ground-truth validation pipeline.

    Args:
        signature_type: 'cytosig', 'secact', or 'all'.
        force: Force overwrite existing outputs.
    """
    t_start = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Check results directory
    if not RESULTS_DIR.exists():
        log(f"ERROR: Results directory not found: {RESULTS_DIR}")
        log("Run parse10m activity inference first.")
        sys.exit(1)

    # Load activity results
    activity_data = load_activity_results()
    if not activity_data:
        log("ERROR: No activity results found")
        sys.exit(1)

    # Filter by signature type
    if signature_type != 'all':
        activity_data = {k: v for k, v in activity_data.items()
                         if k == signature_type}

    # Load or extract metadata
    metadata = load_metadata()

    all_auc = []
    all_celltype = []
    all_donor = []

    for sig_type, adata in activity_data.items():
        log(f"\n{'=' * 60}")
        log(f"SIGNATURE: {sig_type.upper()}")
        log(f"{'=' * 60}")

        obs = adata.obs
        if metadata is not None:
            # Merge external metadata
            common_idx = obs.index.intersection(metadata.index)
            if len(common_idx) > 0:
                for col in metadata.columns:
                    if col not in obs.columns:
                        obs[col] = metadata.loc[common_idx, col]

        # Detect key columns
        treatment_col = detect_treatment_column(obs)
        donor_col = detect_donor_column(obs)
        celltype_col = detect_celltype_column(obs)

        log(f"  Treatment column: {treatment_col}")
        log(f"  Donor column: {donor_col}")
        log(f"  Cell type column: {celltype_col}")

        if treatment_col is None:
            log(f"  ERROR: No treatment column found for {sig_type}")
            continue

        # 1. Self-signature AUC-ROC
        auc_df = compute_self_signature_auc(adata, treatment_col, sig_type)
        all_auc.append(auc_df)

        # 2. Cell type specificity
        if celltype_col is not None:
            celltype_df = compute_celltype_specificity(
                adata, treatment_col, celltype_col, sig_type
            )
            all_celltype.append(celltype_df)

        # 3. Cross-donor consistency
        if donor_col is not None:
            donor_df = compute_donor_consistency(
                adata, treatment_col, donor_col, sig_type
            )
            all_donor.append(donor_df)

        del adata
        gc.collect()

    # Combine results across signature types
    auc_combined = pd.concat(all_auc, ignore_index=True) if all_auc else pd.DataFrame()
    celltype_combined = pd.concat(all_celltype, ignore_index=True) if all_celltype else pd.DataFrame()
    donor_combined = pd.concat(all_donor, ignore_index=True) if all_donor else pd.DataFrame()

    # Save outputs
    log(f"\n{'=' * 60}")
    log("Saving output files...")
    log(f"{'=' * 60}")

    if not auc_combined.empty:
        auc_path = OUTPUT_DIR / 'parse10m_self_signature_auc.csv'
        auc_combined.to_csv(auc_path, index=False)
        log(f"  Saved: {auc_path.name} ({len(auc_combined)} rows)")

    if not donor_combined.empty:
        donor_path = OUTPUT_DIR / 'parse10m_donor_consistency.csv'
        donor_combined.to_csv(donor_path, index=False)
        log(f"  Saved: {donor_path.name} ({len(donor_combined)} rows)")

    # Build and save full validation summary
    summary_df = build_validation_summary(auc_combined, celltype_combined, donor_combined)
    if not summary_df.empty:
        summary_path = OUTPUT_DIR / 'parse10m_ground_truth_validation.csv'
        summary_df.to_csv(summary_path, index=False)
        log(f"  Saved: {summary_path.name} ({len(summary_df)} rows)")

    # Print summary statistics
    elapsed = time.time() - t_start
    log(f"\n{'=' * 60}")
    log(f"PIPELINE COMPLETE ({elapsed / 60:.1f} min)")
    log(f"{'=' * 60}")

    if not auc_combined.empty:
        log(f"\nSelf-signature AUC-ROC summary:")
        for sig_type in auc_combined['signature_type'].unique():
            subset = auc_combined[auc_combined['signature_type'] == sig_type]
            valid_auc = subset['auc_roc'].dropna()
            if len(valid_auc) > 0:
                log(f"  {sig_type}: median={valid_auc.median():.3f}, "
                    f"mean={valid_auc.mean():.3f}, n={len(valid_auc)}")
                high_auc = (valid_auc > 0.7).sum()
                log(f"    AUC > 0.7: {high_auc}/{len(valid_auc)} "
                    f"({high_auc/len(valid_auc)*100:.0f}%)")

    if not donor_combined.empty:
        log(f"\nCross-donor consistency summary:")
        for sig_type in donor_combined['signature_type'].unique():
            subset = donor_combined[donor_combined['signature_type'] == sig_type]
            log(f"  {sig_type}: median rho={subset['median_spearman'].median():.3f}, "
                f"n={len(subset)}")


def main():
    parser = argparse.ArgumentParser(
        description="CytoSig Ground-Truth Validation Using parse_10M"
    )
    parser.add_argument(
        '--signature-type', default='all',
        choices=['cytosig', 'secact', 'all'],
        help='Signature type to validate (default: all)',
    )
    parser.add_argument(
        '--force', action='store_true',
        help='Force overwrite existing output files',
    )

    args = parser.parse_args()

    log(f"Signature type: {args.signature_type}")
    log(f"Force: {args.force}")

    run_pipeline(
        signature_type=args.signature_type,
        force=args.force,
    )


if __name__ == '__main__':
    main()
