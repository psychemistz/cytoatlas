#!/usr/bin/env python3
"""
Tahoe Drug Sensitivity Signature Extraction
=============================================
Extracts drug sensitivity signatures from the Tahoe perturbation dataset,
linking cytokine/secreted protein activity to drug responses across
cancer cell lines.

Key analyses:
1. Drug sensitivity matrix: drug x cell_line x signature activity
2. Known drug-pathway associations: kinase inhibitors, proteasome inhibitors,
   checkpoint modulators, etc.
3. Dose-response analysis from Plate 13: monotonicity test for activity vs dose

Output:
    - tahoe_drug_sensitivity_matrix.csv        (drug x cell_line x signature)
    - tahoe_cytokine_pathway_activation.csv    (pathway-level summary)

Usage:
    # Full analysis
    python scripts/22_tahoe_drug_signatures.py

    # Force overwrite
    python scripts/22_tahoe_drug_signatures.py --force

    # Specific signature type
    python scripts/22_tahoe_drug_signatures.py --signature-type cytosig
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
sys.path.insert(0, '/vf/users/parks34/projects/1ridgesig/SecActpy')
from secactpy import (
    load_cytosig, load_secact,
    ridge_batch, estimate_batch_size,
    CUPY_AVAILABLE
)

# ==============================================================================
# Configuration
# ==============================================================================

RESULTS_DIR = Path('/data/parks34/projects/2secactpy/results/tahoe')
OUTPUT_DIR = Path('/vf/users/parks34/projects/2secactpy/results/tahoe')

# Known drug-pathway associations
# Maps drug names/classes to expected cytokine pathway effects
DRUG_PATHWAY_MAP = {
    # Kinase inhibitors
    'kinase_inhibitor': {
        'keywords': ['kinase', 'tinib', 'imatinib', 'dasatinib', 'nilotinib',
                     'sorafenib', 'sunitinib', 'erlotinib', 'gefitinib',
                     'lapatinib', 'crizotinib', 'vemurafenib', 'trametinib'],
        'expected_signatures': ['TGFB1', 'VEGF', 'PDGF', 'EGF', 'FGF2'],
    },
    # Proteasome inhibitors
    'proteasome_inhibitor': {
        'keywords': ['bortezomib', 'carfilzomib', 'ixazomib', 'proteasome',
                     'MG132', 'MG-132'],
        'expected_signatures': ['TNFA', 'IL6', 'NFKB', 'IL1B'],
    },
    # HDAC inhibitors
    'hdac_inhibitor': {
        'keywords': ['vorinostat', 'panobinostat', 'romidepsin', 'HDAC',
                     'entinostat', 'belinostat', 'SAHA'],
        'expected_signatures': ['IFNG', 'TNFA', 'IL6'],
    },
    # mTOR inhibitors
    'mtor_inhibitor': {
        'keywords': ['rapamycin', 'everolimus', 'temsirolimus', 'mTOR',
                     'torin', 'AZD8055'],
        'expected_signatures': ['IL2', 'IL6', 'VEGF', 'TGFB1'],
    },
    # JAK/STAT inhibitors
    'jak_inhibitor': {
        'keywords': ['ruxolitinib', 'tofacitinib', 'baricitinib', 'JAK'],
        'expected_signatures': ['IFNG', 'IFNA', 'IFNB', 'IL6', 'IL2', 'IL10'],
    },
    # NFkB pathway
    'nfkb_modulator': {
        'keywords': ['nfkb', 'ikk', 'BAY11-7082', 'BAY 11-7082'],
        'expected_signatures': ['TNFA', 'IL1B', 'IL6', 'IL8'],
    },
    # DNA damage / chemotherapy
    'dna_damage': {
        'keywords': ['cisplatin', 'doxorubicin', 'etoposide', 'camptothecin',
                     'topotecan', 'irinotecan', 'bleomycin', 'mitomycin'],
        'expected_signatures': ['TNFA', 'IL6', 'IFNG', 'TGFB1'],
    },
    # Glucocorticoids
    'glucocorticoid': {
        'keywords': ['dexamethasone', 'prednisolone', 'hydrocortisone',
                     'betamethasone', 'cortisol'],
        'expected_signatures': ['IL2', 'IL6', 'TNFA', 'IL1B', 'IFNG'],
    },
}

# Plate 13 is the dose-response plate in the Tahoe experiment
DOSE_RESPONSE_PLATE = 'Plate13'

# GPU settings
BACKEND = 'cupy' if CUPY_AVAILABLE else 'numpy'


# ==============================================================================
# Utility Functions
# ==============================================================================

def log(msg: str):
    """Print timestamped log message."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_activity_results() -> Dict[str, ad.AnnData]:
    """Load pre-computed activity results from results/tahoe/.

    Returns:
        Dict mapping signature type -> AnnData with activity z-scores.
    """
    log("Loading tahoe activity results...")
    results = {}

    for sig_type in ['cytosig', 'secact']:
        candidates = [
            RESULTS_DIR / f'tahoe_{sig_type}.h5ad',
            RESULTS_DIR / f'tahoe_{sig_type}_zscore.h5ad',
            RESULTS_DIR / f'tahoe_activity_{sig_type}.h5ad',
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
    """Load tahoe metadata (drug, cell_line, dose, plate).

    Returns:
        DataFrame with drug/cell_line/dose/plate columns.
    """
    meta_candidates = [
        RESULTS_DIR / 'tahoe_metadata.csv',
        RESULTS_DIR / 'metadata.csv',
        RESULTS_DIR / 'tahoe_obs.csv',
    ]

    for path in meta_candidates:
        if path.exists():
            log(f"  Loading metadata: {path.name}")
            meta = pd.read_csv(path, index_col=0)
            log(f"    Shape: {meta.shape}")
            return meta

    log("  No dedicated metadata file found")
    return None


def detect_drug_column(obs: pd.DataFrame) -> Optional[str]:
    """Detect the column containing drug/compound names."""
    for col in ['drug', 'compound', 'treatment', 'Drug', 'Compound',
                'perturbation', 'drug_name', 'compound_name', 'sm_name']:
        if col in obs.columns:
            return col
    return None


def detect_cellline_column(obs: pd.DataFrame) -> Optional[str]:
    """Detect the column containing cell line names."""
    for col in ['cell_line', 'cellline', 'CellLine', 'cell_line_name',
                'sample', 'cell_type', 'cell_iname']:
        if col in obs.columns:
            return col
    return None


def detect_dose_column(obs: pd.DataFrame) -> Optional[str]:
    """Detect the column containing dose/concentration values."""
    for col in ['dose', 'concentration', 'Dose', 'dose_um', 'conc',
                'sm_dose', 'dose_value']:
        if col in obs.columns:
            return col
    return None


def detect_plate_column(obs: pd.DataFrame) -> Optional[str]:
    """Detect the column containing plate identifiers."""
    for col in ['plate', 'Plate', 'plate_name', 'plate_id', 'batch']:
        if col in obs.columns:
            return col
    return None


def classify_drug(drug_name: str) -> List[str]:
    """Classify a drug into known pathway categories.

    Args:
        drug_name: Drug name string.

    Returns:
        List of matching pathway category names.
    """
    drug_lower = str(drug_name).lower()
    categories = []
    for category, info in DRUG_PATHWAY_MAP.items():
        for keyword in info['keywords']:
            if keyword.lower() in drug_lower:
                categories.append(category)
                break
    return categories if categories else ['unclassified']


# ==============================================================================
# Drug Sensitivity Matrix
# ==============================================================================

def compute_drug_sensitivity_matrix(
    activity_adata: ad.AnnData,
    drug_col: str,
    cellline_col: str,
    sig_type: str,
) -> pd.DataFrame:
    """Build the drug x cell_line x signature activity matrix.

    For each drug-cellline combination, compute mean activity across
    replicates for every signature.

    Args:
        activity_adata: AnnData with activity z-scores.
        drug_col: Column with drug names.
        cellline_col: Column with cell line names.
        sig_type: 'cytosig' or 'secact'.

    Returns:
        DataFrame with columns: drug, cell_line, signature, signature_type,
        mean_activity, std_activity, n_replicates, activity_diff_vs_control.
    """
    log(f"  Computing drug sensitivity matrix ({sig_type})...")

    obs = activity_adata.obs
    drugs = obs[drug_col].dropna().unique()
    cell_lines = obs[cellline_col].dropna().unique()
    signatures = list(activity_adata.var_names)

    act_matrix = activity_adata.X
    if hasattr(act_matrix, 'toarray'):
        act_matrix = act_matrix.toarray()
    act_matrix = act_matrix.astype(np.float64)

    log(f"    {len(drugs)} drugs, {len(cell_lines)} cell lines, "
        f"{len(signatures)} signatures")

    # Find control samples (DMSO, vehicle, untreated)
    control_keywords = ['dmso', 'vehicle', 'control', 'untreated', 'none']
    control_drugs = [d for d in drugs
                     if any(kw in str(d).lower() for kw in control_keywords)]
    log(f"    Control drugs: {control_drugs}")

    # Compute control baselines per cell line
    control_baselines = {}  # cell_line -> mean activity per signature
    for cl in cell_lines:
        cl_mask = obs[cellline_col] == cl
        ctrl_mask = cl_mask & obs[drug_col].isin(control_drugs)

        if ctrl_mask.sum() > 0:
            control_baselines[cl] = np.nanmean(act_matrix[ctrl_mask.values], axis=0)

    rows = []

    for drug in drugs:
        drug_mask = obs[drug_col] == drug
        drug_categories = classify_drug(drug)

        for cl in cell_lines:
            cl_mask = obs[cellline_col] == cl
            combined = drug_mask & cl_mask
            n_reps = combined.sum()

            if n_reps < 1:
                continue

            act_vals = act_matrix[combined.values]  # (replicates x signatures)

            for sig_idx, sig in enumerate(signatures):
                vals = act_vals[:, sig_idx]
                vals = vals[np.isfinite(vals)]
                if len(vals) == 0:
                    continue

                mean_act = float(np.mean(vals))
                std_act = float(np.std(vals)) if len(vals) > 1 else 0.0

                # Compute difference from control
                diff_vs_control = np.nan
                if cl in control_baselines:
                    ctrl_val = control_baselines[cl][sig_idx]
                    if np.isfinite(ctrl_val):
                        diff_vs_control = mean_act - ctrl_val

                rows.append({
                    'drug': drug,
                    'cell_line': cl,
                    'signature': sig,
                    'signature_type': sig_type,
                    'drug_category': ';'.join(drug_categories),
                    'mean_activity': round(mean_act, 4),
                    'std_activity': round(std_act, 4),
                    'n_replicates': int(n_reps),
                    'activity_diff_vs_control': round(diff_vs_control, 4) if np.isfinite(diff_vs_control) else None,
                })

    df = pd.DataFrame(rows)
    log(f"    {len(df)} drug-cellline-signature entries")
    return df


# ==============================================================================
# Drug-Pathway Association Analysis
# ==============================================================================

def compute_pathway_activation(
    sensitivity_df: pd.DataFrame,
) -> pd.DataFrame:
    """Identify known drug-pathway associations from sensitivity data.

    For each drug category, test whether expected signatures show
    significant activity changes compared to other signatures.

    Args:
        sensitivity_df: Drug sensitivity matrix from compute_drug_sensitivity_matrix().

    Returns:
        DataFrame with pathway-level activation statistics.
    """
    log("  Computing cytokine pathway activation...")

    if sensitivity_df.empty:
        return pd.DataFrame()

    rows = []

    for category, info in DRUG_PATHWAY_MAP.items():
        expected_sigs = info['expected_signatures']

        # Filter to drugs in this category
        cat_mask = sensitivity_df['drug_category'].str.contains(category, na=False)
        cat_df = sensitivity_df[cat_mask]

        if len(cat_df) == 0:
            continue

        n_drugs = cat_df['drug'].nunique()
        n_celllines = cat_df['cell_line'].nunique()

        for sig in expected_sigs:
            sig_data = cat_df[cat_df['signature'] == sig]
            other_data = cat_df[cat_df['signature'] != sig]

            if len(sig_data) < 3:
                continue

            # Use activity_diff_vs_control if available, otherwise mean_activity
            if 'activity_diff_vs_control' in sig_data.columns:
                sig_vals = sig_data['activity_diff_vs_control'].dropna().values
                other_vals = other_data['activity_diff_vs_control'].dropna().values
            else:
                sig_vals = sig_data['mean_activity'].values
                other_vals = other_data['mean_activity'].values

            sig_vals = sig_vals[np.isfinite(sig_vals)]
            other_vals = other_vals[np.isfinite(other_vals)]

            if len(sig_vals) < 3:
                continue

            mean_expected = float(np.mean(sig_vals))
            mean_other = float(np.mean(other_vals)) if len(other_vals) > 0 else 0.0
            activity_diff = mean_expected - mean_other

            # Test if expected signature is more activated than background
            if len(other_vals) >= 3:
                try:
                    stat, pval = stats.mannwhitneyu(
                        sig_vals, other_vals, alternative='two-sided'
                    )
                except Exception:
                    pval = np.nan
            else:
                pval = np.nan

            rows.append({
                'drug_category': category,
                'expected_signature': sig,
                'signature_type': sig_data['signature_type'].iloc[0] if len(sig_data) > 0 else 'unknown',
                'mean_activity_expected': round(mean_expected, 4),
                'mean_activity_other': round(mean_other, 4),
                'activity_diff': round(activity_diff, 4),
                'n_observations': len(sig_vals),
                'n_drugs': int(n_drugs),
                'n_celllines': int(n_celllines),
                'pvalue': pval,
            })

    df = pd.DataFrame(rows)

    # FDR correction
    if len(df) > 0 and 'pvalue' in df.columns:
        valid_pvals = df['pvalue'].dropna()
        if len(valid_pvals) > 0:
            _, fdr, _, _ = multipletests(valid_pvals.values, method='fdr_bh')
            df.loc[valid_pvals.index, 'fdr'] = fdr

    log(f"    {len(df)} pathway-signature associations")
    return df


# ==============================================================================
# Dose-Response Analysis
# ==============================================================================

def compute_dose_response(
    activity_adata: ad.AnnData,
    drug_col: str,
    cellline_col: str,
    dose_col: str,
    plate_col: Optional[str],
    sig_type: str,
) -> pd.DataFrame:
    """Analyze dose-response monotonicity from Plate 13.

    Tests whether activity changes monotonically with dose for each
    drug-cellline-signature combination.

    Args:
        activity_adata: AnnData with activity z-scores.
        drug_col: Column with drug names.
        cellline_col: Column with cell line names.
        dose_col: Column with dose values.
        plate_col: Column with plate IDs (filter to Plate 13).
        sig_type: 'cytosig' or 'secact'.

    Returns:
        DataFrame with dose-response monotonicity metrics.
    """
    log(f"  Computing dose-response analysis ({sig_type})...")

    obs = activity_adata.obs.copy()

    # Filter to dose-response plate if plate column exists
    if plate_col is not None and plate_col in obs.columns:
        plate_mask = obs[plate_col].astype(str).str.contains('13|Plate13|plate13',
                                                              case=False, na=False)
        if plate_mask.sum() > 0:
            log(f"    Filtering to dose-response plate: {plate_mask.sum()} samples")
            obs = obs[plate_mask]
        else:
            log(f"    No Plate 13 found, using all samples with dose info")

    if dose_col not in obs.columns:
        log(f"    WARNING: Dose column '{dose_col}' not found")
        return pd.DataFrame()

    # Convert dose to numeric
    obs['_dose_numeric'] = pd.to_numeric(obs[dose_col], errors='coerce')
    obs = obs.dropna(subset=['_dose_numeric'])

    if len(obs) == 0:
        log(f"    WARNING: No valid dose values")
        return pd.DataFrame()

    act_matrix = activity_adata.X
    if hasattr(act_matrix, 'toarray'):
        act_matrix = act_matrix.toarray()
    act_matrix = act_matrix.astype(np.float64)

    signatures = list(activity_adata.var_names)
    drugs = obs[drug_col].dropna().unique()
    cell_lines = obs[cellline_col].dropna().unique()

    rows = []

    for drug in drugs:
        drug_mask = obs[drug_col] == drug

        for cl in cell_lines:
            cl_mask = obs[cellline_col] == cl
            combined = drug_mask & cl_mask

            if combined.sum() < 3:
                continue

            subset = obs[combined]
            doses = subset['_dose_numeric'].values
            unique_doses = np.unique(doses)

            if len(unique_doses) < 3:
                continue

            # Get activity for these samples
            sample_indices = np.where(combined.values)[0]

            for sig_idx, sig in enumerate(signatures):
                vals = act_matrix[sample_indices, sig_idx]

                # Remove NaN
                valid = np.isfinite(vals)
                vals = vals[valid]
                d = doses[valid]

                if len(vals) < 3:
                    continue

                # Spearman correlation of activity vs dose
                rho, pval = stats.spearmanr(d, vals)

                # Monotonicity test: compute mean activity at each dose level
                dose_means = []
                for dose_val in sorted(np.unique(d)):
                    dose_mask = d == dose_val
                    if dose_mask.sum() > 0:
                        dose_means.append(float(np.mean(vals[dose_mask])))

                # Count monotonic steps
                if len(dose_means) >= 3:
                    diffs = np.diff(dose_means)
                    n_increasing = (diffs > 0).sum()
                    n_decreasing = (diffs < 0).sum()
                    n_steps = len(diffs)
                    monotonicity_score = max(n_increasing, n_decreasing) / n_steps
                    direction = 'increasing' if n_increasing > n_decreasing else 'decreasing'
                else:
                    monotonicity_score = np.nan
                    direction = 'unknown'

                # Activity range across doses
                act_range = float(np.max(vals) - np.min(vals))

                rows.append({
                    'drug': drug,
                    'cell_line': cl,
                    'signature': sig,
                    'signature_type': sig_type,
                    'spearman_rho': round(rho, 4) if np.isfinite(rho) else None,
                    'spearman_pval': pval if np.isfinite(pval) else None,
                    'monotonicity_score': round(monotonicity_score, 4) if np.isfinite(monotonicity_score) else None,
                    'direction': direction,
                    'activity_range': round(act_range, 4),
                    'n_doses': len(np.unique(d)),
                    'n_samples': len(vals),
                    'min_activity': round(float(np.min(vals)), 4),
                    'max_activity': round(float(np.max(vals)), 4),
                })

    df = pd.DataFrame(rows)

    # FDR correction
    if len(df) > 0 and 'spearman_pval' in df.columns:
        valid_pvals = df['spearman_pval'].dropna()
        if len(valid_pvals) > 0:
            _, fdr, _, _ = multipletests(valid_pvals.values, method='fdr_bh')
            df.loc[valid_pvals.index, 'fdr'] = fdr

    log(f"    {len(df)} dose-response entries")
    if len(df) > 0:
        monotonic = df[df['monotonicity_score'].notna() & (df['monotonicity_score'] > 0.8)]
        log(f"    Strongly monotonic (score > 0.8): {len(monotonic)}")

    return df


# ==============================================================================
# Main Pipeline
# ==============================================================================

def run_pipeline(
    signature_type: str = 'all',
    force: bool = False,
) -> None:
    """Run the Tahoe drug sensitivity extraction pipeline.

    Args:
        signature_type: 'cytosig', 'secact', or 'all'.
        force: Force overwrite existing outputs.
    """
    t_start = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Check results directory
    if not RESULTS_DIR.exists():
        log(f"ERROR: Results directory not found: {RESULTS_DIR}")
        log("Run tahoe activity inference first.")
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

    # Load metadata
    metadata = load_metadata()

    all_sensitivity = []
    all_dose_response = []

    for sig_type, adata in activity_data.items():
        log(f"\n{'=' * 60}")
        log(f"SIGNATURE: {sig_type.upper()}")
        log(f"{'=' * 60}")

        obs = adata.obs
        if metadata is not None:
            common_idx = obs.index.intersection(metadata.index)
            if len(common_idx) > 0:
                for col in metadata.columns:
                    if col not in obs.columns:
                        obs[col] = metadata.loc[common_idx, col]

        # Detect key columns
        drug_col = detect_drug_column(obs)
        cellline_col = detect_cellline_column(obs)
        dose_col = detect_dose_column(obs)
        plate_col = detect_plate_column(obs)

        log(f"  Drug column: {drug_col}")
        log(f"  Cell line column: {cellline_col}")
        log(f"  Dose column: {dose_col}")
        log(f"  Plate column: {plate_col}")

        if drug_col is None:
            log(f"  ERROR: No drug column found for {sig_type}")
            continue

        if cellline_col is None:
            log(f"  ERROR: No cell line column found for {sig_type}")
            continue

        # 1. Drug sensitivity matrix
        sensitivity_df = compute_drug_sensitivity_matrix(
            adata, drug_col, cellline_col, sig_type
        )
        all_sensitivity.append(sensitivity_df)

        # 2. Dose-response analysis
        if dose_col is not None:
            dr_df = compute_dose_response(
                adata, drug_col, cellline_col, dose_col, plate_col, sig_type
            )
            all_dose_response.append(dr_df)

        del adata
        gc.collect()

    # Combine results
    sensitivity_combined = pd.concat(all_sensitivity, ignore_index=True) if all_sensitivity else pd.DataFrame()
    dr_combined = pd.concat(all_dose_response, ignore_index=True) if all_dose_response else pd.DataFrame()

    # Compute pathway activation from combined sensitivity data
    pathway_df = compute_pathway_activation(sensitivity_combined)

    # Save outputs
    log(f"\n{'=' * 60}")
    log("Saving output files...")
    log(f"{'=' * 60}")

    if not sensitivity_combined.empty:
        sens_path = OUTPUT_DIR / 'tahoe_drug_sensitivity_matrix.csv'
        sensitivity_combined.to_csv(sens_path, index=False)
        log(f"  Saved: {sens_path.name} ({len(sensitivity_combined)} rows)")

    if not pathway_df.empty:
        pathway_path = OUTPUT_DIR / 'tahoe_cytokine_pathway_activation.csv'
        pathway_df.to_csv(pathway_path, index=False)
        log(f"  Saved: {pathway_path.name} ({len(pathway_df)} rows)")

    if not dr_combined.empty:
        dr_path = OUTPUT_DIR / 'tahoe_dose_response.csv'
        dr_combined.to_csv(dr_path, index=False)
        log(f"  Saved: {dr_path.name} ({len(dr_combined)} rows)")

    # Summary
    elapsed = time.time() - t_start
    log(f"\n{'=' * 60}")
    log(f"PIPELINE COMPLETE ({elapsed / 60:.1f} min)")
    log(f"{'=' * 60}")

    if not sensitivity_combined.empty:
        log(f"\nDrug sensitivity summary:")
        log(f"  Drugs: {sensitivity_combined['drug'].nunique()}")
        log(f"  Cell lines: {sensitivity_combined['cell_line'].nunique()}")
        log(f"  Signatures: {sensitivity_combined['signature'].nunique()}")
        log(f"  Total entries: {len(sensitivity_combined)}")

    if not pathway_df.empty:
        log(f"\nPathway activation summary:")
        sig_pathways = pathway_df[pathway_df['fdr'].notna() & (pathway_df['fdr'] < 0.05)]
        log(f"  Significant (FDR < 0.05): {len(sig_pathways)}")
        for _, row in sig_pathways.iterrows():
            log(f"    {row['drug_category']} -> {row['expected_signature']}: "
                f"diff={row['activity_diff']:.3f}, FDR={row['fdr']:.4f}")

    if not dr_combined.empty:
        log(f"\nDose-response summary:")
        monotonic = dr_combined[dr_combined['monotonicity_score'].notna() & (dr_combined['monotonicity_score'] > 0.8)]
        log(f"  Strongly monotonic (score > 0.8): {len(monotonic)}/{len(dr_combined)}")


def main():
    parser = argparse.ArgumentParser(
        description="Tahoe Drug Sensitivity Signature Extraction"
    )
    parser.add_argument(
        '--signature-type', default='all',
        choices=['cytosig', 'secact', 'all'],
        help='Signature type to analyze (default: all)',
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
