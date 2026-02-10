#!/usr/bin/env python3
"""
Integrated Cross-Atlas Analysis
================================
Harmonize cell type annotations and compare cytokine/secreted protein
activity profiles across the three single-cell atlases.

Atlases:
- CIMA: 6.5M cells, 428 samples, healthy donors with metabolomics
- Inflammation Atlas: 6.3M cells, 1,047 samples, 20 diseases
- scAtlas: 6.4M cells (2.3M normal + 4.1M cancer), 35 organs

Analysis objectives:
1. Harmonize cell type annotations across atlases
2. Compare healthy profiles (CIMA vs Inflammation healthy)
3. Identify conserved and atlas-specific signatures
4. Generate summary statistics and visualizations
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
from scipy import stats

# ==============================================================================
# Configuration
# ==============================================================================

# Input paths (pseudo-bulk results from previous scripts)
RESULTS_DIR = Path('/data/parks34/projects/2cytoatlas/results')

CIMA_DIR = RESULTS_DIR / 'cima'
INFLAM_DIR = RESULTS_DIR / 'inflammation'
SCATLAS_DIR = RESULTS_DIR / 'scatlas'

# Output path
OUTPUT_DIR = RESULTS_DIR / 'integrated'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# Utility Functions
# ==============================================================================

def log(msg: str):
    """Print timestamped log message."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


def load_activity_results(h5ad_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load activity results from h5ad file.

    Returns:
        zscore_df: Activity z-scores (proteins x samples)
        meta_df: Sample metadata
    """
    if not h5ad_path.exists():
        log(f"  Warning: File not found: {h5ad_path}")
        return None, None

    adata = ad.read_h5ad(h5ad_path)

    # Extract z-scores (stored in .X)
    zscore_df = pd.DataFrame(
        adata.X,
        index=adata.obs_names,
        columns=adata.var_names
    )

    # Get metadata
    meta_df = adata.var.copy()

    return zscore_df, meta_df


# ==============================================================================
# Cell Type Harmonization
# ==============================================================================

# Cell type mapping dictionaries
CIMA_TO_COMMON = {
    # B cells
    'B_naive': 'B_naive',
    'B_intermediate': 'B_memory',
    'B_memory': 'B_memory',
    'Plasmablast': 'Plasma',

    # CD4 T cells
    'CD4_naive': 'CD4_naive',
    'CD4_TCM': 'CD4_memory',
    'CD4_TEM': 'CD4_effector',
    'CD4_CTL': 'CD4_effector',
    'CD4_Treg': 'Treg',
    'CD4_Proliferating': 'T_proliferating',

    # CD8 T cells
    'CD8_naive': 'CD8_naive',
    'CD8_TCM': 'CD8_memory',
    'CD8_TEM': 'CD8_effector',
    'CD8_Proliferating': 'T_proliferating',
    'MAIT': 'MAIT',
    'gdT': 'gdT',

    # NK cells
    'NK_dim': 'NK',
    'NK_bright': 'NK',
    'NK_Proliferating': 'NK',

    # Myeloid cells
    'CD14_Mono': 'Mono_classical',
    'CD16_Mono': 'Mono_nonclassical',
    'cDC': 'cDC',
    'pDC': 'pDC',

    # ILCs
    'ILC': 'ILC',

    # HSPCs
    'HSPC': 'HSPC',
}

INFLAM_TO_COMMON = {
    # B cells
    'B naive': 'B_naive',
    'B memory': 'B_memory',
    'B intermediate': 'B_memory',
    'Plasmablast': 'Plasma',
    'Plasma cell': 'Plasma',

    # CD4 T cells
    'CD4 naive': 'CD4_naive',
    'CD4 memory': 'CD4_memory',
    'CD4 effector': 'CD4_effector',
    'CD4 CTL': 'CD4_effector',
    'Treg': 'Treg',
    'T proliferating': 'T_proliferating',

    # CD8 T cells
    'CD8 naive': 'CD8_naive',
    'CD8 memory': 'CD8_memory',
    'CD8 effector': 'CD8_effector',
    'MAIT': 'MAIT',
    'gdT': 'gdT',

    # NK cells
    'NK dim': 'NK',
    'NK bright': 'NK',
    'NK': 'NK',

    # Myeloid cells
    'Classical monocyte': 'Mono_classical',
    'Non-classical monocyte': 'Mono_nonclassical',
    'Intermediate monocyte': 'Mono_classical',
    'cDC1': 'cDC',
    'cDC2': 'cDC',
    'pDC': 'pDC',
    'Macrophage': 'Macrophage',

    # Granulocytes
    'Neutrophil': 'Neutrophil',
    'Eosinophil': 'Eosinophil',
    'Basophil': 'Basophil',
    'Mast cell': 'Mast',

    # ILCs
    'ILC': 'ILC',
    'ILC1': 'ILC',
    'ILC2': 'ILC',
    'ILC3': 'ILC',

    # HSPCs
    'HSPC': 'HSPC',
}


def harmonize_cell_types(meta_df: pd.DataFrame, atlas: str) -> pd.DataFrame:
    """
    Map atlas-specific cell types to common annotations.

    Args:
        meta_df: Metadata with cell_type column
        atlas: 'cima', 'inflammation', or 'scatlas'

    Returns:
        DataFrame with added 'common_celltype' column
    """
    meta_df = meta_df.copy()

    if 'cell_type' not in meta_df.columns:
        log(f"  Warning: 'cell_type' column not found for {atlas}")
        return meta_df

    if atlas == 'cima':
        mapping = CIMA_TO_COMMON
    elif atlas == 'inflammation':
        mapping = INFLAM_TO_COMMON
    else:
        # For scAtlas, use cell type directly (already standardized)
        meta_df['common_celltype'] = meta_df['cell_type']
        return meta_df

    # Apply mapping
    meta_df['common_celltype'] = meta_df['cell_type'].map(mapping)

    # Fill unmapped with original
    unmapped = meta_df['common_celltype'].isna()
    if unmapped.any():
        meta_df.loc[unmapped, 'common_celltype'] = meta_df.loc[unmapped, 'cell_type']
        log(f"  Warning: {unmapped.sum()} unmapped cell types for {atlas}")

    return meta_df


# ==============================================================================
# Cross-Atlas Comparison Functions
# ==============================================================================

def compare_healthy_profiles(
    cima_zscore: pd.DataFrame,
    cima_meta: pd.DataFrame,
    inflam_zscore: pd.DataFrame,
    inflam_meta: pd.DataFrame,
    inflam_sample_meta: pd.DataFrame
) -> pd.DataFrame:
    """
    Compare activity profiles between CIMA (all healthy) and Inflammation Atlas healthy samples.

    Returns:
        DataFrame with correlation and differential statistics per cell type
    """
    log("Comparing healthy profiles across atlases...")

    results = []

    # Harmonize cell types
    cima_meta = harmonize_cell_types(cima_meta, 'cima')
    inflam_meta = harmonize_cell_types(inflam_meta, 'inflammation')

    # Get common cell types
    cima_celltypes = set(cima_meta['common_celltype'].dropna().unique())
    inflam_celltypes = set(inflam_meta['common_celltype'].dropna().unique())
    common_celltypes = cima_celltypes & inflam_celltypes

    log(f"  CIMA cell types: {len(cima_celltypes)}")
    log(f"  Inflammation cell types: {len(inflam_celltypes)}")
    log(f"  Common cell types: {len(common_celltypes)}")

    # Get common signatures
    common_sigs = list(set(cima_zscore.columns) & set(inflam_zscore.columns))
    log(f"  Common signatures: {len(common_sigs)}")

    # Filter inflammation to healthy samples
    if inflam_sample_meta is not None and 'disease' in inflam_sample_meta.columns:
        healthy_samples = inflam_sample_meta[
            inflam_sample_meta['disease'] == 'healthy'
        ]['sampleID'].tolist()

        # Map columns to samples
        col_to_sample = inflam_meta['sample'].to_dict() if 'sample' in inflam_meta.columns else {}
        healthy_cols = [c for c in inflam_zscore.index
                       if col_to_sample.get(c) in healthy_samples]
        log(f"  Healthy inflammation columns: {len(healthy_cols)}")
    else:
        healthy_cols = list(inflam_zscore.index)

    # Compare per cell type
    for celltype in common_celltypes:
        # Get CIMA columns for this cell type
        cima_cols = cima_meta[cima_meta['common_celltype'] == celltype].index.tolist()
        cima_cols = [c for c in cima_cols if c in cima_zscore.index]

        # Get Inflammation columns for this cell type
        inflam_celltype_cols = inflam_meta[inflam_meta['common_celltype'] == celltype].index.tolist()
        inflam_cols = [c for c in inflam_celltype_cols if c in healthy_cols and c in inflam_zscore.index]

        if len(cima_cols) < 3 or len(inflam_cols) < 3:
            continue

        # Compute mean profiles
        cima_mean = cima_zscore.loc[cima_cols, common_sigs].mean(axis=0)
        inflam_mean = inflam_zscore.loc[inflam_cols, common_sigs].mean(axis=0)

        # Correlation
        rho, pval = stats.spearmanr(cima_mean, inflam_mean)

        # Per-signature comparison
        for sig in common_sigs:
            cima_vals = cima_zscore.loc[cima_cols, sig].values
            inflam_vals = inflam_zscore.loc[inflam_cols, sig].values

            try:
                stat, sig_pval = stats.mannwhitneyu(cima_vals, inflam_vals, alternative='two-sided')
            except Exception:
                stat, sig_pval = np.nan, np.nan

            results.append({
                'cell_type': celltype,
                'signature': sig,
                'cima_mean': cima_vals.mean(),
                'inflam_mean': inflam_vals.mean(),
                'difference': cima_vals.mean() - inflam_vals.mean(),
                'cima_n': len(cima_cols),
                'inflam_n': len(inflam_cols),
                'pvalue': sig_pval,
                'profile_correlation': rho,
                'profile_pvalue': pval
            })

    result_df = pd.DataFrame(results)

    # FDR correction
    if len(result_df) > 0:
        try:
            from statsmodels.stats.multitest import multipletests
            _, pvals_corrected, _, _ = multipletests(result_df['pvalue'].dropna().values, method='fdr_bh')
            result_df.loc[result_df['pvalue'].notna(), 'qvalue'] = pvals_corrected
        except Exception:
            result_df['qvalue'] = result_df['pvalue']

    log(f"  Computed {len(result_df)} cell type-signature comparisons")

    return result_df


def identify_conserved_signatures(
    cima_results: pd.DataFrame,
    inflam_results: pd.DataFrame,
    scatlas_results: pd.DataFrame,
    qvalue_threshold: float = 0.05
) -> pd.DataFrame:
    """
    Identify signatures that are consistently significant across atlases.

    Returns:
        DataFrame with conserved signature statistics
    """
    log("Identifying conserved signatures...")

    # Collect significant signatures from each atlas
    all_sigs = set()

    if cima_results is not None and 'protein' in cima_results.columns:
        cima_sig = set(cima_results[cima_results.get('qvalue', cima_results['pvalue']) < qvalue_threshold]['protein'])
        all_sigs.update(cima_sig)
        log(f"  CIMA significant: {len(cima_sig)}")

    if inflam_results is not None and 'protein' in inflam_results.columns:
        inflam_sig = set(inflam_results[inflam_results.get('qvalue', inflam_results['pvalue']) < qvalue_threshold]['protein'])
        all_sigs.update(inflam_sig)
        log(f"  Inflammation significant: {len(inflam_sig)}")

    if scatlas_results is not None and 'signature' in scatlas_results.columns:
        scatlas_sig = set(scatlas_results[scatlas_results.get('qvalue', scatlas_results['pvalue']) < qvalue_threshold]['signature'])
        all_sigs.update(scatlas_sig)
        log(f"  scAtlas significant: {len(scatlas_sig)}")

    # Count conservation
    results = []
    for sig in all_sigs:
        in_cima = sig in cima_sig if 'cima_sig' in dir() else False
        in_inflam = sig in inflam_sig if 'inflam_sig' in dir() else False
        in_scatlas = sig in scatlas_sig if 'scatlas_sig' in dir() else False

        results.append({
            'signature': sig,
            'in_cima': in_cima,
            'in_inflammation': in_inflam,
            'in_scatlas': in_scatlas,
            'n_atlases': sum([in_cima, in_inflam, in_scatlas]),
            'conserved': sum([in_cima, in_inflam, in_scatlas]) >= 2
        })

    result_df = pd.DataFrame(results)

    if len(result_df) > 0:
        result_df = result_df.sort_values('n_atlases', ascending=False)

    conserved_count = len(result_df[result_df['conserved']]) if len(result_df) > 0 else 0
    log(f"  Total signatures: {len(result_df)}")
    log(f"  Conserved (>=2 atlases): {conserved_count}")

    return result_df


def generate_summary_statistics() -> Dict:
    """
    Generate summary statistics across all atlases.

    Returns:
        Dictionary with summary statistics
    """
    log("Generating summary statistics...")

    summary = {
        'atlases': {},
        'signatures': {},
        'cell_types': {}
    }

    # CIMA summary
    cima_cytosig = CIMA_DIR / 'CIMA_CytoSig_pseudobulk.h5ad'
    cima_secact = CIMA_DIR / 'CIMA_SecAct_pseudobulk.h5ad'

    if cima_cytosig.exists():
        adata = ad.read_h5ad(cima_cytosig)
        summary['atlases']['cima'] = {
            'n_samples': adata.n_vars,
            'n_cytosig_signatures': adata.n_obs,
            'cell_types': adata.var['cell_type'].nunique() if 'cell_type' in adata.var.columns else 'N/A'
        }
        del adata

    if cima_secact.exists():
        adata = ad.read_h5ad(cima_secact)
        summary['atlases']['cima']['n_secact_signatures'] = adata.n_obs
        del adata

    # Inflammation summary
    inflam_cytosig = INFLAM_DIR / 'main_CytoSig_pseudobulk.h5ad'
    inflam_secact = INFLAM_DIR / 'main_SecAct_pseudobulk.h5ad'

    if inflam_cytosig.exists():
        adata = ad.read_h5ad(inflam_cytosig)
        summary['atlases']['inflammation'] = {
            'n_samples': adata.n_vars,
            'n_cytosig_signatures': adata.n_obs,
            'cell_types': adata.var['cell_type'].nunique() if 'cell_type' in adata.var.columns else 'N/A'
        }
        del adata

    if inflam_secact.exists():
        adata = ad.read_h5ad(inflam_secact)
        summary['atlases']['inflammation']['n_secact_signatures'] = adata.n_obs
        del adata

    # scAtlas summary
    scatlas_normal = SCATLAS_DIR / 'normal_organ_signatures.csv'
    scatlas_cancer = SCATLAS_DIR / 'cancer_type_signatures.csv'

    if scatlas_normal.exists():
        df = pd.read_csv(scatlas_normal)
        summary['atlases']['scatlas_normal'] = {
            'n_organs': df['organ'].nunique(),
            'n_signatures': df['signature'].nunique(),
        }

    if scatlas_cancer.exists():
        df = pd.read_csv(scatlas_cancer)
        summary['atlases']['scatlas_cancer'] = {
            'n_cancer_types': df['organ'].nunique() if 'organ' in df.columns else 'N/A',
            'n_signatures': df['signature'].nunique(),
        }

    log(f"  Summary: {summary}")

    return summary


# ==============================================================================
# Main Analysis Pipeline
# ==============================================================================

def run_integrated_analysis():
    """Run integrated cross-atlas analysis."""
    log("=" * 60)
    log("INTEGRATED CROSS-ATLAS ANALYSIS")
    log("=" * 60)

    # Load CIMA results
    log("\nLoading CIMA results...")
    cima_cytosig_zscore, cima_cytosig_meta = load_activity_results(
        CIMA_DIR / 'CIMA_CytoSig_pseudobulk.h5ad'
    )
    cima_secact_zscore, cima_secact_meta = load_activity_results(
        CIMA_DIR / 'CIMA_SecAct_pseudobulk.h5ad'
    )

    # Load CIMA statistical results
    cima_corr_biochem = None
    cima_corr_biochem_path = CIMA_DIR / 'CIMA_correlation_biochemistry.csv'
    if cima_corr_biochem_path.exists():
        cima_corr_biochem = pd.read_csv(cima_corr_biochem_path)
        log(f"  Loaded CIMA biochemistry correlations: {len(cima_corr_biochem)} rows")

    # Load Inflammation results
    log("\nLoading Inflammation results...")
    inflam_cytosig_zscore, inflam_cytosig_meta = load_activity_results(
        INFLAM_DIR / 'main_CytoSig_pseudobulk.h5ad'
    )
    inflam_secact_zscore, inflam_secact_meta = load_activity_results(
        INFLAM_DIR / 'main_SecAct_pseudobulk.h5ad'
    )

    # Load Inflammation sample metadata
    inflam_sample_meta = None
    inflam_meta_path = Path('/data/Jiang_Lab/Data/Seongyong/Inflammation_Atlas/INFLAMMATION_ATLAS_afterQC_sampleMetadata.csv')
    if inflam_meta_path.exists():
        inflam_sample_meta = pd.read_csv(inflam_meta_path)
        log(f"  Loaded Inflammation sample metadata: {len(inflam_sample_meta)} samples")

    # Load Inflammation statistical results
    inflam_disease_diff = None
    inflam_disease_path = INFLAM_DIR / 'disease_differential.csv'
    if inflam_disease_path.exists():
        inflam_disease_diff = pd.read_csv(inflam_disease_path)
        log(f"  Loaded disease differential: {len(inflam_disease_diff)} rows")

    # Load scAtlas results
    log("\nLoading scAtlas results...")
    scatlas_tumor_adj = None
    scatlas_tumor_path = SCATLAS_DIR / 'cancer_tumor_vs_adjacent.csv'
    if scatlas_tumor_path.exists():
        scatlas_tumor_adj = pd.read_csv(scatlas_tumor_path)
        log(f"  Loaded tumor vs adjacent: {len(scatlas_tumor_adj)} rows")

    # Compare healthy profiles
    log("\n" + "=" * 60)
    log("HEALTHY PROFILE COMPARISON")
    log("=" * 60)

    if cima_cytosig_zscore is not None and inflam_cytosig_zscore is not None:
        healthy_comparison = compare_healthy_profiles(
            cima_cytosig_zscore, cima_cytosig_meta,
            inflam_cytosig_zscore, inflam_cytosig_meta,
            inflam_sample_meta
        )
        healthy_comparison['signature_type'] = 'CytoSig'

        if cima_secact_zscore is not None and inflam_secact_zscore is not None:
            healthy_secact = compare_healthy_profiles(
                cima_secact_zscore, cima_secact_meta,
                inflam_secact_zscore, inflam_secact_meta,
                inflam_sample_meta
            )
            healthy_secact['signature_type'] = 'SecAct'
            healthy_comparison = pd.concat([healthy_comparison, healthy_secact])

        healthy_comparison.to_csv(OUTPUT_DIR / 'healthy_profile_comparison.csv', index=False)
        log(f"Saved healthy comparison: {len(healthy_comparison)} rows")

    # Identify conserved signatures
    log("\n" + "=" * 60)
    log("CONSERVED SIGNATURES")
    log("=" * 60)

    conserved = identify_conserved_signatures(
        cima_corr_biochem,
        inflam_disease_diff,
        scatlas_tumor_adj
    )
    conserved.to_csv(OUTPUT_DIR / 'conserved_signatures.csv', index=False)
    log(f"Saved conserved signatures: {len(conserved)} rows")

    # Generate summary
    log("\n" + "=" * 60)
    log("SUMMARY STATISTICS")
    log("=" * 60)

    summary = generate_summary_statistics()

    # Save summary as JSON
    import json
    with open(OUTPUT_DIR / 'analysis_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    log(f"Saved summary to analysis_summary.json")

    # Create summary table
    summary_rows = []
    for atlas, stats in summary.get('atlases', {}).items():
        row = {'atlas': atlas}
        row.update(stats)
        summary_rows.append(row)

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(OUTPUT_DIR / 'atlas_summary_table.csv', index=False)
        log(f"Saved atlas summary table")

    # Aggregate key findings
    log("\n" + "=" * 60)
    log("KEY FINDINGS SUMMARY")
    log("=" * 60)

    findings = []

    # Top conserved signatures
    if len(conserved) > 0:
        top_conserved = conserved[conserved['n_atlases'] >= 2]['signature'].head(20).tolist()
        findings.append({
            'category': 'Conserved Signatures',
            'description': f"Top signatures found in >=2 atlases: {', '.join(top_conserved[:10])}"
        })

    # Healthy profile correlation
    if 'healthy_comparison' in dir() and len(healthy_comparison) > 0:
        avg_corr = healthy_comparison.groupby('cell_type')['profile_correlation'].first().mean()
        findings.append({
            'category': 'Healthy Profile Agreement',
            'description': f"Average correlation between CIMA and Inflammation healthy profiles: {avg_corr:.3f}"
        })

    # Save findings
    if findings:
        findings_df = pd.DataFrame(findings)
        findings_df.to_csv(OUTPUT_DIR / 'key_findings.csv', index=False)
        log(f"Saved key findings")

        for f in findings:
            log(f"  {f['category']}: {f['description']}")

    log("\nIntegrated analysis complete!")


# ==============================================================================
# Entry Point
# ==============================================================================

def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Integrated Cross-Atlas Analysis')
    parser.add_argument('--check-only', action='store_true',
                       help='Only check for available results')
    args = parser.parse_args()

    log("=" * 60)
    log("INTEGRATED CROSS-ATLAS ANALYSIS")
    log("=" * 60)
    log(f"Output: {OUTPUT_DIR}")

    # Check for available results
    log("\nChecking for available results...")
    available_files = {
        'CIMA CytoSig': CIMA_DIR / 'CIMA_CytoSig_pseudobulk.h5ad',
        'CIMA SecAct': CIMA_DIR / 'CIMA_SecAct_pseudobulk.h5ad',
        'CIMA Biochemistry Corr': CIMA_DIR / 'CIMA_correlation_biochemistry.csv',
        'Inflammation CytoSig': INFLAM_DIR / 'main_CytoSig_pseudobulk.h5ad',
        'Inflammation SecAct': INFLAM_DIR / 'main_SecAct_pseudobulk.h5ad',
        'Inflammation Disease Diff': INFLAM_DIR / 'disease_differential.csv',
        'scAtlas Organ Sigs': SCATLAS_DIR / 'normal_organ_signatures.csv',
        'scAtlas Tumor vs Adjacent': SCATLAS_DIR / 'cancer_tumor_vs_adjacent.csv',
    }

    for name, path in available_files.items():
        status = "Found" if path.exists() else "NOT FOUND"
        log(f"  {name}: {status}")

    if args.check_only:
        log("\nCheck complete (--check-only mode)")
        return

    start_time = time.time()

    run_integrated_analysis()

    total_time = time.time() - start_time
    log(f"\nTotal time: {total_time/60:.1f} minutes")
    log("Done!")


if __name__ == '__main__':
    main()
