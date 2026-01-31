#!/usr/bin/env python3
"""
scAtlas Immune Microenvironment Analysis
=========================================
Compute immune infiltration, T cell exhaustion, and CAF subtype analysis
from scAtlas cancer data for visualization.

This script reads the pre-computed activity data and metadata to generate
JSON files for the web visualization.
"""

import os
import sys
import json
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')

# ==============================================================================
# Configuration
# ==============================================================================

RESULTS_DIR = Path('/vf/users/parks34/projects/2secactpy/results/scatlas')
VIZ_DATA_DIR = Path('/vf/users/parks34/projects/2secactpy/visualization/data')

# Cell type classification patterns
IMMUNE_PATTERNS = {
    'T_cell': ['CD4T', 'CD8T', 'gdT', 'T_cell', 'Tcell', 'T.cells', 'T/NK', 'T_NK', 'MAIT'],
    'CD8_T': ['CD8T', 'CD8+'],
    'CD4_T': ['CD4T', 'CD4+'],
    'Treg': ['Treg', 'FOXP3', 'CD4T_10'],
    'NK': ['NK_', 'NK cell', 'NKT'],
    'B_cell': ['B_', 'B cell', 'B lymph', 'PlasmaB', 'Plasma'],
    'Macrophage': ['Mph_', 'Macrophage', 'TAM', 'McDC'],
    'Monocyte': ['Mo_', 'Mono', 'Monocyte'],
    'DC': ['DC_', 'Dendritic', 'cDC', 'pDC'],
    'Neutrophil': ['Neu_', 'Neutro'],
    'Mast': ['Mast', 'MAST'],
}

# Exhaustion-related patterns
EXHAUSTION_PATTERNS = {
    'exhausted': ['PDCD1', 'PD1', 'CTLA4', 'LAG3', 'TIM3', 'TIGIT', 'Tex', 'exhausted'],
    'cytotoxic': ['GZMB', 'PRF1', 'GNLY', 'GZMK'],
    'memory': ['Tem', 'Tcm', 'memory'],
    'naive': ['naive', 'Tn_', 'CCR7'],
}

# Fibroblast patterns for CAF analysis
FIBROBLAST_PATTERNS = ['Fb_', 'Fibroblast', 'CAF', 'Stromal', 'myoFb']

# CAF subtype markers (based on literature)
CAF_SUBTYPE_MARKERS = {
    'myCAF': ['ACTA2', 'TAGLN', 'MYL9', 'COL1A1', 'COL1A2', 'FN1'],  # ECM/contractile
    'iCAF': ['IL6', 'CXCL12', 'CCL2', 'PDGFRA', 'CFD', 'LUM'],  # Inflammatory
    'apCAF': ['HLA-DRA', 'CD74', 'HLA-DRB1', 'SLPI'],  # Antigen-presenting
}

def log(msg):
    print(f"[INFO] {msg}", flush=True)


# ==============================================================================
# Cell Type Classification
# ==============================================================================

def classify_cell_type(cell_type_str):
    """Classify a cell type string into immune categories."""
    cell_type_str = str(cell_type_str)
    categories = []

    for category, patterns in IMMUNE_PATTERNS.items():
        for pattern in patterns:
            if pattern.lower() in cell_type_str.lower():
                categories.append(category)
                break

    return categories if categories else ['Other']


def is_immune_cell(cell_type_str):
    """Check if cell type is an immune cell."""
    categories = classify_cell_type(cell_type_str)
    return 'Other' not in categories or len(categories) > 1


def is_fibroblast(cell_type_str):
    """Check if cell type is a fibroblast."""
    cell_type_str = str(cell_type_str).lower()
    return any(p.lower() in cell_type_str for p in FIBROBLAST_PATTERNS)


def get_exhaustion_state(cell_type_str):
    """Determine T cell exhaustion state from cell type annotation."""
    cell_type_str = str(cell_type_str)

    # First check if it's a T cell
    is_tcell = any(p.lower() in cell_type_str.lower()
                   for p in ['CD8T', 'CD4T', 'T_cell', 'Tcell'])
    if not is_tcell:
        return None

    # Check for exhaustion markers
    for pattern in EXHAUSTION_PATTERNS['exhausted']:
        if pattern.lower() in cell_type_str.lower():
            return 'exhausted'

    # Check for cytotoxic markers
    for pattern in EXHAUSTION_PATTERNS['cytotoxic']:
        if pattern.lower() in cell_type_str.lower():
            return 'cytotoxic'

    # Check for memory markers
    for pattern in EXHAUSTION_PATTERNS['memory']:
        if pattern.lower() in cell_type_str.lower():
            return 'memory'

    # Check for naive markers
    for pattern in EXHAUSTION_PATTERNS['naive']:
        if pattern.lower() in cell_type_str.lower():
            return 'naive'

    return 'other_tcell'


def get_tcell_subset(cell_type_str):
    """Determine T cell subset (CD4/CD8/other)."""
    cell_type_str = str(cell_type_str).lower()

    if 'cd8' in cell_type_str:
        return 'CD8'
    elif 'cd4' in cell_type_str:
        return 'CD4'
    elif 'treg' in cell_type_str or 'foxp3' in cell_type_str:
        return 'Treg'
    elif 'gdt' in cell_type_str:
        return 'gdT'
    elif 'mait' in cell_type_str:
        return 'MAIT'
    elif any(p in cell_type_str for p in ['t_cell', 'tcell', 't cell', 't_nk', 't/nk']):
        return 'T_mixed'
    return None


# ==============================================================================
# Data Loading
# ==============================================================================

def load_activity_data():
    """Load activity z-scores and metadata."""
    log("Loading activity data...")

    # Load CytoSig z-scores
    cytosig_path = RESULTS_DIR / 'cancer_cytosig_zscore.csv'
    cytosig_df = pd.read_csv(cytosig_path, index_col=0)
    log(f"  CytoSig: {cytosig_df.shape}")

    # Load SecAct z-scores (top proteins)
    secact_path = RESULTS_DIR / 'cancer_secact_zscore.csv'
    secact_df = pd.read_csv(secact_path, index_col=0)
    log(f"  SecAct: {secact_df.shape}")

    # Load metadata
    meta_path = RESULTS_DIR / 'cancer_aggregation_meta.csv'
    meta_df = pd.read_csv(meta_path, index_col=0)
    log(f"  Metadata: {meta_df.shape}")

    return cytosig_df, secact_df, meta_df


# ==============================================================================
# Immune Infiltration Analysis
# ==============================================================================

def compute_immune_infiltration(meta_df, cytosig_df):
    """
    Compute immune cell infiltration proportions and activities per cancer type.
    """
    log("Computing immune infiltration analysis...")

    results = []

    # Add immune classification to metadata
    meta_df = meta_df.copy()
    meta_df['is_immune'] = meta_df['cell_type'].apply(is_immune_cell)
    meta_df['immune_category'] = meta_df['cell_type'].apply(
        lambda x: classify_cell_type(x)[0] if classify_cell_type(x) else 'Other'
    )

    # Get valid columns (samples present in both meta and activity)
    valid_samples = list(set(meta_df.index) & set(cytosig_df.columns))
    meta_df = meta_df.loc[valid_samples]

    # Filter to tumor samples only
    if 'tissue' in meta_df.columns:
        tumor_meta = meta_df[meta_df['tissue'] == 'Tumor'].copy()
    else:
        tumor_meta = meta_df.copy()

    if len(tumor_meta) == 0:
        log("  Warning: No tumor samples found, using all samples")
        tumor_meta = meta_df.copy()

    log(f"  Tumor samples: {len(tumor_meta)}")

    # Get cancer types
    if 'cancerType' not in tumor_meta.columns:
        log("  Warning: No cancerType column")
        return []

    cancer_types = tumor_meta['cancerType'].dropna().unique()
    log(f"  Cancer types: {len(cancer_types)}")

    # Compute per cancer type
    for cancer in cancer_types:
        ct_mask = tumor_meta['cancerType'] == cancer
        ct_meta = tumor_meta[ct_mask]

        if len(ct_meta) < 5:
            continue

        # Cell counts
        total_cells = ct_meta['n_cells'].sum()
        immune_cells = ct_meta[ct_meta['is_immune']]['n_cells'].sum()
        immune_prop = immune_cells / total_cells if total_cells > 0 else 0

        # Immune composition
        immune_composition = {}
        for cat in IMMUNE_PATTERNS.keys():
            cat_mask = ct_meta['immune_category'] == cat
            cat_cells = ct_meta[cat_mask]['n_cells'].sum()
            immune_composition[cat] = cat_cells / total_cells if total_cells > 0 else 0

        # Mean activity for each signature (immune vs non-immune)
        ct_samples = ct_meta.index.tolist()
        ct_activities = cytosig_df[ct_samples].T
        ct_activities['is_immune'] = ct_meta.loc[ct_samples, 'is_immune'].values

        for sig in cytosig_df.index:
            immune_act = ct_activities[ct_activities['is_immune']][sig].mean()
            nonimmune_act = ct_activities[~ct_activities['is_immune']][sig].mean()

            # Correlation between immune proportion and activity
            # Using sample-level data
            sample_props = []
            sample_acts = []
            for idx in ct_samples:
                if idx in cytosig_df.columns:
                    sample_props.append(1 if ct_meta.loc[idx, 'is_immune'] else 0)
                    sample_acts.append(cytosig_df.loc[sig, idx])

            if len(sample_props) > 5:
                try:
                    corr, pval = stats.spearmanr(sample_props, sample_acts)
                except:
                    corr, pval = np.nan, np.nan
            else:
                corr, pval = np.nan, np.nan

            results.append({
                'cancer_type': cancer,
                'signature': sig,
                'immune_proportion': round(immune_prop, 4),
                'total_cells': int(total_cells),
                'immune_cells': int(immune_cells),
                'mean_immune_activity': round(immune_act, 4) if not np.isnan(immune_act) else 0,
                'mean_nonimmune_activity': round(nonimmune_act, 4) if not np.isnan(nonimmune_act) else 0,
                'immune_enrichment': round(immune_act - nonimmune_act, 4) if not (np.isnan(immune_act) or np.isnan(nonimmune_act)) else 0,
                'correlation': round(corr, 4) if not np.isnan(corr) else None,
                'pvalue': round(pval, 6) if not np.isnan(pval) else None,
                **{f'prop_{k}': round(v, 4) for k, v in immune_composition.items()}
            })

    log(f"  Generated {len(results)} infiltration records")
    return results


def generate_immune_composition_summary(meta_df):
    """Generate immune composition summary per cancer type."""
    log("Computing immune composition summary...")

    meta_df = meta_df.copy()
    meta_df['immune_category'] = meta_df['cell_type'].apply(
        lambda x: classify_cell_type(x)[0] if classify_cell_type(x) else 'Other'
    )

    # Filter to tumor samples
    if 'tissue' in meta_df.columns:
        tumor_meta = meta_df[meta_df['tissue'] == 'Tumor'].copy()
    else:
        tumor_meta = meta_df.copy()

    if 'cancerType' not in tumor_meta.columns:
        return []

    results = []
    cancer_types = tumor_meta['cancerType'].dropna().unique()

    for cancer in cancer_types:
        ct_meta = tumor_meta[tumor_meta['cancerType'] == cancer]
        total_cells = ct_meta['n_cells'].sum()

        if total_cells < 100:
            continue

        # Count by immune category
        for cat in list(IMMUNE_PATTERNS.keys()) + ['Other']:
            cat_mask = ct_meta['immune_category'] == cat
            cat_cells = ct_meta[cat_mask]['n_cells'].sum()
            n_samples = cat_mask.sum()

            if cat_cells > 0:
                results.append({
                    'cancer_type': cancer,
                    'cell_category': cat,
                    'cell_count': int(cat_cells),
                    'proportion': round(cat_cells / total_cells, 4),
                    'n_samples': int(n_samples),
                    'total_cells': int(total_cells)
                })

    log(f"  Generated {len(results)} composition records")
    return results


# ==============================================================================
# T Cell Exhaustion Analysis
# ==============================================================================

def compute_tcell_exhaustion(meta_df, cytosig_df):
    """
    Compute T cell exhaustion analysis comparing different T cell states.
    """
    log("Computing T cell exhaustion analysis...")

    meta_df = meta_df.copy()
    meta_df['tcell_subset'] = meta_df['cell_type'].apply(get_tcell_subset)
    meta_df['exhaustion_state'] = meta_df['cell_type'].apply(get_exhaustion_state)

    # Get valid samples
    valid_samples = list(set(meta_df.index) & set(cytosig_df.columns))
    meta_df = meta_df.loc[valid_samples]

    # Filter to T cells only
    tcell_meta = meta_df[meta_df['tcell_subset'].notna()].copy()
    log(f"  T cell samples: {len(tcell_meta)}")

    if len(tcell_meta) < 10:
        log("  Warning: Too few T cell samples")
        return [], []

    results = []

    # Group by exhaustion state and cancer type
    cancer_types = tcell_meta['cancerType'].dropna().unique() if 'cancerType' in tcell_meta.columns else ['All']

    for cancer in cancer_types:
        if cancer != 'All':
            ct_meta = tcell_meta[tcell_meta['cancerType'] == cancer]
        else:
            ct_meta = tcell_meta

        if len(ct_meta) < 5:
            continue

        # Group by exhaustion state
        exhaustion_states = ct_meta['exhaustion_state'].dropna().unique()

        for state in exhaustion_states:
            state_meta = ct_meta[ct_meta['exhaustion_state'] == state]
            state_samples = state_meta.index.tolist()

            if len(state_samples) < 2:
                continue

            # Get activities for this state
            state_activities = cytosig_df[state_samples]

            for sig in cytosig_df.index:
                mean_act = state_activities.loc[sig].mean()
                std_act = state_activities.loc[sig].std()
                n_cells = state_meta['n_cells'].sum()

                results.append({
                    'cancer_type': cancer,
                    'exhaustion_state': state,
                    'signature': sig,
                    'mean_activity': round(mean_act, 4) if not np.isnan(mean_act) else 0,
                    'std_activity': round(std_act, 4) if not np.isnan(std_act) else 0,
                    'n_samples': len(state_samples),
                    'n_cells': int(n_cells)
                })

    # Compute exhausted vs non-exhausted comparison
    comparison_results = []

    for cancer in cancer_types:
        if cancer != 'All':
            ct_meta = tcell_meta[tcell_meta['cancerType'] == cancer]
        else:
            ct_meta = tcell_meta

        exhausted_meta = ct_meta[ct_meta['exhaustion_state'] == 'exhausted']
        nonexhausted_meta = ct_meta[ct_meta['exhaustion_state'].isin(['cytotoxic', 'memory', 'other_tcell'])]

        if len(exhausted_meta) < 2 or len(nonexhausted_meta) < 2:
            continue

        exhausted_samples = exhausted_meta.index.tolist()
        nonexhausted_samples = nonexhausted_meta.index.tolist()

        for sig in cytosig_df.index:
            exhausted_vals = cytosig_df.loc[sig, exhausted_samples].values
            nonexhausted_vals = cytosig_df.loc[sig, nonexhausted_samples].values

            mean_exhausted = np.mean(exhausted_vals)
            mean_nonexhausted = np.mean(nonexhausted_vals)
            log2fc = np.log2((mean_exhausted + 0.01) / (mean_nonexhausted + 0.01))

            try:
                stat, pval = stats.mannwhitneyu(exhausted_vals, nonexhausted_vals, alternative='two-sided')
            except:
                stat, pval = np.nan, np.nan

            comparison_results.append({
                'cancer_type': cancer,
                'signature': sig,
                'mean_exhausted': round(mean_exhausted, 4),
                'mean_nonexhausted': round(mean_nonexhausted, 4),
                'log2fc': round(log2fc, 4) if not np.isnan(log2fc) else 0,
                'pvalue': round(pval, 6) if not np.isnan(pval) else None,
                'n_exhausted': len(exhausted_samples),
                'n_nonexhausted': len(nonexhausted_samples)
            })

    log(f"  Generated {len(results)} state records, {len(comparison_results)} comparison records")
    return results, comparison_results


def compute_exhaustion_by_subset(meta_df, cytosig_df):
    """Compute exhaustion analysis by T cell subset (CD4, CD8, etc.)."""
    log("Computing exhaustion by T cell subset...")

    meta_df = meta_df.copy()
    meta_df['tcell_subset'] = meta_df['cell_type'].apply(get_tcell_subset)
    meta_df['exhaustion_state'] = meta_df['cell_type'].apply(get_exhaustion_state)

    valid_samples = list(set(meta_df.index) & set(cytosig_df.columns))
    meta_df = meta_df.loc[valid_samples]

    tcell_meta = meta_df[meta_df['tcell_subset'].notna()].copy()

    results = []

    for subset in ['CD8', 'CD4', 'Treg']:
        subset_meta = tcell_meta[tcell_meta['tcell_subset'] == subset]

        if len(subset_meta) < 5:
            continue

        # Group by cancer type
        cancer_types = subset_meta['cancerType'].dropna().unique() if 'cancerType' in subset_meta.columns else ['All']

        for cancer in cancer_types:
            if cancer != 'All':
                ct_meta = subset_meta[subset_meta['cancerType'] == cancer]
            else:
                ct_meta = subset_meta

            if len(ct_meta) < 3:
                continue

            ct_samples = ct_meta.index.tolist()
            n_cells = ct_meta['n_cells'].sum()

            # Count exhausted vs non-exhausted
            n_exhausted = ct_meta[ct_meta['exhaustion_state'] == 'exhausted']['n_cells'].sum()
            exhaustion_rate = n_exhausted / n_cells if n_cells > 0 else 0

            for sig in cytosig_df.index:
                mean_act = cytosig_df.loc[sig, ct_samples].mean()

                results.append({
                    'cancer_type': cancer,
                    'tcell_subset': subset,
                    'signature': sig,
                    'mean_activity': round(mean_act, 4) if not np.isnan(mean_act) else 0,
                    'exhaustion_rate': round(exhaustion_rate, 4),
                    'n_cells': int(n_cells),
                    'n_samples': len(ct_samples)
                })

    log(f"  Generated {len(results)} subset records")
    return results


# ==============================================================================
# CAF Analysis
# ==============================================================================

def compute_caf_analysis(meta_df, cytosig_df):
    """
    Compute cancer-associated fibroblast analysis.
    """
    log("Computing CAF analysis...")

    meta_df = meta_df.copy()
    meta_df['is_fibroblast'] = meta_df['cell_type'].apply(is_fibroblast)

    valid_samples = list(set(meta_df.index) & set(cytosig_df.columns))
    meta_df = meta_df.loc[valid_samples]

    # Filter to fibroblasts
    fb_meta = meta_df[meta_df['is_fibroblast']].copy()
    log(f"  Fibroblast samples: {len(fb_meta)}")

    if len(fb_meta) < 5:
        log("  Warning: Too few fibroblast samples")
        return [], []

    results = []

    # Get unique fibroblast subtypes
    fb_subtypes = fb_meta['cell_type'].unique()
    log(f"  Fibroblast subtypes: {len(fb_subtypes)}")

    # Overall CAF activity per cancer type
    cancer_types = fb_meta['cancerType'].dropna().unique() if 'cancerType' in fb_meta.columns else ['All']

    for cancer in cancer_types:
        if cancer != 'All':
            ct_meta = fb_meta[fb_meta['cancerType'] == cancer]
        else:
            ct_meta = fb_meta

        if len(ct_meta) < 2:
            continue

        ct_samples = ct_meta.index.tolist()
        n_cells = ct_meta['n_cells'].sum()

        for sig in cytosig_df.index:
            mean_act = cytosig_df.loc[sig, ct_samples].mean()
            std_act = cytosig_df.loc[sig, ct_samples].std()

            results.append({
                'cancer_type': cancer,
                'caf_subtype': 'All_CAF',
                'signature': sig,
                'mean_activity': round(mean_act, 4) if not np.isnan(mean_act) else 0,
                'std_activity': round(std_act, 4) if not np.isnan(std_act) else 0,
                'n_cells': int(n_cells),
                'n_samples': len(ct_samples)
            })

    # Per fibroblast subtype analysis
    subtype_results = []

    for subtype in fb_subtypes:
        subtype_meta = fb_meta[fb_meta['cell_type'] == subtype]

        if len(subtype_meta) < 2:
            continue

        # Try to classify CAF subtype based on name
        subtype_lower = str(subtype).lower()
        caf_class = 'Other'
        if any(p.lower() in subtype_lower for p in ['cxcl12', 'il6', 'ccl2', 'icaf', 'inflammatory']):
            caf_class = 'iCAF'
        elif any(p.lower() in subtype_lower for p in ['acta2', 'myl', 'col1', 'mycaf', 'myo']):
            caf_class = 'myCAF'
        elif any(p.lower() in subtype_lower for p in ['hla', 'cd74', 'apcaf', 'mhc']):
            caf_class = 'apCAF'

        cancer_types = subtype_meta['cancerType'].dropna().unique() if 'cancerType' in subtype_meta.columns else ['All']

        for cancer in cancer_types:
            if cancer != 'All':
                ct_meta = subtype_meta[subtype_meta['cancerType'] == cancer]
            else:
                ct_meta = subtype_meta

            if len(ct_meta) < 1:
                continue

            ct_samples = ct_meta.index.tolist()
            n_cells = ct_meta['n_cells'].sum()

            for sig in cytosig_df.index:
                if len(ct_samples) > 0 and ct_samples[0] in cytosig_df.columns:
                    mean_act = cytosig_df.loc[sig, ct_samples].mean()
                else:
                    mean_act = np.nan

                subtype_results.append({
                    'cancer_type': cancer,
                    'caf_subtype': subtype,
                    'caf_class': caf_class,
                    'signature': sig,
                    'mean_activity': round(mean_act, 4) if not np.isnan(mean_act) else 0,
                    'n_cells': int(n_cells),
                    'n_samples': len(ct_samples)
                })

    log(f"  Generated {len(results)} overall, {len(subtype_results)} subtype records")
    return results, subtype_results


def compute_caf_proportions(meta_df):
    """Compute CAF proportions per cancer type."""
    log("Computing CAF proportions...")

    meta_df = meta_df.copy()
    meta_df['is_fibroblast'] = meta_df['cell_type'].apply(is_fibroblast)

    # Filter to tumor samples
    if 'tissue' in meta_df.columns:
        tumor_meta = meta_df[meta_df['tissue'] == 'Tumor'].copy()
    else:
        tumor_meta = meta_df.copy()

    results = []

    if 'cancerType' not in tumor_meta.columns:
        return results

    cancer_types = tumor_meta['cancerType'].dropna().unique()

    for cancer in cancer_types:
        ct_meta = tumor_meta[tumor_meta['cancerType'] == cancer]
        total_cells = ct_meta['n_cells'].sum()

        if total_cells < 100:
            continue

        fb_cells = ct_meta[ct_meta['is_fibroblast']]['n_cells'].sum()
        fb_proportion = fb_cells / total_cells

        # Count unique fibroblast subtypes
        fb_subtypes = ct_meta[ct_meta['is_fibroblast']]['cell_type'].unique()

        results.append({
            'cancer_type': cancer,
            'caf_proportion': round(fb_proportion, 4),
            'caf_cells': int(fb_cells),
            'total_cells': int(total_cells),
            'n_subtypes': len(fb_subtypes)
        })

    log(f"  Generated {len(results)} proportion records")
    return results


# ==============================================================================
# JSON Export
# ==============================================================================

def save_json(data, filename):
    """Save data to JSON file."""
    filepath = VIZ_DATA_DIR / filename
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, allow_nan=False)
    log(f"  Saved {filename}: {len(data.get('data', data))} records")


def main():
    """Main entry point."""
    log("=" * 60)
    log("scAtlas Immune Microenvironment Analysis")
    log("=" * 60)

    # Load data
    cytosig_df, secact_df, meta_df = load_activity_data()

    # Immune Infiltration Analysis
    log("\n--- Immune Infiltration ---")
    infiltration_data = compute_immune_infiltration(meta_df, cytosig_df)
    composition_data = generate_immune_composition_summary(meta_df)

    save_json({
        'data': infiltration_data,
        'composition': composition_data,
        'signatures': list(cytosig_df.index),
        'cancer_types': list(meta_df['cancerType'].dropna().unique()) if 'cancerType' in meta_df.columns else []
    }, 'immune_infiltration.json')

    # T Cell Exhaustion Analysis
    log("\n--- T Cell Exhaustion ---")
    exhaustion_state_data, exhaustion_comparison = compute_tcell_exhaustion(meta_df, cytosig_df)
    exhaustion_subset_data = compute_exhaustion_by_subset(meta_df, cytosig_df)

    save_json({
        'data': exhaustion_state_data,
        'comparison': exhaustion_comparison,
        'by_subset': exhaustion_subset_data,
        'signatures': list(cytosig_df.index),
        'exhaustion_states': ['exhausted', 'cytotoxic', 'memory', 'naive', 'other_tcell'],
        'tcell_subsets': ['CD8', 'CD4', 'Treg', 'gdT', 'MAIT']
    }, 'exhaustion.json')

    # CAF Analysis
    log("\n--- CAF Analysis ---")
    caf_overall, caf_subtypes = compute_caf_analysis(meta_df, cytosig_df)
    caf_proportions = compute_caf_proportions(meta_df)

    save_json({
        'data': caf_overall,
        'subtypes': caf_subtypes,
        'proportions': caf_proportions,
        'signatures': list(cytosig_df.index),
        'caf_classes': ['myCAF', 'iCAF', 'apCAF', 'Other']
    }, 'caf_signatures.json')

    log("\n" + "=" * 60)
    log("Analysis complete!")
    log("=" * 60)


if __name__ == '__main__':
    main()
