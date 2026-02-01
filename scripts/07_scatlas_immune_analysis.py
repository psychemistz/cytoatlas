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
# Updated to support both cellType1 (original) and subCluster (standardized) naming
# subCluster format: PREFIX##_Subtype_MARKER (e.g., CD8T02_Tem_GZMK, B12_Plasma_IGHA2)
IMMUNE_PATTERNS = {
    'T_cell': ['CD4T', 'CD8T', 'gdT', 'T_cell', 'Tcell', 'T.cells', 'T/NK', 'T_NK', 'MAIT', 'T cells'],
    'CD8_T': ['CD8T', 'CD8+', 'CD8 T'],
    'CD4_T': ['CD4T', 'CD4+', 'CD4 T'],
    'Treg': ['Treg', 'FOXP3', 'CD4T08'],  # CD4T08_Treg_FOXP3
    'NK': ['NK_', 'NK cell', 'NKT', 'NK-', 'I01_', 'I02_', 'I03_', 'I04_', 'I05_', 'I06_', 'I07_', 'I10_', 'CD16hiNK', 'CD16loNK'],  # I* are ILC/NK
    'ILC': ['I08_', 'I09_', 'ILC', 'gdT'],  # I08=ILC3, I09=gdT
    'B_cell': ['B01_', 'B02_', 'B03_', 'B04_', 'B05_', 'B06_', 'B07_', 'B08_', 'B09_', 'B10_', 'B11_', 'B12_', 'B13_', 'B14_',
               'B cell', 'B lymph', 'PlasmaB', 'Plasma', 'B-cell', 'B_cell'],  # B## subCluster prefixes
    'Macrophage': ['Mph_', 'Macrophage', 'TAM', 'McDC', 'Mac_', 'Mac ',
                   'M07_', 'M08_', 'M09_', 'M10_', 'M11_', 'M12_', 'M17_', 'M18_', 'M19_'],  # M07-M12, M17-M19 are Mph
    'Monocyte': ['Mo_', 'Mono', 'Monocyte', 'M05_', 'M06_'],  # M05=CD14 Mo, M06=FCGR3A Mo
    'DC': ['DC_', 'Dendritic', 'cDC', 'pDC', 'CD1C', 'CLEC9A', 'ASDC', 'LC_', 'Langerhans',
           'M01_', 'M02_', 'M03_', 'M04_', 'M16_'],  # M01-M04, M16 are DCs
    'Neutrophil': ['Neu_', 'Neutro', 'M13_', 'M14_'],  # M13=immNeu, M14=mNeu
    'Mast': ['Mast', 'MAST', 'M15_'],  # M15=Mast
    'Myeloid': ['Myeloid', 'MDSC'],  # Generic myeloid cells
}

# Exhaustion-related patterns
EXHAUSTION_PATTERNS = {
    'exhausted': ['PDCD1', 'PD1', 'CTLA4', 'LAG3', 'TIM3', 'TIGIT', 'Tex', 'exhausted'],
    'cytotoxic': ['GZMB', 'PRF1', 'GNLY', 'GZMK'],
    'memory': ['Tem', 'Tcm', 'memory'],
    'naive': ['naive', 'Tn_', 'CCR7'],
}

# Fibroblast patterns for CAF analysis
# S01-S12 are fibroblasts, S22-S24 are CAF subtypes
FIBROBLAST_PATTERNS = [
    'S01_', 'S02_', 'S03_', 'S04_', 'S05_', 'S06_', 'S07_', 'S08_', 'S09_', 'S10_', 'S11_', 'S12_',
    'S22_', 'S23_', 'S24_',  # iCAF, myCAF, apCAF
    'Fb_', 'Fibroblast', 'CAF', 'Stromal', 'myoFb', 'iCAF', 'myCAF', 'apCAF',
]

# Malignant/Tumor cell patterns
MALIGNANT_PATTERNS = [
    'Malig', 'Tumor', 'Cancer', 'Carcinoma',
    'Epi_',  # Cancer-specific epithelial (e.g., Epi_KIRC)
    'Epithelial',  # Often tumor cells in cancer samples
]

# Stromal cell patterns (non-immune, non-malignant)
# Updated to support subCluster naming: E*=Endothelial, S*=Stromal (Fb, Pericyte, SMC, CAF)
STROMAL_PATTERNS = [
    # subCluster prefixes (must check before generic patterns)
    'E01_', 'E02_', 'E03_', 'E04_', 'E05_', 'E06_',  # Endothelial cells
    'S01_', 'S02_', 'S03_', 'S04_', 'S05_', 'S06_', 'S07_', 'S08_', 'S09_', 'S10_',
    'S11_', 'S12_', 'S13_', 'S14_', 'S15_', 'S16_', 'S17_', 'S18_', 'S19_', 'S20_',
    'S21_', 'S22_', 'S23_', 'S24_',  # Stromal cells (Fb, Pericyte, SMC, CAF, etc.)
    # Original patterns
    'Fb_', 'Fibroblast', 'CAF', 'Stromal', 'myoFb',
    'Endo', 'Endothelial', 'Pericyte', 'Smooth',
    'Adipocyte', 'Mesenchymal',
    'EC_',  # Endothelial cell clusters (e.g., EC_05_KDR)
    'Mu_',  # Muscle/smooth muscle cells (e.g., Mu_01_MYH11)
    'Stellate',  # Hepatic stellate cells
    'Myocyte',  # Muscle cells
    'iCAF', 'myCAF', 'apCAF',  # CAF subtypes
]

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
    """Classify a cell type string into immune categories.

    Handles both original cellType1 annotations and standardized subCluster names.
    subCluster format: PREFIX##_Subtype_MARKER (e.g., CD8T02_Tem_GZMK, B12_Plasma_IGHA2)
    """
    cell_type_str = str(cell_type_str)
    categories = []

    # Handle single-letter or exact match cases
    cell_lower = cell_type_str.lower().strip()
    if cell_lower in ['t', 't cell', 't-cell']:
        return ['T_cell']
    if cell_lower in ['b', 'b cell', 'b-cell']:
        return ['B_cell']
    if cell_lower in ['nk', 'nk cell']:
        return ['NK']
    if cell_lower in ['mac', 'macrophage']:
        return ['Macrophage']
    if cell_lower in ['lc']:
        return ['DC']  # Langerhans cells are DCs

    # For subCluster format, check prefix patterns that must match at start
    # These are more specific and should be checked first
    subcluster_prefixes = {
        'B_cell': ['b01_', 'b02_', 'b03_', 'b04_', 'b05_', 'b06_', 'b07_', 'b08_', 'b09_', 'b10_', 'b11_', 'b12_', 'b13_', 'b14_'],
        'T_cell': ['cd4t', 'cd8t'],
        'NK': ['i01_', 'i02_', 'i03_', 'i04_', 'i05_', 'i06_', 'i07_', 'i10_'],
        'ILC': ['i08_', 'i09_'],
        'Macrophage': ['m07_', 'm08_', 'm09_', 'm10_', 'm11_', 'm12_', 'm17_', 'm18_', 'm19_'],
        'Monocyte': ['m05_', 'm06_'],
        'DC': ['m01_', 'm02_', 'm03_', 'm04_', 'm16_'],
        'Neutrophil': ['m13_', 'm14_'],
        'Mast': ['m15_'],
    }

    for category, prefixes in subcluster_prefixes.items():
        for prefix in prefixes:
            if cell_lower.startswith(prefix):
                categories.append(category)
                break
        if categories:
            break

    # If no subCluster prefix matched, try generic patterns
    if not categories:
        for category, patterns in IMMUNE_PATTERNS.items():
            for pattern in patterns:
                # Skip subCluster prefix patterns (already checked above)
                if pattern.endswith('_') and len(pattern) <= 4:
                    continue
                if pattern.lower() in cell_type_str.lower():
                    categories.append(category)
                    break

    return categories if categories else ['Other']


def is_malignant_cell(cell_type_str):
    """Check if cell type is likely malignant/tumor cell."""
    cell_type_str = str(cell_type_str).lower()
    for pattern in MALIGNANT_PATTERNS:
        if pattern.lower() in cell_type_str:
            return True
    return False


def is_stromal_cell(cell_type_str):
    """Check if cell type is stromal (fibroblast, endothelial, etc.).

    Handles subCluster prefixes: E*=Endothelial, S*=Stromal
    """
    cell_type_str = str(cell_type_str).lower()

    # Check subCluster prefixes first (must match at start)
    stromal_prefixes = ['e01_', 'e02_', 'e03_', 'e04_', 'e05_', 'e06_',
                        's01_', 's02_', 's03_', 's04_', 's05_', 's06_', 's07_', 's08_', 's09_', 's10_',
                        's11_', 's12_', 's13_', 's14_', 's15_', 's16_', 's17_', 's18_', 's19_', 's20_',
                        's21_', 's22_', 's23_', 's24_']
    for prefix in stromal_prefixes:
        if cell_type_str.startswith(prefix):
            return True

    # Fall back to generic pattern matching
    for pattern in STROMAL_PATTERNS:
        # Skip prefix patterns (already checked above)
        if pattern.endswith('_') and len(pattern) <= 4:
            continue
        if pattern.lower() in cell_type_str:
            return True
    return False


def classify_tme_category(cell_type_str, is_immune):
    """Classify cell into TME categories: Malignant, Immune, Stromal, Other."""
    if is_immune:
        return 'Immune'
    elif is_malignant_cell(cell_type_str):
        return 'Malignant'
    elif is_stromal_cell(cell_type_str):
        return 'Stromal'
    else:
        return 'Other'


def is_immune_cell(cell_type_str):
    """Check if cell type is an immune cell."""
    categories = classify_cell_type(cell_type_str)
    return 'Other' not in categories or len(categories) > 1


def is_fibroblast(cell_type_str):
    """Check if cell type is a fibroblast.

    Handles subCluster prefixes: S01-S12=Fibroblasts, S22-S24=CAF subtypes
    """
    cell_type_str = str(cell_type_str).lower()

    # Check subCluster prefixes first
    fb_prefixes = ['s01_', 's02_', 's03_', 's04_', 's05_', 's06_', 's07_', 's08_', 's09_', 's10_', 's11_', 's12_',
                   's22_', 's23_', 's24_']  # S22=iCAF, S23=myCAF, S24=apCAF
    for prefix in fb_prefixes:
        if cell_type_str.startswith(prefix):
            return True

    # Fall back to generic pattern matching
    return any(p.lower() in cell_type_str for p in FIBROBLAST_PATTERNS if not (p.endswith('_') and len(p) <= 4))


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

def compute_immune_infiltration(meta_df, cytosig_df, secact_df=None):
    """
    Compute immune cell infiltration proportions and activities per cancer type.

    Now supports both CytoSig and SecAct signatures with signature_type field.

    Enhanced to include:
    - Immune composition within immune cells (not total)
    - CD8:Treg ratio (immune activation vs suppression)
    - T:Myeloid ratio (TME polarization)
    - Data quality flags
    """
    log("Computing immune infiltration analysis...")

    results = []

    # Add immune classification to metadata
    meta_df = meta_df.copy()
    meta_df['is_immune'] = meta_df['cell_type'].apply(is_immune_cell)
    meta_df['immune_category'] = meta_df['cell_type'].apply(
        lambda x: classify_cell_type(x)[0] if classify_cell_type(x) else 'Other'
    )

    # Add more detailed immune categories
    meta_df['all_immune_categories'] = meta_df['cell_type'].apply(classify_cell_type)

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

        # Immune composition (proportion of total TME)
        immune_composition = {}
        for cat in IMMUNE_PATTERNS.keys():
            cat_mask = ct_meta['immune_category'] == cat
            cat_cells = ct_meta[cat_mask]['n_cells'].sum()
            immune_composition[cat] = cat_cells / total_cells if total_cells > 0 else 0

        # Immune composition WITHIN immune cells (more meaningful)
        immune_within = {}
        for cat in IMMUNE_PATTERNS.keys():
            cat_mask = ct_meta['immune_category'] == cat
            cat_cells = ct_meta[cat_mask]['n_cells'].sum()
            immune_within[cat] = cat_cells / immune_cells if immune_cells > 0 else 0

        # Compute clinically relevant ratios
        # CD8:Treg ratio (high = immune active, low = immunosuppressed)
        cd8_cells = sum(ct_meta[ct_meta['cell_type'].str.contains('CD8', case=False, na=False)]['n_cells'])
        treg_cells = sum(ct_meta[ct_meta['cell_type'].str.contains('Treg|FOXP3', case=False, na=False)]['n_cells'])
        cd8_treg_ratio = cd8_cells / treg_cells if treg_cells > 0 else (float('inf') if cd8_cells > 0 else 0)

        # T cell vs Myeloid ratio (TME polarization)
        t_cells = sum(ct_meta[ct_meta['immune_category'].isin(['T_cell', 'CD8_T', 'CD4_T', 'Treg'])]['n_cells'])
        myeloid_cells = sum(ct_meta[ct_meta['immune_category'].isin(['Macrophage', 'Monocyte', 'DC', 'Neutrophil'])]['n_cells'])
        t_myeloid_ratio = t_cells / myeloid_cells if myeloid_cells > 0 else (float('inf') if t_cells > 0 else 0)

        # Data quality flags
        n_samples = len(ct_meta)
        has_cd8_annotation = cd8_cells > 0 or any('CD8' in str(x) for x in ct_meta['cell_type'])
        has_treg_annotation = treg_cells > 0 or any('Treg' in str(x) or 'FOXP3' in str(x) for x in ct_meta['cell_type'])
        data_quality = 'good' if n_samples >= 10 and has_cd8_annotation else ('partial' if n_samples >= 5 else 'limited')

        # Helper function to process signatures from a dataframe
        def process_signatures(activity_df, signature_type):
            ct_samples_valid = [s for s in ct_samples if s in activity_df.columns]
            if len(ct_samples_valid) == 0:
                return []

            ct_activities = activity_df[ct_samples_valid].T
            ct_activities['is_immune'] = ct_meta.loc[ct_samples_valid, 'is_immune'].values

            sig_results = []
            for sig in activity_df.index:
                immune_act = ct_activities[ct_activities['is_immune']][sig].mean()
                nonimmune_act = ct_activities[~ct_activities['is_immune']][sig].mean()

                # Correlation between immune proportion and activity
                sample_props = []
                sample_acts = []
                for idx in ct_samples_valid:
                    sample_props.append(1 if ct_meta.loc[idx, 'is_immune'] else 0)
                    sample_acts.append(activity_df.loc[sig, idx])

                if len(sample_props) > 5:
                    try:
                        corr, pval = stats.spearmanr(sample_props, sample_acts)
                    except:
                        corr, pval = np.nan, np.nan
                else:
                    corr, pval = np.nan, np.nan

                sig_results.append({
                    'cancer_type': cancer,
                    'signature': sig,
                    'signature_type': signature_type,
                    'immune_proportion': round(immune_prop, 4),
                    'total_cells': int(total_cells),
                    'immune_cells': int(immune_cells),
                    'n_samples': n_samples,
                    'mean_immune_activity': round(immune_act, 4) if not np.isnan(immune_act) else 0,
                    'mean_nonimmune_activity': round(nonimmune_act, 4) if not np.isnan(nonimmune_act) else 0,
                    'immune_enrichment': round(immune_act - nonimmune_act, 4) if not (np.isnan(immune_act) or np.isnan(nonimmune_act)) else 0,
                    'correlation': round(corr, 4) if not np.isnan(corr) else None,
                    'pvalue': round(pval, 6) if not np.isnan(pval) else None,
                    'cd8_treg_ratio': round(cd8_treg_ratio, 2) if cd8_treg_ratio != float('inf') else None,
                    't_myeloid_ratio': round(t_myeloid_ratio, 2) if t_myeloid_ratio != float('inf') else None,
                    'data_quality': data_quality,
                    **{f'prop_{k}': round(v, 4) for k, v in immune_composition.items()},
                    **{f'immune_{k}': round(v, 4) for k, v in immune_within.items()}
                })
            return sig_results

        # Process CytoSig signatures
        ct_samples = ct_meta.index.tolist()
        results.extend(process_signatures(cytosig_df, 'CytoSig'))

        # Process SecAct signatures if provided
        if secact_df is not None:
            results.extend(process_signatures(secact_df, 'SecAct'))

    log(f"  Generated {len(results)} infiltration records")
    return results


def generate_tme_summary(meta_df):
    """
    Generate comprehensive TME (Tumor Microenvironment) summary per cancer type.

    Includes:
    - Full TME composition: Malignant, Immune, Stromal, Other
    - Immune composition WITHIN immune cells
    - Clinical ratios (CD8:Treg, T:Myeloid)
    - Data quality indicators
    """
    log("Computing TME summary...")

    meta_df = meta_df.copy()
    meta_df['is_immune'] = meta_df['cell_type'].apply(is_immune_cell)
    meta_df['is_malignant'] = meta_df['cell_type'].apply(is_malignant_cell)
    meta_df['is_stromal'] = meta_df['cell_type'].apply(is_stromal_cell)
    meta_df['immune_category'] = meta_df['cell_type'].apply(
        lambda x: classify_cell_type(x)[0] if classify_cell_type(x) else 'Other'
    )
    # TME category (mutually exclusive: Immune > Malignant > Stromal > Other)
    meta_df['tme_category'] = meta_df.apply(
        lambda row: classify_tme_category(row['cell_type'], row['is_immune']), axis=1
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

        # TME composition (mutually exclusive categories)
        immune_cells = ct_meta[ct_meta['tme_category'] == 'Immune']['n_cells'].sum()
        malignant_cells = ct_meta[ct_meta['tme_category'] == 'Malignant']['n_cells'].sum()
        stromal_cells = ct_meta[ct_meta['tme_category'] == 'Stromal']['n_cells'].sum()
        other_cells = ct_meta[ct_meta['tme_category'] == 'Other']['n_cells'].sum()

        # Immune composition WITHIN immune cells
        immune_within = {}
        for cat in IMMUNE_PATTERNS.keys():
            cat_mask = ct_meta['immune_category'] == cat
            cat_cells = ct_meta[cat_mask]['n_cells'].sum()
            immune_within[cat] = round(cat_cells / immune_cells, 4) if immune_cells > 0 else 0

        # Clinical ratios
        cd8_cells = sum(ct_meta[ct_meta['cell_type'].str.contains('CD8', case=False, na=False)]['n_cells'])
        treg_cells = sum(ct_meta[ct_meta['cell_type'].str.contains('Treg|FOXP3', case=False, na=False)]['n_cells'])
        t_cells = sum(ct_meta[ct_meta['immune_category'].isin(['T_cell', 'CD8_T', 'CD4_T', 'Treg'])]['n_cells'])
        myeloid_cells = sum(ct_meta[ct_meta['immune_category'].isin(['Macrophage', 'Monocyte', 'DC', 'Neutrophil'])]['n_cells'])

        cd8_treg_ratio = cd8_cells / treg_cells if treg_cells > 0 else None
        t_myeloid_ratio = t_cells / myeloid_cells if myeloid_cells > 0 else None

        # Data quality assessment
        n_samples = len(ct_meta)
        n_donors = ct_meta['donorID'].nunique() if 'donorID' in ct_meta.columns else n_samples
        has_detailed_annotation = cd8_cells > 0 or treg_cells > 0
        has_malignant_annotation = malignant_cells > 0

        # Check for unusual TME compositions and add study-specific notes
        sample_note = None
        # Map cancer types with unusual compositions to their study IDs
        study_notes = {
            'KIRP': ('GSE152938', 'no malignant cells in primary tumor samples'),
            'TGCT': ('GSE197778', 'no malignant cells in primary tumor samples'),
            'NET': ('GSE140312', 'no immune cells (single donor)')
        }
        if cancer in study_notes:
            study_id, note = study_notes[cancer]
            # Only add note if the condition matches
            if cancer in ['KIRP', 'TGCT'] and malignant_cells == 0:
                sample_note = f'{study_id}: {note}'
            elif cancer == 'NET' and immune_cells == 0:
                sample_note = f'{study_id}: {note}'

        if n_samples >= 20 and has_detailed_annotation:
            data_quality = 'high'
        elif n_samples >= 10:
            data_quality = 'medium'
        elif n_samples >= 5:
            data_quality = 'low'
        else:
            data_quality = 'very_low'

        results.append({
            'cancer_type': cancer,
            'total_cells': int(total_cells),
            # TME composition (proportions)
            'malignant_cells': int(malignant_cells),
            'immune_cells': int(immune_cells),
            'stromal_cells': int(stromal_cells),
            'other_cells': int(other_cells),
            'malignant_proportion': round(malignant_cells / total_cells, 4) if total_cells > 0 else 0,
            'immune_proportion': round(immune_cells / total_cells, 4) if total_cells > 0 else 0,
            'stromal_proportion': round(stromal_cells / total_cells, 4) if total_cells > 0 else 0,
            'other_proportion': round(other_cells / total_cells, 4) if total_cells > 0 else 0,
            'n_samples': n_samples,
            'n_donors': n_donors,
            # Clinical ratios
            'cd8_treg_ratio': round(cd8_treg_ratio, 2) if cd8_treg_ratio is not None else None,
            't_myeloid_ratio': round(t_myeloid_ratio, 2) if t_myeloid_ratio is not None else None,
            'cd8_cells': int(cd8_cells),
            'treg_cells': int(treg_cells),
            't_cells': int(t_cells),
            'myeloid_cells': int(myeloid_cells),
            # Immune composition within immune cells
            **{f'immune_{k}': v for k, v in immune_within.items()},
            # Data quality
            'data_quality': data_quality,
            'has_detailed_tcell_annotation': has_detailed_annotation,
            'has_malignant_annotation': has_malignant_annotation,
            'sample_note': sample_note
        })

    log(f"  Generated {len(results)} TME summary records")
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
        immune_cells = ct_meta[ct_meta['cell_type'].apply(is_immune_cell)]['n_cells'].sum()

        if total_cells < 100:
            continue

        # Count by immune category (within immune cells for meaningful comparison)
        for cat in list(IMMUNE_PATTERNS.keys()):
            cat_mask = ct_meta['immune_category'] == cat
            cat_cells = ct_meta[cat_mask]['n_cells'].sum()
            n_samples = cat_mask.sum()

            if cat_cells > 0:
                results.append({
                    'cancer_type': cancer,
                    'cell_category': cat,
                    'cell_count': int(cat_cells),
                    'proportion_of_tme': round(cat_cells / total_cells, 4),
                    'proportion_of_immune': round(cat_cells / immune_cells, 4) if immune_cells > 0 else 0,
                    'n_samples': int(n_samples),
                    'total_cells': int(total_cells),
                    'immune_cells': int(immune_cells)
                })

    log(f"  Generated {len(results)} composition records")
    return results


def compute_study_level_infiltration(meta_df, cytosig_df):
    """
    Compute immune cell infiltration at study/donor level for detailed comparison.
    """
    log("Computing study-level immune infiltration...")

    meta_df = meta_df.copy()
    meta_df['is_immune'] = meta_df['cell_type'].apply(is_immune_cell)
    meta_df['immune_category'] = meta_df['cell_type'].apply(
        lambda x: classify_cell_type(x)[0] if classify_cell_type(x) else 'Other'
    )

    # Extract study from donorID
    meta_df['study'] = meta_df['donorID'].str.extract(r'^([A-Za-z0-9]+)')[0]

    # Filter to tumor samples only
    if 'tissue' in meta_df.columns:
        tumor_meta = meta_df[meta_df['tissue'] == 'Tumor'].copy()
    else:
        tumor_meta = meta_df.copy()

    if len(tumor_meta) == 0:
        return [], [], []

    log(f"  Tumor samples: {len(tumor_meta)}")
    log(f"  Unique studies: {tumor_meta['study'].nunique()}")
    log(f"  Unique donors: {tumor_meta['donorID'].nunique()}")

    # === 1. Study-level summary ===
    study_results = []
    studies = tumor_meta['study'].dropna().unique()

    for study in studies:
        study_meta = tumor_meta[tumor_meta['study'] == study]
        cancer_types = study_meta['cancerType'].unique()

        for cancer in cancer_types:
            ct_meta = study_meta[study_meta['cancerType'] == cancer]
            total_cells = ct_meta['n_cells'].sum()
            n_donors = ct_meta['donorID'].nunique()

            if total_cells < 100:
                continue

            immune_cells = ct_meta[ct_meta['is_immune']]['n_cells'].sum()

            # Composition by category
            composition = {}
            for cat in list(IMMUNE_PATTERNS.keys()):
                cat_cells = ct_meta[ct_meta['immune_category'] == cat]['n_cells'].sum()
                composition[cat] = round(cat_cells / total_cells, 4) if total_cells > 0 else 0

            study_results.append({
                'study': study,
                'cancer_type': cancer,
                'n_donors': int(n_donors),
                'total_cells': int(total_cells),
                'immune_cells': int(immune_cells),
                'immune_proportion': round(immune_cells / total_cells, 4) if total_cells > 0 else 0,
                **{f'prop_{k}': v for k, v in composition.items()}
            })

    log(f"  Study-level records: {len(study_results)}")

    # === 2. Donor-level profiles ===
    donor_results = []
    donors = tumor_meta['donorID'].dropna().unique()

    for donor in donors:
        donor_meta = tumor_meta[tumor_meta['donorID'] == donor]
        study = donor_meta['study'].iloc[0] if len(donor_meta) > 0 else None
        cancer = donor_meta['cancerType'].iloc[0] if 'cancerType' in donor_meta.columns else None
        total_cells = donor_meta['n_cells'].sum()

        if total_cells < 50:
            continue

        immune_cells = donor_meta[donor_meta['is_immune']]['n_cells'].sum()

        # Composition by category
        composition = {}
        for cat in list(IMMUNE_PATTERNS.keys()):
            cat_cells = donor_meta[donor_meta['immune_category'] == cat]['n_cells'].sum()
            composition[cat] = round(cat_cells / total_cells, 4) if total_cells > 0 else 0

        donor_results.append({
            'donor_id': donor,
            'study': study,
            'cancer_type': cancer,
            'total_cells': int(total_cells),
            'immune_cells': int(immune_cells),
            'immune_proportion': round(immune_cells / total_cells, 4) if total_cells > 0 else 0,
            **{f'prop_{k}': v for k, v in composition.items()}
        })

    log(f"  Donor-level records: {len(donor_results)}")

    # === 3. Study metadata ===
    study_meta_list = []
    for study in studies:
        study_data = tumor_meta[tumor_meta['study'] == study]
        cancers = study_data['cancerType'].unique().tolist()
        n_donors = study_data['donorID'].nunique()
        total_cells = study_data['n_cells'].sum()

        study_meta_list.append({
            'study': study,
            'cancer_types': cancers,
            'n_donors': int(n_donors),
            'total_cells': int(total_cells)
        })

    log(f"  Study metadata records: {len(study_meta_list)}")

    return study_results, donor_results, study_meta_list


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

def clean_for_json(obj):
    """Recursively clean object for JSON serialization (handle numpy types, NaN, etc.)."""
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(item) for item in obj]
    elif isinstance(obj, (np.bool_, np.bool)):
        return bool(obj)
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


def save_json(data, filename):
    """Save data to JSON file."""
    filepath = VIZ_DATA_DIR / filename
    cleaned_data = clean_for_json(data)
    with open(filepath, 'w') as f:
        json.dump(cleaned_data, f, indent=2, allow_nan=False)
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
    log(f"  Processing CytoSig ({len(cytosig_df)} signatures) + SecAct ({len(secact_df)} signatures)")
    infiltration_data = compute_immune_infiltration(meta_df, cytosig_df, secact_df)
    composition_data = generate_immune_composition_summary(meta_df)
    tme_summary = generate_tme_summary(meta_df)
    study_data, donor_data, study_meta = compute_study_level_infiltration(meta_df, cytosig_df)

    save_json({
        'data': infiltration_data,
        'composition': composition_data,
        'tme_summary': tme_summary,  # NEW: comprehensive TME breakdown
        'by_study': study_data,
        'by_donor': donor_data,
        'studies': study_meta,
        'cytosig_signatures': list(cytosig_df.index),
        'secact_signatures': list(secact_df.index),
        'signatures': list(cytosig_df.index) + list(secact_df.index),  # Combined for backward compat
        'cancer_types': list(meta_df['cancerType'].dropna().unique()) if 'cancerType' in meta_df.columns else [],
        'immune_categories': list(IMMUNE_PATTERNS.keys()),
        # Metadata for UI
        'data_source': 'Tumor tissue only (excludes Blood, Adjacent, Metastasis)',
        'analysis_note': 'Immune proportions represent tumor-infiltrating immune cells in the TME'
    }, 'immune_infiltration.json')

    # T Cell Exhaustion Analysis
    log("\n--- T Cell Exhaustion ---")

    # CytoSig exhaustion analysis
    cytosig_state_data, cytosig_comparison = compute_tcell_exhaustion(meta_df, cytosig_df)
    cytosig_subset_data = compute_exhaustion_by_subset(meta_df, cytosig_df)

    # Add signature_type field to CytoSig data
    for d in cytosig_state_data:
        d['signature_type'] = 'cytosig'
    for d in cytosig_comparison:
        d['signature_type'] = 'cytosig'
    for d in cytosig_subset_data:
        d['signature_type'] = 'cytosig'

    # SecAct exhaustion analysis (top 100 most variable signatures)
    # Select top signatures by variance across samples
    secact_var = secact_df.var(axis=1).sort_values(ascending=False)
    top_secact = secact_df.loc[secact_var.head(100).index]
    log(f"  Using top 100 SecAct signatures (of {len(secact_df)})")

    secact_state_data, secact_comparison = compute_tcell_exhaustion(meta_df, top_secact)
    secact_subset_data = compute_exhaustion_by_subset(meta_df, top_secact)

    # Add signature_type field to SecAct data
    for d in secact_state_data:
        d['signature_type'] = 'secact'
    for d in secact_comparison:
        d['signature_type'] = 'secact'
    for d in secact_subset_data:
        d['signature_type'] = 'secact'

    # Combine data
    all_state_data = cytosig_state_data + secact_state_data
    all_comparison = cytosig_comparison + secact_comparison
    all_subset_data = cytosig_subset_data + secact_subset_data

    save_json({
        'data': all_state_data,
        'comparison': all_comparison,
        'by_subset': all_subset_data,
        'cytosig_signatures': list(cytosig_df.index),
        'secact_signatures': list(top_secact.index),
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
