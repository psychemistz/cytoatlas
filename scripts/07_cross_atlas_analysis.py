#!/usr/bin/env python3
"""
Cross-Atlas Integration Analysis
================================
Compute statistics for cross-atlas visualization panels.

This script generates real cross-atlas comparison data by:
1. Computing atlas summary statistics (cells, samples, cell types)
2. Analyzing signature overlap across atlases
3. Computing pairwise atlas correlations
4. Harmonizing cell type annotations
5. Performing meta-analysis of age/sex effects
6. Computing signature correlation matrices
7. Mapping cytokines to pathways

Output files (in results/integrated/):
- atlas_summary.json: Overview statistics per atlas
- signature_overlap.csv: Which signatures active in which atlases
- atlas_comparison.csv: Pairwise atlas correlations
- celltype_harmonization.csv: Cell type mapping across atlases
- meta_analysis.csv: Combined effect sizes with heterogeneity
- signature_correlation.csv: Signature-signature correlation matrix
- pathway_enrichment.csv: Cytokine-pathway associations
"""

import os
import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster

warnings.filterwarnings('ignore')

# Add SecActpy to path
sys.path.insert(0, '/data/parks34/projects/1ridgesig/SecActpy')
from secactpy import load_cytosig

# ==============================================================================
# Configuration
# ==============================================================================

RESULTS_DIR = Path('/data/parks34/projects/2cytoatlas/results')
OUTPUT_DIR = RESULTS_DIR / 'integrated'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Cell type harmonization mapping (CIMA L2 -> Common name)
CIMA_TO_COMMON = {
    'CD4_CTL': 'CD4 T',
    'CD4_helper': 'CD4 T',
    'CD4_memory': 'CD4 T',
    'CD4_naive': 'CD4 T',
    'CD4_regulatory': 'Treg',
    'CD56_bright_NK': 'NK',
    'CD56_dim_NK': 'NK',
    'CD8_CTL': 'CD8 T',
    'CD8_memory': 'CD8 T',
    'CD8_naive': 'CD8 T',
    'DC': 'DC',
    'HSPC': 'HSPC',
    'ILC2': 'ILC',
    'Immature_T': 'T cell',
    'MAIT': 'MAIT',
    'MK': 'Megakaryocyte',
    'Memory_B': 'B cell',
    'Mono': 'Monocyte',
    'NKT': 'NKT',
    'Naive_B': 'B cell',
    'Proliferative_NK': 'NK',
    'Proliferative_T': 'T cell',
    'Total_Plasma': 'Plasma',
    'Transitional_B': 'B cell',
    'Transitional_NK': 'NK',
    'gdT': 'gdT',
    'pDC': 'pDC',
}

# Inflammation cell type mapping
INFLAM_TO_COMMON = {
    'B_IFNresponder': 'B cell',
    'B_Memory_ITGAX': 'B cell',
    'B_Memory_switched': 'B cell',
    'B_Memory_unswitched': 'B cell',
    'B_Naive': 'B cell',
    'B_Naive_activated': 'B cell',
    'B_Progenitors': 'B cell',
    'B_Transitional': 'B cell',
    'DC4': 'DC',
    'DC5': 'DC',
    'DC_CCR7': 'DC',
    'DC_Proliferative': 'DC',
    'MAIT': 'MAIT',
    'MAIT_17': 'MAIT',
    'Mono_IFNresponse': 'Monocyte',
    'Mono_FCGR3A': 'Monocyte',
    'Mono_CD14': 'Monocyte',
    'Mono_CD14_S100A': 'Monocyte',
    'Mono_CD14_VCAN': 'Monocyte',
    'Mono_CD16': 'Monocyte',
    'Mono_CCR2': 'Monocyte',
    'Mono_Inflammatory': 'Monocyte',
    'Treg': 'Treg',
    'Treg_Memory': 'Treg',
    'Treg_Naive': 'Treg',
    'NK_CD56bright': 'NK',
    'NK_CD56dim': 'NK',
    'NK_CD16': 'NK',
    'NK_Activated': 'NK',
    'NK_Adaptive': 'NK',
    'NK_Proliferative': 'NK',
    'CD4_CTL': 'CD4 T',
    'CD4_Memory': 'CD4 T',
    'CD4_Naive': 'CD4 T',
    'CD4_Th1': 'CD4 T',
    'CD4_Th17': 'CD4 T',
    'CD4_Th2': 'CD4 T',
    'CD4_TFH': 'CD4 T',
    'CD8_CTL': 'CD8 T',
    'CD8_Cytotoxic': 'CD8 T',
    'CD8_Effector': 'CD8 T',
    'CD8_Memory': 'CD8 T',
    'CD8_Naive': 'CD8 T',
    'CD8_Exhausted': 'CD8 T',
    'CD8_Proliferative': 'CD8 T',
    'Plasma': 'Plasma',
    'Plasma_IgA': 'Plasma',
    'Plasma_IgG': 'Plasma',
    'Plasma_IgM': 'Plasma',
    'pDC': 'pDC',
    'gdT': 'gdT',
    'gdT_Vd1': 'gdT',
    'gdT_Vd2': 'gdT',
    'NKT': 'NKT',
    'ILC': 'ILC',
    'ILC2': 'ILC',
    'ILC3': 'ILC',
    'HSPC': 'HSPC',
    'Megakaryocyte': 'Megakaryocyte',
    'Erythrocyte': 'Erythrocyte',
    'Neutrophil': 'Neutrophil',
    'Basophil': 'Basophil',
    'Eosinophil': 'Eosinophil',
    'Macrophage': 'Macrophage',
    'Macrophage_M1': 'Macrophage',
    'Macrophage_M2': 'Macrophage',
}

# Pathway database (KEGG cytokine pathways)
KEGG_PATHWAYS = {
    'Cytokine-cytokine receptor interaction': ['IFNG', 'TNF', 'IL6', 'IL1B', 'IL10', 'IL17A', 'IL4', 'IL13', 'TGFB1', 'CCL2', 'CXCL10', 'IL2', 'IL21', 'IL12A', 'CSF2', 'IL18'],
    'JAK-STAT signaling': ['IFNG', 'IL6', 'IL10', 'IL4', 'IL13', 'IL2', 'IL21', 'IL12A', 'CSF2', 'IL23A'],
    'NF-kappa B signaling': ['TNF', 'IL1B', 'IL17A', 'LTA', 'CD40LG'],
    'TNF signaling pathway': ['TNF', 'LTA', 'IL1B', 'IL6', 'CCL2', 'CXCL10'],
    'IL-17 signaling pathway': ['IL17A', 'IL17F', 'IL6', 'CCL2', 'CXCL8', 'CSF2', 'CSF3'],
    'Th1 and Th2 cell differentiation': ['IFNG', 'IL4', 'IL12A', 'IL2', 'TNF'],
    'Th17 cell differentiation': ['IL17A', 'IL17F', 'IL21', 'IL23A', 'IL6', 'TGFB1'],
    'T cell receptor signaling': ['IFNG', 'TNF', 'IL2', 'IL4', 'IL10', 'CSF2'],
    'Toll-like receptor signaling': ['TNF', 'IL1B', 'IL6', 'IL12A', 'IFNG', 'CXCL10', 'CCL5'],
    'Chemokine signaling pathway': ['CCL2', 'CCL5', 'CXCL8', 'CXCL10', 'CXCL1', 'CXCL12'],
    'Inflammatory bowel disease': ['IFNG', 'TNF', 'IL6', 'IL1B', 'IL17A', 'IL10', 'IL12A', 'IL23A', 'TGFB1'],
    'Rheumatoid arthritis': ['TNF', 'IL6', 'IL1B', 'IL17A', 'IFNG', 'CSF2', 'CCL2'],
}

REACTOME_PATHWAYS = {
    'Interleukin-6 signaling': ['IL6', 'IL6R', 'IL6ST'],
    'Interferon gamma signaling': ['IFNG', 'IRF1', 'STAT1'],
    'Interferon alpha/beta signaling': ['IFNA1', 'IFNB1'],
    'Signaling by Interleukins': ['IL1B', 'IL2', 'IL4', 'IL6', 'IL10', 'IL12A', 'IL17A', 'IL18', 'IL21', 'IL23A'],
    'Cytokine Signaling in Immune system': ['IFNG', 'TNF', 'IL6', 'IL1B', 'IL10', 'IL4', 'IL13', 'IL2'],
    'TRAF6 mediated NF-kB activation': ['TNF', 'IL1B', 'IL17A'],
    'MyD88 cascade': ['IL1B', 'IL18', 'TNF'],
    'Toll Like Receptor Cascades': ['TNF', 'IL1B', 'IL6', 'IL12A', 'CXCL10'],
}

HALLMARK_PATHWAYS = {
    'INFLAMMATORY_RESPONSE': ['TNF', 'IL1B', 'IL6', 'CCL2', 'CXCL8', 'CXCL10'],
    'INTERFERON_GAMMA_RESPONSE': ['IFNG', 'CXCL10', 'IRF1', 'STAT1'],
    'IL6_JAK_STAT3_SIGNALING': ['IL6', 'IL10', 'IL11', 'LIF', 'OSM'],
    'IL2_STAT5_SIGNALING': ['IL2', 'IL4', 'IL7', 'IL15', 'IL21'],
    'TNFA_SIGNALING_VIA_NFKB': ['TNF', 'IL1B', 'IL6', 'CCL2', 'CXCL8'],
    'COMPLEMENT': ['C3', 'C5', 'CFB', 'CFH'],
    'ALLOGRAFT_REJECTION': ['IFNG', 'TNF', 'IL2', 'GZMB', 'PRF1'],
}


def log(msg: str):
    """Print log message."""
    print(f"[INFO] {msg}", flush=True)


# ==============================================================================
# 1. Atlas Summary Statistics
# ==============================================================================

def compute_atlas_summary():
    """Compute summary statistics for each atlas."""
    log("Computing atlas summary statistics...")

    summary = {}

    # CIMA
    cima_path = RESULTS_DIR / 'cima' / 'CIMA_CytoSig_pseudobulk.h5ad'
    if cima_path.exists():
        cima = ad.read_h5ad(cima_path, backed='r')
        cima_var = pd.DataFrame(cima.var)
        summary['cima'] = {
            'cells': int(cima_var['n_cells'].sum()),
            'samples': int(cima_var['sample'].nunique()),
            'cell_types': int(cima_var['cell_type'].nunique()),
            'cell_type_list': sorted(cima_var['cell_type'].unique().tolist()),
        }
        log(f"  CIMA: {summary['cima']['cells']:,} cells, {summary['cima']['samples']} samples, {summary['cima']['cell_types']} cell types")

    # Inflammation
    inflam_path = RESULTS_DIR / 'inflammation' / 'main_CytoSig_pseudobulk.h5ad'
    if inflam_path.exists():
        inflam = ad.read_h5ad(inflam_path, backed='r')
        inflam_var = pd.DataFrame(inflam.var)
        summary['inflammation'] = {
            'cells': int(inflam_var['n_cells'].sum()),
            'samples': int(inflam_var['sample'].nunique()),
            'cell_types': int(inflam_var['cell_type'].nunique()),
        }
        log(f"  Inflammation: {summary['inflammation']['cells']:,} cells, {summary['inflammation']['samples']} samples")

    # scAtlas Normal
    normal_meta_path = RESULTS_DIR / 'scatlas' / 'normal_aggregation_meta.csv'
    if normal_meta_path.exists():
        normal_meta = pd.read_csv(normal_meta_path)
        summary['scatlas_normal'] = {
            'cells': int(normal_meta['n_cells'].sum()),
            'samples': 0,  # scAtlas doesn't have sample-level info
            'cell_types': int(normal_meta['cell_type'].nunique()),
            'organs': int(normal_meta['tissue'].nunique()),
        }
        log(f"  scAtlas Normal: {summary['scatlas_normal']['cells']:,} cells, {summary['scatlas_normal']['organs']} organs")

    # scAtlas Cancer
    cancer_meta_path = RESULTS_DIR / 'scatlas' / 'cancer_aggregation_meta.csv'
    if cancer_meta_path.exists():
        cancer_meta = pd.read_csv(cancer_meta_path)
        summary['scatlas_cancer'] = {
            'cells': int(cancer_meta['n_cells'].sum()),
            'samples': 0,
            'cell_types': int(cancer_meta['cell_type'].nunique()),
        }
        log(f"  scAtlas Cancer: {summary['scatlas_cancer']['cells']:,} cells")

    # Save
    with open(OUTPUT_DIR / 'atlas_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    return summary


# ==============================================================================
# 2. Signature Overlap Analysis
# ==============================================================================

def compute_signature_overlap():
    """Analyze which signatures are active across atlases."""
    log("Computing signature overlap...")

    # Load CytoSig signatures
    cytosig = load_cytosig()
    all_signatures = list(cytosig.columns)
    log(f"  Total CytoSig signatures: {len(all_signatures)}")

    # Load mean activities from each atlas
    atlas_means = {}

    # CIMA
    cima_path = RESULTS_DIR / 'cima' / 'CIMA_CytoSig_pseudobulk.h5ad'
    if cima_path.exists():
        cima = ad.read_h5ad(cima_path)
        cima_means = pd.Series(cima.X.mean(axis=1), index=cima.obs_names)
        atlas_means['cima'] = cima_means
        log(f"  CIMA: {len(cima_means)} signatures loaded")

    # Inflammation
    inflam_path = RESULTS_DIR / 'inflammation' / 'main_CytoSig_pseudobulk.h5ad'
    if inflam_path.exists():
        inflam = ad.read_h5ad(inflam_path)
        inflam_means = pd.Series(inflam.X.mean(axis=1), index=inflam.obs_names)
        atlas_means['inflammation'] = inflam_means
        log(f"  Inflammation: {len(inflam_means)} signatures loaded")

    # scAtlas (use organ signatures as summary)
    scatlas_path = RESULTS_DIR / 'scatlas' / 'normal_organ_signatures.csv'
    if scatlas_path.exists():
        scatlas = pd.read_csv(scatlas_path)
        scatlas_cytosig = scatlas[scatlas['signature_type'] == 'CytoSig']
        scatlas_means = scatlas_cytosig.groupby('signature')['mean_activity'].mean()
        atlas_means['scatlas'] = scatlas_means
        log(f"  scAtlas: {len(scatlas_means)} signatures loaded")

    # Define "active" threshold (mean |z-score| > 0.5 in at least one cell type)
    threshold = 0.5

    results = []
    for sig in all_signatures:
        in_cima = sig in atlas_means.get('cima', pd.Series()).index and abs(atlas_means['cima'].get(sig, 0)) > threshold
        in_inflam = sig in atlas_means.get('inflammation', pd.Series()).index and abs(atlas_means['inflammation'].get(sig, 0)) > threshold
        in_scatlas = sig in atlas_means.get('scatlas', pd.Series()).index and abs(atlas_means['scatlas'].get(sig, 0)) > threshold

        n_atlases = sum([in_cima, in_inflam, in_scatlas])

        results.append({
            'signature': sig,
            'cima': in_cima,
            'inflammation': in_inflam,
            'scatlas': in_scatlas,
            'n_atlases': n_atlases,
            'cima_mean': float(atlas_means.get('cima', pd.Series()).get(sig, np.nan)),
            'inflammation_mean': float(atlas_means.get('inflammation', pd.Series()).get(sig, np.nan)),
            'scatlas_mean': float(atlas_means.get('scatlas', pd.Series()).get(sig, np.nan)),
        })

    overlap_df = pd.DataFrame(results)
    overlap_df.to_csv(OUTPUT_DIR / 'signature_overlap.csv', index=False)

    # Compute counts
    all_three = len(overlap_df[(overlap_df['cima']) & (overlap_df['inflammation']) & (overlap_df['scatlas'])])
    cima_inflam = len(overlap_df[(overlap_df['cima']) & (overlap_df['inflammation']) & (~overlap_df['scatlas'])])
    cima_scatlas = len(overlap_df[(overlap_df['cima']) & (~overlap_df['inflammation']) & (overlap_df['scatlas'])])
    inflam_scatlas = len(overlap_df[(~overlap_df['cima']) & (overlap_df['inflammation']) & (overlap_df['scatlas'])])
    cima_only = len(overlap_df[(overlap_df['cima']) & (~overlap_df['inflammation']) & (~overlap_df['scatlas'])])
    inflam_only = len(overlap_df[(~overlap_df['cima']) & (overlap_df['inflammation']) & (~overlap_df['scatlas'])])
    scatlas_only = len(overlap_df[(~overlap_df['cima']) & (~overlap_df['inflammation']) & (overlap_df['scatlas'])])

    log(f"  All 3 atlases: {all_three}")
    log(f"  CIMA+Inflam only: {cima_inflam}")
    log(f"  CIMA+scAtlas only: {cima_scatlas}")
    log(f"  Inflam+scAtlas only: {inflam_scatlas}")

    return overlap_df


# ==============================================================================
# 3. Atlas Pairwise Comparison
# ==============================================================================

def compute_atlas_comparison():
    """Compute pairwise correlations between atlases for common cell types."""
    log("Computing atlas comparisons...")

    # Load pseudobulk data
    cima_path = RESULTS_DIR / 'cima' / 'CIMA_CytoSig_pseudobulk.h5ad'
    inflam_path = RESULTS_DIR / 'inflammation' / 'main_CytoSig_pseudobulk.h5ad'
    scatlas_path = RESULTS_DIR / 'scatlas' / 'normal_celltype_signatures.csv'

    cima_data = {}
    inflam_data = {}
    scatlas_data = {}

    # Load CIMA - compute mean activity per cell type
    if cima_path.exists():
        cima = ad.read_h5ad(cima_path)
        cima_var = pd.DataFrame(cima.var)
        cima_df = pd.DataFrame(cima.X, index=cima.obs_names, columns=cima.var_names)

        for ct in cima_var['cell_type'].unique():
            ct_cols = cima_var[cima_var['cell_type'] == ct].index
            cima_data[ct] = cima_df[ct_cols].mean(axis=1)

    # Load Inflammation
    if inflam_path.exists():
        inflam = ad.read_h5ad(inflam_path)
        inflam_var = pd.DataFrame(inflam.var)
        inflam_df = pd.DataFrame(inflam.X, index=inflam.obs_names, columns=inflam.var_names)

        for ct in inflam_var['cell_type'].unique():
            ct_cols = inflam_var[inflam_var['cell_type'] == ct].index
            inflam_data[ct] = inflam_df[ct_cols].mean(axis=1)

    # Load scAtlas
    if os.path.exists(scatlas_path):
        scatlas = pd.read_csv(scatlas_path)
        scatlas_cytosig = scatlas[scatlas['signature_type'] == 'CytoSig']

        # Aggregate by cell type and signature (average across organs)
        scatlas_agg = scatlas_cytosig.groupby(['cell_type', 'signature'])['mean_activity'].mean().reset_index()

        for ct in scatlas_agg['cell_type'].unique():
            ct_data = scatlas_agg[scatlas_agg['cell_type'] == ct]
            scatlas_data[ct] = pd.Series(ct_data['mean_activity'].values, index=ct_data['signature'].values)

    # Map to common cell types
    def map_to_common(data_dict, mapping):
        common_data = {}
        for ct, activities in data_dict.items():
            common_ct = mapping.get(ct, ct)
            if common_ct not in common_data:
                common_data[common_ct] = []
            common_data[common_ct].append(activities)

        # Average multiple cell types mapping to same common type
        result = {}
        for common_ct, activity_list in common_data.items():
            if activity_list:
                result[common_ct] = pd.concat(activity_list, axis=1).mean(axis=1)
        return result

    cima_common = map_to_common(cima_data, CIMA_TO_COMMON)
    inflam_common = map_to_common(inflam_data, INFLAM_TO_COMMON)

    # For scAtlas, create simple mapping
    scatlas_common = {}
    for ct, activities in scatlas_data.items():
        # Simple heuristic mapping
        ct_lower = ct.lower()
        if 'cd4' in ct_lower or 't_cd4' in ct_lower:
            common = 'CD4 T'
        elif 'cd8' in ct_lower or 't_cd8' in ct_lower:
            common = 'CD8 T'
        elif 'nk' in ct_lower and 't' not in ct_lower:
            common = 'NK'
        elif 'b cell' in ct_lower or 'b_cell' in ct_lower:
            common = 'B cell'
        elif 'mono' in ct_lower:
            common = 'Monocyte'
        elif 'macro' in ct_lower:
            common = 'Macrophage'
        elif 'dc' in ct_lower or 'dendritic' in ct_lower:
            common = 'DC'
        elif 'plasma' in ct_lower:
            common = 'Plasma'
        elif 'treg' in ct_lower:
            common = 'Treg'
        else:
            common = ct

        if common not in scatlas_common:
            scatlas_common[common] = []
        scatlas_common[common].append(activities)

    for ct, act_list in scatlas_common.items():
        # Handle potential duplicate indices by averaging
        if len(act_list) == 1:
            scatlas_common[ct] = act_list[0]
        else:
            # Combine into DataFrame, handling duplicates by grouping
            combined = pd.concat(act_list, axis=0)
            scatlas_common[ct] = combined.groupby(level=0).mean()

    # Compute pairwise correlations
    results = []

    # CIMA vs Inflammation
    common_cts_ci = set(cima_common.keys()) & set(inflam_common.keys())
    for ct in common_cts_ci:
        cima_act = cima_common[ct]
        inflam_act = inflam_common[ct]
        common_sigs = cima_act.index.intersection(inflam_act.index)

        if len(common_sigs) >= 5:
            rho, pval = stats.spearmanr(cima_act[common_sigs], inflam_act[common_sigs])

            for sig in common_sigs:
                results.append({
                    'comparison': 'cima_vs_inflammation',
                    'cell_type': ct,
                    'signature': sig,
                    'x': float(cima_act[sig]),
                    'y': float(inflam_act[sig]),
                })

    # CIMA vs scAtlas
    common_cts_cs = set(cima_common.keys()) & set(scatlas_common.keys())
    for ct in common_cts_cs:
        cima_act = cima_common[ct]
        scatlas_act = scatlas_common[ct]
        common_sigs = cima_act.index.intersection(scatlas_act.index)

        if len(common_sigs) >= 5:
            for sig in common_sigs:
                results.append({
                    'comparison': 'cima_vs_scatlas',
                    'cell_type': ct,
                    'signature': sig,
                    'x': float(cima_act[sig]),
                    'y': float(scatlas_act[sig]),
                })

    # Inflammation vs scAtlas
    common_cts_is = set(inflam_common.keys()) & set(scatlas_common.keys())
    for ct in common_cts_is:
        inflam_act = inflam_common[ct]
        scatlas_act = scatlas_common[ct]
        common_sigs = inflam_act.index.intersection(scatlas_act.index)

        if len(common_sigs) >= 5:
            for sig in common_sigs:
                results.append({
                    'comparison': 'inflam_vs_scatlas',
                    'cell_type': ct,
                    'signature': sig,
                    'x': float(inflam_act[sig]),
                    'y': float(scatlas_act[sig]),
                })

    comparison_df = pd.DataFrame(results)
    comparison_df.to_csv(OUTPUT_DIR / 'atlas_comparison.csv', index=False)

    # Compute overall correlations per comparison type
    for comp in ['cima_vs_inflammation', 'cima_vs_scatlas', 'inflam_vs_scatlas']:
        comp_data = comparison_df[comparison_df['comparison'] == comp]
        if len(comp_data) > 5:
            rho, pval = stats.spearmanr(comp_data['x'], comp_data['y'])
            log(f"  {comp}: r = {rho:.3f}, n = {len(comp_data)}")

    return comparison_df


# ==============================================================================
# 4. Cell Type Harmonization
# ==============================================================================

def compute_celltype_harmonization():
    """Compute cell type harmonization across atlases."""
    log("Computing cell type harmonization...")

    # Load cell type info
    cima_path = RESULTS_DIR / 'cima' / 'CIMA_CytoSig_pseudobulk.h5ad'
    inflam_path = RESULTS_DIR / 'inflammation' / 'main_CytoSig_pseudobulk.h5ad'
    scatlas_path = RESULTS_DIR / 'scatlas' / 'normal_aggregation_meta.csv'

    harmonization = []

    # CIMA cell types
    if cima_path.exists():
        cima = ad.read_h5ad(cima_path, backed='r')
        cima_var = pd.DataFrame(cima.var)
        for ct in cima_var['cell_type'].unique():
            n_cells = int(cima_var[cima_var['cell_type'] == ct]['n_cells'].sum())
            harmonization.append({
                'atlas': 'CIMA',
                'original_name': ct,
                'common_name': CIMA_TO_COMMON.get(ct, ct),
                'n_cells': n_cells,
            })

    # Inflammation cell types
    if inflam_path.exists():
        inflam = ad.read_h5ad(inflam_path, backed='r')
        inflam_var = pd.DataFrame(inflam.var)
        for ct in inflam_var['cell_type'].unique():
            n_cells = int(inflam_var[inflam_var['cell_type'] == ct]['n_cells'].sum())
            # Try to match to common name
            common = INFLAM_TO_COMMON.get(ct, ct)
            harmonization.append({
                'atlas': 'Inflammation',
                'original_name': ct,
                'common_name': common,
                'n_cells': n_cells,
            })

    # scAtlas cell types
    if os.path.exists(scatlas_path):
        scatlas_meta = pd.read_csv(scatlas_path)
        ct_counts = scatlas_meta.groupby('cell_type')['n_cells'].sum()
        for ct, n_cells in ct_counts.items():
            # Simple mapping
            ct_lower = str(ct).lower()
            if 'cd4' in ct_lower:
                common = 'CD4 T'
            elif 'cd8' in ct_lower:
                common = 'CD8 T'
            elif 'nk' in ct_lower and 't' not in ct_lower:
                common = 'NK'
            elif 'b cell' in ct_lower or ct_lower.startswith('b_'):
                common = 'B cell'
            elif 'mono' in ct_lower:
                common = 'Monocyte'
            elif 'macro' in ct_lower:
                common = 'Macrophage'
            elif 'dc' in ct_lower or 'dendritic' in ct_lower:
                common = 'DC'
            else:
                common = ct

            harmonization.append({
                'atlas': 'scAtlas',
                'original_name': ct,
                'common_name': common,
                'n_cells': int(n_cells),
            })

    harm_df = pd.DataFrame(harmonization)
    harm_df.to_csv(OUTPUT_DIR / 'celltype_harmonization.csv', index=False)

    # Summarize common cell types
    common_summary = harm_df.groupby(['common_name', 'atlas']).agg({
        'n_cells': 'sum',
        'original_name': 'count'
    }).reset_index()

    log(f"  Total mappings: {len(harm_df)}")
    log(f"  Unique common cell types: {harm_df['common_name'].nunique()}")

    return harm_df


# ==============================================================================
# 5. Meta-Analysis
# ==============================================================================

def compute_meta_analysis():
    """Perform meta-analysis of age/sex effects across atlases."""
    log("Computing meta-analysis...")

    meta_results = []

    # === CIMA Age Correlations (CytoSig + SecAct) ===
    cima_age_path = RESULTS_DIR / 'cima' / 'CIMA_correlation_age.csv'
    if cima_age_path.exists():
        age_corr = pd.read_csv(cima_age_path)

        for sig_type in ['CytoSig', 'SecAct']:
            sig_corr = age_corr[age_corr['signature'] == sig_type]
            for _, row in sig_corr.iterrows():
                n = row['n']
                rho = row['rho']
                z = np.arctanh(rho) if abs(rho) < 1 else 0
                se = 1 / np.sqrt(n - 3) if n > 3 else 0.5

                meta_results.append({
                    'analysis': 'age',
                    'signature': row['protein'],
                    'sig_type': sig_type,
                    'atlas': 'CIMA',
                    'effect': float(rho),
                    'se': float(se),
                    'pvalue': float(row['pvalue']),
                    'n': int(n),
                })
            log(f"  CIMA age correlations ({sig_type}): {len(sig_corr)}")

    # === Inflammation Age Correlations (CytoSig + SecAct) ===
    # Compute from h5ad if sample metadata is available
    inflam_meta_path = Path('/data/Jiang_Lab/Data/Seongyong/Inflammation_Atlas/INFLAMMATION_ATLAS_afterQC_sampleMetadata.csv')

    if inflam_meta_path.exists():
        sample_meta = pd.read_csv(inflam_meta_path)
        meta_age = sample_meta[['sampleID', 'age']].dropna()

        for sig_type in ['CytoSig', 'SecAct']:
            inflam_h5ad_path = RESULTS_DIR / 'inflammation' / f'main_{sig_type}_pseudobulk.h5ad'
            if not inflam_h5ad_path.exists():
                log(f"  Skipping Inflammation {sig_type} age (no pseudobulk file)")
                continue

            log(f"  Computing Inflammation age correlations ({sig_type})...")
            try:
                if len(meta_age) >= 10:
                    inflam_adata = ad.read_h5ad(inflam_h5ad_path)
                    activity_df = pd.DataFrame(
                        inflam_adata.X,
                        index=inflam_adata.obs_names,
                        columns=inflam_adata.var_names
                    )
                    var_info = inflam_adata.var[['sample', 'n_cells']].copy()

                    # Aggregate to sample level (weighted mean)
                    sample_activity = {}
                    for sample in var_info['sample'].unique():
                        sample_cols = var_info[var_info['sample'] == sample].index
                        weights = var_info.loc[sample_cols, 'n_cells'].values
                        total_weight = weights.sum()
                        if total_weight > 0:
                            weighted_mean = (activity_df[sample_cols] * weights).sum(axis=1) / total_weight
                            sample_activity[sample] = weighted_mean

                    sample_activity_df = pd.DataFrame(sample_activity).T

                    # Compute correlations with age
                    count = 0
                    for sig in sample_activity_df.columns:
                        merged = meta_age.merge(
                            sample_activity_df[[sig]].reset_index().rename(columns={'index': 'sampleID', sig: 'activity'}),
                            on='sampleID', how='inner'
                        )
                        if len(merged) >= 10:
                            rho, pval = stats.spearmanr(merged['age'], merged['activity'])
                            n = len(merged)
                            se = 1 / np.sqrt(n - 3) if n > 3 else 0.5

                            meta_results.append({
                                'analysis': 'age',
                                'signature': sig,
                                'sig_type': sig_type,
                                'atlas': 'Inflammation',
                                'effect': float(rho),
                                'se': float(se),
                                'pvalue': float(pval),
                                'n': int(n),
                            })
                            count += 1

                    log(f"  Inflammation age correlations ({sig_type}): {count}")
            except Exception as e:
                log(f"  Warning: Could not compute Inflammation {sig_type} correlations: {e}")

    # === CIMA BMI Correlations (CytoSig + SecAct) ===
    cima_bmi_path = RESULTS_DIR / 'cima' / 'CIMA_correlation_bmi.csv'
    if cima_bmi_path.exists():
        bmi_corr = pd.read_csv(cima_bmi_path)

        for sig_type in ['CytoSig', 'SecAct']:
            sig_corr = bmi_corr[bmi_corr['signature'] == sig_type]
            for _, row in sig_corr.iterrows():
                n = row['n']
                rho = row['rho']
                z = np.arctanh(rho) if abs(rho) < 1 else 0
                se = 1 / np.sqrt(n - 3) if n > 3 else 0.5

                meta_results.append({
                    'analysis': 'bmi',
                    'signature': row['protein'],
                    'sig_type': sig_type,
                    'atlas': 'CIMA',
                    'effect': float(rho),
                    'se': float(se),
                    'pvalue': float(row['pvalue']),
                    'n': int(n),
                })
            log(f"  CIMA BMI correlations ({sig_type}): {len(sig_corr)}")

    # === Inflammation BMI Correlations (CytoSig + SecAct) ===
    if inflam_meta_path.exists():
        meta_bmi = sample_meta[['sampleID', 'BMI']].dropna()

        for sig_type in ['CytoSig', 'SecAct']:
            inflam_h5ad_path = RESULTS_DIR / 'inflammation' / f'main_{sig_type}_pseudobulk.h5ad'
            if not inflam_h5ad_path.exists():
                log(f"  Skipping Inflammation {sig_type} BMI (no pseudobulk file)")
                continue

            log(f"  Computing Inflammation BMI correlations ({sig_type})...")
            try:
                if len(meta_bmi) >= 10:
                    inflam_adata = ad.read_h5ad(inflam_h5ad_path)
                    activity_df = pd.DataFrame(
                        inflam_adata.X,
                        index=inflam_adata.obs_names,
                        columns=inflam_adata.var_names
                    )
                    var_info = inflam_adata.var[['sample', 'n_cells']].copy()

                    # Aggregate to sample level (weighted mean)
                    sample_activity = {}
                    for sample in var_info['sample'].unique():
                        sample_cols = var_info[var_info['sample'] == sample].index
                        weights = var_info.loc[sample_cols, 'n_cells'].values
                        total_weight = weights.sum()
                        if total_weight > 0:
                            weighted_mean = (activity_df[sample_cols] * weights).sum(axis=1) / total_weight
                            sample_activity[sample] = weighted_mean

                    sample_activity_df = pd.DataFrame(sample_activity).T

                    # Compute correlations with BMI
                    count = 0
                    for sig in sample_activity_df.columns:
                        merged = meta_bmi.merge(
                            sample_activity_df[[sig]].reset_index().rename(columns={'index': 'sampleID', sig: 'activity'}),
                            on='sampleID', how='inner'
                        )
                        if len(merged) >= 10:
                            rho, pval = stats.spearmanr(merged['BMI'], merged['activity'])
                            n = len(merged)
                            se = 1 / np.sqrt(n - 3) if n > 3 else 0.5

                            meta_results.append({
                                'analysis': 'bmi',
                                'signature': sig,
                                'sig_type': sig_type,
                                'atlas': 'Inflammation',
                                'effect': float(rho),
                                'se': float(se),
                                'pvalue': float(pval),
                                'n': int(n),
                            })
                            count += 1

                    log(f"  Inflammation BMI correlations ({sig_type}): {count}")
            except Exception as e:
                log(f"  Warning: Could not compute Inflammation {sig_type} BMI correlations: {e}")

    # === CIMA Sex Differences ===
    cima_diff_path = RESULTS_DIR / 'cima' / 'CIMA_differential_demographics.csv'
    if cima_diff_path.exists():
        diff = pd.read_csv(cima_diff_path)
        sex_diff = diff[(diff['comparison'] == 'sex') & (diff['signature'] == 'CytoSig')]

        for _, row in sex_diff.iterrows():
            median_diff = row['median_g1'] - row['median_g2']
            n1, n2 = row['n_g1'], row['n_g2']
            se = np.sqrt(1/n1 + 1/n2) * 1.5

            meta_results.append({
                'analysis': 'sex',
                'signature': row['protein'],
                'sig_type': 'CytoSig',
                'atlas': 'CIMA',
                'effect': float(median_diff),
                'se': float(se),
                'pvalue': float(row['pvalue']),
                'n': int(n1 + n2),
            })

    meta_df = pd.DataFrame(meta_results)

    # === Compute pooled effects and I² for signatures with multiple atlases ===
    # Group by analysis type, signature type, AND signature name
    if len(meta_df) > 0:
        pooled_results = []
        for analysis_type in meta_df['analysis'].unique():
            analysis_df = meta_df[meta_df['analysis'] == analysis_type]

            # Get unique sig_types
            sig_types = analysis_df['sig_type'].unique() if 'sig_type' in analysis_df.columns else ['CytoSig']

            for sig_type in sig_types:
                type_df = analysis_df[analysis_df['sig_type'] == sig_type] if 'sig_type' in analysis_df.columns else analysis_df
                signatures = type_df['signature'].unique()

                for sig in signatures:
                    sig_data = type_df[type_df['signature'] == sig]
                    n_atlases = len(sig_data)

                    if n_atlases == 1:
                        # Single study - no pooling possible
                        row = sig_data.iloc[0]
                        pooled_results.append({
                            **row.to_dict(),
                            'I2': 0.0,
                            'pooled_effect': row['effect'],
                            'pooled_se': row['se'],
                            'ci_low': row['effect'] - 1.96 * row['se'],
                            'ci_high': row['effect'] + 1.96 * row['se'],
                            'n_atlases': 1
                        })
                    else:
                        # Fixed-effects meta-analysis
                        effects = sig_data['effect'].values
                        ses = sig_data['se'].values
                        weights = 1 / (ses ** 2)

                        pooled_effect = np.sum(weights * effects) / np.sum(weights)
                        pooled_se = np.sqrt(1 / np.sum(weights))

                        # Q statistic and I²
                        Q = np.sum(weights * (effects - pooled_effect) ** 2)
                        df = n_atlases - 1
                        I2 = max(0, (Q - df) / Q * 100) if Q > 0 else 0

                        # Add individual atlas results
                        for _, row in sig_data.iterrows():
                            pooled_results.append({
                                **row.to_dict(),
                                'I2': I2,
                                'pooled_effect': pooled_effect,
                                'pooled_se': pooled_se,
                                'ci_low': pooled_effect - 1.96 * pooled_se,
                                'ci_high': pooled_effect + 1.96 * pooled_se,
                                'n_atlases': n_atlases
                            })

        meta_df = pd.DataFrame(pooled_results)

    meta_df.to_csv(OUTPUT_DIR / 'meta_analysis.csv', index=False)

    log(f"  Total meta-analysis records: {len(meta_df)}")
    log(f"  Age effects: {len(meta_df[meta_df['analysis'] == 'age'])} records")
    log(f"  BMI effects: {len(meta_df[meta_df['analysis'] == 'bmi'])} records")
    log(f"  Sex effects: {len(meta_df[meta_df['analysis'] == 'sex'])} records")
    log(f"  Multi-atlas signatures: {len(meta_df[meta_df['n_atlases'] > 1]['signature'].unique()) if 'n_atlases' in meta_df.columns else 0}")

    return meta_df


# ==============================================================================
# 6. Signature Correlation Matrix
# ==============================================================================

def compute_signature_correlation():
    """Compute signature-signature correlation matrix."""
    log("Computing signature correlations...")

    # Load pseudobulk activities from CIMA (most samples)
    cima_path = RESULTS_DIR / 'cima' / 'CIMA_CytoSig_pseudobulk.h5ad'

    if not cima_path.exists():
        log("  CIMA data not found, skipping correlation")
        return None

    cima = ad.read_h5ad(cima_path)

    # Activity matrix: signatures x samples
    activity_df = pd.DataFrame(cima.X, index=cima.obs_names, columns=cima.var_names)

    # Compute Spearman correlation between signatures (across samples)
    corr_matrix = activity_df.T.corr(method='spearman')

    # Save
    corr_matrix.to_csv(OUTPUT_DIR / 'signature_correlation.csv')

    # Detect modules via hierarchical clustering
    # Use 1 - correlation as distance
    dist_matrix = 1 - corr_matrix.values
    np.fill_diagonal(dist_matrix, 0)
    dist_matrix = np.clip(dist_matrix, 0, 2)

    # Cluster
    from scipy.spatial.distance import squareform
    condensed = squareform(dist_matrix)
    Z = linkage(condensed, method='ward')

    # Cut tree at height to get 4-6 modules
    for n_clusters in [5, 4, 6, 3]:
        clusters = fcluster(Z, n_clusters, criterion='maxclust')
        if 3 <= len(set(clusters)) <= 6:
            break

    # Assign module names based on known biology
    modules = pd.DataFrame({
        'signature': corr_matrix.columns,
        'cluster': clusters
    })

    # Name modules based on key members
    module_names = {}
    inflammatory = ['IFNG', 'TNF', 'IL6', 'IL1B', 'IL12A', 'IL18']
    th2 = ['IL4', 'IL13', 'IL5']
    regulatory = ['IL10', 'TGFB1']
    th17 = ['IL17A', 'IL21', 'IL23A', 'IL22']
    chemokines = ['CCL2', 'CCL5', 'CXCL8', 'CXCL10']

    for c in modules['cluster'].unique():
        members = modules[modules['cluster'] == c]['signature'].tolist()

        # Check which category has most members
        scores = {
            'Inflammatory': len(set(members) & set(inflammatory)),
            'Th2': len(set(members) & set(th2)),
            'Regulatory': len(set(members) & set(regulatory)),
            'Th17': len(set(members) & set(th17)),
            'Chemokines': len(set(members) & set(chemokines)),
        }

        best_name = max(scores, key=scores.get)
        if scores[best_name] == 0:
            best_name = f'Module_{c}'

        module_names[c] = best_name

    modules['module_name'] = modules['cluster'].map(module_names)
    modules.to_csv(OUTPUT_DIR / 'signature_modules.csv', index=False)

    log(f"  Correlation matrix: {corr_matrix.shape}")
    log(f"  Detected {len(set(clusters))} modules")

    return corr_matrix, modules


# ==============================================================================
# 7. Pathway Enrichment
# ==============================================================================

def compute_pathway_enrichment():
    """Compute pathway enrichment for cytokine signatures."""
    log("Computing pathway enrichment...")

    # Load CytoSig signatures
    cytosig = load_cytosig()
    signatures = list(cytosig.columns)

    results = []

    # For each pathway database
    for db_name, pathways in [('kegg', KEGG_PATHWAYS), ('reactome', REACTOME_PATHWAYS), ('hallmark', HALLMARK_PATHWAYS)]:
        for pathway, genes in pathways.items():
            # Count overlap with our signatures
            overlap = set(signatures) & set(genes)
            n_overlap = len(overlap)
            n_pathway = len(genes)
            n_total = len(signatures)

            if n_overlap > 0:
                # Simple enrichment score (odds ratio approximation)
                # Fisher's exact test
                a = n_overlap
                b = n_pathway - n_overlap
                c = n_total - n_overlap
                d = 1000 - n_pathway - c  # Background genes

                from scipy.stats import fisher_exact
                try:
                    odds, pval = fisher_exact([[a, b], [c, max(1, d)]], alternative='greater')
                except:
                    pval = 1.0
                    odds = 1.0

                results.append({
                    'database': db_name,
                    'pathway': pathway,
                    'pathway_id': f'{db_name}_{pathway[:20].replace(" ", "_")}',
                    'n_genes': n_overlap,
                    'n_pathway_total': n_pathway,
                    'pvalue': pval,
                    'odds_ratio': odds,
                    'genes': ','.join(sorted(overlap)),
                })

    pathway_df = pd.DataFrame(results)

    # FDR correction per database
    for db in pathway_df['database'].unique():
        mask = pathway_df['database'] == db
        pvals = pathway_df.loc[mask, 'pvalue'].values
        from statsmodels.stats.multitest import multipletests
        _, fdr, _, _ = multipletests(pvals, method='fdr_bh')
        pathway_df.loc[mask, 'fdr'] = fdr

    pathway_df['neg_log_fdr'] = -np.log10(pathway_df['fdr'] + 1e-10)

    pathway_df.to_csv(OUTPUT_DIR / 'pathway_enrichment.csv', index=False)

    log(f"  KEGG pathways: {len(pathway_df[pathway_df['database'] == 'kegg'])}")
    log(f"  Reactome pathways: {len(pathway_df[pathway_df['database'] == 'reactome'])}")
    log(f"  Hallmark pathways: {len(pathway_df[pathway_df['database'] == 'hallmark'])}")

    return pathway_df


# ==============================================================================
# Main
# ==============================================================================

def main():
    """Run all cross-atlas analyses."""
    log("=" * 60)
    log("Cross-Atlas Integration Analysis")
    log("=" * 60)

    # 1. Atlas Summary
    summary = compute_atlas_summary()

    # 2. Signature Overlap
    overlap = compute_signature_overlap()

    # 3. Atlas Comparison
    comparison = compute_atlas_comparison()

    # 4. Cell Type Harmonization
    harmonization = compute_celltype_harmonization()

    # 5. Meta-Analysis
    meta = compute_meta_analysis()

    # 6. Signature Correlation
    corr_result = compute_signature_correlation()

    # 7. Pathway Enrichment
    pathways = compute_pathway_enrichment()

    log("=" * 60)
    log("Cross-atlas analysis complete!")
    log(f"Output directory: {OUTPUT_DIR}")
    log("=" * 60)


if __name__ == '__main__':
    main()
