#!/usr/bin/env python3
"""
Generate Comprehensive Atlas Comparison Data.

Creates multiple comparison views for cross-atlas analysis:
1. Pseudobulk resampled comparison - bootstrap matched samples
2. Single-cell comparison - cell-level activity comparison
3. Cell type aggregated comparison - one point per cell type per condition
4. Prediction concordance - cross-atlas prediction validation

Outputs JSON data for visualization in the Atlas Comparison panel.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad
from scipy import stats
from scipy.spatial.distance import cosine
import warnings
warnings.filterwarnings('ignore')

# Add scripts directory to path for cell_type_mapping
import sys
sys.path.insert(0, str(Path(__file__).parent))
from cell_type_mapping import (
    COARSE_LINEAGES,
    CIMA_TO_COARSE, INFLAMMATION_TO_COARSE,
    CIMA_TO_FINE, INFLAMMATION_TO_FINE,
    FINE_TYPES
)

# scAtlas cellType1 to FINE mapping (for pseudobulk data)
# Maps to standard fine types: CD4_Naive, CD4_Memory, Treg, Tfh, CD8_Naive, CD8_Memory, CD8_Effector,
# MAIT, gdT, NKT, ILC, NK_CD56bright, NK_CD56dim, B_Naive, B_Memory, Plasmablast, Plasma,
# Mono_Classical, Mono_NonClassical, Macrophage, cDC1, cDC2, pDC, Neutrophil, Mast, HSPC
SCATLAS_CELLTYPE1_TO_FINE = {
    # CD4 T cell subtypes
    'T_CD4_naive': 'CD4_Naive', 'naive thymus-derived cd4-positive, alpha-beta t cell': 'CD4_Naive',
    'T_CD4_conv': 'CD4_Memory', 'cd4-positive, alpha-beta memory t cell': 'CD4_Memory',
    'CD4 T cell': 'CD4_Memory', 'T_CD4': 'CD4_Memory', 'abT (CD4)': 'CD4_Memory',
    'cd4-positive alpha-beta t cell': 'CD4_Memory', 'cd4-positive helper t cell': 'CD4_Memory',
    'cd4-positive, alpha-beta t cell': 'CD4_Memory',
    'Th1': 'CD4_Effector', 'Th17': 'CD4_Effector',
    'Treg': 'Treg', 'T_CD4_reg': 'Treg', 'regulatory t cell': 'Treg', 'naive regulatory t cell': 'Treg',
    'T_CD4_fh': 'Tfh', 't follicular helper cell': 'Tfh',
    'Tcm': 'CD4_Memory',
    # CD8 T cell subtypes
    'naive thymus-derived cd8-positive, alpha-beta t cell': 'CD8_Naive',
    'cd8-positive, alpha-beta memory t cell': 'CD8_Memory', 'CD8 T': 'CD8_Memory', 'CD8 T cell': 'CD8_Memory',
    'T_CD8_activated': 'CD8_Effector', 'T_CD8_CTL': 'CD8_Effector', 'abT (CD8)': 'CD8_Effector',
    'cd8-positive alpha-beta t cell': 'CD8_Memory', 'cd8-positive, alpha-beta t cell': 'CD8_Memory',
    'cd8-positive, alpha-beta cytokine secreting effector t cell': 'CD8_Effector',
    'cd8-positive, alpha-beta cytotoxic t cell': 'CD8_Effector', 'Cytotoxic T cells': 'CD8_Effector',
    'T Cell CCL5': 'CD8_Effector', 'T Cell GZMA': 'CD8_Effector', 'T Cell GZMK': 'CD8_Effector',
    'T Cell XCL1': 'CD8_Effector',
    # General T cells (map to memory as default)
    'T cell': 'CD4_Memory', 'T cells': 'CD4_Memory', 't cell': 'CD4_Memory', 'tcells': 'CD4_Memory',
    'T Cell IL7R': 'CD4_Memory', 'T Cell RGS1': 'CD4_Memory',
    'alpha-beta_T_Cells': 'CD4_Memory', 'T_cell_dividing': 'CD4_Memory',
    # Unconventional T cells
    'MAIT': 'MAIT', 'T_CD8_MAIT': 'MAIT',
    'gd T': 'gdT', 'T_CD8_gd': 'gdT', 'gamma-delta_T_Cells_1': 'gdT', 'gamma-delta_T_Cells_2': 'gdT',
    'NKT cell': 'NKT', 'nkt cell': 'NKT', 'NKTcell': 'NKT', 'NK-T cells': 'NKT',
    'NK/T Cell GNLY': 'NKT', 'mature nk t cell': 'NKT', 'cd8b-positive nk t cell': 'NKT',
    'type i nk t cell': 'NKT',
    'ILC': 'ILC', 'ILCT': 'ILC', 'innate lymphoid cell': 'ILC', 'Innate_lymphoid': 'ILC',
    # NK cell subtypes
    'NK_CD160pos': 'NK_CD56bright', 'Resident NK': 'NK_CD56bright',
    'immature natural killer cell': 'NK_CD56bright',
    'NK_FCGR3Apos': 'NK_CD56dim', 'Circulating NK/NKT': 'NK_CD56dim',
    'NK': 'NK_CD56dim', 'NK cell': 'NK_CD56dim', 'NK cells': 'NK_CD56dim', 'nk cell': 'NK_CD56dim',
    'natural killer cell': 'NK_CD56dim', 'NK-like_Cells': 'NK_CD56dim', 'NK_dividing': 'NK_CD56dim',
    # B cell subtypes
    'naive b cell': 'B_Naive', 'B_CD27neg': 'B_Naive',
    'memory b cell': 'B_Memory', 'B_CD27pos': 'B_Memory', 'B_Hypermutation': 'B_Memory',
    'B cell memory': 'B_Memory',
    'B': 'B_Naive', 'B cell': 'B_Naive', 'B cells': 'B_Naive', 'b cell': 'B_Naive', 'bcells': 'B_Naive',
    'B Cell CD79A': 'B_Naive', 'B Cell MS4A1': 'B_Naive', 'B Cell VPREB3': 'B_Naive',
    'B_follicular': 'B_Memory', 'B_mantle': 'B_Memory', 'Follicular B cell': 'B_Memory',
    'Mature_B_Cells': 'B_Memory',
    # Plasma cells
    'Plasmablast': 'Plasmablast',
    'Plasma': 'Plasma', 'Plasma cells': 'Plasma', 'Plasma_Cells': 'Plasma',
    'Plasma B cells': 'Plasma', 'Plasma Cell JCHAIN': 'Plasma',
    'plasma cell': 'Plasma', 'Plasma_IgG': 'Plasma', 'Plasma_IgM': 'Plasma',
    'B cell IgA Plasma': 'Plasma', 'B cell IgG Plasma': 'Plasma',
    # Monocyte subtypes
    'classical monocyte': 'Mono_Classical', 'Monocyte': 'Mono_Classical', 'monocyte': 'Mono_Classical',
    'Monocytes': 'Mono_Classical', 'monocytes': 'Mono_Classical',
    'MNP-a/classical monocyte derived': 'Mono_Classical',
    'non-classical monocyte': 'Mono_NonClassical', 'intermediate monocyte': 'Mono_NonClassical',
    'MNP-b/non-classical monocyte derived': 'Mono_NonClassical',
    'Mono+mono derived cells': 'Mono_Classical',
    # Macrophages
    'Macrophage': 'Macrophage', 'macrophage': 'Macrophage', 'Macrophages': 'Macrophage',
    'Macrophage C1QB': 'Macrophage', 'Macrophage FCN3': 'Macrophage',
    'LYVE1 Macrophage': 'Macrophage', 'Inflammatory_Macrophage': 'Macrophage',
    'Non-inflammatory_Macrophage': 'Macrophage', 'microglial cell': 'Macrophage',
    'Macrophages and DCs': 'Macrophage', 'Mac and DCs': 'Macrophage',
    # Dendritic cells
    'cDC1s': 'cDC1', 'DC_1': 'cDC1', 'cd141-positive myeloid dendritic cell': 'cDC1',
    'cDC2s': 'cDC2', 'DC_2': 'cDC2', 'cd1c-positive myeloid dendritic cell': 'cDC2',
    'DC': 'cDC2', 'dendritic cell': 'cDC2', 'mDC': 'cDC2',
    'pDC': 'pDC', 'pDCs': 'pDC', 'DC_plasmacytoid': 'pDC', 'plasmacytoid dendritic cell': 'pDC',
    # Neutrophils and Mast cells
    'Neutrophils': 'Neutrophil', 'neutrophil': 'Neutrophil', 'granulocyte': 'Neutrophil',
    'cd24 neutrophil': 'Neutrophil', 'nampt neutrophil': 'Neutrophil',
    'Mast': 'Mast', 'Mast cells': 'Mast', 'mast cell': 'Mast',
    'Basophils': 'Mast', 'basophil': 'Mast',
    # Progenitors
    'Progenitor': 'HSPC', 'DP': 'HSPC', 'SP': 'HSPC',
    'dn1 thymic pro-t cell': 'HSPC', 'dn3 thymocyte': 'HSPC', 'thymocyte': 'HSPC',
    'mesenchymal stem cell': 'HSPC',
    # General/Mixed categories (map to most specific possible)
    'Myeloid': 'Mono_Classical', 'myeloid': 'Mono_Classical', 'myeloid cell': 'Mono_Classical',
    'Lymphoid': 'CD4_Memory', 'Lymphocytes': 'CD4_Memory',
    'Immune': 'Mono_Classical', 'immune cell': 'Mono_Classical', 'Leucocytes': 'Mono_Classical',
}

# scAtlas cellType1 to coarse mapping (for pseudobulk data)
# cellType1 uses different naming than subCluster
SCATLAS_CELLTYPE1_TO_COARSE = {
    # B cells
    'B': 'B', 'B cell': 'B', 'B cells': 'B', 'b cell': 'B', 'bcells': 'B',
    'B Cell CD79A': 'B', 'B Cell MS4A1': 'B', 'B Cell VPREB3': 'B',
    'B cell memory': 'B', 'B_CD27neg': 'B', 'B_CD27pos': 'B',
    'B_Hypermutation': 'B', 'B_follicular': 'B', 'B_mantle': 'B',
    'Follicular B cell': 'B', 'Mature_B_Cells': 'B',
    'memory b cell': 'B', 'naive b cell': 'B',
    # Plasma cells
    'Plasma': 'Plasma', 'Plasma cells': 'Plasma', 'Plasma_Cells': 'Plasma',
    'Plasma B cells': 'Plasma', 'Plasma Cell JCHAIN': 'Plasma',
    'plasma cell': 'Plasma', 'Plasmablast': 'Plasma',
    'Plasma_IgG': 'Plasma', 'Plasma_IgM': 'Plasma',
    'B cell IgA Plasma': 'Plasma', 'B cell IgG Plasma': 'Plasma',
    # CD4 T cells
    'CD4 T cell': 'CD4_T', 'T_CD4': 'CD4_T', 'T_CD4_conv': 'CD4_T',
    'T_CD4_fh': 'CD4_T', 'T_CD4_naive': 'CD4_T', 'T_CD4_reg': 'CD4_T',
    'abT (CD4)': 'CD4_T', 'cd4-positive alpha-beta t cell': 'CD4_T',
    'cd4-positive helper t cell': 'CD4_T', 'cd4-positive, alpha-beta memory t cell': 'CD4_T',
    'cd4-positive, alpha-beta t cell': 'CD4_T', 'Treg': 'CD4_T', 'Th1': 'CD4_T', 'Th17': 'CD4_T',
    'regulatory t cell': 'CD4_T', 'naive regulatory t cell': 'CD4_T',
    'naive thymus-derived cd4-positive, alpha-beta t cell': 'CD4_T',
    't follicular helper cell': 'CD4_T', 'Tcm': 'CD4_T',
    # CD8 T cells
    'CD8 T': 'CD8_T', 'CD8 T cell': 'CD8_T', 'T_CD8_CTL': 'CD8_T',
    'T_CD8_activated': 'CD8_T', 'abT (CD8)': 'CD8_T',
    'cd8-positive alpha-beta t cell': 'CD8_T', 'cd8-positive, alpha-beta cytokine secreting effector t cell': 'CD8_T',
    'cd8-positive, alpha-beta cytotoxic t cell': 'CD8_T', 'cd8-positive, alpha-beta memory t cell': 'CD8_T',
    'cd8-positive, alpha-beta t cell': 'CD8_T', 'Cytotoxic T cells': 'CD8_T',
    'naive thymus-derived cd8-positive, alpha-beta t cell': 'CD8_T',
    # General T cells (split to CD4_T by default)
    'T cell': 'CD4_T', 'T cells': 'CD4_T', 't cell': 'CD4_T', 'tcells': 'CD4_T',
    'T Cell CCL5': 'CD8_T', 'T Cell GZMA': 'CD8_T', 'T Cell GZMK': 'CD8_T',
    'T Cell IL7R': 'CD4_T', 'T Cell RGS1': 'CD4_T', 'T Cell XCL1': 'CD8_T',
    'alpha-beta_T_Cells': 'CD4_T', 'T_cell_dividing': 'CD4_T',
    # NK cells
    'NK': 'NK', 'NK cell': 'NK', 'NK cells': 'NK', 'nk cell': 'NK',
    'natural killer cell': 'NK', 'NK-like_Cells': 'NK',
    'NK_CD160pos': 'NK', 'NK_FCGR3Apos': 'NK', 'NK_dividing': 'NK',
    'Resident NK': 'NK', 'Circulating NK/NKT': 'NK',
    'immature natural killer cell': 'NK',
    # Unconventional T cells
    'MAIT': 'Unconventional_T', 'T_CD8_MAIT': 'Unconventional_T',
    'gd T': 'Unconventional_T', 'T_CD8_gd': 'Unconventional_T',
    'gamma-delta_T_Cells_1': 'Unconventional_T', 'gamma-delta_T_Cells_2': 'Unconventional_T',
    'NKT cell': 'Unconventional_T', 'nkt cell': 'Unconventional_T', 'NKTcell': 'Unconventional_T',
    'NK-T cells': 'Unconventional_T', 'NK/T Cell GNLY': 'Unconventional_T',
    'mature nk t cell': 'Unconventional_T', 'cd8b-positive nk t cell': 'Unconventional_T',
    'type i nk t cell': 'Unconventional_T',
    'ILC': 'Unconventional_T', 'ILCT': 'Unconventional_T',
    'innate lymphoid cell': 'Unconventional_T', 'Innate_lymphoid': 'Unconventional_T',
    # Myeloid
    'Myeloid': 'Myeloid', 'myeloid': 'Myeloid', 'myeloid cell': 'Myeloid',
    'Monocyte': 'Myeloid', 'Monocytes': 'Myeloid', 'monocyte': 'Myeloid', 'monocytes': 'Myeloid',
    'classical monocyte': 'Myeloid', 'intermediate monocyte': 'Myeloid', 'non-classical monocyte': 'Myeloid',
    'Mono+mono derived cells': 'Myeloid',
    'Macrophage': 'Myeloid', 'macrophage': 'Myeloid', 'Macrophages': 'Myeloid',
    'Macrophage C1QB': 'Myeloid', 'Macrophage FCN3': 'Myeloid',
    'LYVE1 Macrophage': 'Myeloid', 'Inflammatory_Macrophage': 'Myeloid', 'Non-inflammatory_Macrophage': 'Myeloid',
    'Macrophages and DCs': 'Myeloid', 'Mac and DCs': 'Myeloid',
    'DC': 'Myeloid', 'DC_1': 'Myeloid', 'DC_2': 'Myeloid', 'DC_plasmacytoid': 'Myeloid',
    'dendritic cell': 'Myeloid', 'mDC': 'Myeloid', 'cDC1s': 'Myeloid', 'cDC2s': 'Myeloid',
    'pDC': 'Myeloid', 'pDCs': 'Myeloid', 'plasmacytoid dendritic cell': 'Myeloid',
    'cd141-positive myeloid dendritic cell': 'Myeloid', 'cd1c-positive myeloid dendritic cell': 'Myeloid',
    'MNP-a/classical monocyte derived': 'Myeloid', 'MNP-b/non-classical monocyte derived': 'Myeloid',
    'Mast': 'Myeloid', 'Mast cells': 'Myeloid', 'mast cell': 'Myeloid',
    'Neutrophils': 'Myeloid', 'neutrophil': 'Myeloid', 'granulocyte': 'Myeloid',
    'cd24 neutrophil': 'Myeloid', 'nampt neutrophil': 'Myeloid',
    'Basophils': 'Myeloid', 'basophil': 'Myeloid',
    'microglial cell': 'Myeloid',
    # Progenitor / Other
    'Progenitor': 'Progenitor', 'DP': 'Progenitor', 'SP': 'Progenitor',
    'dn1 thymic pro-t cell': 'Progenitor', 'dn3 thymocyte': 'Progenitor', 'thymocyte': 'Progenitor',
    'mesenchymal stem cell': 'Progenitor',
    # General/Mixed (assign to most likely)
    'Lymphoid': 'CD4_T', 'Lymphocytes': 'CD4_T', 'Immune': 'Myeloid',
    'immune cell': 'Myeloid', 'Leucocytes': 'Myeloid',
}

# Paths
RESULTS_DIR = Path('/data/parks34/projects/2secactpy/results')
VIZ_OUTPUT_DIR = Path('/data/parks34/projects/2secactpy/visualization/data')

# Data files
DATA_FILES = {
    'cima': {
        'cytosig_pb': RESULTS_DIR / 'cima' / 'CIMA_CytoSig_pseudobulk.h5ad',
        'secact_pb': RESULTS_DIR / 'cima' / 'CIMA_SecAct_pseudobulk.h5ad',
        'cytosig_sc': RESULTS_DIR / 'cima' / 'CIMA_CytoSig_singlecell.h5ad',
        'secact_sc': RESULTS_DIR / 'cima' / 'CIMA_SecAct_singlecell.h5ad',
    },
    'inflammation': {
        'cytosig_pb': RESULTS_DIR / 'inflammation' / 'main_CytoSig_pseudobulk.h5ad',
        'secact_pb': RESULTS_DIR / 'inflammation' / 'main_SecAct_pseudobulk.h5ad',
        'cytosig_sc': RESULTS_DIR / 'inflammation' / 'main_CytoSig_singlecell.h5ad',
        'secact_sc': RESULTS_DIR / 'inflammation' / 'main_SecAct_singlecell.h5ad',
    },
    'scatlas': {
        'cytosig_pb': RESULTS_DIR / 'scatlas' / 'scatlas_normal_CytoSig_pseudobulk.h5ad',
        'secact_pb': RESULTS_DIR / 'scatlas' / 'scatlas_normal_SecAct_pseudobulk.h5ad',
        'cytosig_sc': RESULTS_DIR / 'scatlas' / 'scatlas_normal_CytoSig_singlecell.h5ad',
        'secact_sc': RESULTS_DIR / 'scatlas' / 'scatlas_normal_SecAct_singlecell.h5ad',
    }
}


def get_coarse_mapping(atlas: str) -> dict:
    """Get coarse mapping for an atlas."""
    if atlas == 'cima':
        return CIMA_TO_COARSE
    elif atlas == 'inflammation':
        return INFLAMMATION_TO_COARSE
    elif atlas == 'scatlas':
        return SCATLAS_CELLTYPE1_TO_COARSE  # Use cellType1 mapping for pseudobulk
    return {}


def get_fine_mapping(atlas: str) -> dict:
    """Get fine mapping for an atlas."""
    if atlas == 'cima':
        return CIMA_TO_FINE
    elif atlas == 'inflammation':
        return INFLAMMATION_TO_FINE
    elif atlas == 'scatlas':
        return SCATLAS_CELLTYPE1_TO_FINE
    return {}


def load_pseudobulk_data(atlas: str, sig_type: str) -> tuple:
    """
    Load pseudobulk data and return as DataFrame.

    Pseudobulk h5ad structure:
    - obs: signatures (index = signature names)
    - var: samples (with 'cell_type', 'sample', 'n_cells' columns)
    - X: activity matrix (signatures × samples)

    Returns:
        (activity_df, metadata_df) where activity_df has samples as rows, signatures as columns
    """
    key = f'{sig_type.lower()}_pb'
    path = DATA_FILES.get(atlas, {}).get(key)

    if not path or not path.exists():
        print(f"  [WARN] Pseudobulk file not found: {atlas}/{sig_type}")
        return None, None

    print(f"  Loading {path.name}...")
    adata = ad.read_h5ad(path)

    # Extract data
    X = adata.X
    if hasattr(X, 'toarray'):
        X = X.toarray()

    # Signatures are in obs_names, samples in var_names
    signatures = list(adata.obs_names)
    samples = list(adata.var_names)

    # Transpose: signatures × samples -> samples × signatures
    activity_df = pd.DataFrame(X.T, index=samples, columns=signatures)

    # Get metadata from var
    metadata_df = adata.var.copy()
    metadata_df.index = samples

    print(f"    Shape: {activity_df.shape} (samples × signatures)")
    print(f"    Cell types: {metadata_df['cell_type'].nunique()}")

    return activity_df, metadata_df


def load_singlecell_sample(atlas: str, sig_type: str, n_cells: int = 10000,
                           cell_type_filter: str = None) -> tuple:
    """
    Load a sample of single-cell data.

    Single-cell h5ad structure:
    - obs: cells (with metadata like cell_type, sample, etc.)
    - var: signatures (names in var['_index'] or var_names)
    - X: activity matrix (cells × signatures)

    Returns:
        (activity_df, metadata_df) where activity_df has cells as rows, signatures as columns
    """
    key = f'{sig_type.lower()}_sc'
    path = DATA_FILES.get(atlas, {}).get(key)

    if not path or not path.exists():
        print(f"  [WARN] Single-cell file not found: {atlas}/{sig_type}")
        return None, None

    print(f"  Loading sample from {path.name}...")
    adata = ad.read_h5ad(path, backed='r')

    # Get cell indices, optionally filtered by cell type
    if cell_type_filter:
        cell_type_col = 'cell_type' if 'cell_type' in adata.obs.columns else 'Level2'
        if cell_type_col not in adata.obs.columns:
            # Try subCluster for scAtlas
            cell_type_col = 'subCluster' if 'subCluster' in adata.obs.columns else None

        if cell_type_col:
            mask = adata.obs[cell_type_col] == cell_type_filter
            valid_idx = np.where(mask)[0]
        else:
            valid_idx = np.arange(adata.n_obs)
    else:
        valid_idx = np.arange(adata.n_obs)

    # Sample cells
    np.random.seed(42)
    n_sample = min(n_cells, len(valid_idx))
    sampled_idx = np.random.choice(valid_idx, n_sample, replace=False)
    sampled_idx = sorted(sampled_idx)

    # Load data for sampled cells
    X = adata.X[sampled_idx, :]
    if hasattr(X, 'toarray'):
        X = X.toarray()

    # Get signature names
    if '_index' in adata.var.columns:
        signatures = list(adata.var['_index'].values)
    else:
        signatures = list(adata.var_names)

    # Get metadata
    obs_df = adata.obs.iloc[sampled_idx].copy()

    activity_df = pd.DataFrame(X, columns=signatures)

    print(f"    Sampled: {n_sample} cells, {len(signatures)} signatures")

    return activity_df, obs_df


def compute_celltype_aggregated_comparison(atlas1: str, atlas2: str, sig_type: str,
                                            level: str = 'coarse') -> dict:
    """
    Compute cell type aggregated comparison.

    For each harmonized cell type, compute mean activity across all samples,
    then compare between atlases.

    Args:
        atlas1, atlas2: Atlas names
        sig_type: 'CytoSig' or 'SecAct'
        level: 'coarse' or 'fine'

    Returns:
        dict with comparison data
    """
    print(f"\n  Computing {level} cell type aggregated comparison: {atlas1} vs {atlas2}")

    # Load pseudobulk data
    act1, meta1 = load_pseudobulk_data(atlas1, sig_type)
    act2, meta2 = load_pseudobulk_data(atlas2, sig_type)

    # If pseudobulk not available, skip (single-cell lacks cell type metadata)
    if act1 is None:
        print(f"    [SKIP] No pseudobulk data for {atlas1}")
        return {'data': [], 'correlation': None, 'n': 0, 'note': f'No pseudobulk data for {atlas1}'}

    if act2 is None:
        print(f"    [SKIP] No pseudobulk data for {atlas2}")
        return {'data': [], 'correlation': None, 'n': 0, 'note': f'No pseudobulk data for {atlas2}'}

    if act1 is None or act2 is None:
        return {'data': [], 'correlation': None, 'n': 0}

    # Get mapping
    mapping1 = get_coarse_mapping(atlas1) if level == 'coarse' else get_fine_mapping(atlas1)
    mapping2 = get_coarse_mapping(atlas2) if level == 'coarse' else get_fine_mapping(atlas2)

    # Map cell types
    meta1['harmonized'] = meta1['cell_type'].map(mapping1)
    meta2['harmonized'] = meta2['cell_type'].map(mapping2)

    # Get common harmonized types and signatures
    common_types = set(meta1['harmonized'].dropna()) & set(meta2['harmonized'].dropna())
    common_sigs = set(act1.columns) & set(act2.columns)

    if sig_type == 'SecAct':
        # Limit to top 100 most variable signatures for SecAct
        combined_var = act1[list(common_sigs)].var() + act2[list(common_sigs)].var()
        common_sigs = set(combined_var.nlargest(100).index)

    common_sigs = sorted(common_sigs)
    common_types = sorted(common_types)

    print(f"    Common types: {len(common_types)}, Common signatures: {len(common_sigs)}")

    # Compute mean activity per cell type
    comparison_data = []

    for ct in common_types:
        mask1 = meta1['harmonized'] == ct
        mask2 = meta2['harmonized'] == ct

        if mask1.sum() == 0 or mask2.sum() == 0:
            continue

        mean1 = act1.loc[mask1, common_sigs].mean()
        mean2 = act2.loc[mask2, common_sigs].mean()

        for sig in common_sigs:
            comparison_data.append({
                'signature': sig,
                'cell_type': ct,
                'x': float(mean1[sig]),
                'y': float(mean2[sig]),
                'n_samples_x': int(mask1.sum()),
                'n_samples_y': int(mask2.sum())
            })

    # Compute correlation
    if comparison_data:
        x_vals = [d['x'] for d in comparison_data]
        y_vals = [d['y'] for d in comparison_data]

        # Remove NaN
        valid = [(x, y) for x, y in zip(x_vals, y_vals)
                 if not (np.isnan(x) or np.isnan(y))]

        if len(valid) > 2:
            x_clean, y_clean = zip(*valid)
            r, p = stats.spearmanr(x_clean, y_clean)
        else:
            r, p = np.nan, np.nan
    else:
        r, p = np.nan, np.nan

    return {
        'data': comparison_data,
        'correlation': float(r) if not np.isnan(r) else None,
        'pvalue': float(p) if not np.isnan(p) else None,
        'n': len(comparison_data),
        'n_celltypes': len(common_types),
        'n_signatures': len(common_sigs)
    }


def compute_pseudobulk_resampled_comparison(atlas1: str, atlas2: str, sig_type: str,
                                             level: str = 'coarse',
                                             n_bootstrap: int = 100) -> dict:
    """
    Compute pseudobulk comparison with bootstrap resampling.

    For each harmonized cell type:
    1. Get all samples from both atlases
    2. Bootstrap resample to get matched sample sizes
    3. Compute mean and CI for activity differences

    Returns:
        dict with resampled comparison statistics
    """
    print(f"\n  Computing resampled pseudobulk comparison: {atlas1} vs {atlas2}")

    # Load data
    act1, meta1 = load_pseudobulk_data(atlas1, sig_type)
    act2, meta2 = load_pseudobulk_data(atlas2, sig_type)

    if act1 is None or act2 is None:
        return {'data': [], 'n': 0}

    # Get mapping
    mapping1 = get_coarse_mapping(atlas1) if level == 'coarse' else get_fine_mapping(atlas1)
    mapping2 = get_coarse_mapping(atlas2) if level == 'coarse' else get_fine_mapping(atlas2)

    meta1['harmonized'] = meta1['cell_type'].map(mapping1)
    meta2['harmonized'] = meta2['cell_type'].map(mapping2)

    common_types = set(meta1['harmonized'].dropna()) & set(meta2['harmonized'].dropna())
    common_sigs = set(act1.columns) & set(act2.columns)

    if sig_type == 'SecAct':
        combined_var = act1[list(common_sigs)].var() + act2[list(common_sigs)].var()
        common_sigs = set(combined_var.nlargest(100).index)

    common_sigs = sorted(common_sigs)

    print(f"    Common types: {len(common_types)}, Running {n_bootstrap} bootstrap iterations...")

    resampled_data = []

    for ct in sorted(common_types):
        mask1 = meta1['harmonized'] == ct
        mask2 = meta2['harmonized'] == ct

        samples1 = act1.loc[mask1, common_sigs]
        samples2 = act2.loc[mask2, common_sigs]

        n1, n2 = len(samples1), len(samples2)
        if n1 < 2 or n2 < 2:
            continue

        # Bootstrap
        n_resample = min(n1, n2)

        for sig in common_sigs:
            boot_diffs = []

            for _ in range(n_bootstrap):
                idx1 = np.random.choice(n1, n_resample, replace=True)
                idx2 = np.random.choice(n2, n_resample, replace=True)

                mean1 = samples1.iloc[idx1][sig].mean()
                mean2 = samples2.iloc[idx2][sig].mean()
                boot_diffs.append(mean2 - mean1)

            boot_diffs = np.array(boot_diffs)

            resampled_data.append({
                'signature': sig,
                'cell_type': ct,
                'mean_diff': float(np.mean(boot_diffs)),
                'ci_low': float(np.percentile(boot_diffs, 2.5)),
                'ci_high': float(np.percentile(boot_diffs, 97.5)),
                'x_mean': float(samples1[sig].mean()),
                'y_mean': float(samples2[sig].mean()),
                'n_x': n1,
                'n_y': n2
            })

    return {
        'data': resampled_data,
        'n': len(resampled_data),
        'n_bootstrap': n_bootstrap
    }


def compute_singlecell_comparison(atlas1: str, atlas2: str, sig_type: str,
                                   level: str = 'coarse', n_cells: int = 5000) -> dict:
    """
    Compute single-cell level comparison.

    For each harmonized cell type, sample cells from both atlases
    and compare activity distributions.

    Returns:
        dict with per-celltype distribution comparisons
    """
    print(f"\n  Computing single-cell comparison: {atlas1} vs {atlas2}")

    # Get mapping
    mapping1 = get_coarse_mapping(atlas1) if level == 'coarse' else get_fine_mapping(atlas1)
    mapping2 = get_coarse_mapping(atlas2) if level == 'coarse' else get_fine_mapping(atlas2)

    # Get common cell types
    common_types = set(mapping1.values()) & set(mapping2.values())

    if sig_type == 'CytoSig':
        target_sigs = ['IFNG', 'TNF', 'IL6', 'IL10', 'IL17A', 'TGFB1', 'IL1B', 'IL4']
    else:
        target_sigs = None  # Will select top variable

    sc_data = []

    for ct in sorted(common_types)[:6]:  # Limit to 6 cell types for performance
        print(f"    Processing {ct}...")

        # Get original cell types that map to this harmonized type
        orig_types1 = [k for k, v in mapping1.items() if v == ct]
        orig_types2 = [k for k, v in mapping2.items() if v == ct]

        # Load samples
        act1, meta1 = load_singlecell_sample(atlas1, sig_type, n_cells=n_cells)
        act2, meta2 = load_singlecell_sample(atlas2, sig_type, n_cells=n_cells)

        if act1 is None or act2 is None:
            continue

        # Get cell type column
        ct_col1 = 'cell_type' if 'cell_type' in meta1.columns else ('Level2' if 'Level2' in meta1.columns else 'subCluster')
        ct_col2 = 'cell_type' if 'cell_type' in meta2.columns else ('Level2' if 'Level2' in meta2.columns else 'subCluster')

        # Filter to target cell types
        mask1 = meta1[ct_col1].isin(orig_types1)
        mask2 = meta2[ct_col2].isin(orig_types2)

        if mask1.sum() < 10 or mask2.sum() < 10:
            continue

        cells1 = act1.loc[mask1]
        cells2 = act2.loc[mask2]

        # Get common signatures
        common_sigs = set(cells1.columns) & set(cells2.columns)

        if target_sigs:
            common_sigs = common_sigs & set(target_sigs)
        else:
            # Top 20 most variable
            var = cells1[list(common_sigs)].var() + cells2[list(common_sigs)].var()
            common_sigs = set(var.nlargest(20).index)

        for sig in sorted(common_sigs):
            vals1 = cells1[sig].dropna().values
            vals2 = cells2[sig].dropna().values

            if len(vals1) < 10 or len(vals2) < 10:
                continue

            # KS test
            ks_stat, ks_p = stats.ks_2samp(vals1, vals2)

            # Effect size (Cohen's d)
            pooled_std = np.sqrt((vals1.std()**2 + vals2.std()**2) / 2)
            cohens_d = (vals1.mean() - vals2.mean()) / pooled_std if pooled_std > 0 else 0

            sc_data.append({
                'signature': sig,
                'cell_type': ct,
                'x_mean': float(vals1.mean()),
                'y_mean': float(vals2.mean()),
                'x_std': float(vals1.std()),
                'y_std': float(vals2.std()),
                'x_n': int(len(vals1)),
                'y_n': int(len(vals2)),
                'ks_stat': float(ks_stat),
                'ks_pvalue': float(ks_p),
                'cohens_d': float(cohens_d),
                # Distribution percentiles for violin plot
                'x_q25': float(np.percentile(vals1, 25)),
                'x_q50': float(np.percentile(vals1, 50)),
                'x_q75': float(np.percentile(vals1, 75)),
                'y_q25': float(np.percentile(vals2, 25)),
                'y_q50': float(np.percentile(vals2, 50)),
                'y_q75': float(np.percentile(vals2, 75)),
            })

    return {
        'data': sc_data,
        'n': len(sc_data)
    }


def compute_prediction_concordance(atlas1: str, atlas2: str, sig_type: str) -> dict:
    """
    Compute cross-atlas prediction concordance.

    Train a simple predictor on atlas1 cell types, test on atlas2.
    Measures how well cell type signatures generalize.

    Returns:
        dict with concordance metrics
    """
    print(f"\n  Computing prediction concordance: {atlas1} -> {atlas2}")

    # Load cell type aggregated data
    act1, meta1 = load_pseudobulk_data(atlas1, sig_type)
    act2, meta2 = load_pseudobulk_data(atlas2, sig_type)

    if act1 is None or act2 is None:
        # Single-cell files lack cell type metadata, can't do prediction
        print(f"    [SKIP] No pseudobulk data available")
        return {'overall_accuracy': None, 'n_samples': 0, 'note': 'Pseudobulk data required'}

    mapping1 = get_coarse_mapping(atlas1)
    mapping2 = get_coarse_mapping(atlas2)

    meta1['harmonized'] = meta1['cell_type'].map(mapping1)
    meta2['harmonized'] = meta2['cell_type'].map(mapping2)

    common_types = sorted(set(meta1['harmonized'].dropna()) & set(meta2['harmonized'].dropna()))
    common_sigs = sorted(set(act1.columns) & set(act2.columns))

    if sig_type == 'SecAct':
        var = act1[common_sigs].var() + act2[common_sigs].var()
        common_sigs = list(var.nlargest(50).index)

    print(f"    Common types: {len(common_types)}, Signatures: {len(common_sigs)}")

    # Compute centroid for each cell type in atlas1
    centroids1 = {}
    for ct in common_types:
        mask = meta1['harmonized'] == ct
        if mask.sum() > 0:
            centroids1[ct] = act1.loc[mask, common_sigs].mean().values

    # For each sample in atlas2, find nearest centroid
    concordance_details = []
    correct = 0
    total = 0

    for ct in common_types:
        mask2 = meta2['harmonized'] == ct
        if mask2.sum() == 0:
            continue

        samples2 = act2.loc[mask2, common_sigs]

        for idx, row in samples2.iterrows():
            sample_vec = row.values

            # Find nearest centroid
            min_dist = float('inf')
            predicted_ct = None

            for ref_ct, centroid in centroids1.items():
                dist = cosine(sample_vec, centroid)
                if dist < min_dist:
                    min_dist = dist
                    predicted_ct = ref_ct

            is_correct = predicted_ct == ct
            if is_correct:
                correct += 1
            total += 1

            concordance_details.append({
                'true_type': ct,
                'predicted_type': predicted_ct,
                'correct': is_correct,
                'distance': float(min_dist)
            })

    accuracy = correct / total if total > 0 else 0

    # Per-celltype accuracy
    ct_accuracy = {}
    for ct in common_types:
        ct_details = [d for d in concordance_details if d['true_type'] == ct]
        if ct_details:
            ct_accuracy[ct] = sum(d['correct'] for d in ct_details) / len(ct_details)

    return {
        'overall_accuracy': float(accuracy),
        'n_samples': total,
        'per_celltype_accuracy': ct_accuracy,
        'confusion_summary': concordance_details[:100]  # Limit for JSON size
    }


def main():
    print("=" * 70)
    print("Generating Comprehensive Atlas Comparison Data")
    print("=" * 70)

    results = {}

    # Define comparison pairs
    pairs = [
        ('cima', 'inflammation'),
        ('cima', 'scatlas'),
        ('inflammation', 'scatlas')
    ]

    for sig_type in ['CytoSig', 'SecAct']:
        print(f"\n{'='*70}")
        print(f"Processing {sig_type}")
        print("=" * 70)

        sig_key = sig_type.lower()
        results[sig_key] = {}

        for atlas1, atlas2 in pairs:
            pair_key = f"{atlas1}_vs_{atlas2}"
            print(f"\n{'-'*60}")
            print(f"Comparison: {atlas1.upper()} vs {atlas2.upper()}")
            print("-" * 60)

            results[sig_key][pair_key] = {}

            # 1. Cell type aggregated comparison (coarse)
            results[sig_key][pair_key]['celltype_aggregated_coarse'] = \
                compute_celltype_aggregated_comparison(atlas1, atlas2, sig_type, level='coarse')

            # 2. Cell type aggregated comparison (fine)
            results[sig_key][pair_key]['celltype_aggregated_fine'] = \
                compute_celltype_aggregated_comparison(atlas1, atlas2, sig_type, level='fine')

            # 3. Pseudobulk resampled comparison (if available)
            if DATA_FILES.get(atlas1, {}).get(f'{sig_key}_pb') and \
               DATA_FILES.get(atlas2, {}).get(f'{sig_key}_pb'):
                results[sig_key][pair_key]['pseudobulk_resampled'] = \
                    compute_pseudobulk_resampled_comparison(atlas1, atlas2, sig_type,
                                                            level='coarse', n_bootstrap=50)
            else:
                results[sig_key][pair_key]['pseudobulk_resampled'] = {'data': [], 'n': 0}

            # 4. Single-cell comparison (skip for now - metadata not in singlecell h5ad)
            # The singlecell h5ad files don't contain cell type metadata
            # Would need to join with original h5ad files to get cell type
            results[sig_key][pair_key]['singlecell'] = {'data': [], 'n': 0, 'note': 'Metadata not available in singlecell h5ad'}

            # 5. Prediction concordance
            results[sig_key][pair_key]['prediction_concordance'] = \
                compute_prediction_concordance(atlas1, atlas2, sig_type)

    # Save results
    print("\n" + "=" * 70)
    print("Saving results...")
    print("=" * 70)

    # Update cross_atlas.json
    cross_atlas_path = VIZ_OUTPUT_DIR / 'cross_atlas.json'
    if cross_atlas_path.exists():
        with open(cross_atlas_path, 'r') as f:
            cross_atlas = json.load(f)
    else:
        cross_atlas = {}

    # Store under 'atlas_comparison' key
    cross_atlas['atlas_comparison'] = results

    with open(cross_atlas_path, 'w') as f:
        json.dump(cross_atlas, f)

    print(f"Updated: {cross_atlas_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    for sig_key, sig_results in results.items():
        print(f"\n{sig_key.upper()}:")
        for pair_key, pair_results in sig_results.items():
            print(f"\n  {pair_key}:")
            for comp_type, comp_data in pair_results.items():
                if isinstance(comp_data, dict):
                    n = comp_data.get('n', len(comp_data.get('data', [])))
                    corr = comp_data.get('correlation', comp_data.get('overall_accuracy'))
                    if corr is not None:
                        print(f"    {comp_type}: n={n}, r/acc={corr:.3f}")
                    else:
                        print(f"    {comp_type}: n={n}")

    print("\nDone!")


if __name__ == '__main__':
    main()
