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


def compute_lins_ccc(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute Lin's Concordance Correlation Coefficient.

    CCC measures both precision (correlation) and accuracy (bias).
    CCC = 1 means perfect agreement, CCC = 0 means no agreement.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    # Remove NaN pairs
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]

    if len(x) < 3:
        return np.nan

    mean_x, mean_y = np.mean(x), np.mean(y)
    var_x, var_y = np.var(x, ddof=1), np.var(y, ddof=1)
    cov_xy = np.cov(x, y, ddof=1)[0, 1]

    # Lin's CCC formula
    ccc = (2 * cov_xy) / (var_x + var_y + (mean_x - mean_y)**2)

    return float(ccc)


def compute_signature_agreement(atlas1: str, atlas2: str, sig_type: str,
                                 level: str = 'coarse') -> dict:
    """
    Compute signature agreement metrics focusing on SIMILARITY between atlases.

    Metrics computed:
    1. Lin's Concordance Correlation Coefficient (CCC) - overall and per cell type
    2. Equivalence: % of signatures within ±0.5, ±1.0, ±2.0 z-score bounds
    3. Rank agreement: % of signatures with same direction (both positive or both negative)

    Returns:
        dict with agreement metrics
    """
    print(f"\n  Computing signature agreement: {atlas1} vs {atlas2}")

    # Load data
    act1, meta1 = load_pseudobulk_data(atlas1, sig_type)
    act2, meta2 = load_pseudobulk_data(atlas2, sig_type)

    if act1 is None or act2 is None:
        return {'celltype_agreement': [], 'equivalence': {}, 'overall_ccc': None, 'n': 0}

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
    common_types = sorted(common_types)

    print(f"    Common types: {len(common_types)}, Common signatures: {len(common_sigs)}")

    # Collect all paired values for overall metrics
    all_x, all_y = [], []
    all_diffs = []

    # Per-cell type agreement
    celltype_agreement = []

    for ct in common_types:
        mask1 = meta1['harmonized'] == ct
        mask2 = meta2['harmonized'] == ct

        if mask1.sum() == 0 or mask2.sum() == 0:
            continue

        mean1 = act1.loc[mask1, common_sigs].mean()
        mean2 = act2.loc[mask2, common_sigs].mean()

        x_vals = mean1.values
        y_vals = mean2.values

        # Remove NaN
        valid_mask = ~(np.isnan(x_vals) | np.isnan(y_vals))
        x_clean = x_vals[valid_mask]
        y_clean = y_vals[valid_mask]

        if len(x_clean) < 3:
            continue

        all_x.extend(x_clean)
        all_y.extend(y_clean)

        diffs = np.abs(x_clean - y_clean)
        all_diffs.extend(diffs)

        # Per-cell type CCC
        ccc = compute_lins_ccc(x_clean, y_clean)

        # Spearman correlation for this cell type
        if len(x_clean) > 2:
            r, _ = stats.spearmanr(x_clean, y_clean)
        else:
            r = np.nan

        # Direction agreement: % where both have same sign
        same_direction = np.sum((x_clean > 0) == (y_clean > 0)) / len(x_clean)

        # Equivalence for this cell type
        equiv_0_5 = np.sum(diffs <= 0.5) / len(diffs)
        equiv_1_0 = np.sum(diffs <= 1.0) / len(diffs)
        equiv_2_0 = np.sum(diffs <= 2.0) / len(diffs)

        celltype_agreement.append({
            'cell_type': ct,
            'ccc': float(ccc) if not np.isnan(ccc) else None,
            'spearman_r': float(r) if not np.isnan(r) else None,
            'direction_agreement': float(same_direction),
            'equiv_0_5': float(equiv_0_5),
            'equiv_1_0': float(equiv_1_0),
            'equiv_2_0': float(equiv_2_0),
            'n_signatures': len(x_clean),
            'mean_abs_diff': float(np.mean(diffs)),
            'median_abs_diff': float(np.median(diffs))
        })

    # Overall metrics
    all_x = np.array(all_x)
    all_y = np.array(all_y)
    all_diffs = np.array(all_diffs)

    if len(all_x) > 0:
        overall_ccc = compute_lins_ccc(all_x, all_y)
        overall_spearman, _ = stats.spearmanr(all_x, all_y) if len(all_x) > 2 else (np.nan, np.nan)

        # Equivalence bounds
        equivalence = {
            'within_0_5': float(np.sum(all_diffs <= 0.5) / len(all_diffs)),
            'within_1_0': float(np.sum(all_diffs <= 1.0) / len(all_diffs)),
            'within_2_0': float(np.sum(all_diffs <= 2.0) / len(all_diffs)),
            'within_3_0': float(np.sum(all_diffs <= 3.0) / len(all_diffs)),
            'mean_abs_diff': float(np.mean(all_diffs)),
            'median_abs_diff': float(np.median(all_diffs))
        }

        # Direction agreement overall
        direction_agreement = float(np.sum((all_x > 0) == (all_y > 0)) / len(all_x))

        # Difference distribution for histogram
        diff_distribution = {
            'bins': [-4, -3, -2, -1, -0.5, 0.5, 1, 2, 3, 4],
            'counts': []
        }
        bins = diff_distribution['bins']
        signed_diffs = all_y - all_x
        for i in range(len(bins) - 1):
            count = np.sum((signed_diffs >= bins[i]) & (signed_diffs < bins[i+1]))
            diff_distribution['counts'].append(int(count))
        # Add overflow bins
        diff_distribution['below'] = int(np.sum(signed_diffs < bins[0]))
        diff_distribution['above'] = int(np.sum(signed_diffs >= bins[-1]))
    else:
        overall_ccc = None
        overall_spearman = None
        equivalence = {}
        direction_agreement = None
        diff_distribution = {}

    return {
        'celltype_agreement': celltype_agreement,
        'equivalence': equivalence,
        'diff_distribution': diff_distribution,
        'overall_ccc': float(overall_ccc) if overall_ccc and not np.isnan(overall_ccc) else None,
        'overall_spearman': float(overall_spearman) if overall_spearman and not np.isnan(overall_spearman) else None,
        'direction_agreement': direction_agreement,
        'n': len(all_x),
        'n_celltypes': len(celltype_agreement),
        'n_signatures': len(common_sigs)
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


# Original h5ad files with cell type metadata
ORIGINAL_H5AD = {
    'cima': Path('/data/Jiang_Lab/Data/Seongyong/CIMA/Cell_Atlas/CIMA_RNA_6484974cells_36326genes_compressed.h5ad'),
    'inflammation': Path('/data/Jiang_Lab/Data/Seongyong/Inflammation_Atlas/INFLAMMATION_ATLAS_main_afterQC.h5ad'),
    'scatlas': Path('/data/Jiang_Lab/Data/Seongyong/scAtlas_2025/igt_s9_fine_counts.h5ad'),
}

# Cell type column names in original h5ad files
CELLTYPE_COLUMNS = {
    'cima': 'cell_type_l2',  # Level 2 matches pseudobulk granularity
    'inflammation': 'Level2',
    'scatlas': 'cellType1',
}


def compute_singlecell_mean_activity(atlas1: str, atlas2: str, sig_type: str,
                                      level: str = 'coarse', n_cells: int = 50000) -> dict:
    """
    Compute single-cell mean activity comparison between atlases.

    For each atlas:
    1. Sample cells from single-cell activity h5ad
    2. Get cell type from original h5ad (by matching cell index)
    3. Map to harmonized cell types (coarse or fine)
    4. Compute mean activity per harmonized cell type

    Args:
        atlas1, atlas2: Atlas names
        sig_type: 'CytoSig' or 'SecAct'
        level: 'coarse' or 'fine' cell type granularity
        n_cells: Number of cells to sample per atlas

    Returns comparison data for scatter plot.
    """
    print(f"\n  Computing single-cell mean activity ({level}): {atlas1} vs {atlas2}")

    def load_singlecell_with_celltype(atlas: str, sig_type: str, n_sample: int) -> tuple:
        """Load single-cell activity with cell type metadata."""
        # Activity file
        key = f'{sig_type.lower()}_sc'
        act_path = DATA_FILES.get(atlas, {}).get(key)
        if not act_path or not act_path.exists():
            print(f"    [SKIP] Activity file not found for {atlas}")
            return None, None

        # Original h5ad for cell type
        orig_path = ORIGINAL_H5AD.get(atlas)
        if not orig_path or not orig_path.exists():
            print(f"    [SKIP] Original h5ad not found for {atlas}")
            return None, None

        ct_col = CELLTYPE_COLUMNS.get(atlas)

        print(f"    Loading {atlas} single-cell data...")

        try:
            # Load activity h5ad
            act_adata = ad.read_h5ad(act_path, backed='r')
            n_total = act_adata.n_obs
        except Exception as e:
            print(f"    [ERROR] Failed to load activity h5ad: {e}")
            return None, None

        # Sample cell indices
        np.random.seed(42)
        sample_idx = np.random.choice(n_total, min(n_sample, n_total), replace=False)
        sample_idx = np.sort(sample_idx)

        # Load activity for sampled cells
        X = act_adata.X[sample_idx, :]
        if hasattr(X, 'toarray'):
            X = X.toarray()

        # Get signature names
        if '_index' in act_adata.var.columns:
            signatures = list(act_adata.var['_index'].values)
        else:
            signatures = list(act_adata.var_names)

        act_df = pd.DataFrame(X, columns=signatures)

        # Load cell type for sampled cells from original h5ad
        print(f"    Loading cell types from original h5ad...")
        orig_adata = ad.read_h5ad(orig_path, backed='r')

        # Get cell type column
        if ct_col in orig_adata.obs.columns:
            # For backed mode, need to load just the column
            cell_types = orig_adata.obs[ct_col].iloc[sample_idx].values
        else:
            print(f"    [WARN] Cell type column {ct_col} not found")
            return None, None

        meta_df = pd.DataFrame({'cell_type': cell_types})

        print(f"    Loaded {len(act_df)} cells, {len(signatures)} signatures")

        return act_df, meta_df

    # Load data for both atlases
    act1, meta1 = load_singlecell_with_celltype(atlas1, sig_type, n_cells)
    act2, meta2 = load_singlecell_with_celltype(atlas2, sig_type, n_cells)

    if act1 is None or act2 is None:
        return {'data': [], 'n': 0, 'overall_correlation': None}

    # Get mappings based on level
    if level == 'fine':
        mapping1 = get_fine_mapping(atlas1)
        mapping2 = get_fine_mapping(atlas2)
    else:
        mapping1 = get_coarse_mapping(atlas1)
        mapping2 = get_coarse_mapping(atlas2)

    # Map cell types
    meta1['harmonized'] = meta1['cell_type'].map(mapping1)
    meta2['harmonized'] = meta2['cell_type'].map(mapping2)

    # Get common types and signatures
    common_types = set(meta1['harmonized'].dropna().unique()) & set(meta2['harmonized'].dropna().unique())
    common_sigs = set(act1.columns) & set(act2.columns)

    if sig_type == 'SecAct':
        # Limit to top 100 most variable (more for SecAct to show diversity)
        combined_var = act1[list(common_sigs)].var() + act2[list(common_sigs)].var()
        common_sigs = set(combined_var.nlargest(100).index)

    common_sigs = sorted(common_sigs)
    common_types = sorted(common_types)

    print(f"    Common types: {len(common_types)}, Common signatures: {len(common_sigs)}")

    # Compute mean activity per cell type
    comparison_data = []
    all_x, all_y = [], []

    for ct in common_types:
        mask1 = meta1['harmonized'] == ct
        mask2 = meta2['harmonized'] == ct

        n1, n2 = mask1.sum(), mask2.sum()
        if n1 < 10 or n2 < 10:
            continue

        mean1 = act1.loc[mask1, common_sigs].mean()
        mean2 = act2.loc[mask2, common_sigs].mean()

        for sig in common_sigs:
            x_val = float(mean1[sig])
            y_val = float(mean2[sig])

            if np.isnan(x_val) or np.isnan(y_val):
                continue

            comparison_data.append({
                'signature': sig,
                'cell_type': ct,
                'x': x_val,
                'y': y_val,
                'n_cells_x': int(n1),
                'n_cells_y': int(n2)
            })

            all_x.append(x_val)
            all_y.append(y_val)

    # Compute overall correlation
    if len(all_x) > 2:
        r, p = stats.spearmanr(all_x, all_y)
    else:
        r, p = np.nan, np.nan

    return {
        'data': comparison_data,
        'n': len(comparison_data),
        'n_celltypes': len(common_types),
        'n_signatures': len(common_sigs),
        'overall_correlation': float(r) if not np.isnan(r) else None,
        'overall_pvalue': float(p) if not np.isnan(p) else None,
        'n_cells_atlas1': int(meta1['harmonized'].notna().sum()),
        'n_cells_atlas2': int(meta2['harmonized'].notna().sum())
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

            # 3. Single-cell mean activity comparison (coarse level)
            # Join activity data with cell type from original h5ad files
            if DATA_FILES.get(atlas1, {}).get(f'{sig_key}_sc') and \
               DATA_FILES.get(atlas2, {}).get(f'{sig_key}_sc'):
                results[sig_key][pair_key]['singlecell_mean_coarse'] = \
                    compute_singlecell_mean_activity(atlas1, atlas2, sig_type, level='coarse', n_cells=50000)
            else:
                results[sig_key][pair_key]['singlecell_mean_coarse'] = {'data': [], 'n': 0}

            # 4. Single-cell mean activity comparison (fine level)
            if DATA_FILES.get(atlas1, {}).get(f'{sig_key}_sc') and \
               DATA_FILES.get(atlas2, {}).get(f'{sig_key}_sc'):
                results[sig_key][pair_key]['singlecell_mean_fine'] = \
                    compute_singlecell_mean_activity(atlas1, atlas2, sig_type, level='fine', n_cells=50000)
            else:
                results[sig_key][pair_key]['singlecell_mean_fine'] = {'data': [], 'n': 0}

    # Compute per-signature cross-atlas correlations for reliability assessment
    print("\n" + "=" * 70)
    print("Computing per-signature cross-atlas correlations...")
    print("=" * 70)

    signature_reliability = {}
    pair_names = ['cima_vs_inflammation', 'cima_vs_scatlas', 'inflammation_vs_scatlas']
    pair_labels = ['CIMA-Inflam', 'CIMA-scAtlas', 'Inflam-scAtlas']

    for sig_type in ['cytosig', 'secact']:
        print(f"\n{sig_type.upper()}:")
        sig_reliability = {
            'signatures': [],
            'summary': {
                'total': 0,
                'highly_conserved': 0,  # r > 0.7 in 2+ pairs
                'moderately_conserved': 0,  # r > 0.5 in 2+ pairs
                'atlas_specific': 0,  # low correlation in available pairs
                'insufficient_data': 0,  # only 1 pair has data
            },
            'pair_correlations': {}  # Overall correlation for each pair
        }

        # Get data from celltype_aggregated_coarse (most reliable)
        all_signatures = set()
        pair_data = {}
        for pair_key in pair_names:
            pair_result = results.get(sig_type, {}).get(pair_key, {})
            data = pair_result.get('celltype_aggregated_coarse', {}).get('data', [])
            pair_data[pair_key] = data
            # Overall correlation for this pair
            if pair_result.get('celltype_aggregated_coarse', {}).get('correlation'):
                sig_reliability['pair_correlations'][pair_key] = {
                    'correlation': pair_result['celltype_aggregated_coarse']['correlation'],
                    'pvalue': pair_result['celltype_aggregated_coarse'].get('pvalue', 0),
                    'n': pair_result['celltype_aggregated_coarse'].get('n', len(data))
                }
            for pt in data:
                all_signatures.add(pt['signature'])

        # Compute per-signature correlation for each pair
        for sig in sorted(all_signatures):
            sig_info = {
                'signature': sig,
                'correlations': {},
                'n_reliable_pairs': 0,
                'n_pairs_with_data': 0,
                'mean_correlation': 0,
                'category': 'unknown'
            }

            valid_correlations = []
            for pair_key in pair_names:
                data = pair_data[pair_key]
                sig_points = [pt for pt in data if pt['signature'] == sig]
                if len(sig_points) >= 3:
                    xs = [pt['x'] for pt in sig_points]
                    ys = [pt['y'] for pt in sig_points]
                    try:
                        r, p = stats.spearmanr(xs, ys)
                        if not np.isnan(r):
                            sig_info['correlations'][pair_key] = {
                                'r': round(float(r), 3),
                                'p': float(p),
                                'n': len(sig_points)
                            }
                            valid_correlations.append(r)
                            sig_info['n_pairs_with_data'] += 1
                            if r > 0.7:
                                sig_info['n_reliable_pairs'] += 1
                    except Exception:
                        pass

            if valid_correlations:
                sig_info['mean_correlation'] = round(float(np.mean(valid_correlations)), 3)

                # Categorize based on number of pairs and correlation values
                n_pairs = len(valid_correlations)
                n_high = sum(1 for r in valid_correlations if r > 0.7)
                n_moderate = sum(1 for r in valid_correlations if r > 0.5)

                if n_pairs < 2:
                    # Only 1 atlas pair has data - insufficient for cross-atlas assessment
                    sig_info['category'] = 'insufficient_data'
                    sig_reliability['summary']['insufficient_data'] += 1
                elif n_high >= 2:
                    # High correlation in 2+ pairs
                    sig_info['category'] = 'highly_conserved'
                    sig_reliability['summary']['highly_conserved'] += 1
                elif n_moderate >= 2:
                    # Moderate correlation in 2+ pairs
                    sig_info['category'] = 'moderately_conserved'
                    sig_reliability['summary']['moderately_conserved'] += 1
                else:
                    # Low correlation - atlas-specific patterns
                    sig_info['category'] = 'atlas_specific'
                    sig_reliability['summary']['atlas_specific'] += 1

            sig_reliability['signatures'].append(sig_info)

        sig_reliability['summary']['total'] = len(sig_reliability['signatures'])

        # Sort by: 1) number of pairs with data (desc), 2) mean correlation (desc)
        sig_reliability['signatures'].sort(key=lambda x: (-x['n_pairs_with_data'], -x['mean_correlation']))

        signature_reliability[sig_type] = sig_reliability

        print(f"  Total: {sig_reliability['summary']['total']}")
        print(f"  Highly conserved (r>0.7 in 2+ pairs): {sig_reliability['summary']['highly_conserved']}")
        print(f"  Moderately conserved (r>0.5 in 2+ pairs): {sig_reliability['summary']['moderately_conserved']}")
        print(f"  Atlas-specific (low correlation): {sig_reliability['summary']['atlas_specific']}")
        print(f"  Insufficient data (1 pair only): {sig_reliability['summary']['insufficient_data']}")

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

    # Store signature reliability data
    cross_atlas['signature_reliability'] = signature_reliability

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
