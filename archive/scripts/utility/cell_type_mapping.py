#!/usr/bin/env python3
"""
Cell Type Mapping Module for Cross-Atlas Harmonization.

This module provides mapping dictionaries and functions to harmonize cell type
annotations across CIMA, Inflammation Atlas, and scAtlas.

Mapping Levels:
- Coarse (8 lineages): Maximum overlap for cross-atlas comparison
- Fine (~30 types): Balance between resolution and comparability
"""

from typing import Dict, List, Optional, Set
import pandas as pd

# =============================================================================
# Coarse Level Mapping (8 Lineages)
# =============================================================================

COARSE_LINEAGES = [
    'CD4_T', 'CD8_T', 'NK', 'B', 'Plasma',
    'Myeloid', 'Unconventional_T', 'Progenitor'
]

CIMA_TO_COARSE = {
    # CD4 T cells
    'CD4_CTL': 'CD4_T',
    'CD4_helper': 'CD4_T',
    'CD4_memory': 'CD4_T',
    'CD4_naive': 'CD4_T',
    'CD4_regulatory': 'CD4_T',
    # CD8 T cells
    'CD8_CTL': 'CD8_T',
    'CD8_memory': 'CD8_T',
    'CD8_naive': 'CD8_T',
    # NK cells
    'CD56_bright_NK': 'NK',
    'CD56_dim_NK': 'NK',
    'Proliferative_NK': 'NK',
    'Transitional_NK': 'NK',
    # Unconventional T cells
    'NKT': 'Unconventional_T',
    'MAIT': 'Unconventional_T',
    'gdT': 'Unconventional_T',
    'ILC2': 'Unconventional_T',
    # B cells
    'Memory_B': 'B',
    'Naive_B': 'B',
    'Transitional_B': 'B',
    # Plasma cells
    'Total_Plasma': 'Plasma',
    # Myeloid
    'Mono': 'Myeloid',
    'DC': 'Myeloid',
    'pDC': 'Myeloid',
    # Progenitors
    'HSPC': 'Progenitor',
    'MK': 'Progenitor',
    'Immature_T': 'Progenitor',
    'Proliferative_T': 'Progenitor',
}

INFLAMMATION_TO_COARSE = {
    # CD4 T cells
    'T_CD4_Naive': 'CD4_T',
    'T_CD4_CM': 'CD4_T',
    'T_CD4_CM_ribo': 'CD4_T',
    'T_CD4_EM': 'CD4_T',
    'T_CD4_EMRA': 'CD4_T',
    'T_CD4_eff': 'CD4_T',
    'Th0': 'CD4_T',
    'Th1': 'CD4_T',
    'Th2': 'CD4_T',
    'Tregs': 'CD4_T',
    'Tregs_activated': 'CD4_T',
    # CD8 T cells
    'T_CD8_Naive': 'CD8_T',
    'T_CD8_CM': 'CD8_T',
    'T_CD8_CM_stem': 'CD8_T',
    'T_CD8_EM_CX3CR1high': 'CD8_T',
    'T_CD8_EM_CX3CR1int': 'CD8_T',
    'T_CD8_Mem_cytotoxic': 'CD8_T',
    'T_CD8_activated': 'CD8_T',
    'T_CD8_arrested': 'CD8_T',
    'T_CD8_eff_HOBIT': 'CD8_T',
    'T_CD8_IFNresponse': 'CD8_T',
    # NK cells
    'NK_CD16high': 'NK',
    'NK_CD56dimCD16': 'NK',
    'NK_CD56high': 'NK',
    'NK_IFN1response': 'NK',
    'NK_Proliferative': 'NK',
    'NK_adaptive': 'NK',
    'NK_lowRibocontent': 'NK',
    # Unconventional T cells
    'MAIT': 'Unconventional_T',
    'MAIT_17': 'Unconventional_T',
    'gdT_V1': 'Unconventional_T',
    'gdT_V2_Vγ9': 'Unconventional_T',
    # B cells
    'B_Naive': 'B',
    'B_Naive_activated': 'B',
    'B_Transitional': 'B',
    'B_Memory_switched': 'B',
    'B_Memory_unswitched': 'B',
    'B_Memory_ITGAX': 'B',
    'B_IFNresponder': 'B',
    'B_Progenitors': 'B',
    # Plasma cells
    'Plasma_IGHA': 'Plasma',
    'Plasma_IGHG': 'Plasma',
    'Plasma_Proliferative': 'Plasma',
    'Plasma_XBP1': 'Plasma',
    # Myeloid
    'Mono_classical': 'Myeloid',
    'Mono_inflammatory': 'Myeloid',
    'Mono_nonClassical': 'Myeloid',
    'Mono_regulatory': 'Myeloid',
    'Mono_IFNresponse': 'Myeloid',
    'cDC1': 'Myeloid',
    'cDC2': 'Myeloid',
    'cDC3': 'Myeloid',
    'DC4': 'Myeloid',
    'DC5': 'Myeloid',
    'DC_CCR7': 'Myeloid',
    'DC_Proliferative': 'Myeloid',
    'pDC': 'Myeloid',
    # Progenitors
    'HSC_LMP': 'Progenitor',
    'HSC_MEMP': 'Progenitor',
    'HSC_MMP': 'Progenitor',
    'T_Progenitors': 'Progenitor',
    'T_Proliferative': 'Progenitor',
    # Excluded (QC failures)
    'Doublets': None,
    'LowQuality_cells': None,
    'Platelets': None,
    'RBC': None,
    'UTC': None,
}

SCATLAS_TO_COARSE = {
    # CD4 T cells
    'CD4T01_Tn_SOX4': 'CD4_T',
    'CD4T02_Tn_CCR7': 'CD4_T',
    'CD4T03_Tn_NR4A1': 'CD4_T',
    'CD4T04_Tfh_IL6ST': 'CD4_T',
    'CD4T05_Tm_LTB': 'CD4_T',
    'CD4T06_Tm_ANXA1': 'CD4_T',
    'CD4T07_Tem_GZMK': 'CD4_T',
    'CD4T08_Treg_FOXP3': 'CD4_T',
    # CD8 T cells
    'CD8T01_Tn_CCR7': 'CD8_T',
    'CD8T02_Tem_GZMK': 'CD8_T',
    'CD8T03_Trm_ITGA1': 'CD8_T',
    'CD8T04_Trm_HSPA1A': 'CD8_T',
    'CD8T05_Temra_GZMH': 'CD8_T',
    # Unconventional T / ILC
    'CD8T06_MAIT_SLC4A10': 'Unconventional_T',
    'I08_ILC3_KIT': 'Unconventional_T',
    'I09_gdT_ITGA1': 'Unconventional_T',
    # NK cells
    'I01_CD16hiNK_CREM': 'NK',
    'I02_CD16hiNK_SYNE2': 'NK',
    'I03_CD16hiNK_GZMB': 'NK',
    'I04_CD16hiNK_HSPA1A': 'NK',
    'I05_CD16loNK_SELL': 'NK',
    'I06_CD16loNK_NR4A2': 'NK',
    'I07_CD16loNK_CXCR6': 'NK',
    # B cells
    'B01_Bn_IGHM': 'B',
    'B02_Bn_TCL1A': 'B',
    'B03_Bn_NR4A2': 'B',
    'B04_Bm_CD27': 'B',
    'B05_Bm_NR4A2': 'B',
    'B06_Bm_ITGB1': 'B',
    'B07_Bm_HSPA1A': 'B',
    'B08_ABC_FCRL5': 'B',
    'B09_GCB_RGS13': 'B',
    # Plasma cells
    'B10_Plasmablast_MKI67': 'Plasma',
    'B11_Plasma_IGHG2': 'Plasma',
    'B12_Plasma_IGHA2': 'Plasma',
    # Myeloid
    'M01_cDC1_CLEC9A': 'Myeloid',
    'M02_cDC2_CD1C': 'Myeloid',
    'M03_cDC_LAMP3': 'Myeloid',
    'M04_pDC_LILRA4': 'Myeloid',
    'M05_Mo_CD14': 'Myeloid',
    'M06_Mo_FCGR3A': 'Myeloid',
    'M07_Mph_FCN1': 'Myeloid',
    'M08_Mph_NLRP3': 'Myeloid',
    'M09_Mph_FOLR2': 'Myeloid',
    'M10_Mph_CD5L': 'Myeloid',
    'M11_Mph_PPARG': 'Myeloid',
    'M12_Mph_MT1X': 'Myeloid',
    'M13_immNeu_MMP8': 'Myeloid',
    'M14_mNeu_CXCR2': 'Myeloid',
    'M15_Mast_CPA3': 'Myeloid',
}

# =============================================================================
# Fine Level Mapping (~30 types)
# =============================================================================

FINE_TYPES = [
    'CD4_Naive', 'CD4_Memory', 'CD4_Effector', 'Treg', 'Tfh',
    'CD8_Naive', 'CD8_Memory', 'CD8_Effector', 'CD8_Trm',
    'MAIT', 'gdT', 'NKT', 'ILC',
    'NK_CD56bright', 'NK_CD56dim', 'NK_other',
    'B_Naive', 'B_Memory', 'B_Transitional', 'B_GC', 'Plasmablast', 'Plasma',
    'Mono_Classical', 'Mono_NonClassical', 'Macrophage',
    'cDC1', 'cDC2', 'pDC',
    'Neutrophil', 'Mast',
    'HSPC', 'Progenitor_other'
]

CIMA_TO_FINE = {
    'CD4_naive': 'CD4_Naive',
    'CD4_memory': 'CD4_Memory',
    'CD4_CTL': 'CD4_Effector',
    'CD4_helper': 'CD4_Effector',
    'CD4_regulatory': 'Treg',
    'CD8_naive': 'CD8_Naive',
    'CD8_memory': 'CD8_Memory',
    'CD8_CTL': 'CD8_Effector',
    'MAIT': 'MAIT',
    'gdT': 'gdT',
    'NKT': 'NKT',
    'ILC2': 'ILC',
    'CD56_bright_NK': 'NK_CD56bright',
    'CD56_dim_NK': 'NK_CD56dim',
    'Proliferative_NK': 'NK_other',
    'Transitional_NK': 'NK_other',
    'Naive_B': 'B_Naive',
    'Memory_B': 'B_Memory',
    'Transitional_B': 'B_Transitional',
    'Total_Plasma': 'Plasma',
    'Mono': 'Mono_Classical',
    'DC': 'cDC2',
    'pDC': 'pDC',
    'HSPC': 'HSPC',
    'MK': 'Progenitor_other',
    'Immature_T': 'Progenitor_other',
    'Proliferative_T': 'Progenitor_other',
}

INFLAMMATION_TO_FINE = {
    'T_CD4_Naive': 'CD4_Naive',
    'T_CD4_CM': 'CD4_Memory',
    'T_CD4_CM_ribo': 'CD4_Memory',
    'T_CD4_EM': 'CD4_Effector',
    'T_CD4_EMRA': 'CD4_Effector',
    'T_CD4_eff': 'CD4_Effector',
    'Th0': 'Tfh',
    'Th1': 'Tfh',
    'Th2': 'Tfh',
    'Tregs': 'Treg',
    'Tregs_activated': 'Treg',
    'T_CD8_Naive': 'CD8_Naive',
    'T_CD8_CM': 'CD8_Memory',
    'T_CD8_CM_stem': 'CD8_Memory',
    'T_CD8_EM_CX3CR1high': 'CD8_Effector',
    'T_CD8_EM_CX3CR1int': 'CD8_Effector',
    'T_CD8_Mem_cytotoxic': 'CD8_Effector',
    'T_CD8_activated': 'CD8_Effector',
    'T_CD8_arrested': 'CD8_Memory',
    'T_CD8_eff_HOBIT': 'CD8_Effector',
    'T_CD8_IFNresponse': 'CD8_Effector',
    'MAIT': 'MAIT',
    'MAIT_17': 'MAIT',
    'gdT_V1': 'gdT',
    'gdT_V2_Vγ9': 'gdT',
    'NK_CD56high': 'NK_CD56bright',
    'NK_CD16high': 'NK_CD56dim',
    'NK_CD56dimCD16': 'NK_CD56dim',
    'NK_IFN1response': 'NK_other',
    'NK_Proliferative': 'NK_other',
    'NK_adaptive': 'NK_other',
    'NK_lowRibocontent': 'NK_other',
    'B_Naive': 'B_Naive',
    'B_Naive_activated': 'B_Naive',
    'B_Transitional': 'B_Transitional',
    'B_Memory_switched': 'B_Memory',
    'B_Memory_unswitched': 'B_Memory',
    'B_Memory_ITGAX': 'B_Memory',
    'B_IFNresponder': 'B_Memory',
    'B_Progenitors': 'B_Naive',
    'Plasma_IGHA': 'Plasma',
    'Plasma_IGHG': 'Plasma',
    'Plasma_Proliferative': 'Plasmablast',
    'Plasma_XBP1': 'Plasma',
    'Mono_classical': 'Mono_Classical',
    'Mono_inflammatory': 'Mono_Classical',
    'Mono_nonClassical': 'Mono_NonClassical',
    'Mono_regulatory': 'Mono_Classical',
    'Mono_IFNresponse': 'Mono_Classical',
    'cDC1': 'cDC1',
    'cDC2': 'cDC2',
    'cDC3': 'cDC2',
    'DC4': 'cDC2',
    'DC5': 'cDC2',
    'DC_CCR7': 'cDC2',
    'DC_Proliferative': 'cDC2',
    'pDC': 'pDC',
    'HSC_LMP': 'HSPC',
    'HSC_MEMP': 'HSPC',
    'HSC_MMP': 'HSPC',
    'T_Progenitors': 'Progenitor_other',
    'T_Proliferative': 'Progenitor_other',
}

SCATLAS_TO_FINE = {
    'CD4T01_Tn_SOX4': 'CD4_Naive',
    'CD4T02_Tn_CCR7': 'CD4_Naive',
    'CD4T03_Tn_NR4A1': 'CD4_Naive',
    'CD4T04_Tfh_IL6ST': 'Tfh',
    'CD4T05_Tm_LTB': 'CD4_Memory',
    'CD4T06_Tm_ANXA1': 'CD4_Memory',
    'CD4T07_Tem_GZMK': 'CD4_Effector',
    'CD4T08_Treg_FOXP3': 'Treg',
    'CD8T01_Tn_CCR7': 'CD8_Naive',
    'CD8T02_Tem_GZMK': 'CD8_Memory',
    'CD8T03_Trm_ITGA1': 'CD8_Trm',
    'CD8T04_Trm_HSPA1A': 'CD8_Trm',
    'CD8T05_Temra_GZMH': 'CD8_Effector',
    'CD8T06_MAIT_SLC4A10': 'MAIT',
    'I08_ILC3_KIT': 'ILC',
    'I09_gdT_ITGA1': 'gdT',
    'I01_CD16hiNK_CREM': 'NK_CD56dim',
    'I02_CD16hiNK_SYNE2': 'NK_CD56dim',
    'I03_CD16hiNK_GZMB': 'NK_CD56dim',
    'I04_CD16hiNK_HSPA1A': 'NK_CD56dim',
    'I05_CD16loNK_SELL': 'NK_CD56bright',
    'I06_CD16loNK_NR4A2': 'NK_CD56bright',
    'I07_CD16loNK_CXCR6': 'NK_other',
    'B01_Bn_IGHM': 'B_Naive',
    'B02_Bn_TCL1A': 'B_Naive',
    'B03_Bn_NR4A2': 'B_Naive',
    'B04_Bm_CD27': 'B_Memory',
    'B05_Bm_NR4A2': 'B_Memory',
    'B06_Bm_ITGB1': 'B_Memory',
    'B07_Bm_HSPA1A': 'B_Memory',
    'B08_ABC_FCRL5': 'B_GC',
    'B09_GCB_RGS13': 'B_GC',
    'B10_Plasmablast_MKI67': 'Plasmablast',
    'B11_Plasma_IGHG2': 'Plasma',
    'B12_Plasma_IGHA2': 'Plasma',
    'M01_cDC1_CLEC9A': 'cDC1',
    'M02_cDC2_CD1C': 'cDC2',
    'M03_cDC_LAMP3': 'cDC2',
    'M04_pDC_LILRA4': 'pDC',
    'M05_Mo_CD14': 'Mono_Classical',
    'M06_Mo_FCGR3A': 'Mono_NonClassical',
    'M07_Mph_FCN1': 'Macrophage',
    'M08_Mph_NLRP3': 'Macrophage',
    'M09_Mph_FOLR2': 'Macrophage',
    'M10_Mph_CD5L': 'Macrophage',
    'M11_Mph_PPARG': 'Macrophage',
    'M12_Mph_MT1X': 'Macrophage',
    'M13_immNeu_MMP8': 'Neutrophil',
    'M14_mNeu_CXCR2': 'Neutrophil',
    'M15_Mast_CPA3': 'Mast',
}

# =============================================================================
# Mapping Functions
# =============================================================================

def get_mapping_dict(atlas: str, level: str = 'coarse') -> Dict[str, Optional[str]]:
    """Get mapping dictionary for an atlas at specified level.

    Args:
        atlas: 'cima', 'inflammation', or 'scatlas'
        level: 'coarse' (8 lineages) or 'fine' (~30 types)

    Returns:
        Dictionary mapping original cell types to harmonized types
    """
    mappings = {
        ('cima', 'coarse'): CIMA_TO_COARSE,
        ('cima', 'fine'): CIMA_TO_FINE,
        ('inflammation', 'coarse'): INFLAMMATION_TO_COARSE,
        ('inflammation', 'fine'): INFLAMMATION_TO_FINE,
        ('scatlas', 'coarse'): SCATLAS_TO_COARSE,
        ('scatlas', 'fine'): SCATLAS_TO_FINE,
    }

    key = (atlas.lower(), level.lower())
    if key not in mappings:
        raise ValueError(f"Unknown atlas/level combination: {atlas}/{level}")

    return mappings[key]


def map_cell_types(
    cell_types: pd.Series,
    atlas: str,
    level: str = 'coarse',
    unmapped_value: str = 'Unknown'
) -> pd.Series:
    """Map cell types to harmonized annotations.

    Args:
        cell_types: Series of original cell type labels
        atlas: 'cima', 'inflammation', or 'scatlas'
        level: 'coarse' or 'fine'
        unmapped_value: Value to use for unmapped types

    Returns:
        Series of mapped cell type labels
    """
    mapping = get_mapping_dict(atlas, level)
    return cell_types.map(lambda x: mapping.get(x, unmapped_value))


def get_shared_types(level: str = 'coarse') -> Set[str]:
    """Get cell types present in all three atlases at specified level.

    Args:
        level: 'coarse' or 'fine'

    Returns:
        Set of shared cell type names
    """
    cima_types = set(v for v in get_mapping_dict('cima', level).values() if v)
    inflam_types = set(v for v in get_mapping_dict('inflammation', level).values() if v)
    scatlas_types = set(v for v in get_mapping_dict('scatlas', level).values() if v)

    return cima_types & inflam_types & scatlas_types


def get_atlas_specific_types(atlas: str, level: str = 'coarse') -> Set[str]:
    """Get cell types unique to a specific atlas.

    Args:
        atlas: 'cima', 'inflammation', or 'scatlas'
        level: 'coarse' or 'fine'

    Returns:
        Set of atlas-specific cell type names
    """
    all_atlases = ['cima', 'inflammation', 'scatlas']
    other_atlases = [a for a in all_atlases if a != atlas.lower()]

    atlas_types = set(v for v in get_mapping_dict(atlas, level).values() if v)
    other_types = set()
    for other in other_atlases:
        other_types |= set(v for v in get_mapping_dict(other, level).values() if v)

    return atlas_types - other_types


def summarize_mapping(level: str = 'coarse') -> pd.DataFrame:
    """Create summary table of cell type mapping across atlases.

    Args:
        level: 'coarse' or 'fine'

    Returns:
        DataFrame with cell types as rows and atlases as columns
    """
    all_types = set()
    atlas_data = {}

    for atlas in ['cima', 'inflammation', 'scatlas']:
        mapping = get_mapping_dict(atlas, level)
        # Invert mapping: harmonized -> original types
        inverted = {}
        for orig, harm in mapping.items():
            if harm:
                if harm not in inverted:
                    inverted[harm] = []
                inverted[harm].append(orig)
        atlas_data[atlas] = inverted
        all_types |= set(inverted.keys())

    # Build summary table
    rows = []
    for cell_type in sorted(all_types):
        row = {'cell_type': cell_type}
        for atlas in ['cima', 'inflammation', 'scatlas']:
            types = atlas_data[atlas].get(cell_type, [])
            row[f'{atlas}_types'] = ', '.join(sorted(types)) if types else '-'
            row[f'{atlas}_count'] = len(types)
        rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# Main: Print summary
# =============================================================================

if __name__ == '__main__':
    print("Cell Type Mapping Summary")
    print("=" * 60)

    print("\n1. Coarse Level (8 Lineages)")
    print("-" * 40)
    shared_coarse = get_shared_types('coarse')
    print(f"Shared types: {sorted(shared_coarse)}")

    for atlas in ['cima', 'inflammation', 'scatlas']:
        specific = get_atlas_specific_types(atlas, 'coarse')
        if specific:
            print(f"{atlas.upper()}-specific: {sorted(specific)}")

    print("\n2. Fine Level (~30 Types)")
    print("-" * 40)
    shared_fine = get_shared_types('fine')
    print(f"Shared types ({len(shared_fine)}): {sorted(shared_fine)}")

    for atlas in ['cima', 'inflammation', 'scatlas']:
        specific = get_atlas_specific_types(atlas, 'fine')
        if specific:
            print(f"{atlas.upper()}-specific ({len(specific)}): {sorted(specific)}")

    print("\n3. Mapping Summary Table (Coarse)")
    print("-" * 40)
    summary = summarize_mapping('coarse')
    print(summary.to_string(index=False))
