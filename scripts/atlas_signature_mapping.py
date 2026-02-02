#!/usr/bin/env python3
"""
Atlas to Cell-Type-Specific Signature Mapping

Maps cell types from each atlas (CIMA, Inflammation, scAtlas) to the
cell-type-specific cytokine signatures generated from the CytoSig database.

This enables cell-type-aware cytokine activity inference:
- Use the appropriate signature matrix based on the cell type
- Falls back to "universal" signature if no specific match exists

Author: Seongyong Park
Date: 2026-02-02
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional

# ==============================================================================
# Configuration
# ==============================================================================

SIGNATURE_DIR = Path('/data/parks34/projects/2secactpy/results/celltype_signatures')
OUTPUT_DIR = Path('/data/parks34/projects/2secactpy/results/celltype_signatures')

# ==============================================================================
# Atlas Cell Type to Signature Mapping
# ==============================================================================

# Maps coarse atlas cell types to the best matching signature cell type
# Priority: Most specific match first, then broader categories, then None (use universal)

ATLAS_COARSE_TO_SIGNATURE = {
    # Immune cells
    'CD4_T': ['T_CD4', 'T_Cell', 'PBMC'],
    'CD8_T': ['T_CD8', 'T_Cell', 'PBMC'],
    'NK': ['NK_Cell', 'PBMC'],
    'B': ['B_Cell', 'PBMC'],
    'Plasma': ['B_Cell', 'PBMC'],  # Plasma cells are differentiated B cells
    'Myeloid': ['Monocyte', 'Macrophage', 'Dendritic_Cell', 'PBMC'],
    'Unconventional_T': ['T_Cell', 'PBMC'],  # MAIT, gdT, NKT, ILC
    'Progenitor': ['PBMC'],  # HSPC, progenitors
}

# Maps fine atlas cell types to signature cell types
ATLAS_FINE_TO_SIGNATURE = {
    # CD4 T cells
    'CD4_Naive': ['T_CD4', 'T_Cell'],
    'CD4_Memory': ['T_CD4', 'T_Cell'],
    'CD4_Effector': ['T_CD4', 'T_Cell'],
    'Treg': ['T_CD4', 'T_Cell'],  # Tregs are CD4+ regulatory
    'Tfh': ['T_CD4', 'T_Cell'],

    # CD8 T cells
    'CD8_Naive': ['T_CD8', 'T_Cell'],
    'CD8_Memory': ['T_CD8', 'T_Cell'],
    'CD8_Effector': ['T_CD8', 'T_Cell'],
    'CD8_Trm': ['T_CD8', 'T_Cell'],

    # Unconventional T / ILC
    'MAIT': ['T_Cell', 'PBMC'],
    'gdT': ['T_Cell', 'PBMC'],
    'NKT': ['NK_Cell', 'T_Cell', 'PBMC'],
    'ILC': ['NK_Cell', 'PBMC'],  # ILCs share properties with NK

    # NK cells
    'NK_CD56bright': ['NK_Cell', 'PBMC'],
    'NK_CD56dim': ['NK_Cell', 'PBMC'],
    'NK_other': ['NK_Cell', 'PBMC'],

    # B cells
    'B_Naive': ['B_Cell', 'PBMC'],
    'B_Memory': ['B_Cell', 'PBMC'],
    'B_Transitional': ['B_Cell', 'PBMC'],
    'B_GC': ['B_Cell', 'PBMC'],
    'Plasmablast': ['B_Cell', 'PBMC'],
    'Plasma': ['B_Cell', 'PBMC'],

    # Myeloid
    'Mono_Classical': ['Monocyte', 'Macrophage', 'PBMC'],
    'Mono_NonClassical': ['Monocyte', 'Macrophage', 'PBMC'],
    'Macrophage': ['Macrophage', 'Monocyte', 'PBMC'],
    'cDC1': ['Dendritic_Cell', 'Monocyte', 'PBMC'],
    'cDC2': ['Dendritic_Cell', 'Monocyte', 'PBMC'],
    'pDC': ['Dendritic_Cell', 'PBMC'],
    'Neutrophil': ['Neutrophil', 'PBMC'],
    'Mast': ['Basophil', 'PBMC'],  # Mast cells related to basophils

    # Progenitors
    'HSPC': ['PBMC'],
    'Progenitor_other': ['PBMC'],
}

# Direct mapping for specific CIMA cell types
CIMA_CELLTYPE_TO_SIGNATURE = {
    'CD4_CTL': ['T_CD4', 'T_Cell'],
    'CD4_helper': ['T_CD4', 'T_Cell'],
    'CD4_memory': ['T_CD4', 'T_Cell'],
    'CD4_naive': ['T_CD4', 'T_Cell'],
    'CD4_regulatory': ['T_CD4', 'T_Cell'],
    'CD8_CTL': ['T_CD8', 'T_Cell'],
    'CD8_memory': ['T_CD8', 'T_Cell'],
    'CD8_naive': ['T_CD8', 'T_Cell'],
    'CD56_bright_NK': ['NK_Cell', 'PBMC'],
    'CD56_dim_NK': ['NK_Cell', 'PBMC'],
    'Proliferative_NK': ['NK_Cell', 'PBMC'],
    'Transitional_NK': ['NK_Cell', 'PBMC'],
    'NKT': ['NK_Cell', 'T_Cell'],
    'MAIT': ['T_Cell', 'PBMC'],
    'gdT': ['T_Cell', 'PBMC'],
    'ILC2': ['NK_Cell', 'PBMC'],
    'Memory_B': ['B_Cell', 'PBMC'],
    'Naive_B': ['B_Cell', 'PBMC'],
    'Transitional_B': ['B_Cell', 'PBMC'],
    'Total_Plasma': ['B_Cell', 'PBMC'],
    'Mono': ['Monocyte', 'Macrophage', 'PBMC'],
    'DC': ['Dendritic_Cell', 'Monocyte', 'PBMC'],
    'pDC': ['Dendritic_Cell', 'PBMC'],
    'HSPC': ['PBMC'],
    'MK': ['PBMC'],
    'Immature_T': ['T_Cell', 'PBMC'],
    'Proliferative_T': ['T_Cell', 'PBMC'],
}

# Direct mapping for Inflammation atlas cell types
INFLAMMATION_CELLTYPE_TO_SIGNATURE = {
    # CD4 T cells
    'T_CD4_Naive': ['T_CD4', 'T_Cell'],
    'T_CD4_CM': ['T_CD4', 'T_Cell'],
    'T_CD4_CM_ribo': ['T_CD4', 'T_Cell'],
    'T_CD4_EM': ['T_CD4', 'T_Cell'],
    'T_CD4_EMRA': ['T_CD4', 'T_Cell'],
    'T_CD4_eff': ['T_CD4', 'T_Cell'],
    'Th0': ['T_CD4', 'T_Cell'],
    'Th1': ['T_CD4', 'T_Cell'],
    'Th2': ['T_CD4', 'T_Cell'],
    'Tregs': ['T_CD4', 'T_Cell'],
    'Tregs_activated': ['T_CD4', 'T_Cell'],
    # CD8 T cells
    'T_CD8_Naive': ['T_CD8', 'T_Cell'],
    'T_CD8_CM': ['T_CD8', 'T_Cell'],
    'T_CD8_CM_stem': ['T_CD8', 'T_Cell'],
    'T_CD8_EM_CX3CR1high': ['T_CD8', 'T_Cell'],
    'T_CD8_EM_CX3CR1int': ['T_CD8', 'T_Cell'],
    'T_CD8_Mem_cytotoxic': ['T_CD8', 'T_Cell'],
    'T_CD8_activated': ['T_CD8', 'T_Cell'],
    'T_CD8_arrested': ['T_CD8', 'T_Cell'],
    'T_CD8_eff_HOBIT': ['T_CD8', 'T_Cell'],
    'T_CD8_IFNresponse': ['T_CD8', 'T_Cell'],
    # NK cells
    'NK_CD16high': ['NK_Cell', 'PBMC'],
    'NK_CD56dimCD16': ['NK_Cell', 'PBMC'],
    'NK_CD56high': ['NK_Cell', 'PBMC'],
    'NK_IFN1response': ['NK_Cell', 'PBMC'],
    'NK_Proliferative': ['NK_Cell', 'PBMC'],
    'NK_adaptive': ['NK_Cell', 'PBMC'],
    'NK_lowRibocontent': ['NK_Cell', 'PBMC'],
    # Unconventional T
    'MAIT': ['T_Cell', 'PBMC'],
    'MAIT_17': ['T_Cell', 'PBMC'],
    'gdT_V1': ['T_Cell', 'PBMC'],
    'gdT_V2_Vγ9': ['T_Cell', 'PBMC'],
    # B cells
    'B_Naive': ['B_Cell', 'PBMC'],
    'B_Naive_activated': ['B_Cell', 'PBMC'],
    'B_Transitional': ['B_Cell', 'PBMC'],
    'B_Memory_switched': ['B_Cell', 'PBMC'],
    'B_Memory_unswitched': ['B_Cell', 'PBMC'],
    'B_Memory_ITGAX': ['B_Cell', 'PBMC'],
    'B_IFNresponder': ['B_Cell', 'PBMC'],
    'B_Progenitors': ['B_Cell', 'PBMC'],
    # Plasma
    'Plasma_IGHA': ['B_Cell', 'PBMC'],
    'Plasma_IGHG': ['B_Cell', 'PBMC'],
    'Plasma_Proliferative': ['B_Cell', 'PBMC'],
    'Plasma_XBP1': ['B_Cell', 'PBMC'],
    # Myeloid
    'Mono_classical': ['Monocyte', 'Macrophage', 'PBMC'],
    'Mono_inflammatory': ['Monocyte', 'Macrophage', 'PBMC'],
    'Mono_nonClassical': ['Monocyte', 'Macrophage', 'PBMC'],
    'Mono_regulatory': ['Monocyte', 'Macrophage', 'PBMC'],
    'Mono_IFNresponse': ['Monocyte', 'Macrophage', 'PBMC'],
    'cDC1': ['Dendritic_Cell', 'Monocyte', 'PBMC'],
    'cDC2': ['Dendritic_Cell', 'Monocyte', 'PBMC'],
    'cDC3': ['Dendritic_Cell', 'Monocyte', 'PBMC'],
    'DC4': ['Dendritic_Cell', 'Monocyte', 'PBMC'],
    'DC5': ['Dendritic_Cell', 'Monocyte', 'PBMC'],
    'DC_CCR7': ['Dendritic_Cell', 'Monocyte', 'PBMC'],
    'DC_Proliferative': ['Dendritic_Cell', 'Monocyte', 'PBMC'],
    'pDC': ['Dendritic_Cell', 'PBMC'],
    # Progenitors
    'HSC_LMP': ['PBMC'],
    'HSC_MEMP': ['PBMC'],
    'HSC_MMP': ['PBMC'],
    'T_Progenitors': ['T_Cell', 'PBMC'],
    'T_Proliferative': ['T_Cell', 'PBMC'],
}

# Direct mapping for scAtlas cell types
SCATLAS_CELLTYPE_TO_SIGNATURE = {
    # CD4 T cells
    'CD4T01_Tn_SOX4': ['T_CD4', 'T_Cell'],
    'CD4T02_Tn_CCR7': ['T_CD4', 'T_Cell'],
    'CD4T03_Tn_NR4A1': ['T_CD4', 'T_Cell'],
    'CD4T04_Tfh_IL6ST': ['T_CD4', 'T_Cell'],
    'CD4T05_Tm_LTB': ['T_CD4', 'T_Cell'],
    'CD4T06_Tm_ANXA1': ['T_CD4', 'T_Cell'],
    'CD4T07_Tem_GZMK': ['T_CD4', 'T_Cell'],
    'CD4T08_Treg_FOXP3': ['T_CD4', 'T_Cell'],
    # CD8 T cells
    'CD8T01_Tn_CCR7': ['T_CD8', 'T_Cell'],
    'CD8T02_Tem_GZMK': ['T_CD8', 'T_Cell'],
    'CD8T03_Trm_ITGA1': ['T_CD8', 'T_Cell'],
    'CD8T04_Trm_HSPA1A': ['T_CD8', 'T_Cell'],
    'CD8T05_Temra_GZMH': ['T_CD8', 'T_Cell'],
    'CD8T06_MAIT_SLC4A10': ['T_Cell', 'PBMC'],
    # ILC / unconventional
    'I08_ILC3_KIT': ['NK_Cell', 'PBMC'],
    'I09_gdT_ITGA1': ['T_Cell', 'PBMC'],
    # NK cells
    'I01_CD16hiNK_CREM': ['NK_Cell', 'PBMC'],
    'I02_CD16hiNK_SYNE2': ['NK_Cell', 'PBMC'],
    'I03_CD16hiNK_GZMB': ['NK_Cell', 'PBMC'],
    'I04_CD16hiNK_HSPA1A': ['NK_Cell', 'PBMC'],
    'I05_CD16loNK_SELL': ['NK_Cell', 'PBMC'],
    'I06_CD16loNK_NR4A2': ['NK_Cell', 'PBMC'],
    'I07_CD16loNK_CXCR6': ['NK_Cell', 'PBMC'],
    # B cells
    'B01_Bn_IGHM': ['B_Cell', 'PBMC'],
    'B02_Bn_TCL1A': ['B_Cell', 'PBMC'],
    'B03_Bn_NR4A2': ['B_Cell', 'PBMC'],
    'B04_Bm_CD27': ['B_Cell', 'PBMC'],
    'B05_Bm_NR4A2': ['B_Cell', 'PBMC'],
    'B06_Bm_ITGB1': ['B_Cell', 'PBMC'],
    'B07_Bm_HSPA1A': ['B_Cell', 'PBMC'],
    'B08_ABC_FCRL5': ['B_Cell', 'PBMC'],
    'B09_GCB_RGS13': ['B_Cell', 'PBMC'],
    'B10_Plasmablast_MKI67': ['B_Cell', 'PBMC'],
    'B11_Plasma_IGHG2': ['B_Cell', 'PBMC'],
    'B12_Plasma_IGHA2': ['B_Cell', 'PBMC'],
    # Myeloid
    'M01_cDC1_CLEC9A': ['Dendritic_Cell', 'Monocyte', 'PBMC'],
    'M02_cDC2_CD1C': ['Dendritic_Cell', 'Monocyte', 'PBMC'],
    'M03_cDC_LAMP3': ['Dendritic_Cell', 'Monocyte', 'PBMC'],
    'M04_pDC_LILRA4': ['Dendritic_Cell', 'PBMC'],
    'M05_Mo_CD14': ['Monocyte', 'Macrophage', 'PBMC'],
    'M06_Mo_FCGR3A': ['Monocyte', 'Macrophage', 'PBMC'],
    'M07_Mph_FCN1': ['Macrophage', 'Monocyte', 'PBMC'],
    'M08_Mph_NLRP3': ['Macrophage', 'Monocyte', 'PBMC'],
    'M09_Mph_FOLR2': ['Macrophage', 'Monocyte', 'PBMC'],
    'M10_Mph_CD5L': ['Macrophage', 'Monocyte', 'PBMC'],
    'M11_Mph_PPARG': ['Macrophage', 'Monocyte', 'PBMC'],
    'M12_Mph_MT1X': ['Macrophage', 'Monocyte', 'PBMC'],
    'M13_immNeu_MMP8': ['Neutrophil', 'PBMC'],
    'M14_mNeu_CXCR2': ['Neutrophil', 'PBMC'],
    'M15_Mast_CPA3': ['Basophil', 'PBMC'],
}

# Non-immune cell types (for tissue atlases like scAtlas)
NON_IMMUNE_TO_SIGNATURE = {
    # Epithelial (from scAtlas tissues)
    'Epithelial': ['Epithelial', 'Airway_Epithelial', 'Keratinocyte'],
    'Alveolar_Epithelial': ['Airway_Epithelial', 'Epithelial'],
    'Airway_Epithelial': ['Airway_Epithelial', 'Epithelial'],
    'Intestinal_Epithelial': ['Intestinal_Epithelial', 'Epithelial'],
    'Hepatocyte': ['Hepatocyte'],
    'Keratinocyte': ['Keratinocyte', 'Epithelial'],
    'Renal_Epithelial': ['Renal_Epithelial', 'Epithelial'],

    # Stromal
    'Fibroblast': ['Fibroblast', 'Dermal_Fibroblast', 'Lung_Fibroblast'],
    'Myofibroblast': ['Fibroblast'],
    'Smooth_Muscle': ['Smooth_Muscle'],
    'Pericyte': ['Smooth_Muscle', 'Endothelial'],

    # Endothelial
    'Endothelial': ['Endothelial', 'HUVEC'],
    'Vascular_Endothelial': ['Endothelial', 'HUVEC'],
    'Lymphatic_Endothelial': ['Lymphatic_Endothelial', 'Endothelial'],

    # Other
    'Adipocyte': ['Adipocyte'],
    'Neuron': ['Neuron'],
    'Astrocyte': ['Neuron'],  # Neural category
}


def get_available_signatures() -> List[str]:
    """Get list of available signature cell types from the generated signatures."""
    metadata_path = SIGNATURE_DIR / 'metadata.json'
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        return metadata.get('celltypes', [])
    return []


def get_best_signature_match(
    atlas_celltype: str,
    atlas: str = 'cima',
    available_signatures: Optional[List[str]] = None
) -> Optional[str]:
    """
    Find the best matching signature cell type for an atlas cell type.

    Args:
        atlas_celltype: Cell type name from the atlas
        atlas: Atlas name ('cima', 'inflammation', 'scatlas')
        available_signatures: List of available signature cell types

    Returns:
        Best matching signature cell type, or None if no match
    """
    if available_signatures is None:
        available_signatures = get_available_signatures()

    available_set = set(available_signatures)

    # First try direct mapping
    direct_mappings = {
        'cima': CIMA_CELLTYPE_TO_SIGNATURE,
        'inflammation': INFLAMMATION_CELLTYPE_TO_SIGNATURE,
        'scatlas': SCATLAS_CELLTYPE_TO_SIGNATURE,
    }

    if atlas.lower() in direct_mappings:
        mapping = direct_mappings[atlas.lower()]
        if atlas_celltype in mapping:
            candidates = mapping[atlas_celltype]
            for candidate in candidates:
                if candidate in available_set:
                    return candidate

    # Try fine level mapping
    if atlas_celltype in ATLAS_FINE_TO_SIGNATURE:
        candidates = ATLAS_FINE_TO_SIGNATURE[atlas_celltype]
        for candidate in candidates:
            if candidate in available_set:
                return candidate

    # Try coarse level mapping
    if atlas_celltype in ATLAS_COARSE_TO_SIGNATURE:
        candidates = ATLAS_COARSE_TO_SIGNATURE[atlas_celltype]
        for candidate in candidates:
            if candidate in available_set:
                return candidate

    # Try non-immune mapping
    if atlas_celltype in NON_IMMUNE_TO_SIGNATURE:
        candidates = NON_IMMUNE_TO_SIGNATURE[atlas_celltype]
        for candidate in candidates:
            if candidate in available_set:
                return candidate

    # Fallback: try PBMC as universal immune signature
    if 'PBMC' in available_set:
        return 'PBMC'

    return None


def create_atlas_signature_mapping(atlas: str) -> pd.DataFrame:
    """
    Create complete mapping table for an atlas.

    Args:
        atlas: Atlas name ('cima', 'inflammation', 'scatlas')

    Returns:
        DataFrame with columns: atlas_celltype, signature_celltype, match_type
    """
    available_signatures = get_available_signatures()

    # Get all cell types for this atlas
    direct_mappings = {
        'cima': CIMA_CELLTYPE_TO_SIGNATURE,
        'inflammation': INFLAMMATION_CELLTYPE_TO_SIGNATURE,
        'scatlas': SCATLAS_CELLTYPE_TO_SIGNATURE,
    }

    atlas_celltypes = list(direct_mappings.get(atlas.lower(), {}).keys())

    # Create mapping
    rows = []
    for ct in atlas_celltypes:
        sig = get_best_signature_match(ct, atlas, available_signatures)
        match_type = 'direct' if sig else 'none'

        # Determine match type
        if sig:
            mapping = direct_mappings[atlas.lower()]
            if ct in mapping and sig in mapping[ct][:1]:
                match_type = 'primary'
            elif ct in mapping and sig in mapping[ct]:
                match_type = 'secondary'
            elif sig == 'PBMC':
                match_type = 'fallback'

        rows.append({
            'atlas_celltype': ct,
            'signature_celltype': sig,
            'match_type': match_type
        })

    return pd.DataFrame(rows)


def save_all_mappings():
    """Generate and save mapping files for all atlases."""
    print("Generating atlas-to-signature mappings...")

    available_signatures = get_available_signatures()
    print(f"  Available signatures: {len(available_signatures)}")

    all_mappings = {}

    for atlas in ['cima', 'inflammation', 'scatlas']:
        mapping_df = create_atlas_signature_mapping(atlas)

        # Save CSV
        mapping_df.to_csv(OUTPUT_DIR / f'{atlas}_signature_mapping.csv', index=False)

        # Summary
        matched = mapping_df['signature_celltype'].notna().sum()
        total = len(mapping_df)
        print(f"  {atlas.upper()}: {matched}/{total} cell types mapped")

        # Store for JSON
        all_mappings[atlas] = {
            row['atlas_celltype']: {
                'signature': row['signature_celltype'],
                'match_type': row['match_type']
            }
            for _, row in mapping_df.iterrows()
        }

    # Save combined JSON
    with open(OUTPUT_DIR / 'atlas_signature_mapping.json', 'w') as f:
        json.dump(all_mappings, f, indent=2)

    print(f"\nSaved mapping files to {OUTPUT_DIR}")


if __name__ == '__main__':
    save_all_mappings()
