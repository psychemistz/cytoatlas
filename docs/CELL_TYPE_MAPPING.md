# Cell Type Mapping Strategy

## Overview

This document describes the cell type mapping strategy between three single-cell atlases:
- **CIMA**: 27 PBMC cell types from healthy donors
- **Inflammation Atlas**: 66 Level2 cell types from PBMC (disease vs healthy)
- **scAtlas**: 96 subCluster types from tissue-resident cells (including non-immune)

## Atlas Cell Type Summaries

### CIMA (6.5M cells, 27 types)
Focus: Peripheral blood mononuclear cells (PBMC) from healthy donors
```
CD4_CTL, CD4_helper, CD4_memory, CD4_naive, CD4_regulatory
CD8_CTL, CD8_memory, CD8_naive
CD56_bright_NK, CD56_dim_NK, Proliferative_NK, Transitional_NK
NKT, MAIT, gdT, ILC2
Memory_B, Naive_B, Transitional_B, Total_Plasma
Mono, DC, pDC
HSPC, MK, Immature_T, Proliferative_T
```

### Inflammation Atlas (4.9M cells, 66 types)
Focus: PBMC from disease and healthy donors
Level1 (coarse): Mono, T_CD4_Naive, T_CD4_NonNaive, T_CD8_NonNaive, ILC, B, etc.
Level2 (fine): 66 subtypes including T cell states, NK subsets, B cell maturation stages

### scAtlas (2.3M cells, 96 types)
Focus: Tissue-resident cells across organs
Compartments: Immune (1M), Epithelial (0.6M), Stromal (0.4M), Endothelial (0.2M)
majorCluster: CD4T, CD8T, B, ILC, Myeloid, Epithelial, Stromal, Endothelial

## Mapping Strategy

### Recommended Approach: Hierarchical Harmonization

1. **Coarse Level (8 lineages)** - Maximum overlap between atlases
2. **Fine Level (~25-30 types)** - Balance between resolution and comparability
3. **Full Level** - Atlas-specific annotations (no direct mapping)

### Coarse Level Mapping (8 Lineages)

| Lineage | CIMA | Inflammation | scAtlas |
|---------|------|--------------|---------|
| **CD4_T** | CD4_CTL, CD4_helper, CD4_memory, CD4_naive, CD4_regulatory | T_CD4_* (11 types) | CD4T* (8 types) |
| **CD8_T** | CD8_CTL, CD8_memory, CD8_naive | T_CD8_* (10 types) | CD8T* (6 types) |
| **NK** | CD56_bright_NK, CD56_dim_NK, Proliferative_NK, Transitional_NK | NK_* (6 types) | I01-I07 (7 types) |
| **B** | Memory_B, Naive_B, Transitional_B | B_* (8 types) | B01-B09 (9 types) |
| **Plasma** | Total_Plasma | Plasma_* (4 types) | B10-B12 (3 types) |
| **Myeloid** | Mono, DC, pDC | Mono_*, DC_*, cDC*, pDC | M01-M15 (15 types) |
| **Unconventional_T** | NKT, MAIT, gdT, ILC2 | MAIT_*, gdT_* | CD8T06_MAIT, I08-I09 |
| **Progenitor** | HSPC, MK, Immature_T, Proliferative_T | HSC_*, T_Progenitors, Cycling_cells | - |

### Fine Level Mapping (~30 types)

| Fine Type | CIMA | Inflammation | scAtlas |
|-----------|------|--------------|---------|
| **CD4_Naive** | CD4_naive | T_CD4_Naive | CD4T01-03 |
| **CD4_Memory** | CD4_memory | T_CD4_CM, T_CD4_CM_ribo | CD4T05 |
| **CD4_Effector** | CD4_CTL, CD4_helper | T_CD4_EM, T_CD4_EMRA, T_CD4_eff | CD4T06-07 |
| **Treg** | CD4_regulatory | Tregs, Tregs_activated | CD4T08 |
| **Tfh** | - | Th0, Th1, Th2 | CD4T04 |
| **CD8_Naive** | CD8_naive | T_CD8_Naive | CD8T01 |
| **CD8_Memory** | CD8_memory | T_CD8_CM, T_CD8_CM_stem | CD8T02-04 |
| **CD8_Effector** | CD8_CTL | T_CD8_Mem_cytotoxic, T_CD8_eff_HOBIT | CD8T05 |
| **MAIT** | MAIT | MAIT, MAIT_17 | CD8T06_MAIT |
| **gdT** | gdT | gdT_V1, gdT_V2_Vγ9 | I09_gdT |
| **NKT** | NKT | - | - |
| **NK_CD56bright** | CD56_bright_NK | NK_CD56high | I05-I06 |
| **NK_CD56dim** | CD56_dim_NK | NK_CD16high, NK_CD56dimCD16 | I01-I04 |
| **NK_other** | Proliferative_NK, Transitional_NK | NK_Proliferative, NK_adaptive, NK_IFN1response | I07 |
| **ILC** | ILC2 | - | I08_ILC3 |
| **B_Naive** | Naive_B | B_Naive, B_Naive_activated | B01-B03 |
| **B_Memory** | Memory_B | B_Memory_switched, B_Memory_unswitched, B_Memory_ITGAX | B04-B07 |
| **B_Transitional** | Transitional_B | B_Transitional | - |
| **B_GC** | - | - | B08-B09 |
| **Plasmablast** | - | Plasma_Proliferative | B10 |
| **Plasma** | Total_Plasma | Plasma_IGHA, Plasma_IGHG, Plasma_XBP1 | B11-B12 |
| **Mono_Classical** | Mono | Mono_classical, Mono_inflammatory | M05 |
| **Mono_NonClassical** | - | Mono_nonClassical | M06 |
| **Macrophage** | - | - | M07-M12 |
| **cDC1** | - | cDC1 | M01 |
| **cDC2** | DC | cDC2, cDC3, DC4, DC5 | M02-M03 |
| **pDC** | pDC | pDC | M04 |
| **Neutrophil** | - | - | M13-M14 |
| **Mast** | - | - | M15 |
| **HSPC** | HSPC | HSC_LMP, HSC_MEMP, HSC_MMP | - |

## Implementation

### Python Mapping Dictionary

```python
COARSE_MAPPING = {
    # CIMA -> Coarse
    'cima': {
        'CD4_CTL': 'CD4_T', 'CD4_helper': 'CD4_T', 'CD4_memory': 'CD4_T',
        'CD4_naive': 'CD4_T', 'CD4_regulatory': 'CD4_T',
        'CD8_CTL': 'CD8_T', 'CD8_memory': 'CD8_T', 'CD8_naive': 'CD8_T',
        'CD56_bright_NK': 'NK', 'CD56_dim_NK': 'NK',
        'Proliferative_NK': 'NK', 'Transitional_NK': 'NK',
        'NKT': 'Unconventional_T', 'MAIT': 'Unconventional_T',
        'gdT': 'Unconventional_T', 'ILC2': 'Unconventional_T',
        'Memory_B': 'B', 'Naive_B': 'B', 'Transitional_B': 'B',
        'Total_Plasma': 'Plasma',
        'Mono': 'Myeloid', 'DC': 'Myeloid', 'pDC': 'Myeloid',
        'HSPC': 'Progenitor', 'MK': 'Progenitor',
        'Immature_T': 'Progenitor', 'Proliferative_T': 'Progenitor',
    },
    # Inflammation -> Coarse
    'inflammation': {
        'T_CD4_Naive': 'CD4_T', 'T_CD4_CM': 'CD4_T', 'T_CD4_CM_ribo': 'CD4_T',
        'T_CD4_EM': 'CD4_T', 'T_CD4_EMRA': 'CD4_T', 'T_CD4_eff': 'CD4_T',
        'Th0': 'CD4_T', 'Th1': 'CD4_T', 'Th2': 'CD4_T',
        'Tregs': 'CD4_T', 'Tregs_activated': 'CD4_T',
        'T_CD8_Naive': 'CD8_T', 'T_CD8_CM': 'CD8_T', 'T_CD8_CM_stem': 'CD8_T',
        'T_CD8_EM_CX3CR1high': 'CD8_T', 'T_CD8_EM_CX3CR1int': 'CD8_T',
        'T_CD8_Mem_cytotoxic': 'CD8_T', 'T_CD8_activated': 'CD8_T',
        'T_CD8_arrested': 'CD8_T', 'T_CD8_eff_HOBIT': 'CD8_T',
        'T_CD8_IFNresponse': 'CD8_T',
        'NK_CD16high': 'NK', 'NK_CD56dimCD16': 'NK', 'NK_CD56high': 'NK',
        'NK_IFN1response': 'NK', 'NK_Proliferative': 'NK',
        'NK_adaptive': 'NK', 'NK_lowRibocontent': 'NK',
        'MAIT': 'Unconventional_T', 'MAIT_17': 'Unconventional_T',
        'gdT_V1': 'Unconventional_T', 'gdT_V2_Vγ9': 'Unconventional_T',
        'B_Naive': 'B', 'B_Naive_activated': 'B', 'B_Transitional': 'B',
        'B_Memory_switched': 'B', 'B_Memory_unswitched': 'B',
        'B_Memory_ITGAX': 'B', 'B_IFNresponder': 'B', 'B_Progenitors': 'B',
        'Plasma_IGHA': 'Plasma', 'Plasma_IGHG': 'Plasma',
        'Plasma_Proliferative': 'Plasma', 'Plasma_XBP1': 'Plasma',
        'Mono_classical': 'Myeloid', 'Mono_inflammatory': 'Myeloid',
        'Mono_nonClassical': 'Myeloid', 'Mono_regulatory': 'Myeloid',
        'Mono_IFNresponse': 'Myeloid',
        'cDC1': 'Myeloid', 'cDC2': 'Myeloid', 'cDC3': 'Myeloid',
        'DC4': 'Myeloid', 'DC5': 'Myeloid', 'DC_CCR7': 'Myeloid',
        'DC_Proliferative': 'Myeloid', 'pDC': 'Myeloid',
        'HSC_LMP': 'Progenitor', 'HSC_MEMP': 'Progenitor',
        'HSC_MMP': 'Progenitor', 'T_Progenitors': 'Progenitor',
        'T_Proliferative': 'Progenitor',
    },
    # scAtlas -> Coarse (immune cells only)
    'scatlas': {
        'CD4T01_Tn_SOX4': 'CD4_T', 'CD4T02_Tn_CCR7': 'CD4_T',
        'CD4T03_Tn_NR4A1': 'CD4_T', 'CD4T04_Tfh_IL6ST': 'CD4_T',
        'CD4T05_Tm_LTB': 'CD4_T', 'CD4T06_Tm_ANXA1': 'CD4_T',
        'CD4T07_Tem_GZMK': 'CD4_T', 'CD4T08_Treg_FOXP3': 'CD4_T',
        'CD8T01_Tn_CCR7': 'CD8_T', 'CD8T02_Tem_GZMK': 'CD8_T',
        'CD8T03_Trm_ITGA1': 'CD8_T', 'CD8T04_Trm_HSPA1A': 'CD8_T',
        'CD8T05_Temra_GZMH': 'CD8_T',
        'CD8T06_MAIT_SLC4A10': 'Unconventional_T',
        'I01_CD16hiNK_CREM': 'NK', 'I02_CD16hiNK_SYNE2': 'NK',
        'I03_CD16hiNK_GZMB': 'NK', 'I04_CD16hiNK_HSPA1A': 'NK',
        'I05_CD16loNK_SELL': 'NK', 'I06_CD16loNK_NR4A2': 'NK',
        'I07_CD16loNK_CXCR6': 'NK',
        'I08_ILC3_KIT': 'Unconventional_T', 'I09_gdT_ITGA1': 'Unconventional_T',
        'B01_Bn_IGHM': 'B', 'B02_Bn_TCL1A': 'B', 'B03_Bn_NR4A2': 'B',
        'B04_Bm_CD27': 'B', 'B05_Bm_NR4A2': 'B', 'B06_Bm_ITGB1': 'B',
        'B07_Bm_HSPA1A': 'B', 'B08_ABC_FCRL5': 'B', 'B09_GCB_RGS13': 'B',
        'B10_Plasmablast_MKI67': 'Plasma', 'B11_Plasma_IGHG2': 'Plasma',
        'B12_Plasma_IGHA2': 'Plasma',
        'M01_cDC1_CLEC9A': 'Myeloid', 'M02_cDC2_CD1C': 'Myeloid',
        'M03_cDC_LAMP3': 'Myeloid', 'M04_pDC_LILRA4': 'Myeloid',
        'M05_Mo_CD14': 'Myeloid', 'M06_Mo_FCGR3A': 'Myeloid',
        'M07_Mph_FCN1': 'Myeloid', 'M08_Mph_NLRP3': 'Myeloid',
        'M09_Mph_FOLR2': 'Myeloid', 'M10_Mph_CD5L': 'Myeloid',
        'M11_Mph_PPARG': 'Myeloid', 'M12_Mph_MT1X': 'Myeloid',
        'M13_immNeu_MMP8': 'Myeloid', 'M14_mNeu_CXCR2': 'Myeloid',
        'M15_Mast_CPA3': 'Myeloid',
    }
}
```

## Recommendations

### 1. For Cross-Atlas Comparison
Use **Coarse Level (8 lineages)** for direct comparison:
- Maximizes cell type overlap
- Reduces noise from annotation inconsistencies
- Comparable sample sizes per group

### 2. For Disease-Specific Analysis
Use **Fine Level (~30 types)** within single atlas:
- Better resolution for biological insights
- Map results to coarse level for cross-atlas validation

### 3. For Tissue-Specific Analysis (scAtlas)
Use **Full Annotations** with tissue context:
- Tissue-resident macrophages vs circulating monocytes
- Resident memory T cells vs effector T cells

### 4. Handling Missing Types
Some cell types exist only in specific atlases:
- **Macrophages**: scAtlas only (tissue-resident)
- **Neutrophils**: scAtlas only
- **NKT cells**: CIMA only (not well-annotated in others)
- **Progenitors**: CIMA/Inflammation only (PBMC)

## Quality Control

1. Validate mapping with marker gene expression
2. Check cell counts per mapped type (minimum threshold: 100 cells)
3. Use silhouette scores or kNN accuracy to assess mapping quality
4. Perform sensitivity analysis with alternative mappings

## References

- CIMA: Comprehensive Immune Monitoring Atlas
- Inflammation Atlas: Disease-associated immune states
- scAtlas: Human Cell Atlas tissue reference
