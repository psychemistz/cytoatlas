# scAtlas Dataset

The scAtlas provides comprehensive single-cell coverage of normal human organs and pan-cancer tumor microenvironments.

## Overview

| Property | Value |
|----------|-------|
| **Total Cells** | 6,440,926 |
| **Normal Organs** | 35+ |
| **Cancer Types** | 15+ |
| **Cell Types** | 100+ (subCluster level) |
| **Gene Symbols** | ~19,000 |

## Dataset Summary

| Dataset | Cells | Organs/Types | Focus |
|---------|-------|--------------|-------|
| Normal | 2,293,951 | 35 organs | Organ-specific signatures |
| PanCancer | 4,146,975 | 15+ cancers | Tumor microenvironment |

## File Paths

```python
# Normal organ counts (raw counts for activity computation)
NORMAL_COUNTS = '/data/Jiang_Lab/Data/Seongyong/scAtlas_2025/igt_s9_fine_counts.h5ad'

# Pan-cancer counts
CANCER_COUNTS = '/data/Jiang_Lab/Data/Seongyong/scAtlas_2025/PanCancer_igt_s9_fine_counts.h5ad'
```

## Cell Observations (`.obs`)

### Normal Organs

| Column | Type | Description | Example Values |
|--------|------|-------------|----------------|
| `tissue` | str | Organ/tissue name | Lung, Liver, Heart, Kidney |
| `cellType1` | category | Coarse cell type | Epithelial, Immune, Stromal |
| `subCluster` | category | Standardized cell type | Alveolar macrophage, Hepatocyte |
| `cellType2` | category | Fine annotation | Alveolar Type 2, Kupffer cell |
| `donorID` | str | Donor identifier | D001, D002, ... |

### PanCancer

| Column | Type | Description | Example Values |
|--------|------|-------------|----------------|
| `tissue` | str | Tissue status | Tumor, Adjacent, Normal |
| `cancerType` | str | Cancer type | Lung adenocarcinoma, Colorectal |
| `subCluster` | category | Cell type annotation | CAF, TAM, CD8+ Tex |
| `donorID` | str | Donor identifier | D001, D002, ... |
| `sample` | str | Sample identifier | S001, S002, ... |

## Organ Coverage (Normal)

| Organ System | Organs |
|--------------|--------|
| Respiratory | Lung, Trachea, Nasal |
| Cardiovascular | Heart, Blood vessel, Spleen |
| Digestive | Liver, Stomach, Small intestine, Colon, Pancreas |
| Urinary | Kidney, Bladder |
| Reproductive | Testis, Ovary, Uterus, Placenta |
| Endocrine | Thyroid, Adrenal |
| Nervous | Brain, Spinal cord |
| Musculoskeletal | Muscle, Bone marrow, Adipose |
| Skin | Skin |
| Immune | Lymph node, Thymus, Tonsil |

## Cancer Types (PanCancer)

| Cancer Type | Abbreviation | Cells (approx) |
|-------------|--------------|----------------|
| Lung adenocarcinoma | LUAD | 500K |
| Colorectal cancer | CRC | 400K |
| Breast cancer | BRCA | 350K |
| Liver hepatocellular | LIHC | 300K |
| Gastric cancer | STAD | 250K |
| Pancreatic cancer | PAAD | 200K |
| Kidney renal | KIRC | 200K |
| Head and neck | HNSC | 180K |
| Ovarian cancer | OV | 150K |
| Melanoma | SKCM | 150K |
| Glioblastoma | GBM | 120K |
| Others | - | ~1M |

## Cell Type Hierarchy (subCluster)

```
Epithelial
├── Alveolar Type 1
├── Alveolar Type 2
├── Hepatocyte
├── Enterocyte
├── Goblet cell
└── ...

Immune
├── T cells
│   ├── CD4+ Naive
│   ├── CD4+ Memory
│   ├── CD8+ Naive
│   ├── CD8+ Memory
│   ├── CD8+ Tex (exhausted)
│   ├── Treg
│   ├── γδ T
│   └── MAIT
├── B cells
│   ├── Naive B
│   ├── Memory B
│   └── Plasma cell
├── NK cells
│   ├── CD56bright NK
│   └── CD56dim NK
├── Myeloid
│   ├── Monocyte
│   ├── Macrophage
│   ├── Dendritic cell
│   └── Neutrophil
└── ILCs

Stromal
├── Fibroblast
├── Cancer-associated fibroblast (CAF)
├── Endothelial
├── Pericyte
└── Smooth muscle

Other
├── Neuronal
├── Oligodendrocyte
└── Erythroid
```

## Tumor vs Adjacent Analysis

The PanCancer dataset includes paired Tumor and Adjacent samples from the same donors:

| Analysis Type | Comparison | Statistical Method |
|---------------|------------|-------------------|
| Paired | Tumor vs Adjacent (same donor) | Paired t-test |
| Unpaired | All Tumor vs All Normal | Wilcoxon rank-sum |

Key columns for paired analysis:
- `donorID`: Links Tumor and Adjacent samples
- `tissue`: "Tumor" or "Adjacent"
- `cancerType`: Cancer type for stratification

## Usage Examples

### Loading the dataset

```python
import anndata as ad

# Normal organs (backed mode)
normal = ad.read_h5ad(
    '/data/Jiang_Lab/Data/Seongyong/scAtlas_2025/igt_s9_fine_counts.h5ad',
    backed='r'
)
print(f"Normal: {normal.n_obs:,} cells, {normal.n_vars:,} genes")

# PanCancer (backed mode)
cancer = ad.read_h5ad(
    '/data/Jiang_Lab/Data/Seongyong/scAtlas_2025/PanCancer_igt_s9_fine_counts.h5ad',
    backed='r'
)
print(f"Cancer: {cancer.n_obs:,} cells")
```

### Organ-specific analysis

```python
# Distribution across organs
print(normal.obs['tissue'].value_counts())

# Cell types in lung
lung_mask = normal.obs['tissue'] == 'Lung'
print(normal.obs.loc[lung_mask, 'subCluster'].value_counts())
```

### Paired tumor-adjacent analysis

```python
# Find donors with both Tumor and Adjacent samples
donor_tissues = cancer.obs.groupby('donorID')['tissue'].apply(set)
paired_donors = donor_tissues[
    donor_tissues.apply(lambda x: 'Tumor' in x and 'Adjacent' in x)
].index.tolist()

print(f"Donors with paired samples: {len(paired_donors)}")
```

### T cell exhaustion analysis

```python
# Identify exhausted T cells
exhausted_markers = ['CD8+ Tex', 'Exhausted', 'Dysfunctional']
tex_mask = cancer.obs['subCluster'].str.contains('|'.join(exhausted_markers), na=False)
print(f"Exhausted T cells: {tex_mask.sum():,}")
```

## Related Pipelines

- [scAtlas Analysis](../pipelines/scatlas/analysis.md)
- [Organ Signatures](../pipelines/scatlas/panels/organs.md)
- [Cancer Comparison](../pipelines/scatlas/panels/cancer.md)
- [T Cell Exhaustion](../pipelines/scatlas/panels/exhaustion.md)
- [Immune Infiltration](../pipelines/scatlas/immune.md)

## Output Files

### Normal Organs
| File | Description |
|------|-------------|
| `normal_organ_signatures.csv` | Organ × signature activity matrix |
| `normal_top_organ_signatures.csv` | Top organ-specific signatures |
| `normal_celltype_signatures.csv` | Cell type × signature matrix |

### PanCancer
| File | Description |
|------|-------------|
| `cancer_comparison.csv` | Tumor vs Adjacent differential |
| `cancer_celltype_activity.csv` | Cell type activities in TME |
| `exhaustion_signatures.csv` | Exhausted vs non-exhausted T cells |

### Single-cell Activities
| File | Description |
|------|-------------|
| `scatlas_cancer_CytoSig_singlecell.h5ad` | Per-cell CytoSig activities |
| `scatlas_cancer_SecAct_singlecell.h5ad` | Per-cell SecAct activities |

## Notes

- Primary cell type annotation uses `subCluster` (standardized across organs)
- Raw counts in `.X` (count matrix from counts H5AD files)
- Paired Tumor-Adjacent analysis requires filtering by `donorID`
- Single-cell activity files are large (~10GB each)
- CAF (Cancer-Associated Fibroblast) analysis available for tumor stroma
