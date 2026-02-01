# CIMA Dataset

The Chinese Immune Multi-omics Atlas (CIMA) is a comprehensive single-cell atlas of healthy human immune cells with paired biochemistry and metabolomics data.

## Overview

| Property | Value |
|----------|-------|
| **Cells** | 6,484,974 |
| **Samples** | 428 |
| **Genes** | 36,326 |
| **Source** | Chinese Immune Multi-omics Atlas |
| **File Size** | ~45 GB (compressed H5AD) |

## File Paths

```python
# Main single-cell data
H5AD_PATH = '/data/Jiang_Lab/Data/Seongyong/CIMA/Cell_Atlas/CIMA_RNA_6484974cells_36326genes_compressed.h5ad'

# Metadata files
SAMPLE_META_PATH = '/data/Jiang_Lab/Data/Seongyong/CIMA/Metadata/CIMA_Sample_Information_Metadata.csv'
BIOCHEMISTRY_PATH = '/data/Jiang_Lab/Data/Seongyong/CIMA/Cell_Atlas/CIMA_Sample_Blood_Biochemistry_Results.csv'
METABOLITES_PATH = '/data/Jiang_Lab/Data/Seongyong/CIMA/Cell_Atlas/CIMA_Sample_Plasma_Metabolites_and_Lipids_Results.csv'
```

## Cell Observations (`.obs`)

| Column | Type | Description | Example Values |
|--------|------|-------------|----------------|
| `sample` | str | Sample identifier | CIMA001, CIMA002, ... |
| `cell_type_l1` | category | Coarse cell type (7 types) | T cell, B cell, Myeloid |
| `cell_type_l2` | category | Intermediate cell type (27 types) | CD4+ T, CD8+ T, NK |
| `cell_type_l3` | category | Fine cell type (100+ types) | Naive CD4+ T, Memory CD8+ T |
| `donor_id` | str | Donor identifier | D001, D002, ... |

## Sample Metadata

### Sample Information (`CIMA_Sample_Information_Metadata.csv`)

| Column | Type | Description | Example Values |
|--------|------|-------------|----------------|
| `Sample_name` | str | Sample identifier | CIMA001 |
| `age` | float | Donor age in years | 25, 45, 65 |
| `sex` | str | Biological sex | Male, Female |
| `bmi` | float | Body mass index | 22.5, 28.3 |
| `smoking` | str | Smoking status | Never, Former, Current |
| `blood_type` | str | ABO blood type | A, B, O, AB |

### Blood Biochemistry (`CIMA_Sample_Blood_Biochemistry_Results.csv`)

399 samples with 19 biochemical markers:

| Column | Type | Description | Unit |
|--------|------|-------------|------|
| `ALT` | float | Alanine aminotransferase | U/L |
| `AST` | float | Aspartate aminotransferase | U/L |
| `ALP` | float | Alkaline phosphatase | U/L |
| `GGT` | float | Gamma-glutamyl transferase | U/L |
| `TBIL` | float | Total bilirubin | μmol/L |
| `ALB` | float | Albumin | g/L |
| `TP` | float | Total protein | g/L |
| `GLU` | float | Glucose | mmol/L |
| `BUN` | float | Blood urea nitrogen | mmol/L |
| `CREA` | float | Creatinine | μmol/L |
| `UA` | float | Uric acid | μmol/L |
| `TC` | float | Total cholesterol | mmol/L |
| `TG` | float | Triglycerides | mmol/L |
| `HDL` | float | HDL cholesterol | mmol/L |
| `LDL` | float | LDL cholesterol | mmol/L |
| `WBC` | float | White blood cell count | 10^9/L |
| `RBC` | float | Red blood cell count | 10^12/L |
| `HGB` | float | Hemoglobin | g/L |
| `PLT` | float | Platelet count | 10^9/L |

### Plasma Metabolites (`CIMA_Sample_Plasma_Metabolites_and_Lipids_Results.csv`)

390 samples with 1,549 metabolite features covering:
- Amino acids and derivatives
- Lipids and fatty acids
- Carbohydrates
- Nucleotides
- Organic acids

## Cell Type Hierarchy

```
Level 1 (7 types)
├── T cell
│   ├── CD4+ T
│   │   ├── Naive CD4+ T
│   │   ├── Memory CD4+ T
│   │   ├── Th1
│   │   ├── Th17
│   │   └── Treg
│   ├── CD8+ T
│   │   ├── Naive CD8+ T
│   │   ├── Memory CD8+ T
│   │   └── Cytotoxic CD8+ T
│   └── Other T
├── B cell
│   ├── Naive B
│   ├── Memory B
│   └── Plasma cell
├── NK cell
│   ├── CD56bright NK
│   └── CD56dim NK
├── Myeloid
│   ├── Classical Monocyte
│   ├── Non-classical Monocyte
│   ├── cDC1
│   ├── cDC2
│   └── pDC
├── Erythroid
├── Platelet
└── Other
```

## Usage Examples

### Loading the dataset

```python
import anndata as ad
import pandas as pd

# Backed mode (memory efficient, ~2GB RAM)
adata = ad.read_h5ad(
    '/data/Jiang_Lab/Data/Seongyong/CIMA/Cell_Atlas/CIMA_RNA_6484974cells_36326genes_compressed.h5ad',
    backed='r'
)

# Full load (requires ~50GB RAM)
# adata = ad.read_h5ad(H5AD_PATH)
```

### Accessing metadata

```python
# Cell type distribution
print(adata.obs['cell_type_l2'].value_counts())

# Sample metadata
sample_meta = pd.read_csv(SAMPLE_META_PATH)
sample_meta = sample_meta.rename(columns={'Sample_name': 'sample'})
```

### Aggregating by sample and cell type

```python
# Group cells for pseudo-bulk analysis
groups = adata.obs.groupby(['sample', 'cell_type_l2'], observed=True).groups
print(f"Found {len(groups)} sample-celltype combinations")
```

## Related Pipelines

- [CIMA Activity Analysis](../pipelines/cima/activity.md)
- [Age/BMI Correlations](../pipelines/cima/panels/correlations.md)
- [Biochemistry Analysis](../pipelines/cima/panels/correlations.md#biochemistry)
- [Metabolite Analysis](../pipelines/cima/panels/metabolites.md)
- [Sex/Smoking Differential](../pipelines/cima/panels/differential.md)

## Output Files

| File | Description |
|------|-------------|
| `CIMA_correlation_age.csv` | Activity × age Spearman correlations |
| `CIMA_correlation_bmi.csv` | Activity × BMI Spearman correlations |
| `CIMA_correlation_biochemistry.csv` | Activity × biochemistry correlations |
| `CIMA_correlation_metabolites.csv` | Activity × metabolite correlations |
| `CIMA_differential_demographics.csv` | Sex/smoking differential analysis |

## Notes

- Uses `cell_type_l2` (27 types) as primary analysis level
- Sample column is `sample` (renamed from `Sample_name` in metadata)
- Raw counts available in `.X` (compressed H5AD)
- GPU acceleration available with CuPy for activity computation
