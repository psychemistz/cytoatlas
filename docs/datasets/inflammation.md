# Inflammation Atlas Dataset

The Inflammation Atlas is a comprehensive single-cell atlas spanning 20 inflammatory and autoimmune diseases with treatment response data across three cohorts.

## Overview

| Property | Value |
|----------|-------|
| **Total Cells** | 6,340,934 |
| **Samples** | 1,047 |
| **Diseases** | 20 |
| **Disease Groups** | 6 |
| **Cohorts** | 3 (Main, Validation, External) |
| **Treatment Response Data** | Yes |

## Cohort Summary

| Cohort | Cells | Samples | Genes | Purpose |
|--------|-------|---------|-------|---------|
| Main | 4,918,140 | 817 | 22,826 | Primary discovery |
| Validation | 849,922 | 144 | 22,826 | Internal validation |
| External | 572,872 | 86 | 37,124 | External validation |

## File Paths

```python
# H5AD files (post-QC)
MAIN_H5AD = '/data/Jiang_Lab/Data/Seongyong/Inflammation_Atlas/INFLAMMATION_ATLAS_main_afterQC.h5ad'
VAL_H5AD = '/data/Jiang_Lab/Data/Seongyong/Inflammation_Atlas/INFLAMMATION_ATLAS_validation_afterQC.h5ad'
EXT_H5AD = '/data/Jiang_Lab/Data/Seongyong/Inflammation_Atlas/INFLAMMATION_ATLAS_external_afterQC.h5ad'

# Sample metadata
SAMPLE_META_PATH = '/data/Jiang_Lab/Data/Seongyong/Inflammation_Atlas/INFLAMMATION_ATLAS_afterQC_sampleMetadata.csv'
```

## Cell Observations (`.obs`)

| Column | Type | Description | Example Values |
|--------|------|-------------|----------------|
| `sampleID` | str | Sample identifier | INF001, INF002, ... |
| `Level1` | category | Coarse cell type | T cell, B cell, Myeloid |
| `Level2` | category | Fine cell type (66 types) | Naive CD4+ T, Memory B |
| `Level3` | category | Very fine annotation | Th17.1, IgG+ Plasma |
| `disease` | str | Disease diagnosis | Healthy, RA, IBD, SLE |
| `diseaseGroup` | str | Disease category | Healthy, Autoimmune, Infectious |
| `tissue` | str | Tissue source | Blood, Synovium, Intestine |

## Sample Metadata

### Sample Information (`INFLAMMATION_ATLAS_afterQC_sampleMetadata.csv`)

| Column | Type | Description | Example Values |
|--------|------|-------------|----------------|
| `sampleID` | str | Sample identifier | INF001 |
| `disease` | str | Disease diagnosis | Healthy, RA, IBD, SLE |
| `diseaseGroup` | str | Disease category (6 groups) | Healthy, Autoimmune |
| `therapyResponse` | str | Treatment response | R (responder), NR (non-responder), None |
| `therapy` | str | Treatment type | Anti-TNF, JAK inhibitor |
| `tissue` | str | Tissue source | Blood, Synovium |
| `age` | float | Patient age | 35, 52, 68 |
| `sex` | str | Biological sex | Male, Female |
| `cohort` | str | Study cohort | Main, Validation, External |

## Disease Groups

| Disease Group | Diseases | Samples |
|---------------|----------|---------|
| Healthy | Healthy controls | ~150 |
| Autoimmune | RA, SLE, Sjogren's, PSA | ~300 |
| Inflammatory Bowel | Crohn's, UC | ~200 |
| Infectious | COVID-19, Sepsis | ~150 |
| Allergic | Asthma, Atopic dermatitis | ~100 |
| Other | Various | ~150 |

## Treatment Response Data

The atlas includes longitudinal treatment response data for inflammatory diseases:

| Response Status | Description | Samples |
|-----------------|-------------|---------|
| R (Responder) | Clinical response to therapy | ~200 |
| NR (Non-responder) | No clinical response | ~100 |
| None | No treatment data | ~750 |

Therapies tracked:
- Anti-TNF (infliximab, adalimumab, etanercept)
- JAK inhibitors (tofacitinib, baricitinib)
- Anti-IL-6 (tocilizumab)
- Anti-CD20 (rituximab)

## Cell Type Hierarchy

```
Level 1 (Coarse)
├── T cell
│   └── Level 2 (Fine)
│       ├── Naive CD4+ T
│       ├── Memory CD4+ T
│       ├── Th1
│       ├── Th17
│       ├── Th17.1
│       ├── Treg
│       ├── Naive CD8+ T
│       ├── Memory CD8+ T
│       ├── Cytotoxic CD8+ T
│       ├── γδ T
│       └── MAIT
├── B cell
│   ├── Naive B
│   ├── Memory B
│   ├── Atypical Memory B
│   ├── Plasmablast
│   └── Plasma cell
├── NK cell
│   ├── CD56bright NK
│   ├── CD56dim NK
│   └── Adaptive NK
├── Myeloid
│   ├── Classical Monocyte
│   ├── Intermediate Monocyte
│   ├── Non-classical Monocyte
│   ├── cDC1
│   ├── cDC2
│   ├── pDC
│   ├── Macrophage
│   └── Neutrophil
└── Innate Lymphoid
    ├── ILC1
    ├── ILC2
    └── ILC3
```

## Usage Examples

### Loading the dataset

```python
import anndata as ad
import pandas as pd

# Load main cohort (backed mode)
adata = ad.read_h5ad(
    '/data/Jiang_Lab/Data/Seongyong/Inflammation_Atlas/INFLAMMATION_ATLAS_main_afterQC.h5ad',
    backed='r'
)

print(f"Main cohort: {adata.n_obs:,} cells")
```

### Accessing gene symbols

```python
# Inflammation Atlas uses Ensembl IDs as index, gene symbols in var['symbol']
if 'symbol' in adata.var.columns:
    gene_symbols = adata.var['symbol'].values
```

### Disease analysis

```python
# Sample metadata
sample_meta = pd.read_csv(SAMPLE_META_PATH)

# Disease distribution
print(sample_meta['disease'].value_counts())

# Treatment response samples
response_samples = sample_meta[sample_meta['therapyResponse'].isin(['R', 'NR'])]
print(f"Samples with response data: {len(response_samples)}")
```

### Cross-cohort validation

```python
# Load all cohorts
main = ad.read_h5ad(MAIN_H5AD, backed='r')
val = ad.read_h5ad(VAL_H5AD, backed='r')
ext = ad.read_h5ad(EXT_H5AD, backed='r')

# Compare cell type distributions
for cohort, adata in [('Main', main), ('Validation', val), ('External', ext)]:
    print(f"\n{cohort}: {adata.n_obs:,} cells")
    print(adata.obs['Level2'].value_counts().head(5))
```

## Related Pipelines

- [Inflammation Activity Analysis](../pipelines/inflammation/activity.md)
- [Disease Activity Differential](../pipelines/inflammation/panels/disease.md)
- [Treatment Response Prediction](../pipelines/inflammation/panels/treatment.md)
- [Cross-Cohort Validation](../pipelines/inflammation/panels/validation.md)

## Output Files

| File | Description |
|------|-------------|
| `main_CytoSig_pseudobulk.h5ad` | CytoSig activities (main cohort) |
| `main_SecAct_pseudobulk.h5ad` | SecAct activities (main cohort) |
| `disease_differential.csv` | Disease vs healthy differential |
| `response_prediction_results.csv` | Treatment response ML predictions |
| `cohort_validation_correlation.csv` | Main vs validation correlation |

## Notes

- Primary analysis uses `Level2` (66 fine cell types)
- Sample column is `sampleID`
- Gene symbols in `adata.var['symbol']`, indices are Ensembl IDs
- Treatment response prediction uses sample-level aggregation
- Cross-cohort validation confirms generalizability
