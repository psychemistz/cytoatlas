# parse_10M Dataset

The parse_10M dataset contains 9.7 million cytokine-treated human PBMCs from 12 donors across 90 cytokine conditions plus PBS controls. This is the **ground truth** validation dataset for CytoSig — cells treated with a known cytokine should show elevated predicted activity for that cytokine's signature.

## Overview

| Property | Value |
|----------|-------|
| **Cells** | 9,697,974 |
| **Samples** | 1,092 (12 donors x 91 conditions) |
| **Genes** | 40,352 |
| **Source** | Parse Biosciences Mega-scale PBMC perturbation |
| **File Size** | ~212 GB (single H5AD) |
| **Species** | Human |
| **Tissue** | PBMC (peripheral blood mononuclear cells) |

## File Paths

```python
# Main single-cell data
H5AD_PATH = '/data/Jiang_Lab/Data/Seongyong/parse_10M/Parse_10M_PBMC_cytokines.h5ad'

# Cytokine treatment metadata
CYTOKINE_META_PATH = '/data/Jiang_Lab/Data/Seongyong/parse_10M/cytokine_origin_parse10M.csv'
```

## Cell Observations (`.obs`)

| Column | Type | Description | Example Values |
|--------|------|-------------|----------------|
| `donor_id` | str | Donor identifier (12 donors) | D1, D2, ..., D12 |
| `cytokine` | category | Cytokine treatment condition | IL-17A, IFN-gamma, PBS |
| `cell_type` | category | PBMC cell type (18 types) | CD4+ T, Monocyte, NK |
| `condition` | str | Treatment condition label | IL-17A_100ng, PBS_control |
| `concentration` | float | Cytokine concentration (ng/mL) | 10, 50, 100 |
| `time_point` | str | Treatment duration | 24h |

## Gene Variables (`.var`)

| Column | Type | Description |
|--------|------|-------------|
| `gene_name` | str | HGNC gene symbol |
| `gene_id` | str | Ensembl gene ID |
| `feature_type` | str | Gene Biotype (Gene Expression) |

## Cytokine Treatment Metadata (`cytokine_origin_parse10M.csv`)

90 cytokine conditions + PBS control:

| Column | Type | Description | Example Values |
|--------|------|-------------|----------------|
| `cytokine` | str | Cytokine name | IL-17A, IFN-gamma, TNF-alpha |
| `family` | str | Cytokine family grouping | Interleukin, Interferon, TNF |
| `concentration` | float | Treatment concentration (ng/mL) | 10, 50, 100 |
| `vendor` | str | Reagent vendor | PeproTech, R&D Systems |
| `catalog_number` | str | Vendor catalog number | 200-17, 285-IF |

## Experimental Design

```
12 Donors
├── Each donor receives:
│   ├── 90 cytokine treatments (varying concentrations)
│   └── 1 PBS control (vehicle)
│   = 91 conditions per donor
│
├── Total: 12 x 91 = 1,092 sample groups
├── ~8,900 cells per sample group (average)
└── 18 cell types per sample (PBMC composition)
```

## Cell Type Hierarchy

```
PBMC (18 types)
├── T cell
│   ├── CD4+ T naive
│   ├── CD4+ T memory
│   ├── CD4+ T effector
│   ├── Th1
│   ├── Th17
│   ├── Treg
│   ├── CD8+ T naive
│   ├── CD8+ T memory
│   ├── CD8+ T effector
│   └── gamma-delta T
├── B cell
│   ├── Naive B
│   ├── Memory B
│   └── Plasmablast
├── NK cell
│   ├── CD56bright NK
│   └── CD56dim NK
├── Monocyte
│   ├── Classical monocyte
│   └── Non-classical monocyte
└── Dendritic cell
```

## Usage Examples

### Loading the dataset

```python
import anndata as ad
import pandas as pd

# Backed mode (memory efficient, ~4GB RAM)
adata = ad.read_h5ad(
    '/data/Jiang_Lab/Data/Seongyong/parse_10M/Parse_10M_PBMC_cytokines.h5ad',
    backed='r'
)

# Check dimensions
print(f"Cells: {adata.n_obs:,}")   # 9,697,974
print(f"Genes: {adata.n_vars:,}")   # 40,352
```

### Accessing treatment metadata

```python
# Cell type distribution
print(adata.obs['cell_type'].value_counts())

# Cytokine conditions
print(adata.obs['cytokine'].value_counts())

# Cytokine metadata
cytokine_meta = pd.read_csv(CYTOKINE_META_PATH)
print(f"Cytokines: {len(cytokine_meta)}")
```

### Pseudobulk aggregation (donor x cytokine x cell_type)

```python
# Group cells for pseudo-bulk analysis
groups = adata.obs.groupby(['donor_id', 'cytokine', 'cell_type'], observed=True).groups
print(f"Found {len(groups)} pseudobulk groups")
# Expected: ~1,092 conditions x 18 cell types = ~19,656 groups
```

## Ground Truth Validation Logic

This dataset enables **direct experimental validation** of CytoSig predictions:

```python
# For each cytokine c in 90 treatments:
#   For each cell_type t:
#     treated = activity[donor, cytokine=c, cell_type=t]
#     control = activity[donor, cytokine=PBS, cell_type=t]
#     response = treated - control  # treatment effect
#     ground_truth[c, t] = response[c]  # self-signature response
#     # If CytoSig works: self-signature response >> non-self response
```

### CytoSig overlap with 90 tested cytokines

Of the 44 CytoSig signatures, many overlap with the 90 cytokines tested in parse_10M. This direct overlap enables gold-standard validation: cells treated with cytokine X should show elevated CytoSig-predicted activity for signature X.

## Related Pipelines

- [parse_10M Activity Analysis](../pipelines/perturbation/parse10m.md)
- Ground Truth Validation (`scripts/21_parse10m_ground_truth.py`)

## Output Files

| File | Description |
|------|-------------|
| `parse10m_pseudobulk_activity.h5ad` | Pseudobulk activity (donor x cytokine x cell_type) |
| `parse10m_treatment_vs_control.csv` | Treatment - PBS control differential |
| `parse10m_cytokine_response_matrix.csv` | Cytokine x cell_type activity heatmap |
| `parse10m_ground_truth_validation.csv` | Predicted vs actual cytokine response |

## Notes

- Single H5AD file (212 GB) — **must** use `backed='r'` mode
- Chunked processing by donor x cytokine groups (1,092 groups)
- PBS control available per donor for paired comparisons
- 40,352 genes provides excellent overlap (>80%) with CytoSig and SecAct signatures
- This is the most scientifically valuable dataset for validating the signature matrices
