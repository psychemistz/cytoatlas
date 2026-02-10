# Tahoe-100M Dataset

The Tahoe-100M dataset contains 100.6 million drug-perturbed cancer cells across 50 cell lines treated with 95 drugs, organized into 14 experimental plates. This enables mapping of drug-induced cytokine/secreted protein pathway changes across diverse cancer types.

## Overview

| Property | Value |
|----------|-------|
| **Cells** | ~100,600,000 |
| **Samples** | 14 plates |
| **Genes** | 62,710 |
| **Cell Lines** | 50 cancer cell lines |
| **Drugs** | 95 compounds |
| **Source** | Tahoe-100M Drug Perturbation Atlas |
| **File Size** | ~314 GB (14 plate H5ADs) |
| **Species** | Human |
| **Tissue** | Cancer cell lines (multi-tissue origin) |

## File Paths

```python
# Base directory
TAHOE_DIR = '/data/Jiang_Lab/Data/Seongyong/tahoe/'

# 14 plate files (each 4.7-35 GB)
# plate{1-14}_filt_Vevo_Tahoe100M_WServicesFrom_ParseGigalab.h5ad
PLATE_FILES = [
    f'{TAHOE_DIR}/plate{i}_filt_Vevo_Tahoe100M_WServicesFrom_ParseGigalab.h5ad'
    for i in range(1, 15)
]
```

## Cell Observations (`.obs`)

| Column | Type | Description | Example Values |
|--------|------|-------------|----------------|
| `cell_line` | category | Cancer cell line name | A549, MCF7, HeLa |
| `drug` | category | Drug/compound name | Bortezomib, Trametinib, DMSO_TF |
| `plate` | int | Plate number (1-14) | 1, 7, 13 |
| `dose` | float | Drug concentration | 0.1, 1.0, 10.0 |
| `cell_type` | str | Assigned cell type (if annotated) | epithelial, mesenchymal |
| `cancer_type` | str | Cancer type of origin | lung, breast, colon |

## Gene Variables (`.var`)

| Column | Type | Description |
|--------|------|-------------|
| `gene_name` | str | HGNC gene symbol |
| `gene_id` | str | Ensembl gene ID |
| `feature_type` | str | Gene Biotype |

## Experimental Design

```
14 Plates
├── Plates 1-12: Standard drug screen
│   ├── 50 cell lines x 95 drugs x 1 dose
│   └── DMSO_TF controls per cell line per plate
│
├── Plate 13: Dose-response
│   ├── 50 cell lines x 25 drugs x 3 doses
│   └── Enables dose-response curve fitting
│
└── Plate 14: Replicate plate
    └── Technical replicates for quality control

Total: ~100.6M cells across all plates
Average: ~7.2M cells per plate
```

## Plate Summary

| Plate | Cells (approx.) | Drugs | Cell Lines | Special |
|-------|-----------------|-------|------------|---------|
| 1-12 | ~7M each | 95 | 50 | Standard screen |
| 13 | ~7M | 25 | 50 | 3-dose response |
| 14 | ~7M | 95 | 50 | Replicates |

## Cell Line Metadata

50 cancer cell lines spanning multiple cancer types:

| Cancer Type | Cell Lines (examples) |
|-------------|----------------------|
| Lung | A549, NCI-H1299, NCI-H460 |
| Breast | MCF7, MDA-MB-231, T47D |
| Colon | HCT116, HT29, SW480 |
| Ovarian | SKOV3, A2780, OVCAR3 |
| Melanoma | A375, SK-MEL-28 |
| Pancreatic | PANC-1, MiaPaCa-2 |
| Leukemia | K562, HL-60, Jurkat |

## Drug Categories

95 drugs organized by mechanism of action:

| Category | Examples | Expected Pathway Effects |
|----------|----------|-------------------------|
| Kinase inhibitors | Trametinib (MEK), Sorafenib (multi-kinase) | MAPK/ERK suppression |
| Proteasome inhibitors | Bortezomib, Carfilzomib | NF-kB pathway modulation |
| HDAC inhibitors | Vorinostat, Panobinostat | Chromatin remodeling |
| mTOR inhibitors | Rapamycin, Everolimus | PI3K/AKT/mTOR suppression |
| Cytotoxic agents | Doxorubicin, Cisplatin | Stress/apoptosis response |
| Targeted therapy | Imatinib, Gefitinib | Pathway-specific suppression |
| Immunomodulators | Lenalidomide, Thalidomide | Immune signaling modulation |

## Usage Examples

### Loading a single plate

```python
import anndata as ad
import pandas as pd

# Load one plate in backed mode
plate1 = ad.read_h5ad(
    '/data/Jiang_Lab/Data/Seongyong/tahoe/plate1_filt_Vevo_Tahoe100M_WServicesFrom_ParseGigalab.h5ad',
    backed='r'
)

print(f"Cells: {plate1.n_obs:,}")   # ~7M
print(f"Genes: {plate1.n_vars:,}")  # 62,710
```

### Accessing drug/cell line metadata

```python
# Drug distribution
print(plate1.obs['drug'].value_counts().head(10))

# Cell line distribution
print(plate1.obs['cell_line'].value_counts())

# Control cells
controls = plate1.obs[plate1.obs['drug'] == 'DMSO_TF']
print(f"Control cells: {len(controls):,}")
```

### Processing plate-by-plate

```python
import os

TAHOE_DIR = '/data/Jiang_Lab/Data/Seongyong/tahoe/'
for plate_num in range(1, 15):
    plate_file = f'{TAHOE_DIR}/plate{plate_num}_filt_Vevo_Tahoe100M_WServicesFrom_ParseGigalab.h5ad'
    if os.path.exists(plate_file):
        adata = ad.read_h5ad(plate_file, backed='r')
        print(f"Plate {plate_num}: {adata.n_obs:,} cells")
```

## Related Pipelines

- [Tahoe Drug Response Analysis](../pipelines/perturbation/tahoe.md)
- Drug Signature Extraction (`scripts/22_tahoe_drug_signatures.py`)

## Output Files

| File | Description |
|------|-------------|
| `tahoe_pseudobulk_activity.h5ad` | Drug x cell_line pseudobulk activity |
| `tahoe_drug_vs_control.csv` | Drug - DMSO differential analysis |
| `tahoe_drug_sensitivity_matrix.csv` | Drug x cell_line x signature activity matrix |
| `tahoe_dose_response.csv` | Plate 13 dose-response curves |
| `tahoe_cytokine_pathway_activation.csv` | Drug-induced cytokine pathway changes |

## Notes

- Process plate-by-plate to manage memory (each plate 4.7-35 GB)
- DMSO_TF is the vehicle control for all drug treatments
- Plate 13 has 3 dose levels for 25 drugs — enables dose-response analysis
- 62,710 genes provides excellent overlap (>80%) with both CytoSig and SecAct
- Parallelizable via SLURM array jobs (one job per plate)
- Key scientific question: which cytokine/secreted protein pathways do specific drugs activate or suppress?
