# Dataset Documentation

This directory contains detailed documentation for all datasets used in the CytoAtlas project.

## Atlases Overview

| Atlas | Cells | Samples | Genes | Primary Focus |
|-------|-------|---------|-------|---------------|
| [CIMA](cima.md) | 6,484,974 | 428 | 36,326 | Healthy aging, metabolism, biochemistry |
| [Inflammation](inflammation.md) | 6,340,934 | 1,047 | 22,826+ | Disease activity, treatment response |
| [scAtlas](scatlas.md) | 6,440,926 | 35+ organs | 19,000+ | Organ signatures, cancer comparison |
| [SpatialCorpus](spatial_corpus.md) | ~110,000,000 | 251 files | 150-20K | Spatial transcriptomics (8 technologies) |
| [parse_10M](parse_10m.md) | 9,697,974 | 1,092 | 40,352 | Cytokine perturbation (ground truth) |
| [Tahoe-100M](tahoe.md) | ~100,600,000 | 14 plates | 62,710 | Drug perturbation (50 cancer lines) |

## Signature Matrices

| Signature | Proteins | Description |
|-----------|----------|-------------|
| [CytoSig](signatures.md#cytosig) | 44 | Cytokine response signatures |
| [SecAct](signatures.md#secact) | 1,249 | Comprehensive secreted protein signatures |

## Data Locations

All H5AD files are stored on the Jiang Lab data partition:
```
/data/Jiang_Lab/Data/Seongyong/
├── CIMA/
│   ├── Cell_Atlas/
│   └── Metadata/
├── Inflammation_Atlas/
├── scAtlas_2025/
├── SpatialCorpus-110M/     # 251 H5AD files (Visium, Xenium, MERFISH, etc.)
├── parse_10M/              # Single 212 GB H5AD + metadata CSV
└── tahoe/                  # 14 plate H5AD files
```

## Common Patterns

### Cell Type Hierarchies

Each atlas uses hierarchical cell type annotations:

| Atlas | Coarse | Intermediate | Fine |
|-------|--------|--------------|------|
| CIMA | cell_type_l1 | cell_type_l2 | cell_type_l3 |
| Inflammation | Level1 | Level2 | Level3 |
| scAtlas | cellType1 | subCluster | cellType2 |

### Data Access Patterns

```python
import anndata as ad

# Memory-efficient backed mode (recommended for exploration)
adata = ad.read_h5ad(path, backed='r')

# Full load (required for some operations)
adata = ad.read_h5ad(path)

# Access counts from layers
if 'counts' in adata.layers:
    raw_counts = adata.layers['counts']
```

## Quick Reference

### Sample Columns by Atlas

| Column | CIMA | Inflammation | scAtlas | parse_10M | Tahoe | SpatialCorpus |
|--------|------|--------------|---------|-----------|-------|---------------|
| Sample ID | `sample` | `sampleID` | `donorID` | `sample_id` | `plate` | varies |
| Cell Type | `cell_type_l2` | `Level2` | `subCluster` | `cell_type` | `cell_line` | `cell_type` |
| Condition | - | `disease` | `cancerType` | `cytokine` | `drug` | `tissue` |
| Control | - | - | - | `PBS` | `DMSO_TF` | - |
