# SpatialCorpus-110M Dataset

The SpatialCorpus-110M is a collection of 251 spatial transcriptomics datasets totaling approximately 110 million cells across 8 technologies (Visium, Xenium, MERFISH, MERSCOPE, CosMx, ISS, Slide-seq, and 10x Chromium spatial). This is the largest integrated spatial transcriptomics resource for cytokine activity analysis.

## Overview

| Property | Value |
|----------|-------|
| **Cells** | ~110,000,000 |
| **Datasets** | 251 H5AD files |
| **Genes** | Variable (150-20,000 per technology) |
| **Technologies** | 8 spatial platforms |
| **Source** | SpatialCorpus (NicheFormer collection) |
| **File Size** | ~377 GB total |
| **Species** | Human (244 files) + Mouse (7 files, excluded) |
| **Tissues** | 30+ tissue/organ types |

## File Paths

```python
# Base directory
SPATIAL_DIR = '/data/Jiang_Lab/Data/Seongyong/SpatialCorpus-110M/'

# 251 H5AD files organized by technology subdirectories
# Example files:
# {SPATIAL_DIR}/visium/V1_Human_Lymph_Node.h5ad
# {SPATIAL_DIR}/xenium/Xenium_Human_Breast_Cancer.h5ad
# {SPATIAL_DIR}/merfish/MERFISH_Human_Brain.h5ad
```

## Technology Tiers

Gene panel size determines analysis strategy:

| Tier | Technologies | Files | Genes | Strategy |
|------|-------------|-------|-------|----------|
| **A** (Full inference) | Visium | 171 | 15,000-20,000 | Full CytoSig + SecAct ridge regression |
| **B** (Targeted scoring) | Xenium, MERFISH, MERSCOPE, CosMx | 51 | 150-1,000 | Subset signatures with sufficient gene overlap |
| **C** (Skip) | ISS (3 files), mouse datasets (9 files) | 12 | <150 or wrong species | Excluded from analysis |

## Technology Summary

| Technology | Files | Cells (approx.) | Genes | Resolution |
|------------|-------|-----------------|-------|------------|
| Visium | 171 | ~60M | 15,000-20,000 | 55 um spots |
| Xenium | 23 | ~25M | 300-500 | Subcellular |
| MERFISH | 15 | ~10M | 150-400 | Subcellular |
| MERSCOPE | 8 | ~5M | 300-500 | Subcellular |
| CosMx | 5 | ~3M | 900-1,000 | Subcellular |
| ISS | 3 | ~1M | 100-200 | In situ |
| Slide-seq | 7 (mouse) | ~4M | 20,000 | 10 um beads |
| 10x amb | 2 (mouse) | ~2M | 15,000 | Variable |

## Cell Observations (`.obs`)

Columns vary by technology and dataset, but common fields include:

| Column | Type | Description | Availability |
|--------|------|-------------|--------------|
| `cell_type` | str | Cell type annotation | Most datasets |
| `tissue` | str | Tissue/organ type | All datasets |
| `technology` | str | Spatial technology | All datasets |
| `x_coord` | float | Spatial X coordinate | All datasets |
| `y_coord` | float | Spatial Y coordinate | All datasets |
| `sample_id` | str | Sample/section identifier | All datasets |
| `disease` | str | Disease status | Some datasets |
| `region` | str | Tissue region/zone | Some datasets |
| `niche_cluster` | int | Spatial neighborhood cluster | NicheFormer-annotated |
| `nicheformer_embedding` | array | NicheFormer latent embedding | Some datasets |

## Gene Variables (`.var`)

| Column | Type | Description |
|--------|------|-------------|
| `gene_name` | str | Gene symbol (varies by platform) |
| `gene_id` | str | Ensembl gene ID (where available) |
| `in_panel` | bool | Whether gene is in technology's panel |

## Tissue Distribution

| Tissue Category | Datasets | Technologies |
|----------------|----------|-------------|
| Brain | 35 | Visium, MERFISH, Slide-seq |
| Breast | 28 | Visium, Xenium, CosMx |
| Lung | 22 | Visium, Xenium, MERSCOPE |
| Liver | 18 | Visium, MERFISH |
| Kidney | 15 | Visium, MERSCOPE |
| Heart | 12 | Visium |
| Colon/Intestine | 14 | Visium, Xenium |
| Skin | 10 | Visium, CosMx |
| Lymph node | 8 | Visium, Xenium |
| Tumor (various) | 40+ | Visium, Xenium, MERFISH |
| Other organs | 49 | Mixed |

## Species Filter

**Human only** — skip 9 mouse datasets (~103 GB):
- 7 mouse Slide-seq brain files
- 2 mouse 10x amb files

This aligns with existing CytoAtlas atlases (all human) and reduces processing from 377 GB to ~274 GB.

## Usage Examples

### Discovering available datasets

```python
import os
import anndata as ad
from pathlib import Path

SPATIAL_DIR = Path('/data/Jiang_Lab/Data/Seongyong/SpatialCorpus-110M/')

# List all H5AD files
h5ad_files = sorted(SPATIAL_DIR.rglob('*.h5ad'))
print(f"Total datasets: {len(h5ad_files)}")
```

### Loading a Visium dataset

```python
# Visium dataset (full gene panel, Tier A)
adata = ad.read_h5ad(
    f'{SPATIAL_DIR}/visium/V1_Human_Lymph_Node.h5ad',
    backed='r'
)

print(f"Cells/spots: {adata.n_obs:,}")
print(f"Genes: {adata.n_vars:,}")  # ~15,000-20,000

# Spatial coordinates
if 'x_coord' in adata.obs.columns:
    coords = adata.obs[['x_coord', 'y_coord']]
elif 'spatial' in adata.obsm:
    coords = adata.obsm['spatial']
```

### Checking gene panel coverage

```python
from secactpy import load_cytosig

cytosig = load_cytosig()
cytosig_genes = set(cytosig.index)
dataset_genes = set(adata.var_names)

overlap = cytosig_genes & dataset_genes
coverage = len(overlap) / len(cytosig_genes) * 100
print(f"CytoSig gene coverage: {coverage:.1f}% ({len(overlap)}/{len(cytosig_genes)})")
# Visium: ~80%+, MERFISH: ~5-15%, Xenium: ~10-20%
```

### Technology-stratified processing

```python
# Group files by technology
from collections import defaultdict
tech_files = defaultdict(list)
for f in h5ad_files:
    tech = f.parent.name  # technology subdirectory
    tech_files[tech].append(f)

for tech, files in sorted(tech_files.items()):
    print(f"{tech}: {len(files)} files")
```

## Spatial-Specific Considerations

### Gene Panel Coverage

The key limitation of spatial transcriptomics for activity inference is gene panel size:

- **Visium** (Tier A): 15K+ genes, >80% overlap with CytoSig/SecAct. Full ridge regression feasible.
- **Xenium/MERFISH/CosMx** (Tier B): 150-1,000 genes, 5-30% overlap. Only signatures with sufficient gene coverage can be scored. Use targeted signature scoring with bootstrap confidence intervals.
- **ISS** (Tier C): <150 genes, insufficient for any signature inference.

### Spatial Coordinates

Stored in `.obs` (x_coord, y_coord) or `.obsm['spatial']`. For visualization, coordinates are downsampled to ~10,000 points per dataset to keep API response sizes manageable.

### NicheFormer Embeddings

Some datasets include NicheFormer latent space embeddings that capture spatial neighborhood context. These can be used for:
- Spatial clustering (niche identification)
- Neighborhood activity patterns
- Cross-technology comparison of spatial niches

## Related Pipelines

- [Spatial Activity Analysis](../pipelines/spatial/activity.md)
- Spatial Neighborhood Analysis (`scripts/23_spatial_neighborhood.py`)

## Output Files

| File | Description |
|------|-------------|
| `spatial_activity_by_technology.h5ad` | Technology-stratified activity results |
| `spatial_activity_by_tissue.csv` | Tissue-level activity summary |
| `spatial_neighborhood_activity.csv` | Niche-level activity patterns |
| `spatial_technology_comparison.csv` | Cross-technology reproducibility |

## Notes

- 251 files with heterogeneous schemas — process per-technology then integrate
- Gene panel coverage varies dramatically by technology
- Spatial coordinates are essential for neighborhood analysis
- Human-only filtering removes 9 mouse datasets, saving ~103 GB of processing
- Visium datasets are the primary focus for full activity inference
- Targeted scoring for Tier B datasets uses only signatures where >50% of genes are in the panel
