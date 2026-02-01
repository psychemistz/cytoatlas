# JSON to API Endpoint Mapping

Maps visualization JSON files to their corresponding REST API endpoints.

## Overview

The CytoAtlas API (`cytoatlas-api/`) serves data from the JSON files through structured REST endpoints.

## Mapping Table

### CIMA Endpoints

| JSON File | API Endpoint | Method | Description |
|-----------|--------------|--------|-------------|
| `cima_correlations.json` | `/api/v1/cima/correlations` | GET | Age/BMI/biochemistry correlations |
| `cima_correlations.json` | `/api/v1/cima/correlations/age` | GET | Age correlations only |
| `cima_correlations.json` | `/api/v1/cima/correlations/bmi` | GET | BMI correlations only |
| `cima_differential.json` | `/api/v1/cima/differential` | GET | Sex/smoking differential |
| `cima_metabolites_top.json` | `/api/v1/cima/metabolites` | GET | Top metabolite correlations |
| `cima_celltype.json` | `/api/v1/cima/celltypes` | GET | Cell type activities |
| `cima_biochem_scatter.json` | `/api/v1/cima/biochemistry/scatter` | GET | Scatter plot data |
| `cima_eqtl.json` | `/api/v1/cima/eqtl` | GET | eQTL associations |

### Inflammation Endpoints

| JSON File | API Endpoint | Method | Description |
|-----------|--------------|--------|-------------|
| `inflammation_celltype.json` | `/api/v1/inflammation/celltypes` | GET | Cell type activities |
| `inflammation_disease_filtered.json` | `/api/v1/inflammation/disease` | GET | Disease differential |
| `inflammation_disease_filtered.json` | `/api/v1/inflammation/disease/{name}` | GET | Specific disease |
| `inflammation_severity_filtered.json` | `/api/v1/inflammation/severity` | GET | Disease severity |
| `treatment_response.json` | `/api/v1/inflammation/treatment` | GET | Treatment response |
| `cohort_validation.json` | `/api/v1/inflammation/validation` | GET | Cross-cohort validation |
| `inflammation_longitudinal.json` | `/api/v1/inflammation/longitudinal` | GET | Longitudinal data |

### scAtlas Endpoints

| JSON File | API Endpoint | Method | Description |
|-----------|--------------|--------|-------------|
| `scatlas_organs.json` | `/api/v1/scatlas/organs` | GET | Organ signatures |
| `scatlas_organs.json` | `/api/v1/scatlas/organs/{organ}` | GET | Specific organ |
| `scatlas_organs_top.json` | `/api/v1/scatlas/organs/top` | GET | Top organ markers |
| `scatlas_celltypes.json` | `/api/v1/scatlas/celltypes` | GET | Cell type signatures |
| `cancer_comparison.json` | `/api/v1/scatlas/cancer/comparison` | GET | Tumor vs Adjacent |
| `cancer_types.json` | `/api/v1/scatlas/cancer/types` | GET | Cancer type activities |
| `exhaustion.json` | `/api/v1/scatlas/exhaustion` | GET | T cell exhaustion |
| `immune_infiltration.json` | `/api/v1/scatlas/immune` | GET | Immune infiltration |
| `caf_signatures.json` | `/api/v1/scatlas/caf` | GET | CAF signatures |

### Cross-Atlas Endpoints

| JSON File | API Endpoint | Method | Description |
|-----------|--------------|--------|-------------|
| `cross_atlas.json` | `/api/v1/cross-atlas/comparison` | GET | Atlas comparison |
| `search_index.json` | `/api/v1/search` | GET | Global search |
| `summary_stats.json` | `/api/v1/stats` | GET | Project summary |

## API Response Format

All endpoints return JSON with consistent structure:

```json
{
  "data": [...],
  "meta": {
    "total": 1234,
    "page": 1,
    "page_size": 100,
    "signature_type": "CytoSig"
  }
}
```

## Query Parameters

### Filtering

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `cell_type` | string | Filter by cell type | `?cell_type=CD4+ T` |
| `signature` | string | Filter by signature | `?signature=IL6` |
| `signature_type` | string | CytoSig or SecAct | `?signature_type=CytoSig` |
| `organ` | string | Filter by organ | `?organ=Lung` |
| `disease` | string | Filter by disease | `?disease=RA` |

### Pagination

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `page` | int | 1 | Page number |
| `page_size` | int | 100 | Results per page |
| `offset` | int | 0 | Skip N results |
| `limit` | int | 100 | Max results |

### Sorting

| Parameter | Type | Description |
|-----------|------|-------------|
| `sort_by` | string | Column to sort by |
| `order` | string | `asc` or `desc` |

## Service Functions

Each endpoint is backed by a service function in `cytoatlas-api/app/services/`:

| Router | Service File | Functions |
|--------|--------------|-----------|
| CIMA | `cima_service.py` | `get_correlations()`, `get_differential()` |
| Inflammation | `inflammation_service.py` | `get_disease_differential()`, `get_treatment_response()` |
| scAtlas | `scatlas_service.py` | `get_organ_signatures()`, `get_cancer_comparison()` |
| Cross-Atlas | `cross_atlas_service.py` | `get_atlas_comparison()` |
| Search | `search_service.py` | `search_global()` |

## Example API Calls

### Get CIMA age correlations for IL6

```bash
curl "http://localhost:8000/api/v1/cima/correlations/age?signature=IL6"
```

### Get disease differential for RA in monocytes

```bash
curl "http://localhost:8000/api/v1/inflammation/disease/RA?cell_type=Classical%20Monocyte"
```

### Get organ signatures with top specificity

```bash
curl "http://localhost:8000/api/v1/scatlas/organs/top?signature_type=CytoSig&limit=20"
```

### Search for TGFB across all atlases

```bash
curl "http://localhost:8000/api/v1/search?q=TGFB"
```
