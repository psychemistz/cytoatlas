# CytoAtlas Documentation

Comprehensive documentation for the Pan-Disease Single-Cell Cytokine Activity Atlas project.

**Start here**: [DEPLOYMENT.md](DEPLOYMENT.md) (setup) → [USER_GUIDE.md](USER_GUIDE.md) (usage) → [API_REFERENCE.md](API_REFERENCE.md) (endpoints) → [ARCHITECTURE.md](ARCHITECTURE.md) (details)

## Quick Links

| Document | Purpose |
|----------|---------|
| **[DEPLOYMENT.md](DEPLOYMENT.md)** | **START HERE** - Setup guide for development, HPC, production |
| **[USER_GUIDE.md](USER_GUIDE.md)** | How to explore CIMA, Inflammation, scAtlas data |
| **[PROJECT_STATUS.md](PROJECT_STATUS.md)** | Current project status, what's done, what remains |
| **[API_REFERENCE.md](API_REFERENCE.md)** | All 260 REST API endpoints with curl examples |
| **[ARCHITECTURE.md](ARCHITECTURE.md)** | System design: components, data flow, technology stack |
| **[CLAUDE.md](../CLAUDE.md)** | Project context for Claude Code |

## Overview

This project computes cytokine and secreted protein activity signatures across 240M+ human cells from six datasets:

| Atlas | Cells | Samples | Focus |
|-------|-------|---------|-------|
| [CIMA](datasets/cima.md) | 6.5M | 428 | Healthy aging, metabolism |
| [Inflammation](datasets/inflammation.md) | 6.3M | 1,047 | Disease activity, treatment response |
| [scAtlas](datasets/scatlas.md) | 6.4M | 35+ organs | Organ signatures, cancer comparison |
| [parse_10M](datasets/parse_10m.md) | 9.7M | 1,092 | Cytokine perturbation, ground truth |
| [Tahoe-100M](datasets/tahoe.md) | 100.6M | 14 plates | Drug perturbation, 50 cancer lines |
| [SpatialCorpus](datasets/spatial_corpus.md) | ~110M | 251 files | Spatial transcriptomics (8 technologies) |

## Quick Facts

- **Total cells**: 240M+ across 6 datasets
- **REST API endpoints**: 260 across 17 routers
- **Signature types**: CytoSig (44), SecAct (1,249)
- **Web UI**: 12 React pages (React 19 + TypeScript + Vite 6 + Tailwind CSS v4)
- **Analysis scripts**: 22 Python pipelines (13 completed, 8 pending GPU execution)
- **See**: [PROJECT_STATUS.md](PROJECT_STATUS.md) for full audit

## Full Documentation Structure

```
docs/
├── README.md                    # This file
├── PROJECT_STATUS.md            # Current project status audit
│
├── datasets/                    # Dataset documentation
│   ├── README.md                # Dataset index
│   ├── cima.md                  # CIMA atlas (6.5M cells)
│   ├── inflammation.md          # Inflammation Atlas (6.3M cells)
│   ├── scatlas.md               # scAtlas (6.4M cells)
│   ├── parse_10m.md             # parse_10M (9.7M cells, cytokine perturbation)
│   ├── tahoe.md                 # Tahoe-100M (100M cells, drug perturbation)
│   ├── spatial_corpus.md        # SpatialCorpus-110M (110M cells, spatial)
│   ├── signatures.md            # CytoSig + SecAct signatures
│   └── bulk.md                  # Bulk RNA-seq validation (GTEx, TCGA)
│
├── pipelines/                   # Pipeline documentation
│   ├── README.md                # Pipeline index + dependency graph
│   ├── cima/                    # CIMA analysis
│   ├── inflammation/            # Inflammation Atlas analysis
│   ├── scatlas/                 # scAtlas analysis
│   ├── perturbation/            # parse_10M + Tahoe analysis
│   ├── spatial/                 # SpatialCorpus analysis
│   └── visualization/           # Visualization preprocessing
│
├── decisions/                   # Architecture Decision Records
│   ├── ADR-001-parquet-over-json.md
│   ├── ADR-002-repository-pattern.md
│   ├── ADR-003-rbac-model.md
│   └── ADR-004-multi-dataset-storage.md
│
├── outputs/                     # Output file documentation
│   └── visualization/           # JSON files for web dashboard
│
├── registry.json                # Machine-readable documentation registry
│
└── templates/                   # Documentation templates
    ├── dataset.template.md
    └── pipeline.template.md
```

## All Guides

### Getting Started
- [DEPLOYMENT.md](DEPLOYMENT.md) - **Development, HPC, and production deployment**
- [USER_GUIDE.md](USER_GUIDE.md) - **How to use CytoAtlas (atlases, chat, exports)**
- [API_REFERENCE.md](API_REFERENCE.md) - **All 260 REST API endpoints**

### Technical Documentation
- [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture (17 sections)
  - Chat system (RAG, LLM abstraction, tool calling)
  - Frontend architecture (SPA, chart components, state management)
  - Pipeline management (dependency graph, execution, validation)
  - Data layer (repository pattern, tiered caching)
  - Security (RBAC, audit logging, prompt injection defense)

### Datasets & Pipelines
- [datasets/README.md](datasets/README.md) - Dataset index
  - [CIMA Atlas](datasets/cima.md) - 6.5M cells, 428 samples, healthy donors with biochemistry/metabolomics
  - [Inflammation Atlas](datasets/inflammation.md) - 6.3M cells, 20 diseases, treatment response data
  - [scAtlas](datasets/scatlas.md) - 6.4M cells, 35+ organs, normal and cancer tissues
  - [parse_10M](datasets/parse_10m.md) - 9.7M cells, cytokine perturbation, ground-truth validation
  - [Tahoe-100M](datasets/tahoe.md) - 100.6M cells, 50 cancer lines x 95 drugs
  - [SpatialCorpus-110M](datasets/spatial_corpus.md) - ~110M cells, 8 spatial technologies
  - [Signature Matrices](datasets/signatures.md) - CytoSig (44 cytokines) + SecAct (1,249 secreted proteins)
  - [Bulk RNA-seq](datasets/bulk.md) - GTEx/TCGA validation data

- [pipelines/README.md](pipelines/README.md) - Pipeline overview and dependency graph
  - [CIMA Pipeline](pipelines/cima/activity.md) - Age/BMI correlations, biochemistry, metabolites
  - [Inflammation Pipeline](pipelines/inflammation/activity.md) - Disease activity, treatment response
  - [scAtlas Pipeline](pipelines/scatlas/analysis.md) - Organ signatures, cancer comparison
  - [parse_10M Pipeline](pipelines/perturbation/parse10m.md) - Cytokine perturbation activity
  - [Tahoe Pipeline](pipelines/perturbation/tahoe.md) - Drug response activity
  - [Spatial Pipeline](pipelines/spatial/activity.md) - Technology-stratified spatial activity
  - [Visualization Preprocessing](pipelines/visualization/preprocess.md) - JSON generation for web dashboard

### Outputs & Analysis
- [JSON Catalog](outputs/visualization/index.md) - Complete list of visualization files
  - [API Mapping](outputs/visualization/api_mapping.md) - JSON to API endpoint mapping
  - [UI Panel Mapping](outputs/visualization/panel_mapping.md) - JSON to UI component mapping

## Key Concepts

### Activity Difference (not Log2FC)

Activity values are z-scores (can be negative), so we use **simple difference** for comparisons:

```python
activity_diff = group1_mean - group2_mean
```

Field name: `activity_diff` (not `log2fc`)
UI label: "Δ Activity"

### Pseudo-bulk Analysis

Primary analysis level: aggregate cells by (sample × cell type) before computing activities:
1. Sum raw counts per group
2. TPM normalize and log2 transform
3. Compute differential expression (subtract row mean)
4. Run ridge regression against signature matrices

### Signature Matrices

| Signature | Proteins | Source | Usage |
|-----------|----------|--------|-------|
| CytoSig | 44 | Jiang et al. | Cytokine activities |
| SecAct | 1,249 | Secreted proteins | Comprehensive secretome |

## Machine-Readable Access

The `registry.json` file provides programmatic access to documentation:

```json
{
  "files": {
    "cima_correlations.json": {
      "type": "visualization",
      "atlas": "CIMA",
      "panels": ["age-correlation", "bmi-correlation"],
      "source_script": "scripts/06_preprocess_viz_data.py",
      "api_endpoints": ["/api/v1/cima/correlations"]
    }
  }
}
```

## MCP Tools

Documentation is accessible via MCP tools in the CytoAtlas API:

- `get_data_lineage(file)` - Trace how any output file was generated
- `get_column_definition(file, column)` - Get column descriptions
- `find_source_script(output)` - Find which script generates a file
- `list_panel_outputs(panel)` - List all outputs for an analysis panel

### Architecture & Decisions
- [ARCHITECTURE.md](ARCHITECTURE.md) - Complete system architecture (20 sections)
- [decisions/README.md](decisions/README.md) - Architecture Decision Records summary
  - [ADR-001: Parquet over JSON](decisions/ADR-001-parquet-over-json.md)
  - [ADR-002: Repository Pattern](decisions/ADR-002-repository-pattern.md)
  - [ADR-003: RBAC Model](decisions/ADR-003-rbac-model.md)
  - [ADR-004: Multi-Dataset Storage](decisions/ADR-004-multi-dataset-storage.md)

### Validation & Credibility
- [ATLAS_VALIDATION.md](ATLAS_VALIDATION.md) - 5-type validation framework
- [CELL_TYPE_MAPPING.md](CELL_TYPE_MAPPING.md) - Cell type harmonization across atlases
- [EMBEDDED_DATA_CHECKLIST.md](EMBEDDED_DATA_CHECKLIST.md) - **IMPORTANT**: JSON files required in frontend

### Legacy & Archive
- [archive/](archive/) - Historical planning documents
- [archive/plans/](archive/plans/) - Old session plans (preserved for reference)

## How to Use This Documentation

1. **First time?** Start with [DEPLOYMENT.md](DEPLOYMENT.md)
2. **Want to explore data?** Read [USER_GUIDE.md](USER_GUIDE.md)
3. **Building on the API?** Check [API_REFERENCE.md](API_REFERENCE.md)
4. **Need to understand architecture?** Read [ARCHITECTURE.md](ARCHITECTURE.md)
5. **Claude Code session?** See [../CLAUDE.md](../CLAUDE.md)

## Contributing

1. Use templates in `templates/` for new documentation
2. Update `registry.json` when adding new outputs
3. Keep pipeline docs in sync with code changes
4. Update relevant sections in [ARCHITECTURE.md](ARCHITECTURE.md) for system changes
5. Add new documentation links to this README.md
