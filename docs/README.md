# CytoAtlas Documentation

Comprehensive documentation for the Pan-Disease Single-Cell Cytokine Activity Atlas project.

## Overview

This project computes cytokine and secreted protein activity signatures across 12+ million human immune cells from three major single-cell atlases:

| Atlas | Cells | Samples | Focus |
|-------|-------|---------|-------|
| [CIMA](datasets/cima.md) | 6.5M | 428 | Healthy aging, metabolism |
| [Inflammation](datasets/inflammation.md) | 6.3M | 1,047 | Disease activity, treatment response |
| [scAtlas](datasets/scatlas.md) | 6.4M | 35+ organs | Organ signatures, cancer comparison |

## Documentation Structure

```
docs/
├── README.md                    # This file
├── OVERVIEW.md                  # High-level project architecture
│
├── datasets/                    # Dataset documentation
│   ├── README.md                # Dataset index
│   ├── cima.md                  # CIMA atlas (6.5M cells)
│   ├── inflammation.md          # Inflammation Atlas (6.3M cells)
│   ├── scatlas.md               # scAtlas (6.4M cells)
│   └── signatures.md            # CytoSig + SecAct signatures
│
├── pipelines/                   # Pipeline documentation
│   ├── README.md                # Pipeline index + dependency graph
│   ├── cima/                    # CIMA analysis
│   ├── inflammation/            # Inflammation Atlas analysis
│   ├── scatlas/                 # scAtlas analysis
│   └── visualization/           # Visualization preprocessing
│
├── outputs/                     # Output file documentation
│   ├── README.md                # Output index
│   ├── results/                 # CSV/H5AD files
│   └── visualization/           # JSON files for web dashboard
│
├── registry.json                # Machine-readable documentation registry
│
└── templates/                   # Documentation templates
    ├── dataset.template.md
    ├── pipeline.template.md
    └── panel.template.md
```

## Quick Links

### Datasets
- [CIMA Atlas](datasets/cima.md) - 6.5M cells, 428 samples, healthy donors with biochemistry/metabolomics
- [Inflammation Atlas](datasets/inflammation.md) - 6.3M cells, 20 diseases, treatment response data
- [scAtlas](datasets/scatlas.md) - 6.4M cells, 35+ organs, normal and cancer tissues
- [Signature Matrices](datasets/signatures.md) - CytoSig (44 cytokines) + SecAct (1,249 secreted proteins)

### Pipelines
- [Pipeline Overview](pipelines/README.md) - Data flow and dependencies
- [CIMA Pipeline](pipelines/cima/activity.md) - Age/BMI correlations, biochemistry, metabolites
- [Inflammation Pipeline](pipelines/inflammation/activity.md) - Disease activity, treatment response
- [scAtlas Pipeline](pipelines/scatlas/analysis.md) - Organ signatures, cancer comparison
- [Visualization Preprocessing](pipelines/visualization/preprocess.md) - JSON generation for web dashboard

### Outputs
- [Output Overview](outputs/README.md) - File structure and lineage
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

## Contributing

1. Use templates in `templates/` for new documentation
2. Update `registry.json` when adding new outputs
3. Keep pipeline docs in sync with code changes
