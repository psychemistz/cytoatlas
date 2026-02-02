# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Critical Instructions

**Git commits:** Do not include "Co-Authored-By" lines in commit messages.

**Sensitive data:** Never commit personally sensitive information to git. This includes API keys (Claude/Anthropic, OpenAI, etc.), passwords, tokens, and credentials. Keep such values in `.env` files that are gitignored or use environment variables.

**Data handling:** Always ask the user for data paths. Never use mock data for validation. When implementing services, sample from real datasets for development.

## Project Overview

Pan-Disease Single-Cell Cytokine Activity Atlas - computes cytokine and secreted protein activity signatures across 12+ million human immune cells from three major single-cell atlases (CIMA, Inflammation Atlas, scAtlas) to identify disease-specific and conserved signaling patterns.

## Documentation

Comprehensive documentation is available in the `docs/` directory:

| Document | Description |
|----------|-------------|
| [docs/README.md](docs/README.md) | Documentation index |
| [docs/OVERVIEW.md](docs/OVERVIEW.md) | High-level project architecture |
| [docs/datasets/](docs/datasets/) | Dataset specifications (CIMA, Inflammation, scAtlas) |
| [docs/pipelines/](docs/pipelines/) | Analysis pipeline documentation |
| [docs/outputs/](docs/outputs/) | Output file catalog and API mapping |
| [docs/registry.json](docs/registry.json) | Machine-readable documentation registry |
| [docs/EMBEDDED_DATA_CHECKLIST.md](docs/EMBEDDED_DATA_CHECKLIST.md) | **IMPORTANT**: Checklist of JSON files required in embedded_data.js |

### MCP Documentation Tools

The API includes MCP tools for programmatic documentation access:

```python
# Available tools in cytoatlas-api/app/services/mcp_tools.py
get_data_lineage(file_name)      # Trace file generation
get_column_definition(file, col)  # Get column descriptions
find_source_script(output_file)   # Find generating script
list_panel_outputs(panel_name)    # List panel outputs
get_dataset_info(dataset_name)    # Get dataset details
```

## Development Environment

```bash
source ~/bin/myconda
conda activate secactpy
```

Required external package: `secactpy` from `/vf/users/parks34/projects/1ridgesig/SecActpy/`

## Running Analyses

### SLURM Job Submission (HPC)

```bash
sbatch scripts/slurm/run_all.sh           # Full pipeline
sbatch scripts/slurm/run_all.sh --pilot   # Pilot only (~2 hours)
sbatch scripts/slurm/run_all.sh --main    # Main analyses only

# Individual analyses
sbatch scripts/slurm/run_pilot.sh         # Pilot validation
sbatch scripts/slurm/run_cima.sh          # CIMA 6.5M cells
sbatch scripts/slurm/run_inflam.sh        # Inflammation 6.3M cells
sbatch scripts/slurm/run_scatlas.sh       # scAtlas 6.4M cells
sbatch scripts/slurm/run_integrated.sh    # Cross-atlas comparison
```

### Direct Execution

```bash
cd /data/parks34/projects/2secactpy
python scripts/00_pilot_analysis.py --n-cells 100000 --seed 42
python scripts/01_cima_activity.py --mode pseudobulk
python scripts/02_inflam_activity.py --mode both
python scripts/05_figures.py --all
python scripts/06_preprocess_viz_data.py
```

## Pipeline Architecture

| Phase | Script | Description |
|-------|--------|-------------|
| 0 | `00_pilot_analysis.py` | Pilot validation on 100K cell subsets |
| 1 | `01_cima_activity.py` | CIMA: 6.5M cells, biochemistry/metabolomics correlations |
| 2 | `02_inflam_activity.py` | Inflammation Atlas: 6.3M cells, treatment response prediction |
| 3 | `03_scatlas_analysis.py` | scAtlas: 6.4M cells (normal organs + cancer) |
| 4 | `04_integrated.py` | Cross-atlas integration |
| 5 | `05_figures.py`, `06_preprocess_viz_data.py` | Publication figures and web visualization |

### Key Design Patterns

- GPU acceleration via CuPy (10-34x speedup) with NumPy fallback
- Pseudo-bulk aggregation (cell type × sample) as primary analysis level
- Single-cell batch processing (10K cells/batch) for detailed analysis
- Backed mode (`ad.read_h5ad(..., backed='r')`) for memory efficiency

## Data Paths

```python
# CIMA
CIMA_H5AD = '/data/Jiang_Lab/Data/Seongyong/CIMA/Cell_Atlas/CIMA_RNA_6484974cells_36326genes_compressed.h5ad'
CIMA_BIOCHEM = '/data/Jiang_Lab/Data/Seongyong/CIMA/Cell_Atlas/CIMA_Sample_Blood_Biochemistry_Results.csv'
CIMA_METAB = '/data/Jiang_Lab/Data/Seongyong/CIMA/Cell_Atlas/CIMA_Sample_Plasma_Metabolites_and_Lipids_Results.csv'

# Inflammation Atlas
MAIN_H5AD = '/data/Jiang_Lab/Data/Seongyong/Inflammation_Atlas/INFLAMMATION_ATLAS_main_afterQC.h5ad'
VAL_H5AD = '/data/Jiang_Lab/Data/Seongyong/Inflammation_Atlas/INFLAMMATION_ATLAS_validation_afterQC.h5ad'
EXT_H5AD = '/data/Jiang_Lab/Data/Seongyong/Inflammation_Atlas/INFLAMMATION_ATLAS_external_afterQC.h5ad'

# scAtlas
NORMAL_COUNTS = '/data/Jiang_Lab/Data/Seongyong/scAtlas_2025/igt_s9_fine_counts.h5ad'
CANCER_COUNTS = '/data/Jiang_Lab/Data/Seongyong/scAtlas_2025/PanCancer_igt_s9_fine_counts.h5ad'
```

## Signature Matrices

```python
from secactpy import load_cytosig, load_secact
cytosig = load_cytosig()  # genes × 44 cytokines
secact = load_secact()    # genes × 1,249 secreted proteins
```

## Output Structure

```
results/
├── pilot/           # Pilot validation
├── cima/            # CIMA activity results, correlations, differential
├── inflammation/    # Inflammation Atlas results, predictions
├── scatlas/         # scAtlas organ/celltype signatures
├── integrated/      # Cross-atlas comparisons
└── figures/         # Publication figures

visualization/
├── data/            # JSON files for web dashboard
└── index.html       # Interactive visualization
```

## Statistical Methods

| Method | Use Case | Function |
|--------|----------|----------|
| Spearman correlation | Continuous variables (age, BMI, metabolites) | `correlation_analysis()` |
| Wilcoxon rank-sum | Categorical comparisons (disease vs healthy) | `differential_analysis()` |
| Benjamini-Hochberg FDR | Multiple testing correction | `multipletests(method='fdr_bh')` |
| Logistic Regression / Random Forest | Treatment response prediction | `build_response_predictor()` |

## Validation Strategy

1. **Pilot analysis:** Validate expected biology (IL-17 in Th17, IFNγ in CD8/NK, TNF in monocytes)
2. **Cross-cohort validation:** Main → validation → external cohort generalization
3. **Output verification:** Activity z-scores in -3 to +3 range, gene overlap >80%, correlation r > 0.9

## Activity Difference (not Log2FC)

**Fixed (2026-01-31):** Activity values are z-scores (can be negative), so we use simple difference, not log2 fold-change.

### Calculation

```python
# activity_diff = group1_mean - group2_mean
activity_diff = mean_a - mean_b

# Example: exhausted=-2, non-exhausted=-4
# activity_diff = -2 - (-4) = +2 (correctly indicates higher in exhausted)
```

### Field Name

All differential analyses use `activity_diff` field (renamed from `log2fc`):

| Analysis | Scripts |
|----------|---------|
| Disease vs healthy | `02_inflam_activity.py` |
| Responder vs non-responder | `02_inflam_activity.py` |
| Tumor vs adjacent | `03_scatlas_analysis.py` |
| Cancer vs normal | `03_scatlas_analysis.py` |
| Exhausted vs non-exhausted | `03_scatlas_analysis.py`, `07_scatlas_immune_analysis.py` |
| Sex/smoking differential | `06_preprocess_viz_data.py` |

### Visualization Labels

UI labels show "Δ Activity" to reflect the calculation (difference, not ratio).

## CytoAtlas REST API (188+ endpoints)

```bash
cd cytoatlas-api
pip install -e .
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Key Routers

| Router | Endpoints | Description |
|--------|-----------|-------------|
| CIMA | ~32 | Age/BMI correlations, biochemistry, metabolites, eQTL |
| Inflammation | ~44 | Disease activity, treatment response, cohort validation |
| scAtlas | ~36 | Organ signatures, cancer comparison, immune infiltration |
| Cross-Atlas | ~28 | Atlas comparison, conserved signatures |
| Validation | ~12 | 5-type credibility assessment |
| Search | ~4 | Global search |
| Chat | ~4 | Claude AI assistant |
| Submit | ~4 | Dataset submission |

### Current Status (2026-01-31)

- **API Backend**: 95% complete (14 routers, all services implemented)
- **Frontend SPA**: 90% complete (8 pages)
- **Validation Data**: Complete (generated for all 3 atlases)
- **User Auth**: Scaffolding only
- **Dataset Submission**: Scaffolding only

See `cytoatlas-api/ARCHITECTURE.md` for detailed API documentation.

## Master Plan

For comprehensive project status and implementation details, see:
`/home/parks34/.claude/plans/cytoatlas-master-plan.md`

### Critical TODOs

1. ~~**Generate Validation JSON Data**~~ ✅ Complete (2026-01-31)
   - Generated: `visualization/data/validation/*.json` for all 3 atlases

2. **Production Hardening** (Priority 1)
   - JWT authentication
   - Prometheus metrics
   - Load testing

3. **Dataset Submission** (Priority 2)
   - Chunked file upload
   - Celery background processing

## Git Configuration

```bash
git config user.email "seongyong.park@nih.gov"
git config user.name "Seongyong Park"
```

## Lessons Learned

> **Self-updating section**: When Claude makes a mistake or learns something project-specific, add it here to prevent repeating errors.

### Data Handling
- Activity values are z-scores (can be negative) → use `activity_diff` not `log2fc`
- Gene mapping: CytoSig names (e.g., `TNFA`) differ from HGNC symbols (e.g., `TNF`) - always check `signature_gene_mapping.json`
- JSON files with `*_complete.json` suffix are duplicates - delete them

### API Development
- Always test endpoints with real data paths, not mocks
- Use `get_signature_names()` helper for bidirectional gene name lookup
- Pydantic v2 syntax: use `field_validator` not `validator`

### Frontend
- Check `docs/EMBEDDED_DATA_CHECKLIST.md` before adding new JSON files
- Use "Δ Activity" label (not "Log2FC") in UI for differential displays

## Workflow Tips

### Starting a Session
```bash
# Quick context refresh
claude -c  # Continue last session
claude -r  # Resume specific session
```

### Before Complex Tasks
1. Use `/plan` mode for multi-step implementations
2. Break large tasks into smaller units (A→A1→A2→A3 not A→B directly)
3. For parallel work, consider git worktrees

### Context Management
- Use `/compact` proactively before auto-compaction kicks in
- Create `/handoff` documents before ending long sessions
- Fresh conversations work better for unrelated topics
