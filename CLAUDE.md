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
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | **MAIN**: Complete system architecture (component inventory, data flow, technology stack, API design, DDD roadmap) |
| [docs/OVERVIEW.md](docs/OVERVIEW.md) | Quick project overview (redirect to ARCHITECTURE.md) |
| [docs/datasets/](docs/datasets/) | Dataset specifications (CIMA, Inflammation, scAtlas) |
| [docs/pipelines/](docs/pipelines/) | Analysis pipeline documentation |
| [docs/outputs/](docs/outputs/) | Output file catalog and API mapping |
| [docs/decisions/](docs/decisions/) | Architecture Decision Records (ADRs): Parquet, repository pattern, RBAC |
| [docs/archive/](docs/archive/) | Archived plans from earlier project phases |
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

## Data Layer Architecture

### Data Storage Layers (Tiered Caching)

| Tier | Name | Medium | TTL | Size | Use Case |
|------|------|--------|-----|------|----------|
| 1 | Hot Data | Redis or in-memory dict | 1 hour | ~100MB | Frequently accessed correlations, disease activity |
| 2 | Warm Data | JSON files (visualization/data/) | Persistent | ~500MB | Pre-computed results, on-demand loading |
| 3 | Cold Data | CSV files (results/*/*.csv) | Persistent | ~50GB | Raw analysis outputs, validation, regeneration |

### Repository Pattern (Planned for Round 3)

Protocol-based abstraction for testability and backend swappability:

```python
class DataRepository(Protocol):
    async def get_correlations(self, gene: str) -> CorrelationData: ...
    async def get_disease_activity(self, disease: str) -> ActivityData: ...

# Implementations
class JSONRepository(DataRepository):      # Current
    # Load from visualization/data/*.json

class ParquetRepository(DataRepository):   # Future
    # Load from results/*.parquet (faster for large files)

class PostgreSQLRepository(DataRepository): # Future
    # Query from database tables
```

### Parquet Backend (Planned for Round 3)

Current limitation: JSON files up to 1.8GB loaded entirely into memory. Future improvement:
- Store large files (validation_cima.json, validation_inflammation.json) as Parquet
- Use PyArrow predicate pushdown for efficient filtering
- Reduce memory footprint by 60-80%
- Maintain backward compatibility through repository abstraction

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

### Current Status (2026-02-09, Round 1 Complete)

**Analysis & Data Generation**
- ✅ All 7 analysis pipelines complete (pilot, CIMA, Inflammation, scAtlas, integrated, figures, immune)
- ✅ 30+ JSON visualization files generated (~500MB)
- ✅ Validation data complete for all 3 atlases (~175-336MB each)

**API Backend**
- ✅ 188+ endpoints across 14 routers (100% functional)
- ✅ All 12 services implemented (JSON loading, caching, filtering)
- ✅ Validation service (5-type credibility assessment, 636 lines)
- ✅ Chat service (Claude API integration)
- ⏳ Database models created (PostgreSQL integration not yet active)
- ⏳ Rate limiting scaffolding
- ✅ In-memory cache with Redis fallback

**Web Portal**
- ✅ 8-page SPA (Landing, Explore, Compare, Validate, Submit, Chat, About, Contact)
- ✅ 40+ interactive visualization panels (Plotly, D3.js)
- ✅ Responsive design (mobile, tablet, desktop)

**Documentation (Round 1)**
- ✅ Comprehensive ARCHITECTURE.md created (14 sections, DDD roadmap)
- ✅ OVERVIEW.md refactored (brief overview, redirects to ARCHITECTURE)
- ✅ Archive system created (docs/archive/ for stale plans)
- ⏳ ADRs in progress (docs/decisions/)

**Security & Hardening (Planned for Round 2)**
- ⏳ Full JWT authentication
- ⏳ RBAC model (5 roles: anonymous, viewer, researcher, data_curator, admin)
- ⏳ Audit logging (all data access)
- ⏳ Rate limiting enforcement

See `docs/ARCHITECTURE.md` for detailed system documentation.

## Security Model (Planned for Round 2)

### Role-Based Access Control (RBAC)

Five-role model to be implemented:

| Role | Permissions | Use Case |
|------|-------------|----------|
| **anonymous** | Read public data, search, basic API endpoints | Unauthenticated public access |
| **viewer** | Read all public datasets, access dashboard | Registered users |
| **researcher** | Download data, access advanced analytics | Academic researchers |
| **data_curator** | Submit custom datasets, manage metadata | Dataset maintainers |
| **admin** | System administration, user management, audit logs | System operators |

### Default Security Posture

- API endpoints default to **anonymous** role (most permissive)
- Sensitive operations (data export, submission) require **researcher+** role
- System administration requires **admin** role
- All role assignments logged to audit trail

### Audit Logging

Will be implemented in Round 2:
- All data access (user, timestamp, IP, endpoint, dataset)
- Sensitive operations (downloads, exports, submissions)
- Authentication events (login, logout, token refresh)
- Administrative actions (role changes, dataset registration)
- Retention: 90 days in PostgreSQL, 30 days in logs

## Master Plan

For comprehensive project status and implementation details, see:
`/home/parks34/.claude/plans/cytoatlas-master-plan.md`

### Critical TODOs (by Priority)

#### Round 1: Documentation Cleanup ✅ Complete (2026-02-09)
- ✅ Archive stale plans (docs/archive/)
- ✅ Create ARCHITECTURE.md (comprehensive, 14 sections)
- ✅ Update CLAUDE.md (current status, data layer, security)
- ✅ Create docs/decisions/ with ADRs (Parquet, repository, RBAC)
- ✅ Clean docs/ directory (consolidate overlapping files)
- ✅ Auto-doc generation script (optional)

#### Round 2: Security Hardening (Priority 1)
- [ ] Full JWT authentication (RFC 7519 compliant)
- [ ] RBAC enforcement (5-role model)
- [ ] Audit logging (database + file)
- [ ] Rate limiting (per user/IP)
- [ ] Prometheus metrics (response time, error rate, cache hit)
- [ ] OAuth providers (Google, ORCID)
- [ ] Load testing (k6 or Locust)

#### Round 3: Data Layer Migration (Priority 2)
- [ ] Repository pattern implementation (protocol-based abstraction)
- [ ] Parquet backend for large files (validation_*.json)
- [ ] PostgreSQL backend option (for persistence)
- [ ] Query optimization (predicate pushdown, column selection)

#### Round 4: Extensibility & Scaling (Priority 3)
- [ ] User-submitted datasets (CELEX-style workflow)
- [ ] Chunked file upload (multipart/form-data)
- [ ] Celery background processing (async activity inference)
- [ ] API versioning (v1, v2 support)
- [ ] GraphQL option (complementary to REST)
- [ ] External data integration (cellxgene, GEO)

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
- Large JSON files (>500MB) will become bottleneck in memory → plan Parquet migration early

### API Development
- Always test endpoints with real data paths, not mocks
- Use `get_signature_names()` helper for bidirectional gene name lookup
- Pydantic v2 syntax: use `field_validator` not `validator`
- Repository pattern improves testability: plan abstract interfaces before implementation
- In-memory cache with Redis fallback works well on HPC (no external dependencies required)

### Frontend
- Check `docs/EMBEDDED_DATA_CHECKLIST.md` before adding new JSON files
- Use "Δ Activity" label (not "Log2FC") in UI for differential displays
- Vanilla JS SPA works well for moderate complexity; consider framework if >50 pages

### Architecture & Security
- Security defaults to most permissive (anonymous) → explicit role checks required
- Audit logging must include context (user, IP, timestamp, dataset) for compliance
- Documentation-first approach (ARCHITECTURE.md) prevents architectural drift
- ADRs enable team alignment on design trade-offs (Parquet vs. JSON, repository pattern, RBAC)
- Tiered caching strategy (hot/warm/cold) is essential for HPC environments

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

### Context Management
- Use `/compact` proactively before auto-compaction kicks in
- Create `/handoff` documents before ending long sessions
- Fresh conversations work better for unrelated topics
