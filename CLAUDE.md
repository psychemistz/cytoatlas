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
| [docs/README.md](docs/README.md) | **Master index** - Start here for all documentation |
| [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) | **Deployment Guide** - HPC/SLURM setup, environment variables, troubleshooting |
| [docs/API_REFERENCE.md](docs/API_REFERENCE.md) | **API Reference** - 188+ endpoints grouped by domain, curl examples |
| [docs/USER_GUIDE.md](docs/USER_GUIDE.md) | **User Guide** - How to use CytoAtlas (atlases, chat, exports, comparisons) |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | **Architecture** - System design (14 sections: overview, components, data flow, tech stack, DDD) |
| [docs/OVERVIEW.md](docs/OVERVIEW.md) | Quick overview (redirects to ARCHITECTURE.md) |
| [docs/ATLAS_VALIDATION.md](docs/ATLAS_VALIDATION.md) | Validation methodology and results |
| [docs/CELL_TYPE_MAPPING.md](docs/CELL_TYPE_MAPPING.md) | Cell type harmonization across atlases |
| [docs/QA_LOG.md](docs/QA_LOG.md) | Quality assurance log |
| [docs/datasets/](docs/datasets/) | Dataset specifications (CIMA, Inflammation, scAtlas) |
| [docs/pipelines/](docs/pipelines/) | Analysis pipeline documentation (incl. AlphaGenome eQTL plan) |
| [docs/outputs/](docs/outputs/) | Output file catalog and API mapping |
| [docs/decisions/](docs/decisions/) | Architecture Decision Records (ADRs): Parquet, repository pattern, RBAC |
| [docs/templates/](docs/templates/) | Document templates |
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
| 7 | `07_scatlas_immune_analysis.py` | scAtlas immune analysis (exhaustion, CAF, infiltration) |
| 12 | `12_pseudobulk_activity.py` | Pseudobulk aggregation + activity inference (all SC atlases) |
| 13 | `13_cross_sample_correlation_analysis.py` | Spearman correlation analysis per target |
| 14 | `14_preprocess_bulk_validation.py` | JSON + Parquet preprocessing for validation visualization |
| 15 | `15_bulk_rnaseq_validation.py` | GTEx/TCGA bulk RNA-seq activity + correlations |
| 16 | `16_resampled_validation.py` | Bootstrap resampled activity inference + CIs |
| — | `convert_json_to_parquet.py` | Convert large JSON files to Parquet format |
| — | `create_data_lite.py` | Generate lite dataset for development |

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
├── pilot/                       # Pilot validation
├── cima/                        # CIMA activity results, correlations, differential
├── inflammation/                # Inflammation Atlas results, predictions
├── scatlas/                     # scAtlas organ/celltype signatures
├── integrated/                  # Cross-atlas comparisons
├── figures/                     # Publication figures
├── cross_sample_validation/     # Validation H5AD files + correlation CSVs
│   ├── cima/                    # 20 H5AD + 12 resampled
│   ├── inflammation_{main,val,ext}/  # 12 H5AD each
│   ├── scatlas_{normal,cancer}/  # 12 H5AD + 6 resampled each
│   ├── gtex/                    # 9 H5AD
│   ├── tcga/                    # 8 H5AD
│   └── correlations/            # 18 CSV files (standard + resampled + summary)
└── atlas_validation/            # Resampled pseudobulk H5AD (bootstrap inputs)

visualization/
├── data/                        # JSON files for web dashboard (~1.5GB)
│   └── parquet/                 # Parquet files (generated by convert_json_to_parquet.py)
└── index.html                   # Interactive visualization SPA

cytoatlas-pipeline/              # GPU-accelerated pipeline package (scaffolded)
├── src/cytoatlas_pipeline/
├── tests/
└── pyproject.toml
```

## Data Layer Architecture

### Data Storage Layers (Tiered Caching)

| Tier | Name | Medium | TTL | Size | Use Case |
|------|------|--------|-----|------|----------|
| 1 | Hot Data | Redis or in-memory dict | 1 hour | ~100MB | Frequently accessed correlations, disease activity |
| 2 | Warm Data | JSON files (visualization/data/) | Persistent | ~500MB | Pre-computed results, on-demand loading |
| 3 | Cold Data | CSV files (results/*/*.csv) | Persistent | ~50GB | Raw analysis outputs, validation, regeneration |

### Repository Pattern (Implemented)

Protocol-based abstraction for testability and backend swappability:

```python
class DataRepository(Protocol):
    async def get_correlations(self, gene: str) -> CorrelationData: ...
    async def get_disease_activity(self, disease: str) -> ActivityData: ...

# Implementations
class JSONRepository(DataRepository):      # Active — primary backend
    # Load from visualization/data/*.json

class ParquetRepository(DataRepository):   # Ready — flat layout, JSON fallback
    # Load from visualization/data/parquet/{name}.parquet

class PostgreSQLRepository(DataRepository): # Scaffolded — models created
    # Query from database tables
```

### Parquet Backend (Infrastructure Ready)

Conversion script and repository are implemented. Run to generate files:
```bash
python scripts/convert_json_to_parquet.py --all           # snappy (fast reads)
python scripts/convert_json_to_parquet.py --all --compression zstd  # max compression
```
- Targets: activity_boxplot (392MB), inflammation_disease (275MB), age_bmi_boxplots (234MB), singlecell_activity (155MB), scatlas_celltypes (126MB), inflammation_disease_filtered (56MB)
- Handles nested JSON: `extract_key` for dict-wrapped, `flatten_func` for deeply nested
- bulk_donor_correlations.json (5.5GB) handled separately via scatter metadata Parquet in `14_preprocess_bulk_validation.py`
- Output: `visualization/data/parquet/{name}.parquet`

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

### Current Status (2026-02-09, Round 1-3 Complete)

**Analysis & Data Generation**
- ✅ All 7 analysis pipelines complete (pilot, CIMA, Inflammation, scAtlas, integrated, figures, immune)
- ✅ 30+ JSON visualization files generated (~500MB)
- ✅ Validation data complete for all 3 atlases (~175-336MB each)
- ✅ TCGA bulk RNA-seq validation complete (8 H5AD, correlations CSV)
- ✅ GTEx bulk RNA-seq validation complete (by_tissue + donor_only)
- ✅ bulk_rnaseq_validation.json generated (91 MB)
- ✅ Resampled bootstrap validation: CIMA (4 levels), scAtlas normal/cancer (2 levels each)
- ✅ 18 correlation CSVs in results/cross_sample_validation/correlations/

**Validation Matrix Status**
- ✅ Standard cross-sample correlations: All 8 sources complete (CIMA, Inflammation x3, scAtlas x2, GTEx, TCGA)
- ✅ LinCytoSig cell-type-restricted correlations: All 8 sources complete
- ✅ Single-cell validation JSONs: All 3 atlases (662 MB total)
- ✅ Resampled bootstrap: CIMA (L1-L4), scAtlas normal (celltype, donor_celltype), scAtlas cancer (celltype, donor_celltype)
- ⬜ Resampled bootstrap: Inflammation main/val/ext (resampled pseudobulk exists, activity inference not run)

**API Backend**
- ✅ 188+ endpoints across 14 routers (100% functional)
- ✅ All 12 services implemented (JSON loading, caching, filtering)
- ✅ Repository pattern framework (abstraction layer, protocol-based)
- ✅ Tiered caching (hot/warm/cold data layers)
- ✅ Validation service (5-type credibility assessment, 636 lines)
- ✅ Chat service (RAG-powered Claude integration)
- ✅ Pipeline management router (dependency graph, orchestration)
- ✅ Database models created (PostgreSQL integration scaffolding)
- ✅ Rate limiting scaffolding
- ✅ In-memory cache with Redis fallback
- ✅ Security headers and middleware
- ✅ Audit logging framework

**Parquet Infrastructure**
- ✅ ParquetRepository implemented (flat layout: `visualization/data/parquet/{name}.parquet`)
- ✅ `parquet_data_path` config setting added
- ✅ Conversion script fixed (`scripts/convert_json_to_parquet.py`) — handles flat arrays, extract_key, nested flatten
- ✅ Scatter metadata Parquet output added to `14_preprocess_bulk_validation.py`
- ⬜ Actual Parquet files not yet generated (run `python scripts/convert_json_to_parquet.py --all`)

**Web Portal**
- ✅ 8-page SPA (Landing, Explore, Compare, Validate, Submit, Chat, About, Contact)
- ✅ 40+ interactive visualization panels (Plotly, D3.js)
- ✅ Responsive design (mobile, tablet, desktop)
- ✅ Chart components (line, scatter, heatmap, violin, box)
- ✅ State management and data loader
- ✅ Single-cell validation: "Hide non-expressing" checkbox (matches other tabs)

**Documentation (Round 3)**
- ✅ CLAUDE.md updated (current status, security model, architecture patterns)
- ✅ DEPLOYMENT.md created (HPC/SLURM, development setup, environment variables)
- ✅ API_REFERENCE.md created (14 router groups, 188+ endpoints with examples)
- ✅ USER_GUIDE.md created (atlas overview, chat interface, exports)
- ✅ ARCHITECTURE.md updated (chat system, frontend, pipeline management, data layer)
- ✅ docs/README.md consolidated (master index with links)
- ✅ ATLAS_VALIDATION.md, CELL_TYPE_MAPPING.md, QA_LOG.md

**Security & Hardening (Round 2-3)**
- ✅ JWT authentication (RFC 7519 compliant)
- ✅ RBAC model (5 roles: anonymous, viewer, researcher, data_curator, admin)
- ✅ Audit logging framework (audit_log_path in config)
- ✅ Rate limiting (per user/IP)
- ✅ Prompt injection defense (RAG-powered chat)
- ✅ Security headers (CORS, CSP, HSTS scaffolding)

**GPU Pipeline Package**
- ✅ `cytoatlas-pipeline/` package scaffolded (src, tests, config, pyproject.toml)
- ⬜ Module implementations not started (core, ingest, aggregation, activity, etc.)

See `docs/ARCHITECTURE.md` for detailed system documentation and `docs/DEPLOYMENT.md` for deployment guide.

## Architecture Patterns (Rounds 2-3)

### Repository Pattern (Implemented)

Data abstraction layer for testability and backend swappability:

```python
# Protocol-based abstraction (runtime-checkable)
class DataRepository(Protocol):
    async def get_correlations(self, gene: str) -> CorrelationData: ...
    async def get_disease_activity(self, disease: str) -> ActivityData: ...

# Implementations
class JSONRepository(DataRepository):      # Active — primary backend
class ParquetRepository(DataRepository):   # Ready — flat layout, auto JSON fallback
class PostgreSQLRepository(DataRepository): # Scaffolded — models created
```

**ParquetRepository** uses flat directory `visualization/data/parquet/{data_type}.parquet`, with automatic fallback to JSONRepository when a Parquet file doesn't exist.

### Tiered Caching Strategy

Three-tier architecture for memory and performance optimization:

```
Tier 1 (Hot):   Redis/In-memory dict   (1 hour TTL, ~100MB, 80%+ hit rate)
Tier 2 (Warm):  JSON files             (persistent, ~500MB, on-demand)
Tier 3 (Cold):  CSV files              (persistent, ~50GB, rare access)
```

**Hit rate target**: >80% for frequently accessed correlations/disease_activity

### RBAC Model (Implemented)

Five-role model with explicit permission checks:

| Role | Permissions |
|------|-------------|
| **anonymous** | Read public data, search, basic API endpoints |
| **viewer** | Read all public datasets, access dashboard |
| **researcher** | Download data, access advanced analytics |
| **data_curator** | Submit custom datasets, manage metadata |
| **admin** | System administration, user management, audit logs |

**Security Default**: Most permissive (anonymous) - explicit role checks required per endpoint.

### Audit Logging (Implemented)

All data access logged to JSONL file:

```
{
  "timestamp": "2026-02-09T10:30:45.123Z",
  "user_id": 42,
  "email": "user@example.com",
  "ip_address": "192.0.2.1",
  "method": "GET",
  "endpoint": "/api/v1/cima/correlations",
  "status": 200,
  "dataset": "cima_correlations",
  "action": "read"
}
```

**Config**: `audit_log_path` in config.py, TTL 90 days in DB, 30 days in files.

## Testing (Round 3)

### Running Tests

```bash
# Install test dependencies
cd cytoatlas-api
pip install -e ".[dev]"

# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_routers/test_cima.py -v

# Run with coverage
pytest tests/ --cov=app --cov-report=html

# Run integration tests (requires data files)
pytest tests/ -m integration

# Run only unit tests (no data dependencies)
pytest tests/ -m "not integration"
```

### Test Structure

```
tests/
├── test_routers/          # API endpoint tests
│  ├── test_cima.py
│  ├── test_inflammation.py
│  ├── test_scatlas.py
│  └── test_chat.py
├── test_services/         # Service business logic
│  ├── test_cima_service.py
│  └── test_validation_service.py
├── test_schemas/          # Pydantic model validation
├── conftest.py            # Shared fixtures
└── fixtures/              # Mock data, test databases
```

### Key Test Fixtures

- `mock_json_service`: Mock JSON data loading
- `test_db`: SQLAlchemy test database
- `test_cache`: In-memory test cache
- `test_client`: FastAPI TestClient

## Security (Round 2-3)

### Security Checklist

- [x] **Secrets Management**: All sensitive values in `.env` (not committed)
- [x] **JWT Tokens**: RFC 7519 compliant, 30-minute default expiration
- [x] **RBAC Enforcement**: 5-role model with explicit permission checks
- [x] **Audit Logging**: JSONL file with context (user, IP, endpoint, action)
- [x] **Rate Limiting**: Per-user and per-IP throttling
- [x] **Prompt Injection Defense**: RAG-powered chat validates all LLM inputs
- [x] **Security Headers**: CORS, CSP scaffolding
- [x] **Password Hashing**: Bcrypt with configurable rounds
- [x] **API Key Rotation**: Per-user API key generation and revocation

### Deployment Security

1. **Set SECRET_KEY**: `export SECRET_KEY=$(openssl rand -hex 32)`
2. **Use environment**: `export ENVIRONMENT=production` in HPC jobs
3. **Disable debug mode**: `export DEBUG=false`
4. **Enable auth**: `export REQUIRE_AUTH=true` for sensitive deployments
5. **Monitor logs**: Check `logs/audit.jsonl` for suspicious activity

### Prompt Injection Prevention (Chat)

All chat inputs go through RAG validation:
1. Query embeddings computed (all-MiniLM-L6-v2)
2. Top-K documents retrieved from semantic DB
3. Response grounded in retrieved context
4. System prompt enforces CytoAtlas-specific responses

## Security Model (Implemented)

### Role-Based Access Control (RBAC)

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

### Audit Logging (Implemented)

- All data access (user, timestamp, IP, endpoint, dataset)
- Sensitive operations (downloads, exports, submissions)
- Authentication events (login, logout, token refresh)
- Administrative actions (role changes, dataset registration)
- Retention: 90 days in PostgreSQL, 30 days in logs

## Plans & TODOs

### Plans (in `~/.claude/plans/`)

| Plan | File | Status |
|------|------|--------|
| GPU Pipeline Package | `precious-enchanting-sphinx.md` | Scaffolded (`cytoatlas-pipeline/`), modules not started |
| Compare Menu (SPA) | `witty-crunching-book.md` | Implemented in `visualization/index.html` (5 cross-atlas tabs) |
| Validate Menu (SPA) | `cytoatlas-validation-plan.md` | Implemented in `visualization/index.html` (4 validation tabs) |
| Validation Matrix | `validation-matrix-plan.md` | Mostly complete (see status below) |
| AlphaGenome eQTL | `docs/pipelines/alphagenome_plan.md` | Documented, not started |

### Completed Rounds

#### Round 1: Documentation Cleanup ✅ Complete
#### Round 2: Security Hardening ✅ Complete
#### Round 3: Data Layer Migration & Documentation ✅ Complete

### Round 3.5: Validation & Parquet (In Progress)

**Completed:**
- [x] TCGA bulk RNA-seq full pipeline (activity + correlations + JSON)
- [x] scAtlas cancer regeneration (zscore bug fix, all levels)
- [x] Standard cross-sample correlations: all 8 sources × all levels × 3 sigs
- [x] Resampled bootstrap: CIMA L1-L4, scAtlas normal/cancer celltype + donor_celltype
- [x] Parquet infrastructure (conversion script, repository, config)
- [x] Single-cell "Hide non-expressing" checkbox (consistent across all tabs)
- [x] Scatter metadata Parquet output in preprocessing script

**Remaining:**
- [ ] Resampled bootstrap: Inflammation main/val/ext (pseudobulk exists, run `16_resampled_validation.py`)
- [ ] Generate Parquet files (`python scripts/convert_json_to_parquet.py --all`)
- [ ] Regenerate bulk_donor_correlations.json with Parquet metadata (full HPC rebuild)

#### Round 4: Extensibility & Scaling (Future)
- [ ] GPU pipeline module implementations (`cytoatlas-pipeline/src/`)
- [ ] User-submitted datasets (CELEX-style workflow)
- [ ] Chunked file upload (multipart/form-data)
- [ ] Celery background processing (async activity inference)
- [ ] API versioning (v1, v2 support alongside v1)
- [ ] External data integration (cellxgene, GEO)
- [ ] AlphaGenome eQTL analysis

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
