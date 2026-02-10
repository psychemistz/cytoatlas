# CLAUDE.md

This file provides guidance to Claude Code when working with this repository, structured around Domain-Driven Design (DDD) principles with bounded contexts aligned to the system's core domains.

## Critical Instructions

**Git commits:** Do not include "Co-Authored-By" lines in commit messages.

**Sensitive data:** Never commit personally sensitive information to git. This includes API keys (Claude/Anthropic, OpenAI, etc.), passwords, tokens, and credentials. Keep such values in `.env` files that are gitignored or use environment variables.

**Data handling:** Always ask the user for data paths. Never use mock data for validation. When implementing services, sample from real datasets for development.

---

## Ubiquitous Language

Consistent terminology used across all bounded contexts:

| Term | Definition |
|------|-----------|
| **Activity** | Z-score from ridge regression of gene expression against signature matrix; can be negative |
| **Activity Difference** | Simple difference (`mean_a - mean_b`) between group means; NOT log2 fold-change (field: `activity_diff`) |
| **Atlas** | A single-cell RNA-seq dataset collection (CIMA, Inflammation, scAtlas) |
| **CytoSig** | 44-cytokine signature matrix (genes x cytokines) |
| **SecAct** | 1,249-secreted-protein signature matrix (genes x proteins) |
| **Pseudobulk** | Aggregated expression profile (cell type x sample) — primary analysis level |
| **Donor-level** | Cross-sample validation at the donor/subject granularity |
| **Celltype-level** | Cross-sample validation stratified by cell type |
| **Resampled** | Bootstrap-resampled pseudobulk aggregation for confidence intervals |
| **Signature** | A column in CytoSig or SecAct; represents a cytokine or secreted protein target |
| **Target** | Synonym for signature in the validation/correlation context |
| **Hot/Warm/Cold** | Tiered caching: in-memory (1h TTL) / JSON files (persistent) / CSV archives (persistent) |

---

## Domain Overview

Pan-Disease Single-Cell Cytokine Activity Atlas — computes cytokine and secreted protein activity signatures across 12+ million human immune cells from three major single-cell atlases (CIMA, Inflammation Atlas, scAtlas) to identify disease-specific and conserved signaling patterns.

### Bounded Contexts

```
┌─────────────────────────────────────────────────────────────────┐
│                     CytoAtlas System                            │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌───────────────────────┐ │
│  │   Science    │  │   Pipeline   │  │      API Gateway       │ │
│  │   Domain     │──│   Domain     │──│       Domain           │ │
│  │              │  │              │  │                         │ │
│  │ • Analysis   │  │ • Ingest     │  │ • REST Endpoints       │ │
│  │ • Statistics │  │ • Process    │  │ • Repository Layer     │ │
│  │ • Validation │  │ • Export     │  │ • Service Layer        │ │
│  │ • Signatures │  │ • Orchestr.  │  │ • Auth/RBAC            │ │
│  └──────────────┘  └──────────────┘  └───────────────────────┘ │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐                             │
│  │ Visualization│  │   Data       │                             │
│  │   Domain     │──│   Domain     │                             │
│  │              │  │              │                             │
│  │ • Web Portal │  │ • DuckDB     │                             │
│  │ • Charts     │  │ • JSON files │                             │
│  │ • SPA Pages  │  │ • H5AD/CSV   │                             │
│  └──────────────┘  └──────────────┘                             │
└─────────────────────────────────────────────────────────────────┘
```

---

## Context 1: Science Domain

The core analytical domain — activity inference, statistical methods, and validation.

### Signature Matrices

```python
from secactpy import load_cytosig, load_secact
cytosig = load_cytosig()  # genes × 44 cytokines
secact = load_secact()    # genes × 1,249 secreted proteins
```

Required external package: `secactpy` from `/vf/users/parks34/projects/1ridgesig/SecActpy/`

### Statistical Methods

| Method | Use Case | Function |
|--------|----------|----------|
| Spearman correlation | Continuous variables (age, BMI, metabolites) | `correlation_analysis()` |
| Wilcoxon rank-sum | Categorical comparisons (disease vs healthy) | `differential_analysis()` |
| Benjamini-Hochberg FDR | Multiple testing correction | `multipletests(method='fdr_bh')` |
| Logistic Regression / Random Forest | Treatment response prediction | `build_response_predictor()` |

### Activity Difference (not Log2FC)

Activity values are z-scores (can be negative), so we use simple difference, not log2 fold-change.

```python
activity_diff = mean_a - mean_b
# Example: exhausted=-2, non-exhausted=-4 → activity_diff = +2 (higher in exhausted)
```

All differential analyses use `activity_diff` field. UI labels show "Δ Activity".

### Validation Strategy

1. **Pilot analysis:** Validate expected biology (IL-17 in Th17, IFNγ in CD8/NK, TNF in monocytes)
2. **Cross-cohort validation:** Main → validation → external cohort generalization
3. **Output verification:** Activity z-scores in -3 to +3 range, gene overlap >80%, correlation r > 0.9

### Data Paths

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

---

## Context 2: Pipeline Domain

GPU-accelerated processing pipeline (`cytoatlas-pipeline/` package).

### Analysis Scripts (Active)

| Phase | Script | Description |
|-------|--------|-------------|
| 0 | `00_pilot_analysis.py` | Pilot validation on 100K cell subsets |
| 1 | `01_cima_activity.py` | CIMA: 6.5M cells, biochemistry/metabolomics correlations |
| 2 | `02_inflam_activity.py` | Inflammation Atlas: 6.3M cells, treatment response prediction |
| 3 | `03_scatlas_analysis.py` | scAtlas: 6.4M cells (normal organs + cancer) |
| 4 | `04_integrated.py` | Cross-atlas integration |
| 5 | `05_figures.py` | Publication figures (matplotlib) |
| 6 | `06_preprocess_viz_data.py` | Web visualization JSON preprocessing |
| 7 | `07_cross_atlas_analysis.py` | Cross-atlas comparison |
| 8 | `08_scatlas_immune_analysis.py` | scAtlas immune analysis (exhaustion, CAF, infiltration) |
| 10 | `10_atlas_validation_pipeline.py` | Multi-level atlas validation |
| 11 | `11_donor_level_pipeline.py` | Donor-level analysis pipeline |
| 14 | `14_preprocess_bulk_validation.py` | JSON preprocessing for validation visualization |
| 15 | `15_bulk_validation.py` | GTEx/TCGA bulk RNA-seq activity + correlations |
| 16 | `16_resampled_validation.py` | Bootstrap resampled activity inference + CIs |
| 17 | `17_preprocess_validation_summary.py` | Validation summary preprocessing |
| — | `convert_data_to_duckdb.py` | Convert JSON/CSV data to DuckDB |
| — | `create_data_lite.py` | Generate lite dataset for development |
| — | `build_rag_index.py` | Build RAG semantic index |

### Key Design Patterns

- GPU acceleration via CuPy (10-34x speedup) with NumPy fallback
- Pseudo-bulk aggregation (cell type x sample) as primary analysis level
- Single-cell batch processing (10K cells/batch) for detailed analysis
- Backed mode (`ad.read_h5ad(..., backed='r')`) for memory efficiency

### Pipeline Package (`cytoatlas-pipeline/`)

Fully implemented GPU-accelerated pipeline modules with 18 subpackages:

| Module | Purpose |
|--------|---------|
| `core/` | GPU manager, checkpoint, memory estimation, cache |
| `activity/` | Ridge regression activity inference |
| `aggregation/` | Pseudobulk aggregation, resampling |
| `correlation/` | Spearman, continuous, biochemistry correlations |
| `differential/` | Wilcoxon, t-test, effect size, FDR, stratified |
| `validation/` | Sample-level, celltype-level, biological validation |
| `disease/` | Disease activity, treatment response |
| `cancer/` | Exhaustion, infiltration, tumor-adjacent |
| `organ/` | Organ-specific signature analysis |
| `cross_atlas/` | Harmonization, celltype mapping, meta-analysis |
| `batch/` | Multi-level batch aggregation |
| `search/` | Search indexing |
| `ingest/` | Local H5AD, cellxgene, remote H5AD loaders |
| `export/` | JSON, CSV, H5AD, Parquet, DuckDB writers |
| `orchestration/` | Scheduler, recovery, Celery tasks |

**CLI entry point:** `cytoatlas-pipeline` (defined in `pyproject.toml`, implemented in `cli.py`)

### SLURM Job Submission (HPC)

```bash
sbatch scripts/slurm/run_all.sh           # Full pipeline
sbatch scripts/slurm/run_all.sh --pilot   # Pilot only (~2 hours)
sbatch scripts/slurm/run_all.sh --main    # Main analyses only
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

---

## Context 3: API Gateway Domain

CytoAtlas REST API — 217 endpoints across 15 routers.

### Quick Start

```bash
cd cytoatlas-api
pip install -e .
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Layered Architecture

```
┌─────────────────────────────┐
│    Routers (API Layer)      │  FastAPI endpoints, request/response schemas
├─────────────────────────────┤
│    Services (Application)   │  Business logic, caching, data transformation
├─────────────────────────────┤
│    Repositories (Domain)    │  Data access abstraction (Protocol-based)
├─────────────────────────────┤
│    Infrastructure           │  DuckDB, JSON files, SQLite, Redis
└─────────────────────────────┘
```

### Repository Pattern

Protocol-based abstraction (PEP 544) for backend swappability:

```python
class AtlasRepository(Protocol):
    async def get_activity(self, atlas, signature_type, **filters) -> list[dict]: ...
    async def get_correlations(self, atlas, variable, **filters) -> list[dict]: ...
    async def get_data(self, data_type, **filters) -> list[dict]: ...

# Active implementations
class DuckDBRepository(AtlasRepository):   # Primary — query atlas_data.duckdb
class JSONRepository(AtlasRepository):      # Fallback — load visualization/data/*.json
```

**DuckDBRepository** queries `atlas_data.duckdb` with parameterized SQL, returning pandas DataFrames. Falls back to JSON if DuckDB unavailable.

**Note:** `atlas_data.duckdb` has been generated (51 tables, 9.6M rows, 590MB). Regenerate with `python scripts/convert_data_to_duckdb.py --all` after data updates.

### Key Routers

| Router | Endpoints | Description |
|--------|-----------|-------------|
| Atlas Management | 5 | Atlas registry: list, register, delete, info, features |
| CIMA (`/atlases/cima`) | 28 | Age/BMI correlations, biochemistry, metabolites, eQTL |
| Inflammation (`/atlases/inflammation`) | 42 | Disease activity, treatment response, cohort validation |
| scAtlas (`/atlases/scatlas`) | 31 | Organ signatures, cancer comparison, immune infiltration |
| Cross-Atlas | 20 | Atlas comparison, conserved signatures |
| Validation | 28 | 5-type credibility assessment |
| Search | 6 | Global search |
| Chat | 8 | Claude AI assistant (RAG-powered) |
| Submit | 9 | Dataset submission |
| Auth | 5 | JWT + API key authentication |
| Export | 9 | Data export (CSV, JSON, HDF5) |
| Gene | 11 | Gene-centric views and analysis |
| Health | 4 | Health checks, system status |
| Pipeline | 4 | Pipeline status and management |
| WebSocket | 3 | Real-time updates |

### RBAC Model

| Role | Permissions |
|------|-------------|
| **anonymous** | Read public data, search, basic API endpoints |
| **viewer** | Read all public datasets, access dashboard |
| **researcher** | Download data, access advanced analytics |
| **data_curator** | Submit custom datasets, manage metadata |
| **admin** | System administration, user management, audit logs |

**Security Default**: Most permissive (anonymous) — explicit role checks required per endpoint.

### Testing

```bash
cd cytoatlas-api && pip install -e ".[dev]"
pytest tests/ -v                           # All tests
pytest tests/ -m "not integration"         # Unit tests only
pytest tests/ --cov=app --cov-report=html  # With coverage
```

---

## Context 4: Data Domain

Data storage, flow, and persistence architecture.

### Storage Architecture

| Layer | Medium | Purpose |
|-------|--------|---------|
| **DuckDB** | `atlas_data.duckdb` | Primary science data (activity, correlations, differential, scatter, validation) |
| **SQLite** | `app.db` | App state (users, conversations, jobs) |
| **JSON** | `visualization/data/*.json` | Fallback / web portal data (~1.5GB) |
| **H5AD/CSV** | `results/` | Raw analysis outputs, archival (~50GB) |

### Tiered Caching

```
Tier 1 (Hot):   Redis/In-memory dict   (1 hour TTL, ~100MB, 80%+ hit rate)
Tier 2 (Warm):  JSON files             (persistent, ~500MB, on-demand)
Tier 3 (Cold):  CSV files              (persistent, ~50GB, rare access)
```

### DuckDB Generation

```bash
python scripts/convert_data_to_duckdb.py --all --output atlas_data.duckdb
python scripts/convert_data_to_duckdb.py --table activity  # Individual table
```

### Output Structure

```
results/
├── pilot/                       # Pilot validation
├── cima/                        # CIMA activity results, correlations, differential
├── inflammation/                # Inflammation Atlas results, predictions
├── scatlas/                     # scAtlas organ/celltype signatures
├── integrated/                  # Cross-atlas comparisons
├── figures/                     # Publication figures
├── cross_sample_validation/     # Validation H5AD files + correlation CSVs
└── atlas_validation/            # Resampled pseudobulk H5AD (bootstrap inputs)

visualization/
├── data/                        # JSON files for web dashboard
└── index.html                   # Interactive visualization SPA

cytoatlas-pipeline/              # GPU-accelerated pipeline package
├── src/cytoatlas_pipeline/      # 18 subpackages (~18.7K lines)
├── tests/
└── pyproject.toml
```

---

## Context 5: Visualization Domain

Web portal — 8-page SPA with 40+ interactive panels.

- **Stack:** Vanilla JS, Plotly, D3.js
- **Pages:** Landing, Explore, Compare, Validate, Submit, Chat, About, Contact
- **Charts:** Line, scatter, heatmap, violin, box
- **Labels:** Use "Δ Activity" (not "Log2FC") for differential displays
- **Data checklist:** See `docs/EMBEDDED_DATA_CHECKLIST.md` before adding new JSON files

---

## Context Map: Integration Patterns

| Producer → Consumer | Integration Pattern | Mechanism |
|---------------------|---------------------|-----------|
| Science → Pipeline | Conformist | Scripts call secactpy directly |
| Pipeline → Data | Published Language | CSV/H5AD/JSON output files |
| Data → API | Repository (ACL) | DuckDBRepository / JSONRepository abstract raw storage |
| API → Visualization | Open Host Service | REST endpoints serve JSON to SPA |
| Pipeline → API | Async Event | Celery tasks notify pipeline status |

---

## Development Environment

```bash
source ~/bin/myconda
conda activate secactpy
```

### Git Configuration

```bash
git config user.email "seongyong.park@nih.gov"
git config user.name "Seongyong Park"
```

---

## Security Aggregate

### Checklist

- [x] Secrets in `.env` (gitignored)
- [x] JWT tokens: RFC 7519, 30-min expiration
- [x] RBAC: 5-role model with explicit checks
- [x] Audit logging: JSONL (user, IP, endpoint, action)
- [x] Rate limiting: per-user and per-IP
- [x] Prompt injection defense: RAG-grounded chat
- [x] Security headers: CORS, CSP scaffolding
- [x] Password hashing: bcrypt
- [x] API key rotation per user

### Deployment Security

1. `export SECRET_KEY=$(openssl rand -hex 32)`
2. `export ENVIRONMENT=production`
3. `export DEBUG=false`
4. `export REQUIRE_AUTH=true`
5. Monitor `logs/audit.jsonl`

### Audit Logging

All data access logged to JSONL: `{timestamp, user_id, email, ip_address, method, endpoint, status, dataset, action}`. Retention: 90 days in DB, 30 days in files.

---

## Documentation

| Document | Description |
|----------|-------------|
| [docs/README.md](docs/README.md) | Master index |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | System design (14 sections) |
| [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) | HPC/SLURM setup, environment variables |
| [docs/API_REFERENCE.md](docs/API_REFERENCE.md) | 217 endpoints with curl examples |
| [docs/USER_GUIDE.md](docs/USER_GUIDE.md) | How to use CytoAtlas |
| [docs/ATLAS_VALIDATION.md](docs/ATLAS_VALIDATION.md) | Validation methodology |
| [docs/CELL_TYPE_MAPPING.md](docs/CELL_TYPE_MAPPING.md) | Cell type harmonization |
| [docs/EMBEDDED_DATA_CHECKLIST.md](docs/EMBEDDED_DATA_CHECKLIST.md) | JSON files for embedded_data.js |
| [docs/datasets/](docs/datasets/) | Dataset specifications |
| [docs/pipelines/](docs/pipelines/) | Pipeline documentation |
| [docs/decisions/](docs/decisions/) | Architecture Decision Records |

---

## Current Status

### Completed

- All 7 analysis pipelines (pilot, CIMA, Inflammation, scAtlas, integrated, figures, immune)
- 209 API endpoints across 15 routers (100% functional)
- 8-page SPA with 40+ visualization panels
- Pipeline package: 18 subpackages, ~18.7K lines implemented
- Validation: standard + resampled + single-cell + bulk RNA-seq
- Security: JWT, RBAC, audit logging, rate limiting
- Maintenance infrastructure: `scripts/maintenance/audit_clutter.py`, equivalence test harness
- DuckDB migration: `atlas_data.duckdb` generated (51 tables, 9.6M rows, 590MB)
- DuckDB API integration: `_KNOWN_TABLES` / `_JSON_TO_TABLE` synced, services route through `load_json()`
- Bulk validation split files: `donor_scatter/`, `celltype_scatter/`, `bulk_rnaseq/`, `bulk_donor_meta.json`
- JSONRepository deprecation warning active (directs to DuckDBRepository)

### In Progress

- [ ] DuckDB test coverage: unit tests for DuckDBRepository, JSON→DuckDB equivalence tests
- [ ] Resampled bootstrap: Inflammation main/val/ext (pseudobulk exists, run `16_resampled_validation.py`)
- [ ] Pipeline CLI entry point (`cytoatlas-pipeline` command)
- [ ] Script-to-pipeline equivalence tests (harness ready in `cytoatlas-pipeline/tests/equivalence/`)
- [ ] SLURM wrapper consolidation (3 layers → 1 parameterized template)

### Future Work

- NicheFormer spatial transcriptomics integration (~30M cells)
- scGPT cohort integration (~35M cells)
- cellxgene Census cohort integration
- AlphaGenome eQTL analysis (relocated to `/data/parks34/projects/4germicb/data/alphagenome_eqtl/cytoatlas_outputs/`)

### Archive

Retired scripts, agents, docs, and legacy code in `archive/` — see `archive/README.md` for index.

---

## Routine Maintenance

Run `python scripts/maintenance/audit_clutter.py --report` after major updates or every ~50 commits. See `scripts/maintenance/README.md` for the full checklist.

**Quick check:**
```bash
python scripts/maintenance/audit_clutter.py --report
```

**Equivalence tests:**
```bash
pytest cytoatlas-pipeline/tests/equivalence/ -v --tb=short
```

---

## Lessons Learned

> **Self-updating section**: Add project-specific lessons here to prevent repeating errors.

### Data Handling
- Activity values are z-scores (can be negative) → use `activity_diff` not `log2fc`
- Gene mapping: CytoSig names (e.g., `TNFA`) differ from HGNC symbols (e.g., `TNF`) — check `signature_gene_mapping.json`
- Large JSON files (>500MB) bottleneck memory → DuckDB migration addresses this

### API Development
- Always test endpoints with real data paths, not mocks
- Use `get_signature_names()` helper for bidirectional gene name lookup
- Pydantic v2 syntax: use `field_validator` not `validator`
- Repository pattern improves testability: plan abstract interfaces before implementation
- In-memory cache with Redis fallback works well on HPC (no external dependencies required)

### Architecture
- Security defaults to most permissive (anonymous) → explicit role checks required
- Audit logging must include context (user, IP, timestamp, dataset) for compliance
- Tiered caching (hot/warm/cold) is essential for HPC environments
- DDD bounded contexts prevent cross-domain coupling

## Workflow Tips

### Starting a Session
```bash
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
