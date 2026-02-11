# Project Status

**Last updated:** 2026-02-10

Pan-Disease Single-Cell Cytokine Activity Atlas — computes cytokine and secreted protein activity signatures across 240M+ human cells from six datasets to identify disease-specific and conserved signaling patterns, validate signatures with perturbation ground truth, and map drug-induced pathway changes.

---

## System Summary

| Codebase | Location | Files | LOC |
|----------|----------|-------|-----|
| Analysis scripts | `scripts/` | 28 | 28,985 |
| Pipeline package | `cytoatlas-pipeline/` | ~150 | 22,248 |
| API backend | `cytoatlas-api/app/` | 58 | 25,819 |
| React frontend | `cytoatlas-api/frontend/` | 133 | 11,929 |
| Documentation | `docs/` | 42 | 8,506 |
| **Total** | | **~411** | **~97,500** |

---

## 1. Datasets

| Dataset | Cells | Samples | Focus | Status |
|---------|-------|---------|-------|--------|
| CIMA | 6.5M | 428 | Healthy aging, biochemistry, metabolomics | Fully analyzed |
| Inflammation Atlas | 6.3M | 1,047 | 20 diseases, treatment response prediction | Fully analyzed |
| scAtlas | 6.4M | 35+ organs | Normal organ signatures, cancer comparison | Fully analyzed |
| parse_10M | 9.7M | 1,092 | Cytokine perturbation, ground-truth validation | Scripts ready, not yet executed |
| Tahoe-100M | 100.6M | 14 plates | Drug perturbation, 50 cancer lines x 95 drugs | Scripts ready, not yet executed |
| SpatialCorpus-110M | ~110M | 251 files | 8 spatial technologies, tissue-level activity | Scripts ready, not yet executed |

---

## 2. Analysis Scripts

### Completed (13 scripts)

| Script | LOC | Description | Output |
|--------|-----|-------------|--------|
| 01_cima_activity.py | 827 | CIMA: 6.5M cells, biochemistry/metabolomics correlations | 7 files, 61 MB |
| 02_inflam_activity.py | 1,568 | Inflammation Atlas: 6.3M cells, disease activity | 2 files, 101 KB |
| 03_scatlas_analysis.py | 1,682 | scAtlas: 6.4M cells, organ/cancer signatures | 9 files, 59 MB |
| 04_integrated.py | 671 | Cross-atlas integration | 8 files, 963 KB |
| 05_figures.py | 658 | Publication figures | 5 files, 592 KB |
| 06_preprocess_viz_data.py | 3,553 | Web visualization JSON preprocessing | 60+ files, 7.7 GB |
| 07_cross_atlas_analysis.py | 1,080 | Cross-atlas comparison | Signature overlap |
| 08_scatlas_immune_analysis.py | 1,255 | Immune infiltration, exhaustion, CAF analysis | JSON files |
| 10_atlas_validation_pipeline.py | 611 | Multi-level atlas validation | 194 files, 298 GB |
| 11_donor_level_pipeline.py | 727 | Donor-level analysis pipeline | Donor stratification |
| 14_preprocess_bulk_validation.py | 1,435 | Bulk RNA-seq validation preprocessing | 164 files, 20 GB |
| 15_bulk_validation.py | 822 | GTEx/TCGA bulk RNA-seq correlations | Correlation tables |
| 16_resampled_validation.py | 508 | Bootstrap resampled validation with CIs | 36 scatter JSONs, 14 CSVs |

### Pending execution (8 scripts)

| Script | LOC | Description | Dependencies |
|--------|-----|-------------|--------------|
| 18_parse10m_activity.py | 854 | parse_10M: 9.7M cells, cytokine perturbation activity | None |
| 19_tahoe_activity.py | 1,220 | Tahoe: 100M cells, drug-response activity (14 plates) | None |
| 20_spatial_activity.py | 810 | SpatialCorpus: 110M cells, technology-stratified activity | None |
| 21_parse10m_ground_truth.py | 779 | CytoSig ground-truth validation (predicted vs actual) | 18 |
| 22_tahoe_drug_signatures.py | 763 | Drug sensitivity signature extraction | 19 |
| 23_spatial_neighborhood.py | 1,166 | Spatial neighborhood activity analysis (NicheFormer) | 20 |
| 24_preprocess_perturbation_viz.py | 587 | JSON/DuckDB preprocessing for perturbation viz | 21, 22 |
| 25_preprocess_spatial_viz.py | 576 | JSON/DuckDB preprocessing for spatial viz | 23 |

**Execution dependency chains:**
```
18 → 21 → 24   (parse_10M → ground truth → viz preprocessing)
19 → 22 → 24   (Tahoe → drug signatures → viz preprocessing)
20 → 23 → 25   (SpatialCorpus → neighborhoods → viz preprocessing)
```

**Estimated GPU time:** ~200 hours across 3 parallel chains.

### Utility scripts

| Script | LOC | Status |
|--------|-----|--------|
| convert_data_to_duckdb.py | 1,249 | Run (atlas_data.duckdb generated) |
| convert_perturbation_to_duckdb.py | 299 | Pending (awaiting scripts 18-22) |
| convert_spatial_to_duckdb.py | 294 | Pending (awaiting script 20) |
| create_data_lite.py | 1,249 | Run |
| build_rag_index.py | 441 | Run |

---

## 3. Pipeline Package

**Location:** `cytoatlas-pipeline/` (19,563 LOC source + 2,685 LOC tests)

| Subpackage | Purpose |
|------------|---------|
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

Testing: 15 test files (2,685 LOC) + 2 equivalence tests (1,187 LOC for perturbation + spatial data format validation).

---

## 4. API Backend

**Location:** `cytoatlas-api/` | **Endpoints:** 260 across 17 routers

### Router inventory

| Router | Endpoints | LOC | Description |
|--------|-----------|-----|-------------|
| Inflammation | 42 | 713 | Disease activity, treatment response, cohort validation |
| Validation | 39 | 832 | 5-type credibility assessment |
| scAtlas | 31 | 478 | Organ signatures, cancer comparison, immune infiltration |
| CIMA | 28 | 436 | Age/BMI correlations, biochemistry, metabolites, eQTL |
| Perturbation | 23 | 357 | Cytokine response (parse_10M), drug sensitivity (Tahoe) |
| Cross-Atlas | 20 | 391 | Atlas comparison, conserved signatures |
| Spatial | 15 | 295 | Spatial activity, neighborhood, technology comparison |
| Gene | 11 | 307 | Gene-centric views and analysis |
| Export | 9 | 227 | Data export (CSV, JSON, HDF5) |
| Submit | 9 | 300 | Dataset submission with chunked upload |
| Chat | 8 | 464 | Claude AI assistant (RAG-powered, SSE streaming) |
| Search | 6 | 197 | Global search |
| Atlases | 5 | 175 | Atlas registry |
| Auth | 5 | 203 | JWT + API key authentication |
| Health | 4 | 97 | Health checks, system status |
| Pipeline | 4 | 120 | Pipeline status and management |
| WebSocket | 1 | 255 | Real-time updates |

### Service layer (18 services, 11,561 LOC)

Key services: gene_service (1,140), mcp_tools (1,678), inflammation_service (968), validation_service (923), cross_atlas_service (802).

### Repository layer (5 files, 1,191 LOC)

- **DuckDBRepository** (607 LOC) — primary, queries `atlas_data.duckdb` / `perturbation_data.duckdb` / `spatial_data.duckdb`
- **JSONRepository** (210 LOC) — fallback, loads `visualization/data/*.json`
- Protocol-based abstraction (PEP 544) for backend swappability

### Test coverage (18 files, 7,220 LOC)

- 10 unit tests (2,030 LOC): DuckDB, cache, chat, security, services, streaming
- 2 integration tests (566 LOC): API endpoints, DuckDB equivalence
- 2 data validation tests (350 LOC): JSON integrity, schema
- 4 perturbation/spatial tests (4,078 LOC): router + service for both domains

---

## 5. React Frontend

**Location:** `cytoatlas-api/frontend/` | **Stack:** React 19 + Vite 6 + TypeScript + Tailwind CSS v4 + TanStack Query + Zustand

### Pages (12)

| Page | Route | LOC | Description |
|------|-------|-----|-------------|
| Home | `/` | 189 | Hero, atlas cards, stats, features |
| About | `/about` | 209 | Static content |
| Explore | `/explore` | 86 | Atlas exploration with filters |
| Search | `/search` | 121 | Global search with autocomplete |
| Atlas Detail | `/atlas/:name` | 36 + tabs | 30 tab panels (CIMA/Inflammation/scAtlas) |
| Validate | `/validate` | 97 | 5-tab validation dashboard |
| Compare | `/compare` | 58 | Cross-atlas comparison (5 tabs) |
| Gene Detail | `/gene/:symbol` | 101 | Gene-centric view (5 tabs) |
| Perturbation | `/perturbation` | 97 | Cytokine/drug perturbation (5 tabs) |
| Spatial | `/spatial` | 89 | Spatial transcriptomics (5 tabs) |
| Chat | `/chat` | 193 | AI assistant with SSE streaming |
| Submit | `/submit` | 655 | Dataset upload with chunked transfer |

### Component inventory (76 components)

| Category | Count | Key components |
|----------|-------|----------------|
| Charts | 13 | plotly-chart, scatter, heatmap, bar, boxplot, violin, volcano, lollipop, forest-plot, sankey |
| UI | 9 | tab-panel, filter-bar, signature-toggle, export-button, loading-skeleton, error-boundary, search-input, atlas-card, stat-card |
| Atlas panels | 22 | 5 shared + 6 CIMA + 6 Inflammation + 5 scAtlas |
| Page tabs | 20 | 5 validation + 5 compare + 5 gene + 5 perturbation |
| Spatial tabs | 5 | overview, tissue-activity, tech-comparison, gene-coverage, spatial-map |
| Chat | 5 | sidebar, messages, input, viz, suggestion-chips |
| Layout | 2 | header, footer |

### API layer (21 files)

- 9 hooks: use-atlas, use-cima, use-inflammation, use-scatlas, use-validation, use-cross-atlas, use-gene, use-perturbation, use-spatial, use-chat
- 8 type definitions: atlas, activity, common, validation, gene, perturbation, spatial, chat
- 1 API client with auth token injection
- 3 stores: app-store (signatureType), atlas-store, filter-store, validation-store

### Testing

- **Unit tests (Vitest):** 6 files, 40 tests — client, stores, plotly-chart, scatter-chart, tab-panel, filter-bar
- **E2E tests (Playwright):** 4 specs — home, atlas-detail, validation, search

### Build

- `npm run build` outputs to `../static/` (79 chunks, 5.3 MB minified)
- `base: '/static/'` for correct asset resolution under FastAPI StaticFiles mount
- Multi-stage Dockerfile: Node.js frontend build + Python backend + production

---

## 6. Data Infrastructure

### Storage

| Layer | Location | Size | Status |
|-------|----------|------|--------|
| atlas_data.duckdb | `/data/parks34/projects/2cytoatlas/` | 590 MB (51 tables, 9.6M rows) | Generated |
| perturbation_data.duckdb | (not yet generated) | ~3-5 GB expected | Pending scripts 18-22 |
| spatial_data.duckdb | (not yet generated) | ~2-4 GB expected | Pending script 20 |
| Visualization JSON | `visualization/data/` | 15 GB (1,412 files) | Generated |
| Validation results | `results/atlas_validation/` | 298 GB (194 H5AD/CSV) | Generated |
| Cross-sample validation | `results/cross_sample_validation/` | 20 GB (164 files) | Generated |
| Static data | `cytoatlas-api/static/data/` | 123 MB (3 JSON files) | Generated |
| Build assets | `cytoatlas-api/static/assets/` | 5.3 MB (79 JS/CSS) | Generated (gitignored) |

**Total generated data:** ~334 GB

### Tiered caching

| Tier | Medium | TTL | Purpose |
|------|--------|-----|---------|
| Hot | Redis / in-memory dict | 1 hour | Frequently accessed API responses |
| Warm | JSON files | Persistent | Dashboard visualization data |
| Cold | CSV / H5AD archives | Persistent | Raw analysis outputs |

---

## 7. SLURM / HPC

**Configuration:** `scripts/slurm/jobs.yaml` with 6 resource profiles and 28+ job definitions.

| Profile | GPU | Memory | CPUs | Use case |
|---------|-----|--------|------|----------|
| gpu_heavy | A100 x1 | 128 GB | 16 | Large atlas analyses (24-48h) |
| gpu_medium | A100 x1 | 128 GB | 8 | Moderate GPU work (4-8h) |
| gpu_lite | A100 x1 | 128 GB | 8 | Pilot, quick GPU jobs (2h) |
| gpu_xlarge | A100 x1 | 256 GB | 16 | Extra-large jobs |
| cpu_normal | None | 64 GB | 8 | Integration, figures (2-4h) |
| cpu_heavy | None | 128 GB | 16 | Preprocessing (6h) |

---

## 8. Documentation

| Category | Files | LOC | Contents |
|----------|-------|-----|----------|
| Main docs | 8 | 4,627 | ARCHITECTURE, API_REFERENCE, DEPLOYMENT, USER_GUIDE, ATLAS_VALIDATION, CELL_TYPE_MAPPING, EMBEDDED_DATA_CHECKLIST |
| Dataset specs | 9 | 1,557 | CIMA, Inflammation, scAtlas, parse_10M, Tahoe, SpatialCorpus, signatures, bulk |
| Pipeline docs | 7 | 1,434 | Per-atlas pipeline specs, visualization preprocessing |
| ADRs | 5 | 888 | Parquet/JSON, Repository pattern, RBAC, Multi-database |
| Project instructions | 1 | ~550 | CLAUDE.md |

---

## 9. Security

- JWT authentication with 30-min expiration
- 5-role RBAC model (anonymous, viewer, researcher, data_curator, admin)
- Audit logging to JSONL (user, IP, endpoint, action)
- Rate limiting per-user and per-IP
- RAG-grounded chat with prompt injection defense
- Password hashing with bcrypt
- API key rotation per user

---

## 10. Remaining Work

### High priority — GPU pipeline execution

Run scripts 18-25 on HPC cluster. All code is written and tested; only execution is needed.

```bash
# Submit all perturbation + spatial jobs
cd /data/parks34/projects/2cytoatlas
sbatch scripts/slurm/run_all.sh --phases 20-24
```

After completion:
```bash
python scripts/convert_perturbation_to_duckdb.py   # → perturbation_data.duckdb
python scripts/convert_spatial_to_duckdb.py         # → spatial_data.duckdb
```

### Low priority

| Item | Impact |
|------|--------|
| Install Playwright browsers (`npx playwright install`) and run E2E tests | Testing |
| Run pilot script (00) for onboarding validation | Optional |
| scGPT cohort integration (~35M cells) | Future dataset |
| cellxgene Census integration | Future dataset |

---

## Quick Reference

### Development workflow

```bash
# Terminal 1: API backend
cd cytoatlas-api && uvicorn app.main:app --port 8000 --reload

# Terminal 2: React dev server
cd cytoatlas-api/frontend && npm run dev    # localhost:3000, proxies /api → :8000

# Run tests
cd cytoatlas-api && pytest tests/ -v                      # API tests
cd cytoatlas-api/frontend && npm test                      # Frontend unit tests
cd cytoatlas-api/frontend && npx playwright test           # Frontend E2E tests
cd cytoatlas-pipeline && pytest tests/ -v                  # Pipeline tests
```

### Key commands

```bash
# Regenerate DuckDB
python scripts/convert_data_to_duckdb.py --all

# Rebuild frontend
cd cytoatlas-api/frontend && npm run build

# Run maintenance audit
python scripts/maintenance/audit_clutter.py --report

# Run equivalence tests
pytest cytoatlas-pipeline/tests/equivalence/ -v --tb=short
```
