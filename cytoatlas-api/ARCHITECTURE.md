# CytoAtlas API Architecture

## Overview

The CytoAtlas API is a FastAPI-based REST service that provides programmatic access to pre-computed cytokine and secreted protein activity signatures. The system is designed to be **atlas-agnostic** and extensible:

- **Built-in atlases**: CIMA, Inflammation Atlas, scAtlas
- **User-registered atlases**: Support for custom datasets
- **Dynamic API**: Unified endpoints that work with any atlas

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CytoAtlas API                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│   │   CIMA      │    │ Inflammation│    │   scAtlas   │    │Cross-Atlas  │  │
│   │   Router    │    │   Router    │    │   Router    │    │   Router    │  │
│   │  (~32 eps)  │    │  (~44 eps)  │    │  (~36 eps)  │    │  (~28 eps)  │  │
│   └──────┬──────┘    └──────┬──────┘    └──────┬──────┘    └──────┬──────┘  │
│          │                  │                  │                  │         │
│          └──────────────────┴──────────────────┴──────────────────┘         │
│                                      │                                       │
│                            ┌─────────▼─────────┐                            │
│                            │   Service Layer   │                            │
│                            │  (Business Logic) │                            │
│                            └─────────┬─────────┘                            │
│                                      │                                       │
│          ┌───────────────────────────┼───────────────────────────┐          │
│          │                           │                           │          │
│   ┌──────▼──────┐           ┌────────▼────────┐         ┌───────▼───────┐  │
│   │    Cache    │           │   JSON Files    │         │  PostgreSQL   │  │
│   │ (In-Memory/ │           │ (visualization/ │         │  (Optional)   │  │
│   │   Redis)    │           │     data/)      │         │               │  │
│   └─────────────┘           └─────────────────┘         └───────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Extensible Atlas System

The API supports registering new atlases dynamically:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Atlas Registry                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Built-in Atlases (always available):                               │
│  ┌──────────┐  ┌──────────────┐  ┌──────────┐                      │
│  │   CIMA   │  │ Inflammation │  │  scAtlas │                      │
│  │ 6.5M    │  │    4.9M     │  │   6.4M   │                      │
│  │  cells   │  │   cells      │  │  cells   │                      │
│  └──────────┘  └──────────────┘  └──────────┘                      │
│                                                                      │
│  User-Registered Atlases:                                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │  my_atlas_1  │  │  my_atlas_2  │  │     ...      │              │
│  │   (custom)   │  │   (custom)   │  │              │              │
│  └──────────────┘  └──────────────┘  └──────────────┘              │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Unified API Pattern

Instead of atlas-specific endpoints, use the unified API:

```bash
# Old pattern (still supported for backward compatibility):
GET /api/v1/cima/cell-types
GET /api/v1/inflammation/diseases

# New unified pattern (works for ANY atlas):
GET /api/v1/atlases                           # List all atlases
GET /api/v1/atlases/{atlas}/summary           # Any atlas summary
GET /api/v1/atlases/{atlas}/cell-types        # Any atlas cell types
GET /api/v1/atlases/{atlas}/features          # What's available
GET /api/v1/atlases/{atlas}/activity          # Activity data
GET /api/v1/atlases/{atlas}/correlations/age  # Correlations (if available)
```

### Registering a New Atlas

```bash
# Register a new atlas
POST /api/v1/atlases/register
{
  "name": "my_immune_atlas",
  "display_name": "My Immune Cell Atlas",
  "description": "Custom single-cell RNA-seq dataset",
  "h5ad_path": "/path/to/data.h5ad",
  "data_dir": "/path/to/precomputed/json/",
  "atlas_type": "immune",
  "species": "human"
}

# After registration, all unified endpoints work:
GET /api/v1/atlases/my_immune_atlas/summary
GET /api/v1/atlases/my_immune_atlas/activity
```

### Atlas Features

Each atlas declares its available features:

| Feature | Description | Example Atlases |
|---------|-------------|-----------------|
| `cell_type_activity` | Basic activity data | All |
| `age_correlation` | Age correlations | CIMA, Inflammation |
| `bmi_correlation` | BMI correlations | CIMA, Inflammation |
| `disease_activity` | Disease-specific data | Inflammation |
| `organ_signatures` | Organ patterns | scAtlas |
| `eqtl` | Genetic associations | CIMA |
| `treatment_response` | Treatment prediction | Inflammation |

---

## Directory Structure

```
cytoatlas-api/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application factory
│   ├── config.py               # Pydantic settings configuration
│   │
│   ├── core/                   # Infrastructure components
│   │   ├── cache.py            # Redis/in-memory caching
│   │   ├── database.py         # SQLAlchemy async engine
│   │   ├── security.py         # JWT & API key authentication
│   │   └── rate_limit.py       # Request rate limiting
│   │
│   ├── models/                 # SQLAlchemy ORM models (optional DB)
│   │   ├── atlas.py            # Atlas metadata
│   │   ├── sample.py           # Sample information
│   │   ├── cell_type.py        # Cell type definitions
│   │   ├── signature.py        # Signature definitions
│   │   ├── computed_stat.py    # Pre-computed statistics
│   │   ├── validation_metric.py # Validation results
│   │   └── user.py             # User accounts
│   │
│   ├── schemas/                # Pydantic request/response schemas
│   │   ├── common.py           # Shared schemas (pagination, errors)
│   │   ├── cima.py             # CIMA-specific schemas
│   │   ├── inflammation.py     # Inflammation-specific schemas
│   │   ├── scatlas.py          # scAtlas-specific schemas
│   │   ├── cross_atlas.py      # Cross-atlas comparison schemas
│   │   └── validation.py       # Validation panel schemas
│   │
│   ├── services/               # Business logic layer
│   │   ├── base.py             # Base service with common methods
│   │   ├── cima_service.py     # CIMA data access
│   │   ├── inflammation_service.py
│   │   ├── scatlas_service.py
│   │   ├── cross_atlas_service.py
│   │   ├── validation_service.py
│   │   └── h5ad_service.py     # H5AD file access (future)
│   │
│   └── routers/                # API endpoint definitions
│       ├── health.py           # Health check endpoints
│       ├── auth.py             # Authentication endpoints
│       ├── cima.py             # CIMA endpoints
│       ├── inflammation.py     # Inflammation endpoints
│       ├── scatlas.py          # scAtlas endpoints
│       ├── cross_atlas.py      # Cross-atlas endpoints
│       ├── validation.py       # Validation panel endpoints
│       └── export.py           # Data export endpoints
│
├── scripts/
│   ├── run_server.sh           # Start server (HPC)
│   ├── seed_database.py        # Populate database
│   └── slurm/                  # SLURM job scripts
│
├── tests/
│   ├── unit/                   # Unit tests
│   ├── integration/            # Integration tests
│   └── e2e/                    # End-to-end tests
│
├── alembic/                    # Database migrations
│   └── versions/
│
├── .env.hpc                    # HPC environment template
├── pyproject.toml              # Python project configuration
└── README.md
```

---

## Data Flow

### Current Implementation (JSON-based)

```
┌──────────────────────────────────────────────────────────────────┐
│                     Data Preprocessing Pipeline                   │
│                                                                   │
│  Raw H5AD Files (282GB)                                          │
│       │                                                           │
│       ▼                                                           │
│  scripts/06_preprocess_viz_data.py                               │
│       │                                                           │
│       ▼                                                           │
│  JSON Files (71MB)  ──────────────────────────────────────────┐  │
│  visualization/data/                                           │  │
│    ├── cima_correlations.json                                 │  │
│    ├── cima_celltype.json                                     │  │
│    ├── cima_eqtl_top.json                                     │  │
│    ├── inflammation_disease.json                              │  │
│    ├── inflammation_celltype.json                             │  │
│    ├── scatlas_organs.json                                    │  │
│    ├── scatlas_celltypes.json                                 │  │
│    └── ...                                                    │  │
└───────────────────────────────────────────────────────────────┼──┘
                                                                │
                            ┌───────────────────────────────────┘
                            ▼
┌──────────────────────────────────────────────────────────────────┐
│                        FastAPI Server                             │
│                                                                   │
│  Request: GET /api/v1/cima/correlations/age                      │
│       │                                                           │
│       ▼                                                           │
│  Router (cima.py)                                                │
│       │                                                           │
│       ▼                                                           │
│  Service (cima_service.py)                                       │
│       │                                                           │
│       ├──► Check Cache ──► Cache Hit? ──► Return cached          │
│       │                                                           │
│       ▼ (Cache Miss)                                             │
│  BaseService.load_json("cima_correlations.json")                 │
│       │                                                           │
│       ▼                                                           │
│  Filter/Transform Data                                           │
│       │                                                           │
│       ▼                                                           │
│  Cache Result ──► Return Response                                │
└──────────────────────────────────────────────────────────────────┘
```

---

## API Endpoints by Atlas

### CIMA Atlas (6.5M cells, 421 samples)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/cima/summary` | GET | Atlas statistics |
| `/api/v1/cima/cell-types` | GET | List cell types |
| `/api/v1/cima/signatures` | GET | List signatures |
| `/api/v1/cima/activity` | GET | Cell type activity |
| `/api/v1/cima/correlations/age` | GET | Age correlations |
| `/api/v1/cima/correlations/bmi` | GET | BMI correlations |
| `/api/v1/cima/correlations/biochemistry` | GET | Biochemistry correlations |
| `/api/v1/cima/correlations/metabolites` | GET | Metabolite correlations |
| `/api/v1/cima/differential` | GET | Differential analysis |
| `/api/v1/cima/eqtl` | GET | eQTL browser |
| `/api/v1/cima/eqtl/top` | GET | Top eQTL results |
| `/api/v1/cima/boxplots/age/{signature}` | GET | Age boxplot data |
| `/api/v1/cima/boxplots/bmi/{signature}` | GET | BMI boxplot data |

### Inflammation Atlas (4.9M cells, 817 samples, 20 diseases)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/inflammation/summary` | GET | Atlas statistics |
| `/api/v1/inflammation/diseases` | GET | List diseases |
| `/api/v1/inflammation/cell-types` | GET | List cell types |
| `/api/v1/inflammation/disease-activity` | GET | Disease activity |
| `/api/v1/inflammation/activity` | GET | Cell type activity |
| `/api/v1/inflammation/treatment-response` | GET | Treatment prediction |
| `/api/v1/inflammation/roc-curves` | GET | ROC curve data |
| `/api/v1/inflammation/feature-importance` | GET | Feature importance |
| `/api/v1/inflammation/cohort-validation` | GET | Cross-cohort validation |
| `/api/v1/inflammation/disease-sankey` | GET | Sankey diagram data |

### scAtlas (6.4M cells, normal + cancer)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/scatlas/summary` | GET | Atlas statistics |
| `/api/v1/scatlas/organs` | GET | Organ activity |
| `/api/v1/scatlas/cell-types` | GET | Cell type activity |
| `/api/v1/scatlas/cancer-comparison` | GET | Normal vs cancer |
| `/api/v1/scatlas/cancer-types` | GET | Cancer type list |

### Cross-Atlas Comparison

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/cross-atlas/atlases` | GET | List atlases |
| `/api/v1/cross-atlas/comparison` | GET | Atlas comparison |
| `/api/v1/cross-atlas/conserved-signatures` | GET | Conserved patterns |

### Validation Panel

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/validation/summary` | GET | Validation summary |
| `/api/v1/validation/expression-vs-activity` | GET | Expression correlation |
| `/api/v1/validation/gene-coverage/{signature}` | GET | Gene coverage |
| `/api/v1/validation/cv-stability` | GET | CV stability |

---

## Configuration

### Environment Variables

```bash
# Application
APP_NAME=CytoAtlas API
APP_VERSION=0.1.0
ENVIRONMENT=production          # development, staging, production
DEBUG=false

# API
API_V1_PREFIX=/api/v1
ALLOWED_ORIGINS=*               # CORS origins

# Database (optional)
DATABASE_URL=                   # postgresql+asyncpg://...

# Cache (optional)
REDIS_URL=                      # redis://localhost:6379

# Data Paths
VIZ_DATA_PATH=/vf/users/parks34/projects/2secactpy/visualization/data
RESULTS_BASE_PATH=/vf/users/parks34/projects/2secactpy/results

# Security
SECRET_KEY=your-secret-key
RATE_LIMIT_REQUESTS=1000
RATE_LIMIT_WINDOW=60
```

---

## Development Roadmap

### Phase 1: Foundation ✅ COMPLETE
- [x] Project structure
- [x] FastAPI application factory
- [x] Pydantic settings
- [x] In-memory caching
- [x] Basic authentication scaffolding
- [x] Health check endpoints

### Phase 2: Core Services ✅ COMPLETE
- [x] Base service with JSON loading
- [x] Caching decorator
- [x] CIMA service
- [x] Inflammation service
- [x] scAtlas service
- [x] Cross-atlas service

### Phase 3: Routers ✅ MOSTLY COMPLETE
- [x] CIMA router (32 endpoints)
- [x] Inflammation router (44 endpoints)
- [x] scAtlas router (36 endpoints)
- [x] Cross-atlas router (28 endpoints)
- [x] Validation router (scaffolding)
- [x] Export router (scaffolding)

### Phase 4: Data Alignment 🔄 IN PROGRESS
- [x] Fix schema mismatches (InflammationDiseaseActivity)
- [x] Fix eQTL endpoint
- [ ] Verify all endpoints return valid data
- [ ] Add missing JSON data files for some endpoints
- [ ] Handle edge cases (empty results, missing data)

### Phase 5: Validation Panel 📋 TODO
- [ ] Implement expression-vs-activity correlation
- [ ] Implement gene coverage analysis
- [ ] Implement CV stability metrics
- [ ] Implement biological association validation
- [ ] Add validation data generation scripts

### Phase 6: Export & Integration 📋 TODO
- [ ] CSV export for all data types
- [ ] Bulk download endpoints
- [ ] WebSocket for long-running queries (future)

### Phase 7: Production Hardening 📋 TODO
- [ ] Comprehensive error handling
- [ ] Request logging
- [ ] Prometheus metrics
- [ ] Rate limiting enforcement
- [ ] API key management
- [ ] Load testing

---

## Testing Strategy

### Unit Tests
```bash
pytest tests/unit/ -v
```
- Service method tests
- Schema validation tests
- Utility function tests

### Integration Tests
```bash
pytest tests/integration/ -v
```
- Full request/response cycle
- Database operations (when enabled)
- Cache behavior

### Manual Testing
```bash
# Start server
./scripts/run_server.sh

# Test endpoints
curl http://localhost:8000/api/v1/health
curl http://localhost:8000/api/v1/cima/summary
curl http://localhost:8000/api/v1/inflammation/diseases
```

---

## Deployment

### HPC (Biowulf/SLURM)
```bash
# Interactive node
sinteractive --mem=32g --cpus-per-task=4

# Run server
cd /vf/users/parks34/projects/2secactpy/cytoatlas-api
./scripts/run_server.sh

# Or submit batch job
sbatch scripts/slurm/run_api.sh
```

### Production Considerations
1. **Reverse Proxy**: Use nginx for SSL termination
2. **Multiple Workers**: `--workers 4` for production
3. **Process Manager**: Use systemd or supervisord
4. **Database**: Enable PostgreSQL for persistence
5. **Caching**: Enable Redis for distributed caching

---

## Key Design Decisions

1. **JSON-First Approach**: Pre-computed JSON files provide fast responses without database dependency

2. **Optional Database**: PostgreSQL is optional; system works fully with JSON files only

3. **In-Memory Cache Fallback**: Works without Redis on HPC nodes

4. **Pydantic v2**: Modern schema validation with better performance

5. **Async Throughout**: All I/O operations are async for scalability

6. **Service Layer Pattern**: Business logic separated from routing

7. **HPC Compatibility**: Environment variable handling for SLURM/batch systems

---

## Common Issues & Solutions

### "ENVIRONMENT=BATCH" Error
The HPC sets `ENVIRONMENT=BATCH`. Config validators normalize this to "production".

### Port Already in Use
```bash
pkill -f "uvicorn app.main"
```

### Missing Dependencies
```bash
pip install -e .
```

### Schema Mismatch Errors
Check that JSON data structure matches Pydantic schema. Use validators to transform data if needed.
