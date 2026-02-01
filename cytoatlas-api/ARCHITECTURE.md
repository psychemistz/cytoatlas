# CytoAtlas API Architecture

**Last Updated**: 2026-01-31

## Overview

The CytoAtlas API is a FastAPI-based REST service providing programmatic access to pre-computed cytokine and secreted protein activity signatures across 17+ million immune cells. The system is designed to be **atlas-agnostic** and extensible.

**Key Stats:**
- 188+ API endpoints
- 14 routers
- 30+ JSON data files (~500MB)
- 3 built-in atlases + user registration support

---

## High-Level Architecture

```
+-----------------------------------------------------------------------------+
|                           CytoAtlas Web Portal                               |
+-----------------------------------------------------------------------------+
|                                                                              |
|   +---------------------------------------------------------------------+   |
|   |  FRONTEND (Single Page Application - 8 pages)                        |   |
|   |  +---------+ +---------+ +---------+ +---------+ +---------+ +----+ |   |
|   |  | Landing | | Explore | | Compare | |Validate | | Submit  | |Chat| |   |
|   |  +---------+ +---------+ +---------+ +---------+ +---------+ +----+ |   |
|   +---------------------------------------------------------------------+   |
|                                      |                                       |
|                                      v                                       |
|   +---------------------------------------------------------------------+   |
|   |  FastAPI Backend (14 routers)                                        |   |
|   |  +----------+ +----------+ +----------+ +----------+ +----------+   |   |
|   |  |  Atlas   | |Validation| |  Search  | |   Chat   | |  Export  |   |   |
|   |  |   API    | |   API    | |   API    | |   API    | |   API    |   |   |
|   |  +----------+ +----------+ +----------+ +----------+ +----------+   |   |
|   +---------------------------------------------------------------------+   |
|                                      |                                       |
|          +---------------------------+---------------------------+          |
|          v                           v                           v          |
|   +--------------+          +--------------+          +--------------+      |
|   |    Cache     |          |  JSON Files  |          |  PostgreSQL  |      |
|   | (In-Memory/  |          | (30+ files,  |          |  (Optional)  |      |
|   |   Redis)     |          |   ~500MB)    |          |              |      |
|   +--------------+          +--------------+          +--------------+      |
|                                                                              |
+-----------------------------------------------------------------------------+
```

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
│   │   ├── logging.py          # Request logging
│   │   └── rate_limit.py       # Request rate limiting
│   │
│   ├── models/                 # SQLAlchemy ORM models
│   │   ├── atlas.py            # Atlas metadata
│   │   ├── sample.py           # Sample information
│   │   ├── cell_type.py        # Cell type definitions
│   │   ├── signature.py        # Signature definitions
│   │   ├── computed_stat.py    # Pre-computed statistics
│   │   ├── validation_metric.py # Validation results
│   │   ├── user.py             # User accounts
│   │   ├── conversation.py     # Chat conversations
│   │   └── job.py              # Background job tracking
│   │
│   ├── schemas/                # Pydantic request/response schemas
│   │   ├── common.py           # Shared schemas (pagination, errors)
│   │   ├── atlas.py            # Atlas registration schemas
│   │   ├── cima.py             # CIMA-specific schemas
│   │   ├── inflammation.py     # Inflammation-specific schemas
│   │   ├── scatlas.py          # scAtlas-specific schemas
│   │   ├── cross_atlas.py      # Cross-atlas comparison schemas
│   │   ├── validation.py       # 5-type validation schemas
│   │   ├── chat.py             # Chat/conversation schemas
│   │   ├── search.py           # Search schemas
│   │   └── submit.py           # Atlas submission schemas
│   │
│   ├── services/               # Business logic layer
│   │   ├── base.py             # Base service (JSON loading, caching)
│   │   ├── cima_service.py     # CIMA data access
│   │   ├── inflammation_service.py
│   │   ├── scatlas_service.py
│   │   ├── cross_atlas_service.py
│   │   ├── validation_service.py  # 5-type validation (636 lines)
│   │   ├── atlas_registry.py   # Dynamic atlas registration
│   │   ├── generic_atlas_service.py
│   │   ├── search_service.py
│   │   ├── submit_service.py
│   │   ├── chat_service.py     # Claude API integration
│   │   ├── context_manager.py
│   │   └── mcp_tools.py        # Claude tool definitions
│   │
│   ├── routers/                # API endpoint definitions
│   │   ├── health.py           # Health check endpoints
│   │   ├── auth.py             # Authentication endpoints
│   │   ├── atlases.py          # Unified dynamic API
│   │   ├── cima.py             # CIMA endpoints (~32)
│   │   ├── inflammation.py     # Inflammation endpoints (~44)
│   │   ├── scatlas.py          # scAtlas endpoints (~36)
│   │   ├── cross_atlas.py      # Cross-atlas endpoints (~28)
│   │   ├── validation.py       # Validation endpoints (~12)
│   │   ├── export.py           # Data export endpoints
│   │   ├── search.py           # Search endpoints
│   │   ├── submit.py           # Atlas submission endpoints
│   │   ├── chat.py             # Chat endpoints
│   │   └── websocket.py        # WebSocket endpoints
│   │
│   └── tasks/                  # Background task processing
│       ├── celery_app.py       # Celery configuration
│       └── process_atlas.py    # Atlas processing tasks
│
├── static/                     # Frontend assets
│   ├── index.html              # SPA entry point
│   ├── css/
│   ├── js/
│   │   ├── api.js              # API client
│   │   ├── app.js              # Main application
│   │   ├── router.js           # Client-side routing
│   │   ├── components/         # Reusable components
│   │   └── pages/              # Page views (8 files)
│   └── assets/
│
├── alembic/                    # Database migrations
├── tests/                      # Test suite
├── scripts/                    # Deployment scripts
├── singularity/                # Container definition
├── nginx/                      # Reverse proxy config
│
├── ARCHITECTURE.md             # This file
├── README.md                   # Quick start guide
├── pyproject.toml              # Python package config
├── docker-compose.yml
├── Dockerfile
└── alembic.ini
```

---

## API Endpoints by Category

### Atlas-Specific Endpoints

| Router | Endpoints | Description |
|--------|-----------|-------------|
| CIMA | ~32 | Correlations (age/BMI/biochemistry/metabolites), eQTL, differential |
| Inflammation | ~44 | Disease activity, treatment response, cohort validation |
| scAtlas | ~36 | Organ/celltype signatures, cancer comparison, immune infiltration |
| Cross-Atlas | ~28 | Atlas comparison, conserved signatures, meta-analysis |

### Unified Atlas API

```bash
GET  /api/v1/atlases                      # List all atlases
GET  /api/v1/atlases/{atlas}/summary      # Atlas statistics
GET  /api/v1/atlases/{atlas}/cell-types   # Cell types
GET  /api/v1/atlases/{atlas}/features     # Available features
POST /api/v1/atlases/register             # Register new atlas
```

### 5-Type Validation System

| Endpoint | Type | Description |
|----------|------|-------------|
| `/validation/sample-level/{atlas}/{sig}` | 1 | Sample pseudobulk vs activity |
| `/validation/celltype-level/{atlas}/{sig}` | 2 | Cell type vs activity |
| `/validation/pseudobulk-vs-singlecell/{atlas}/{sig}` | 3 | Aggregation comparison |
| `/validation/singlecell-direct/{atlas}/{sig}` | 4 | Expressing vs non-expressing |
| `/validation/biological-associations/{atlas}` | 5 | Known marker validation |
| `/validation/gene-coverage/{atlas}/{sig}` | - | Gene detection analysis |
| `/validation/cv-stability/{atlas}` | - | Cross-validation stability |
| `/validation/summary/{atlas}` | - | Overall quality grade |

### Other Endpoints

| Category | Endpoints | Description |
|----------|-----------|-------------|
| Health | 2 | Health check, readiness |
| Auth | 4 | Login, register, verify |
| Search | 4 | Global signature/cell type search |
| Chat | 4 | Claude AI conversation |
| Submit | 4 | Dataset submission workflow |
| Export | 6 | CSV/JSON data export |
| WebSocket | 2 | Real-time streaming |

---

## Data Flow

### Current Implementation (JSON-based)

```
Raw H5AD Files (282GB)
       |
       v
scripts/06_preprocess_viz_data.py
       |
       v
JSON Files (visualization/data/, ~500MB)
       |
       v
FastAPI Service
       |
       +---> Check Cache ---> Cache Hit? ---> Return cached
       |
       v (Cache Miss)
BaseService.load_json()
       |
       v
Filter/Transform Data
       |
       v
Cache Result ---> Return Response
```

---

## Configuration

### Environment Variables

```bash
# Application
APP_NAME=CytoAtlas API
APP_VERSION=0.1.0
ENVIRONMENT=production

# API
API_V1_PREFIX=/api/v1
ALLOWED_ORIGINS=*

# Database (optional)
DATABASE_URL=postgresql+asyncpg://...

# Cache (optional)
REDIS_URL=redis://localhost:6379

# Data Paths
VIZ_DATA_PATH=/vf/users/parks34/projects/2secactpy/visualization/data
RESULTS_BASE_PATH=/vf/users/parks34/projects/2secactpy/results

# Claude API (for chat)
ANTHROPIC_API_KEY=sk-ant-...

# Security
SECRET_KEY=your-secret-key
RATE_LIMIT_REQUESTS=1000
RATE_LIMIT_WINDOW=60
```

---

## Development Status

### Complete

- [x] FastAPI application factory with lifespan
- [x] Pydantic v2 settings with HPC validators
- [x] In-memory cache (Redis optional)
- [x] All 14 routers implemented
- [x] All services implemented
- [x] Validation service (5 types, 636 lines)
- [x] Frontend SPA (8 pages)
- [x] 30+ JSON data files

### In Progress

- [ ] Validation JSON data generation
- [ ] Chat streaming stability
- [ ] scAtlas immune analysis completion

### Planned

- [ ] Full JWT authentication
- [ ] OAuth providers (Google, ORCID)
- [ ] Dataset submission with Celery
- [ ] Prometheus metrics
- [ ] Load testing

---

## Key Design Decisions

1. **JSON-First Approach**: Pre-computed JSON provides fast responses without database
2. **Optional Database**: PostgreSQL available but not required
3. **In-Memory Cache Fallback**: Works on HPC without Redis
4. **Atlas-Agnostic Design**: Unified API works with any registered atlas
5. **Async Throughout**: All I/O operations are async for scalability
6. **Service Layer Pattern**: Business logic separated from routing
7. **HPC Compatibility**: Environment variable handling for SLURM

---

## How to Run

```bash
# Activate environment
source ~/bin/myconda
conda activate secactpy

# Navigate to API
cd /vf/users/parks34/projects/2secactpy/cytoatlas-api

# Install dependencies
pip install -e .

# Copy environment config
cp .env.example .env

# Run server
./scripts/run_server.sh
# Or: python -m uvicorn app.main:app --host 0.0.0.0 --port 8000

# Access
# - API docs: http://localhost:8000/docs
# - Web UI: http://localhost:8000/
```

---

## Testing

```bash
# Health check
curl http://localhost:8000/api/v1/health

# CIMA summary
curl http://localhost:8000/api/v1/cima/summary

# Inflammation diseases
curl http://localhost:8000/api/v1/inflammation/diseases

# scAtlas organs
curl http://localhost:8000/api/v1/scatlas/organs
```

---

## References

- **Master Plan**: `/home/parks34/.claude/plans/cytoatlas-master-plan.md`
- **Project Instructions**: `/vf/users/parks34/projects/2secactpy/CLAUDE.md`
- **Session Log**: `/vf/users/parks34/projects/2secactpy/cytoatlas-api/SESSION_LOG.md`
