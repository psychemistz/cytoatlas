# CytoAtlas API

FastAPI server for the CytoAtlas single-cell cytokine activity visualization platform.

## Overview

This API provides REST endpoints for accessing computed cytokine and secreted protein activity signatures across three major human immune cell atlases:

- **CIMA**: 6.5M cells from healthy donors with rich phenotypic data (age, BMI, biochemistry, metabolites)
- **Inflammation Atlas**: 6.3M cells across 3 cohorts with disease comparison and treatment response prediction
- **scAtlas**: 6.4M cells from normal organs and cancer with tumor-adjacent tissue comparison

## Quick Start

### Option 1: Direct Python (Recommended for HPC)

```bash
cd /vf/users/parks34/projects/2secactpy/cytoatlas-api

# Activate conda environment
source ~/bin/myconda
conda activate secactpy

# Install dependencies
pip install -e .

# Copy environment file
cp .env.example .env

# Run development server
./scripts/run_server.sh

# Or manually:
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Option 2: SLURM Job (Production)

```bash
cd /vf/users/parks34/projects/2secactpy/cytoatlas-api

# Submit as batch job (runs for 7 days)
sbatch scripts/slurm/run_api.sh

# Check job status
squeue -u $USER

# View logs
tail -f logs/api_*.out

# Get connection info from log, then SSH tunnel:
ssh -L 8000:<node>:8000 biowulf

# Access at: http://localhost:8000/docs
```

### Option 3: Interactive Session

```bash
# Request interactive node with tunnel
sinteractive --mem=32g --cpus-per-task=4 --time=8:00:00 --tunnel

# Then run the server
cd /vf/users/parks34/projects/2secactpy/cytoatlas-api
source ~/bin/myconda && conda activate secactpy
pip install -e .
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Option 4: Singularity Container

```bash
cd /vf/users/parks34/projects/2secactpy/cytoatlas-api/singularity

# Build container (one time)
singularity build --fakeroot cytoatlas-api.sif cytoatlas-api.def

# Run with data mounts
singularity run \
    --bind /data/Jiang_Lab/Data/Seongyong:/data/Jiang_Lab/Data/Seongyong:ro \
    --bind /vf/users/parks34/projects/2secactpy/results:/vf/users/parks34/projects/2secactpy/results:ro \
    --bind /vf/users/parks34/projects/2secactpy/visualization/data:/vf/users/parks34/projects/2secactpy/visualization/data:ro \
    cytoatlas-api.sif
```

## API Endpoints

### Health & Metadata
- `GET /api/v1/health` - Health check
- `GET /api/v1/health/live` - Liveness probe
- `GET /api/v1/health/ready` - Readiness probe

### CIMA Atlas (~32 endpoints)
- `GET /api/v1/cima/summary` - Summary statistics
- `GET /api/v1/cima/cell-types` - Available cell types
- `GET /api/v1/cima/activity` - Cell type activity
- `GET /api/v1/cima/correlations/age` - Age correlations
- `GET /api/v1/cima/correlations/bmi` - BMI correlations
- `GET /api/v1/cima/correlations/biochemistry` - Biochemistry correlations
- `GET /api/v1/cima/correlations/metabolites` - Metabolite correlations
- `GET /api/v1/cima/differential` - Differential analysis
- `GET /api/v1/cima/boxplots/age/{signature}` - Age boxplot data
- `GET /api/v1/cima/boxplots/bmi/{signature}` - BMI boxplot data
- `GET /api/v1/cima/eqtl` - eQTL analysis
- `GET /api/v1/cima/heatmap/activity` - Activity heatmap

### Inflammation Atlas (~44 endpoints)
- `GET /api/v1/inflammation/summary` - Summary statistics
- `GET /api/v1/inflammation/diseases` - Available diseases
- `GET /api/v1/inflammation/disease-comparison` - Disease vs healthy
- `GET /api/v1/inflammation/treatment-response` - Treatment response summary
- `GET /api/v1/inflammation/treatment-response/roc` - ROC curves
- `GET /api/v1/inflammation/treatment-response/features` - Feature importance
- `GET /api/v1/inflammation/cohort-validation` - Cross-cohort validation
- `GET /api/v1/inflammation/celltype-stratified` - Cell type stratified analysis
- `GET /api/v1/inflammation/driving-populations` - Driving cell populations
- `GET /api/v1/inflammation/conserved-programs` - Conserved programs
- `GET /api/v1/inflammation/sankey` - Sankey diagram data

### scAtlas (~36 endpoints)
- `GET /api/v1/scatlas/summary` - Summary statistics
- `GET /api/v1/scatlas/organs` - Available organs
- `GET /api/v1/scatlas/organ-signatures` - Organ signatures
- `GET /api/v1/scatlas/celltype-signatures` - Cell type signatures
- `GET /api/v1/scatlas/cancer-comparison` - Tumor vs adjacent
- `GET /api/v1/scatlas/cancer-types-analysis` - Cancer type analysis
- `GET /api/v1/scatlas/immune-infiltration` - Immune infiltration
- `GET /api/v1/scatlas/exhaustion` - T cell exhaustion
- `GET /api/v1/scatlas/caf-signatures` - CAF signatures
- `GET /api/v1/scatlas/heatmap/organ` - Organ heatmap

### Cross-Atlas (~28 endpoints)
- `GET /api/v1/cross-atlas/atlases` - Available atlases
- `GET /api/v1/cross-atlas/comparison` - Atlas comparison
- `GET /api/v1/cross-atlas/correlations` - Cross-atlas correlations
- `GET /api/v1/cross-atlas/cell-type-mappings` - Cell type mappings
- `GET /api/v1/cross-atlas/conserved-signatures` - Conserved signatures
- `GET /api/v1/cross-atlas/meta-analysis` - Meta-analysis
- `GET /api/v1/cross-atlas/pathway-enrichment` - Pathway enrichment

### Validation (~30 endpoints)
- `GET /api/v1/validation/full-report` - Complete validation report
- `GET /api/v1/validation/summary` - Quality summary
- `GET /api/v1/validation/expression-vs-activity` - Expression-activity correlation
- `GET /api/v1/validation/gene-coverage` - Gene coverage analysis
- `GET /api/v1/validation/cv-stability` - CV stability
- `GET /api/v1/validation/biological-associations` - Biological validation
- `GET /api/v1/validation/compare-atlases` - Atlas comparison

### Data Export
- `GET /api/v1/export/cima/{type}` - Export CIMA data (CSV/JSON)
- `GET /api/v1/export/inflammation/{type}` - Export Inflammation data
- `GET /api/v1/export/scatlas/{type}` - Export scAtlas data

## Common Query Parameters

- `signature_type`: `"CytoSig"` (44 cytokines) or `"SecAct"` (1,249 secreted proteins)
- `cell_type`: Filter by cell type (e.g., `"CD4_T"`, `"Monocytes"`)
- `disease`: Filter by disease (Inflammation Atlas)
- `organ`: Filter by organ (scAtlas)

## Response Format

All responses follow a consistent structure:

```json
{
  "data": [...],
  "total": 100,
  "offset": 0,
  "limit": 100,
  "has_more": false
}
```

Error responses:
```json
{
  "success": false,
  "error": "Error message",
  "detail": "Additional details"
}
```

## Authentication

The API supports two authentication methods:

1. **JWT Bearer Token**: For user sessions
   ```
   Authorization: Bearer <token>
   ```

2. **API Key**: For programmatic access
   ```
   X-API-Key: <api_key>
   ```

Public endpoints (health, summary) don't require authentication.

## Rate Limiting

- Default: 100 requests per minute per IP
- Authenticated users: Based on user quota
- Headers returned: `X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset`

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test file
pytest tests/unit/test_services.py -v
```

### Database Migrations

```bash
# Create migration
alembic revision --autogenerate -m "Description"

# Apply migrations
alembic upgrade head

# Rollback
alembic downgrade -1
```

### Code Quality

```bash
# Lint
ruff check app tests

# Type check
mypy app
```

## Architecture

```
cytoatlas-api/
├── app/
│   ├── main.py           # FastAPI application factory
│   ├── config.py         # Pydantic settings
│   ├── core/             # Infrastructure (db, cache, auth)
│   ├── models/           # SQLAlchemy ORM models
│   ├── schemas/          # Pydantic request/response schemas
│   ├── services/         # Business logic layer
│   └── routers/          # API endpoints
├── tests/
│   ├── unit/            # Unit tests
│   ├── integration/     # Integration tests
│   └── e2e/             # End-to-end tests
├── alembic/              # Database migrations
├── singularity/          # Singularity container definition
└── scripts/              # Run scripts (SLURM, direct Python)
```

## Data Flow

1. **Request** → Router validates parameters
2. **Router** → Calls service method
3. **Service** → Loads from cache or JSON/CSV files
4. **Response** → Serialized via Pydantic schema

## Caching

| Data Type | Cache Layer | TTL |
|-----------|-------------|-----|
| Summary stats | Redis | 24h |
| Heatmaps | Redis | 1h |
| Filtered results | Redis | 5min |

## License

MIT
