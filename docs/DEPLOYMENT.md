# CytoAtlas Deployment Guide

Complete guide for deploying CytoAtlas in development, HPC, and production environments.

**Last Updated**: 2026-02-09

---

## 1. Quick Start

### Development (Local)

```bash
# Clone and setup
cd /vf/users/parks34/projects/2secactpy/cytoatlas-api
pip install -e .

# Run API server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Open dashboard
open http://localhost:8000/
```

### HPC (SLURM)

```bash
# Submit unified vLLM + API job
cd /vf/users/parks34/projects/2secactpy
sbatch scripts/slurm/run_vllm.sh

# Check job status
squeue -j $JOB_ID

# View output
tail -f logs/vllm_$JOB_ID.log
```

---

## 2. Development Environment Setup

### Prerequisites

```bash
# Conda environment
source ~/bin/myconda
conda activate secactpy

# Verify Python
python --version  # Should be 3.10+
```

### Installation (No Database Mode)

```bash
cd /vf/users/parks34/projects/2secactpy/cytoatlas-api

# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Verify installation
python -c "from app.main import app; print('OK')"
```

### Running with Mock Data (No H5AD Required)

Set environment to skip data file validation:

```bash
export ENVIRONMENT=development
export DEBUG=true
export VIZ_DATA_PATH=/vf/users/parks34/projects/2secactpy/visualization/data
export RESULTS_BASE_PATH=/vf/users/parks34/projects/2secactpy/results

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

API will start even if H5AD/CSV files are unavailable (graceful degradation).

### Using In-Memory Cache (No Redis)

```bash
# Leave REDIS_URL unset - uses in-memory dict fallback
export REDIS_URL=
uvicorn app.main:app --port 8000
```

Cache behavior: Persists within process lifetime, resets on server restart.

---

## 3. HPC/SLURM Deployment

### Unified vLLM + API Job

The `scripts/slurm/run_vllm.sh` script orchestrates:

1. **vLLM GPU Server** (port 8001): Runs Mistral-Small-3.1-24B for chat
2. **Uvicorn API** (port 8000): FastAPI with 4 workers
3. **Health Check Orchestration**: Monitors both processes

### Configuration

Edit `/vf/users/parks34/projects/2secactpy/scripts/slurm/run_vllm.sh`:

```bash
# GPU allocation
#SBATCH --gres=gpu:a100:1          # Change to a100:2, h100:1, etc.

# Memory and CPU
#SBATCH --mem=96G                  # Reduce for smaller GPUs
#SBATCH --cpus-per-task=12         # Adjust based on node capacity

# Time limit
#SBATCH --time=7-00:00:00          # 7 days - adjust as needed

# Port configuration
VLLM_PORT=8001
API_PORT="${PORT:-8000}"            # Override with PORT env var
API_WORKERS="${WORKERS:-4}"         # Override with WORKERS env var
```

### Submission

```bash
# Standard 7-day job
sbatch scripts/slurm/run_vllm.sh

# Custom duration
sbatch --time=24:00:00 scripts/slurm/run_vllm.sh

# Custom ports
sbatch -e "PORT=8080,WORKERS=2" scripts/slurm/run_vllm.sh

# Get job ID from output
JOB_ID=$(sbatch scripts/slurm/run_vllm.sh | awk '{print $NF}')
echo $JOB_ID
```

### Accessing the API

#### SSH Tunnel

```bash
# Establish tunnel (from your local machine)
ssh -L 8000:node001.biowulf.helix.nih.gov:8000 biowulf

# Then access locally
curl http://localhost:8000/api/v1/health
open http://localhost:8000/docs
```

#### Direct Connection (From Biowulf)

```bash
# Check node and port from logs
scontrol show job $JOB_ID | grep "NodeList"
tail -f logs/vllm_$JOB_ID.log | grep "API:"

# Direct access from Biowulf login node
curl http://node001:8000/api/v1/health
```

### Health Verification

```bash
# Check if API is ready
curl -s http://localhost:8000/api/v1/health | jq '.'

# Expected response
{
  "status": "healthy",
  "version": "0.1.0",
  "database": "not configured",
  "cache": "in-memory",
  "environment": "production"
}

# Check readiness probe
curl -s http://localhost:8000/api/v1/health/ready | jq '.'

# Check metrics
curl -s http://localhost:8000/api/v1/health/metrics | jq '.'
```

### Monitoring

```bash
# Watch live log
tail -f logs/vllm_$JOB_ID.log

# Check vLLM health specifically
curl -sf http://localhost:8001/health

# Check API health specifically
curl -sf http://localhost:8000/api/v1/health/ready
```

### Shutting Down

```bash
# Cancel job
scancel $JOB_ID

# Graceful shutdown (vLLM + API will cleanup)
kill -SIGTERM $JOB_ID
```

---

## 4. Environment Variables Reference

All settings from `app/config.py`:

### Application

```bash
# Basic app settings
APP_NAME="CytoAtlas API"
APP_VERSION="0.1.0"
ENVIRONMENT="development"          # development, staging, production
DEBUG=false                         # Enable debug mode (verbose logging)
```

### API Configuration

```bash
# API settings
API_V1_PREFIX="/api/v1"
ALLOWED_ORIGINS="http://localhost:8000,http://localhost:3000"
MAX_REQUEST_BODY_MB=100
```

### Data Paths

```bash
# Base paths (auto-configured in HPC scripts)
H5AD_BASE_PATH="/data/Jiang_Lab/Data/Seongyong"
RESULTS_BASE_PATH="/vf/users/parks34/projects/2secactpy/results"
VIZ_DATA_PATH="/vf/users/parks34/projects/2secactpy/visualization/data"

# CIMA paths
CIMA_H5AD="/data/Jiang_Lab/Data/Seongyong/CIMA/Cell_Atlas/CIMA_RNA_6484974cells_36326genes_compressed.h5ad"
CIMA_BIOCHEM="/data/Jiang_Lab/Data/Seongyong/CIMA/Cell_Atlas/CIMA_Sample_Blood_Biochemistry_Results.csv"
CIMA_METABOLITES="/data/Jiang_Lab/Data/Seongyong/CIMA/Cell_Atlas/CIMA_Sample_Plasma_Metabolites_and_Lipids_Results.csv"

# Inflammation Atlas paths
INFLAMMATION_MAIN_H5AD="/data/Jiang_Lab/Data/Seongyong/Inflammation_Atlas/INFLAMMATION_ATLAS_main_afterQC.h5ad"
INFLAMMATION_VALIDATION_H5AD="/data/Jiang_Lab/Data/Seongyong/Inflammation_Atlas/INFLAMMATION_ATLAS_validation_afterQC.h5ad"
INFLAMMATION_EXTERNAL_H5AD="/data/Jiang_Lab/Data/Seongyong/Inflammation_Atlas/INFLAMMATION_ATLAS_external_afterQC.h5ad"

# scAtlas paths
SCATLAS_NORMAL_H5AD="/data/Jiang_Lab/Data/Seongyong/scAtlas_2025/igt_s9_fine_counts.h5ad"
SCATLAS_CANCER_H5AD="/data/Jiang_Lab/Data/Seongyong/scAtlas_2025/PanCancer_igt_s9_fine_counts.h5ad"
```

### Database (Optional)

```bash
# Leave empty for no database
DATABASE_URL=                      # e.g., postgresql+asyncpg://user:pass@host/db
DATABASE_POOL_SIZE=5
DATABASE_MAX_OVERFLOW=10
DATABASE_POOL_TIMEOUT=30

# To use database (production)
DATABASE_URL="postgresql+asyncpg://cytoatlas:password@db.example.com/cytoatlas"
```

### Cache (Optional)

```bash
# Leave empty for in-memory cache fallback
REDIS_URL=                         # e.g., redis://localhost:6379/0
REDIS_CACHE_TTL=3600              # 1 hour default
```

### Authentication

```bash
# Security
SECRET_KEY=                         # Required for JWT (generate: openssl rand -hex 32)
ALGORITHM="HS256"
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Authentication enforcement
REQUIRE_AUTH=false                 # Set to true in production
AUDIT_ENABLED=true
AUDIT_LOG_PATH="logs/audit.jsonl"

# API key header
API_KEY_HEADER="X-API-Key"
```

### Rate Limiting

```bash
RATE_LIMIT_REQUESTS=100            # Requests per window
RATE_LIMIT_WINDOW=60               # Seconds
ANON_CHAT_LIMIT_PER_DAY=5          # Anonymous chat messages/day
AUTH_CHAT_LIMIT_PER_DAY=1000       # Authenticated chat messages/day
```

### LLM Configuration

```bash
# vLLM (primary - OpenAI-compatible)
LLM_BASE_URL="http://localhost:8001/v1"
LLM_API_KEY="not-needed"            # vLLM doesn't require auth
CHAT_MODEL="mistralai/Mistral-Small-3.1-24B-Instruct-2503"
CHAT_MAX_TOKENS=4096

# Anthropic fallback (if vLLM unavailable)
ANTHROPIC_API_KEY=                 # Set for Claude fallback
ANTHROPIC_CHAT_MODEL="claude-sonnet-4-5-20250929"
```

### RAG Configuration

```bash
RAG_ENABLED=true
RAG_DB_PATH="rag_db"               # Local semantic database
RAG_EMBEDDING_MODEL="all-MiniLM-L6-v2"
RAG_TOP_K=5                        # Top K documents to retrieve
```

### File Upload

```bash
MAX_UPLOAD_SIZE_GB=50
UPLOAD_DIR="/data/cytoatlas/uploads"
```

### Celery (Async Tasks)

```bash
CELERY_BROKER_URL="redis://localhost:6379/1"
CELERY_RESULT_BACKEND="redis://localhost:6379/2"
```

---

## 5. Production Deployment

### Docker Deployment

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY cytoatlas-api/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY cytoatlas-api /app

# Install package
RUN pip install -e .

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Run app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose

```yaml
version: '3.8'

services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: cytoatlas
      POSTGRES_USER: cytoatlas
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  api:
    build: ./cytoatlas-api
    ports:
      - "8000:8000"
    environment:
      ENVIRONMENT: production
      DATABASE_URL: postgresql+asyncpg://cytoatlas:${DB_PASSWORD}@postgres/cytoatlas
      REDIS_URL: redis://redis:6379/0
      SECRET_KEY: ${SECRET_KEY}
    depends_on:
      - redis
      - postgres
    volumes:
      - /vf/users/parks34/projects/2secactpy/visualization/data:/data/viz:ro
      - /vf/users/parks34/projects/2secactpy/results:/data/results:ro

volumes:
  redis_data:
  postgres_data:
```

### Nginx Reverse Proxy

```nginx
upstream api_backend {
    server localhost:8000;
}

server {
    listen 443 ssl http2;
    server_name cytoatlas.example.com;

    ssl_certificate /etc/ssl/certs/cytoatlas.crt;
    ssl_certificate_key /etc/ssl/private/cytoatlas.key;

    # Security headers
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-XSS-Protection "1; mode=block" always;

    # Gzip compression
    gzip on;
    gzip_types application/json application/javascript text/plain;

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api_limit:10m rate=100r/s;
    limit_req zone=api_limit burst=200 nodelay;

    location / {
        proxy_pass http://api_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }

    # WebSocket support
    location /ws {
        proxy_pass http://api_backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

---

## 6. Troubleshooting

### API Won't Start

**Problem**: "Address already in use"

```bash
# Find process using port
lsof -i :8000

# Kill process
kill -9 <PID>

# Or use different port
uvicorn app.main:app --port 8001
```

**Problem**: "ImportError: No module named 'app'"

```bash
# Install in editable mode
cd cytoatlas-api
pip install -e .
```

### vLLM Health Check Timeout

**Problem**: "vLLM did not become healthy in 10 minutes"

```bash
# Check vLLM logs
tail -f logs/vllm_$JOB_ID.log

# Common causes:
# - GPU memory insufficient (reduce gpu_memory_utilization)
# - Model download timeout (check HuggingFace connectivity)
# - Port already in use

# Fix: Increase health check timeout in run_vllm.sh
for i in $(seq 1 300); do  # Changed from 120 to 300 (25 minutes)
    if curl -sf http://localhost:$VLLM_PORT/health > /dev/null 2>&1; then
        break
    fi
    sleep 5
done
```

### Cache/Redis Issues

**Problem**: "Redis connection refused"

```bash
# Check if Redis is running
redis-cli ping

# Start Redis
redis-server --port 6379

# Or disable Redis (use in-memory)
export REDIS_URL=
```

### JSON Data Not Loading

**Problem**: "FileNotFoundError: visualization/data/*.json"

```bash
# Verify paths exist
ls -la /vf/users/parks34/projects/2secactpy/visualization/data/

# Set correct path
export VIZ_DATA_PATH=/vf/users/parks34/projects/2secactpy/visualization/data

# Check if data files were generated
python scripts/06_preprocess_viz_data.py
```

### Database Connection Error

**Problem**: "Connection refused" or "Authentication failed"

```bash
# Check if database is running
psql -h localhost -U cytoatlas -d cytoatlas -c "SELECT 1"

# If using Docker
docker ps | grep postgres

# Check connection string
echo $DATABASE_URL

# To disable database
export DATABASE_URL=
```

### High Memory Usage

**Problem**: "MemoryError" or "killed by OOM"

```bash
# Check memory usage
ps aux | grep uvicorn

# Reduce workers
export WORKERS=2

# Reduce cache TTL
export REDIS_CACHE_TTL=1800  # 30 minutes instead of 60

# Enable Parquet backend (future)
# Currently uses JSON - migrate to Parquet for large files
```

---

## 7. Performance Optimization

### Cache Hit Rate

Monitor in logs:

```bash
curl -s http://localhost:8000/api/v1/health/metrics | jq '.cache'
```

Target: >80% hit rate

**Optimization**:
- Increase cache TTL: `REDIS_CACHE_TTL=7200` (2 hours)
- Pre-warm cache on startup
- Migrate large JSON to Parquet (see ADR-001)

### Response Time

```bash
# Test endpoint latency
time curl -s http://localhost:8000/api/v1/cima/summary > /dev/null

# Check Prometheus metrics (future)
curl -s http://localhost:8000/metrics | grep http_request_duration
```

Target: <200ms p95

**Optimization**:
- Add read-only PostgreSQL replica for queries
- Use Parquet with PyArrow predicate pushdown
- Implement query result caching

### Throughput

```bash
# Test with Apache Bench
ab -n 1000 -c 100 http://localhost:8000/api/v1/health

# Or with wrk
wrk -t4 -c100 -d30s http://localhost:8000/api/v1/health
```

Target: >1000 req/s

**Optimization**:
- Increase worker processes: `--workers 8` (2x CPU cores)
- Use PyPy for fast startup
- Enable HTTP/2

---

## 8. Monitoring & Logging

### API Logs

```bash
# Access logs (Uvicorn)
tail -f logs/api.log

# Structured audit logs
tail -f logs/audit.jsonl | jq '.'

# Example audit entry
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

### Metrics

```bash
# Get summary metrics
curl -s http://localhost:8000/api/v1/health/metrics | jq '.'

# Contains
{
  "total_requests": 1542,
  "total_errors": 3,
  "cache_hits": 1200,
  "cache_misses": 342,
  "avg_response_time_ms": 145
}
```

### vLLM Logs

```bash
# vLLM server logs (during HPC job)
tail -f logs/vllm_$JOB_ID.log

# Check model loading
grep "Loading model" logs/vllm_$JOB_ID.log

# Check inference requests
grep "Processing request" logs/vllm_$JOB_ID.log
```

---

## 9. Backup & Recovery

### Configuration Backup

```bash
# Backup .env file
cp cytoatlas-api/.env cytoatlas-api/.env.backup-$(date +%s)

# Backup database
pg_dump -h localhost -U cytoatlas cytoatlas > cytoatlas-$(date +%Y%m%d).sql
```

### Data Recovery

```bash
# Restore from database backup
psql -h localhost -U cytoatlas cytoatlas < cytoatlas-20260209.sql

# Clear cache (fresh data load)
redis-cli FLUSHDB

# Regenerate visualization JSON
python scripts/06_preprocess_viz_data.py
```

---

## Next Steps

1. **Development**: Follow "Quick Start" → "Development Environment Setup"
2. **HPC Testing**: Follow "HPC/SLURM Deployment" with 1-day time limit
3. **Production**: Follow "Production Deployment" with Docker/K8s
4. **Security**: Set `SECRET_KEY`, enable `REQUIRE_AUTH`, configure SSL/TLS
5. **Monitoring**: Enable Prometheus metrics and alerting

For questions, see [CLAUDE.md](CLAUDE.md) or [API_REFERENCE.md](API_REFERENCE.md).
