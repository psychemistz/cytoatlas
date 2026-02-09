# CytoAtlas Project Overview

Brief overview of the Pan-Disease Single-Cell Cytokine Activity Atlas project.

**For comprehensive architecture details, see [ARCHITECTURE.md](ARCHITECTURE.md).**

## What is CytoAtlas?

Pan-Disease Single-Cell Cytokine Activity Atlas - computes cytokine and secreted protein activity signatures across **12+ million human immune cells** from three major single-cell atlases:

| Atlas | Cells | Focus |
|-------|-------|-------|
| CIMA | 6.5M | Healthy aging, metabolism |
| Inflammation | 6.3M | Disease activity, treatment response |
| scAtlas | 6.4M | Organ signatures, cancer comparison |

## Quick Facts

- **Total cells analyzed**: 17M+
- **REST API endpoints**: 188+ across 14 routers
- **Signature types**: CytoSig (44), LinCytoSig (178), SecAct (1,249)
- **Web UI pages**: 8 interactive pages with Plotly + D3.js
- **Analysis scripts**: 7 Python pipelines + 5 SLURM batch jobs

## Key Components

1. **Analysis Pipeline** (`scripts/00-07_*.py`)
   - GPU-accelerated activity inference (CuPy)
   - Pseudo-bulk aggregation (sample × cell type)
   - Ridge regression against signature matrices

2. **REST API** (`cytoatlas-api/`)
   - FastAPI backend with 188+ endpoints
   - JSON-based data access with caching
   - Optional PostgreSQL + Redis for scaling

3. **Web Portal** (`cytoatlas-api/static/`)
   - Single-Page Application (8 pages)
   - Interactive visualizations (Plotly, D3.js)
   - Real-time search and chat integration

4. **Validation System**
   - 5-type credibility assessment
   - 175-336MB validation data per atlas
   - Quality metrics dashboard

## High-Level Data Flow

```
Raw H5AD (282GB)
    ↓
Python Scripts (GPU acceleration)
    ↓
Results CSV + H5AD
    ↓
Preprocessing (JSON generation)
    ↓
visualization/data/ (30+ JSON files)
    ↓
REST API (caching layer)
    ↓
Web Dashboard (visualization)
```

## Getting Started

### View Dashboard Locally

```bash
cd /vf/users/parks34/projects/2secactpy
cd cytoatlas-api
pip install -e .
uvicorn app.main:app --reload
# Open http://localhost:8000/
```

### Run Analysis Pipeline

```bash
cd /data/parks34/projects/2secactpy
python scripts/00_pilot_analysis.py --n-cells 100000
python scripts/01_cima_activity.py --mode pseudobulk
```

### Submit SLURM Job (HPC)

```bash
sbatch scripts/slurm/run_all.sh           # Full pipeline
sbatch scripts/slurm/run_all.sh --pilot   # Pilot only
```

## Documentation Map

| Document | Purpose |
|----------|---------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | **Start here**: Comprehensive system architecture |
| [CLAUDE.md](../CLAUDE.md) | Quick reference, TODOs, lessons learned |
| [datasets/README.md](datasets/README.md) | Data source specifications |
| [pipelines/README.md](pipelines/README.md) | Analysis pipeline details |
| [outputs/README.md](outputs/README.md) | Output file structure |
| [decisions/](decisions/) | Architecture Decision Records (ADRs) |
| [archive/](archive/) | Archived plans from earlier phases |

## Key Resources

- **Activity Method**: Ridge regression (z-score activities, not log2FC)
- **Signature Matrices**: CytoSig (44 cytokines) + SecAct (1,249 secreted proteins)
- **HPC Cluster**: NIH Biowulf (SLURM)
- **GPU Acceleration**: CuPy with NumPy fallback
- **Environment Setup**: `conda activate secactpy`

---

**For detailed architecture, design decisions, and implementation details, see [ARCHITECTURE.md](ARCHITECTURE.md).**
