#!/bin/bash
#SBATCH --job-name=cytoatlas-api
#SBATCH --partition=norm
#SBATCH --time=7-00:00:00
#SBATCH --mem=32g
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/api_%j.out
#SBATCH --error=logs/api_%j.err

# CytoAtlas API SLURM job script
# Runs the FastAPI server on a compute node

set -e

# Configuration
PORT="${PORT:-8000}"
WORKERS="${WORKERS:-4}"
PROJECT_DIR="/data/parks34/projects/2cytoatlas/cytoatlas-api"

echo "=========================================="
echo "  CytoAtlas API Server (SLURM)           "
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE"
echo ""

# Create logs directory
mkdir -p "$PROJECT_DIR/logs"

# Activate conda environment
source ~/bin/myconda
conda activate secactpy

# Change to project directory
cd "$PROJECT_DIR"

# Install/update dependencies if needed
pip install -e . --quiet

# Get node hostname for connection info
NODE_IP=$(hostname -i)
echo "=========================================="
echo "  Server starting on:"
echo "  http://${NODE_IP}:${PORT}"
echo "  http://${SLURM_NODELIST}:${PORT}"
echo "=========================================="
echo ""
echo "To connect from login node, use SSH tunnel:"
echo "  ssh -L ${PORT}:${SLURM_NODELIST}:${PORT} biowulf"
echo ""
echo "Then access: http://localhost:${PORT}/docs"
echo ""

# Export environment variables
export DEBUG=false
export ENVIRONMENT=production
export VIZ_DATA_PATH=/data/parks34/projects/2cytoatlas/visualization/data
export RESULTS_BASE_PATH=/data/parks34/projects/2cytoatlas/results
export H5AD_BASE_PATH=/data/Jiang_Lab/Data/Seongyong
export LLM_BASE_URL=http://cn0084:8001/v1

# Start the server
exec python -m uvicorn app.main:app \
    --host 0.0.0.0 \
    --port "$PORT" \
    --workers "$WORKERS" \
    --log-level info
