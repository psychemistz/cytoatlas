#!/bin/bash
# Request an interactive session and run the API server
# Usage: ./run_api_interactive.sh

set -e

PORT="${PORT:-8000}"

echo "Requesting interactive session for CytoAtlas API..."
echo ""

# Request interactive session with sinteractive
sinteractive \
    --mem=32g \
    --cpus-per-task=4 \
    --time=8:00:00 \
    --tunnel \
    --gres=lscratch:10 \
    << 'EOF'

# Inside the interactive session
cd /data/parks34/projects/2cytoatlas/cytoatlas-api

# Activate environment
source ~/bin/myconda
conda activate secactpy

# Install dependencies
pip install -e . --quiet

# Print connection info
echo ""
echo "=========================================="
echo "  CytoAtlas API Server"
echo "=========================================="
echo "  Node: $(hostname)"
echo "  Port: $PORT"
echo ""
echo "  Access via tunnel or:"
echo "  http://$(hostname):$PORT/docs"
echo "=========================================="
echo ""

# Start server
python -m uvicorn app.main:app --host 0.0.0.0 --port $PORT --reload

EOF
