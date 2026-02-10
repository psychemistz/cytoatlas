#!/bin/bash
# Run CytoAtlas API server with Singularity

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SINGULARITY_DIR="$PROJECT_DIR/singularity"
SIF_FILE="$SINGULARITY_DIR/cytoatlas-api.sif"

PORT="${PORT:-8000}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  CytoAtlas API (Singularity)          ${NC}"
echo -e "${GREEN}========================================${NC}"

# Check if Singularity is available
if ! command -v singularity &> /dev/null; then
    echo -e "${RED}Error: Singularity is not installed or not in PATH${NC}"
    exit 1
fi

# Build if SIF doesn't exist
if [[ ! -f "$SIF_FILE" ]]; then
    echo -e "${YELLOW}Building Singularity image...${NC}"
    cd "$SINGULARITY_DIR"
    singularity build --fakeroot cytoatlas-api.sif cytoatlas-api.def
fi

echo -e "Using image: $SIF_FILE"
echo -e "Port: $PORT"
echo ""

# Run with bind mounts for data access
exec singularity run \
    --bind /data/Jiang_Lab/Data/Seongyong:/data/Jiang_Lab/Data/Seongyong:ro \
    --bind /data/parks34/projects/2cytoatlas/results:/data/parks34/projects/2cytoatlas/results:ro \
    --bind /data/parks34/projects/2cytoatlas/visualization/data:/data/parks34/projects/2cytoatlas/visualization/data:ro \
    --env PORT="$PORT" \
    --env VIZ_DATA_PATH=/data/parks34/projects/2cytoatlas/visualization/data \
    --env RESULTS_BASE_PATH=/data/parks34/projects/2cytoatlas/results \
    "$SIF_FILE"
