#!/bin/bash
# Run CytoAtlas API server directly with Python (no container)

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
WORKERS="${WORKERS:-1}"
LOG_LEVEL="${LOG_LEVEL:-info}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  CytoAtlas API Server                 ${NC}"
echo -e "${GREEN}========================================${NC}"

# Check if conda environment is active
if [[ -z "$CONDA_DEFAULT_ENV" ]]; then
    echo -e "${YELLOW}Activating conda environment...${NC}"
    source ~/bin/myconda
    conda activate secactpy
fi

echo -e "Python: $(which python)"
echo -e "Environment: ${CONDA_DEFAULT_ENV:-system}"

# Change to project directory
cd "$PROJECT_DIR"

# Check if dependencies are installed
if ! python -c "import fastapi" 2>/dev/null; then
    echo -e "${YELLOW}Installing dependencies...${NC}"
    pip install -e .
fi

# Create .env file if it doesn't exist
if [[ ! -f .env ]]; then
    echo -e "${YELLOW}Creating .env file from HPC template...${NC}"
    if [[ -f .env.hpc ]]; then
        cp .env.hpc .env
    else
        cp .env.example .env
    fi
    echo -e "${YELLOW}Please edit .env with your settings${NC}"
fi

# Export environment variables from .env (handle quotes properly)
if [[ -f .env ]]; then
    while IFS= read -r line || [[ -n "$line" ]]; do
        # Skip comments and empty lines
        [[ "$line" =~ ^#.*$ ]] && continue
        [[ -z "$line" ]] && continue
        # Export the variable
        export "$line" 2>/dev/null || true
    done < .env
fi

# For HPC without database, disable it
if [[ -z "$DATABASE_URL" ]]; then
    echo -e "${YELLOW}Note: Running without database (no persistence)${NC}"
fi

if [[ -z "$REDIS_URL" ]]; then
    echo -e "${YELLOW}Note: Running without Redis (using in-memory cache)${NC}"
fi

# Set defaults only if not already set
# Note: HPC may set ENVIRONMENT=BATCH, which config.py normalizes to "production"
if [[ -z "$DEBUG" ]]; then
    export DEBUG=false
fi
if [[ -z "$ENVIRONMENT" ]]; then
    export ENVIRONMENT=development
fi

echo ""
echo -e "Starting server on ${HOST}:${PORT}"
echo -e "Workers: ${WORKERS}"
echo -e "Log level: ${LOG_LEVEL}"
echo -e "Debug: ${DEBUG}"
echo ""

# Start the server
if [[ "$DEBUG" == "true" ]]; then
    # Development mode with auto-reload
    exec python -m uvicorn app.main:app \
        --host "$HOST" \
        --port "$PORT" \
        --reload \
        --log-level "$LOG_LEVEL"
else
    # Production mode
    exec python -m uvicorn app.main:app \
        --host "$HOST" \
        --port "$PORT" \
        --workers "$WORKERS" \
        --log-level "$LOG_LEVEL"
fi
