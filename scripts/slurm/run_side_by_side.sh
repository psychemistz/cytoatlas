#!/bin/bash
#SBATCH --job-name=cytoatlas-side-by-side
#SBATCH --partition=norm
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --output=logs/side_by_side_%j.log

set -e

source ~/bin/myconda
conda activate secactpy

# ── Configuration ──
OLD_PORT=8001
NEW_PORT=8000
PROJECT_DIR="/vf/users/parks34/projects/2cytoatlas"
WORKTREE_DIR="/tmp/cytoatlas-old-${SLURM_JOB_ID}"
# Last commit before React was added — fully working vanilla JS/CSS app
# with correct project paths and no str() bug (DuckDB works)
OLD_COMMIT="41dadad"

echo "=========================================="
echo "  CytoAtlas: Side-by-Side Comparison      "
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "Start:  $(date)"
echo ""

# ── 1. Create git worktree for the old version ──
echo "[$(date)] Creating git worktree at ${OLD_COMMIT} ..."
cd "$PROJECT_DIR"
git worktree add "$WORKTREE_DIR" "$OLD_COMMIT" --detach 2>/dev/null || {
    echo "[$(date)] Worktree already exists, resetting ..."
    git worktree remove "$WORKTREE_DIR" --force 2>/dev/null
    git worktree add "$WORKTREE_DIR" "$OLD_COMMIT" --detach
}

# Copy .env so the old version picks up current data paths
cp "$PROJECT_DIR/cytoatlas-api/.env" "$WORKTREE_DIR/cytoatlas-api/.env"

echo "[$(date)] Worktree created at: $WORKTREE_DIR"

# ── 2. Start OLD CytoAtlas (vanilla JS/CSS) on port $OLD_PORT ──
echo "[$(date)] Starting OLD CytoAtlas (vanilla JS/CSS) on port $OLD_PORT ..."

export DEBUG=false
export ENVIRONMENT=production
export VIZ_DATA_PATH=/data/parks34/projects/2cytoatlas/visualization/data
export RESULTS_BASE_PATH=/data/parks34/projects/2cytoatlas/results

cd "$WORKTREE_DIR/cytoatlas-api"
python -m uvicorn app.main:app \
    --host 0.0.0.0 \
    --port "$OLD_PORT" \
    --workers 2 \
    --log-level warning &
OLD_PID=$!

# ── 3. Start NEW CytoAtlas (React) on port $NEW_PORT ──
echo "[$(date)] Starting NEW CytoAtlas (React) on port $NEW_PORT ..."

cd "$PROJECT_DIR/cytoatlas-api"
python -m uvicorn app.main:app \
    --host 0.0.0.0 \
    --port "$NEW_PORT" \
    --workers 2 \
    --log-level warning &
NEW_PID=$!

# ── 4. Wait for both to come up ──
echo "[$(date)] Waiting for servers to start ..."
sleep 5

for PORT in $OLD_PORT $NEW_PORT; do
    for i in $(seq 1 30); do
        if curl -sf "http://localhost:$PORT/api/v1/health" > /dev/null 2>&1; then
            echo "[$(date)] Port $PORT is healthy (attempt $i)"
            break
        fi
        sleep 2
    done
done

# ── 5. Print connection info ──
echo ""
echo "=========================================="
echo "  SERVERS RUNNING"
echo "=========================================="
echo ""
echo "  OLD (vanilla JS/CSS):  http://${SLURM_NODELIST}:${OLD_PORT}"
echo "  NEW (React):           http://${SLURM_NODELIST}:${NEW_PORT}"
echo ""
echo "  SSH tunnel command:"
echo "    ssh -L ${NEW_PORT}:${SLURM_NODELIST}:${NEW_PORT} -L ${OLD_PORT}:${SLURM_NODELIST}:${OLD_PORT} biowulf"
echo ""
echo "  Then open in browser:"
echo "    OLD: http://localhost:${OLD_PORT}"
echo "    NEW: http://localhost:${NEW_PORT}"
echo ""
echo "  Health checks:"
echo "    OLD: http://localhost:${OLD_PORT}/api/v1/health"
echo "    NEW: http://localhost:${NEW_PORT}/api/v1/health"
echo ""
echo "=========================================="
echo "  Ctrl+C or scancel $SLURM_JOB_ID to stop"
echo "=========================================="
echo ""

# ── 6. Wait for either process to exit ──
cleanup() {
    echo ""
    echo "[$(date)] Shutting down ..."
    kill $OLD_PID 2>/dev/null
    kill $NEW_PID 2>/dev/null
    echo "[$(date)] Removing worktree ..."
    cd "$PROJECT_DIR"
    git worktree remove "$WORKTREE_DIR" --force 2>/dev/null
    wait 2>/dev/null
    echo "[$(date)] Done."
}
trap cleanup EXIT SIGTERM SIGINT

wait -n $OLD_PID $NEW_PID
echo "[$(date)] A process exited, shutting down ..."
