#!/bin/bash
#SBATCH --job-name=cytoatlas-vllm
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=12
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/vllm_%j.log

set -e

module load CUDA/12.8.1 cuDNN

export HF_HOME=/data/parks34/.cache/huggingface

source ~/bin/myconda
conda activate secactpy

# ── Configuration ──
VLLM_PORT=8001
API_PORT="${PORT:-8000}"
API_WORKERS="${WORKERS:-4}"
API_DIR="/vf/users/parks34/projects/2secactpy/cytoatlas-api"

echo "=========================================="
echo "  CytoAtlas: vLLM + API Server (SLURM)   "
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo ""

# ── 1. Start vLLM in background ──
echo "[$(date)] Starting vLLM server on port $VLLM_PORT ..."
python -m vllm.entrypoints.openai.api_server \
    --model mistralai/Mistral-Small-3.1-24B-Instruct-2503 \
    --host 0.0.0.0 \
    --port $VLLM_PORT \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 32768 \
    --enable-auto-tool-choice \
    --tool-call-parser mistral \
    --enable-prefix-caching &
VLLM_PID=$!

# ── 2. Wait for vLLM to become healthy ──
echo "[$(date)] Waiting for vLLM health check ..."
for i in $(seq 1 120); do
    if curl -sf http://localhost:$VLLM_PORT/health > /dev/null 2>&1; then
        echo "[$(date)] vLLM is healthy (attempt $i)"
        break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "[$(date)] ERROR: vLLM process died"
        exit 1
    fi
    sleep 5
done

if ! curl -sf http://localhost:$VLLM_PORT/health > /dev/null 2>&1; then
    echo "[$(date)] ERROR: vLLM did not become healthy in 10 minutes"
    kill $VLLM_PID 2>/dev/null
    exit 1
fi

# ── 3. Start API server ──
echo "[$(date)] Starting CytoAtlas API on port $API_PORT ..."

export DEBUG=false
export ENVIRONMENT=production
export VIZ_DATA_PATH=/vf/users/parks34/projects/2secactpy/visualization/data
export RESULTS_BASE_PATH=/vf/users/parks34/projects/2secactpy/results
export H5AD_BASE_PATH=/data/Jiang_Lab/Data/Seongyong
export LLM_BASE_URL=http://localhost:$VLLM_PORT/v1

cd "$API_DIR"
pip install -e . --quiet

NODE_IP=$(hostname -i)
echo "=========================================="
echo "  vLLM:  http://localhost:$VLLM_PORT"
echo "  API:   http://${SLURM_NODELIST}:$API_PORT"
echo "=========================================="
echo ""
echo "SSH tunnel:"
echo "  ssh -L ${API_PORT}:${SLURM_NODELIST}:${API_PORT} biowulf"
echo "  Then: http://localhost:${API_PORT}/docs"
echo ""

python -m uvicorn app.main:app \
    --host 0.0.0.0 \
    --port "$API_PORT" \
    --workers "$API_WORKERS" \
    --log-level info &
API_PID=$!

# ── 4. Wait for either process to exit ──
cleanup() {
    echo "[$(date)] Shutting down ..."
    kill $API_PID 2>/dev/null
    kill $VLLM_PID 2>/dev/null
    wait
}
trap cleanup EXIT SIGTERM SIGINT

wait -n $VLLM_PID $API_PID
echo "[$(date)] A process exited, shutting down ..."
