#!/bin/bash
#SBATCH --job-name=vllm-mistral
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=8:00:00
#SBATCH --output=logs/vllm_server_%j.log

set -e

source ~/bin/myconda
conda activate secactpy

export HF_HUB_CACHE="/data/parks34/.cache/huggingface/hub"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

MODEL="mistralai/Mistral-Small-3.1-24B-Instruct-2503"
PORT=8001
HOST=0.0.0.0

echo "=========================================="
echo "  vLLM Server: Mistral Small 3.1 24B"
echo "=========================================="
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $SLURM_NODELIST"
echo "GPU:     $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Model:   $MODEL"
echo "Port:    $PORT"
echo "Start:   $(date)"
echo ""
echo "Connect with:"
echo "  ssh -L ${PORT}:${SLURM_NODELIST}:${PORT} biowulf"
echo ""
echo "Test with:"
echo "  curl http://localhost:${PORT}/v1/models"
echo ""
echo "=========================================="
echo ""

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --host "$HOST" \
    --port "$PORT" \
    --max-model-len 16384 \
    --enable-auto-tool-choice \
    --tool-call-parser mistral \
    --chat-template /data/parks34/.cache/huggingface/hub/mistral_chat_template.jinja \
    --gpu-memory-utilization 0.90 \
    --dtype auto
