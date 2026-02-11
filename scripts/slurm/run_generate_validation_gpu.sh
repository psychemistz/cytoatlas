#!/bin/bash
#SBATCH --job-name=gen_val_gpu_%a
#SBATCH --output=logs/gen_validation_gpu_%A_%a.out
#SBATCH --error=logs/gen_validation_gpu_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=120G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-2

# Pan-Disease Cytokine Activity Atlas
# GPU-accelerated validation data generation using batch processing
# with streaming H5AD reads.
#
# Key optimizations over CPU version (run_generate_validation.sh):
#   1. Row-batch streaming: reads all signature columns in one pass
#      instead of n_signatures * 2 column reads (huge I/O improvement)
#   2. GPU-accelerated stats via CuPy when available
#   3. More memory on GPU nodes (350GB vs 200GB), handles CIMA without OOM
#   4. Larger batch sizes (1M cells) for better throughput
#
# Usage: sbatch scripts/slurm/run_generate_validation_gpu.sh
# Outputs: visualization/data/validation/{atlas}_validation.json

set -e

ATLASES=(cima inflammation scatlas)
ATLAS=${ATLASES[$SLURM_ARRAY_TASK_ID]}

echo "========================================"
echo "GENERATE VALIDATION DATA (GPU BATCH)"
echo "Atlas: $ATLAS"
echo "Job ID: $SLURM_JOB_ID (array task $SLURM_ARRAY_TASK_ID)"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start: $(date)"
echo "========================================"

source ~/bin/myconda
conda activate secactpy

cd /data/parks34/projects/2cytoatlas
mkdir -p logs

echo "Memory available: $(free -h | grep Mem | awk '{print $7}')"
echo "GPU memory: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"

python -u cytoatlas-api/scripts/generate_validation_data.py \
    --atlas "$ATLAS" \
    --signature-type all \
    --batch \
    --batch-size 1000000 \
    --viz-data-path /vf/users/parks34/projects/2cytoatlas/visualization/data \
    --results-path /data/parks34/projects/2cytoatlas/results

echo "========================================"
echo "Atlas: $ATLAS"
echo "End: $(date)"
echo "========================================"
