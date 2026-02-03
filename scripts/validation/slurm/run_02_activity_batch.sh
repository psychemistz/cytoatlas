#!/bin/bash
#SBATCH --job-name=act_batch
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=/vf/users/parks34/projects/2secactpy/logs/activity_batch_%j.out
#SBATCH --error=/vf/users/parks34/projects/2secactpy/logs/activity_batch_%j.err

# Stage 2: Run ALL Activity Inference in a single job
# Runs all 33 configs sequentially (11 pseudobulk × 3 signatures)
# Usage: sbatch run_02_activity_batch.sh
# Note: Run AFTER run_01_pseudobulk_batch.sh completes

set -e

# Load CUDA modules
module load CUDA/12.1
module load cuDNN/8.9.2/CUDA-12

export CUDA_LAUNCH_BLOCKING=0
export PYTHONUNBUFFERED=1

source ~/bin/myconda
conda activate secactpy

cd /vf/users/parks34/projects/2secactpy
mkdir -p logs results/validation/activity

echo "========================================"
echo "Activity Inference - BATCH MODE"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Date: $(date)"
echo "========================================"

# Run all configs in a single Python process
python scripts/validation/02_run_activity_inference.py --all

echo "========================================"
echo "All activity inference completed: $(date)"
echo "========================================"

# List generated files
echo ""
echo "Generated activity files:"
ls -lh results/validation/activity/*.h5ad 2>/dev/null || echo "No activity files generated"

echo ""
echo "Generated validation files:"
ls -lh results/validation/activity/*.csv 2>/dev/null || echo "No validation files generated"
