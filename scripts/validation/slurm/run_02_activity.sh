#!/bin/bash
#SBATCH --job-name=activity_inf
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=/vf/users/parks34/projects/2cytoatlas/logs/activity_%A_%a.out
#SBATCH --error=/vf/users/parks34/projects/2cytoatlas/logs/activity_%A_%a.err
#SBATCH --array=0-32

# Stage 2: Activity Inference on Pseudobulk Data
# 33 configs: 11 pseudobulk files × 3 signatures
# Usage: sbatch run_02_activity.sh
# Note: Run AFTER run_01_pseudobulk.sh completes

set -e

# Load CUDA modules
module load CUDA/12.1
module load cuDNN/8.9.2/CUDA-12

# Set environment variables
export CUDA_LAUNCH_BLOCKING=0
export PYTHONUNBUFFERED=1

source ~/bin/myconda
conda activate secactpy

cd /vf/users/parks34/projects/2cytoatlas
mkdir -p logs results/validation/activity

echo "========================================"
echo "Activity Inference"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Date: $(date)"
echo "========================================"

python scripts/validation/02_run_activity_inference.py --config-index $SLURM_ARRAY_TASK_ID

echo "========================================"
echo "Completed: $(date)"
echo "========================================"
