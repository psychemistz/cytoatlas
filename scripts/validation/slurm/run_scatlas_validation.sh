#!/bin/bash
#SBATCH --job-name=scatlas_validation
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=/vf/users/parks34/projects/2secactpy/logs/scatlas_validation_%j.out
#SBATCH --error=/vf/users/parks34/projects/2secactpy/logs/scatlas_validation_%j.err
#SBATCH --array=0-35

# scAtlas Validation - 36 tests (2 datasets × 3 signatures × 3 aggregations × 2 levels)
# Usage: sbatch run_scatlas_validation.sh

set -e

# Load CUDA modules
module load CUDA/12.1
module load cuDNN/8.9.2/CUDA-12

# Set environment variables
export CUDA_LAUNCH_BLOCKING=0
export PYTHONUNBUFFERED=1

source ~/bin/myconda
conda activate secactpy

cd /vf/users/parks34/projects/2secactpy
mkdir -p logs

echo "========================================"
echo "scAtlas Validation"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Date: $(date)"
echo "========================================"

# scAtlas tests are indices 63-98 (normal: 63-80, cancer: 81-98)
# Offset by 63 for scAtlas
ACTUAL_INDEX=$((63 + $SLURM_ARRAY_TASK_ID))

python scripts/validation/run_validation.py --config-index $ACTUAL_INDEX

echo "Completed: $(date)"
