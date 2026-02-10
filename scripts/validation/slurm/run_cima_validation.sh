#!/bin/bash
#SBATCH --job-name=cima_validation
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=/vf/users/parks34/projects/2cytoatlas/logs/cima_validation_%j.out
#SBATCH --error=/vf/users/parks34/projects/2cytoatlas/logs/cima_validation_%j.err
#SBATCH --array=0-26

# CIMA Validation - 27 tests (3 signatures × 3 aggregations × 3 levels)
# Usage: sbatch run_cima_validation.sh

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
mkdir -p logs

echo "========================================"
echo "CIMA Validation"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Date: $(date)"
echo "========================================"

# CIMA tests are indices 0-26
python scripts/validation/run_validation.py --config-index $SLURM_ARRAY_TASK_ID

echo "Completed: $(date)"
