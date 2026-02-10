#!/bin/bash
#SBATCH --job-name=cytoatlas_validation
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --output=/vf/users/parks34/projects/2cytoatlas/logs/validation_%j.out
#SBATCH --error=/vf/users/parks34/projects/2cytoatlas/logs/validation_%j.err
#SBATCH --array=0-98

# GPU Validation - All 99 tests as job array
# Usage: sbatch run_all_validation.sh

set -e

# Environment setup
source ~/bin/myconda
conda activate secactpy

# Working directory
cd /vf/users/parks34/projects/2cytoatlas

# Create log directory
mkdir -p logs

echo "========================================"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Date: $(date)"
echo "========================================"

# Run validation for this array index
python scripts/validation/run_validation.py --config-index $SLURM_ARRAY_TASK_ID

echo "========================================"
echo "Completed: $(date)"
echo "========================================"
