#!/bin/bash
#SBATCH --job-name=inflam_validation
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=/vf/users/parks34/projects/2secactpy/logs/inflam_validation_%j.out
#SBATCH --error=/vf/users/parks34/projects/2secactpy/logs/inflam_validation_%j.err
#SBATCH --array=0-17

# Inflammation Atlas Validation - 18 tests (3 signatures × 3 aggregations × 2 levels)
# Usage: sbatch run_inflammation_validation.sh

set -e

source ~/bin/myconda
conda activate secactpy

cd /vf/users/parks34/projects/2secactpy
mkdir -p logs

echo "========================================"
echo "Inflammation Atlas Validation"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Date: $(date)"
echo "========================================"

# Inflammation tests are indices 27-44 (main) and 45-62 (validation)
# Offset by 27 for inflammation main dataset
ACTUAL_INDEX=$((27 + $SLURM_ARRAY_TASK_ID))

python scripts/validation/run_validation.py --config-index $ACTUAL_INDEX

echo "Completed: $(date)"
