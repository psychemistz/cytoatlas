#!/bin/bash
#SBATCH --job-name=pseudobulk_gen
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=/vf/users/parks34/projects/2cytoatlas/logs/pseudobulk_%A_%a.out
#SBATCH --error=/vf/users/parks34/projects/2cytoatlas/logs/pseudobulk_%A_%a.err
#SBATCH --array=0-10

# Stage 1: Generate Pseudobulk Expression Data
# 11 configs: CIMA (3) + Inflammation (4) + scAtlas (4)
# Usage: sbatch run_01_pseudobulk.sh

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
mkdir -p logs results/validation/pseudobulk

echo "========================================"
echo "Pseudobulk Generation"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date: $(date)"
echo "========================================"

python scripts/validation/01_generate_pseudobulk.py --config-index $SLURM_ARRAY_TASK_ID

echo "========================================"
echo "Completed: $(date)"
echo "========================================"
