#!/bin/bash
#SBATCH --job-name=pb_batch
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=/vf/users/parks34/projects/2cytoatlas/logs/pseudobulk_batch_%j.out
#SBATCH --error=/vf/users/parks34/projects/2cytoatlas/logs/pseudobulk_batch_%j.err

# Stage 1: Generate ALL Pseudobulk Expression Data in a single job
# Runs all 11 configs sequentially
# Usage: sbatch run_01_pseudobulk_batch.sh

set -e

# Load CUDA modules
module load CUDA/12.1
module load cuDNN/8.9.2/CUDA-12

export CUDA_LAUNCH_BLOCKING=0
export PYTHONUNBUFFERED=1

source ~/bin/myconda
conda activate secactpy

cd /vf/users/parks34/projects/2cytoatlas
mkdir -p logs results/validation/pseudobulk

echo "========================================"
echo "Pseudobulk Generation - BATCH MODE"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date: $(date)"
echo "========================================"

# Run all configs in a single Python process
python scripts/validation/01_generate_pseudobulk.py --all

echo "========================================"
echo "All pseudobulk generation completed: $(date)"
echo "========================================"

# List generated files
echo ""
echo "Generated files:"
ls -lh results/validation/pseudobulk/*.h5ad 2>/dev/null || echo "No files generated"
