#!/bin/bash
#SBATCH --job-name=cima_activity
#SBATCH --output=logs/cima_%j.out
#SBATCH --error=logs/cima_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16

# Pan-Disease Cytokine Activity Atlas
# CIMA Analysis Script
# Computes CytoSig and SecAct activities on 6.5M cells

echo "========================================"
echo "CIMA ACTIVITY ANALYSIS"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"
echo "========================================"

# Load modules
module load CUDA/12.8.1
module load cuDNN/9.12.0/CUDA-12
source ~/bin/myconda
conda activate secactpy

cd /data/parks34/projects/2secactpy
mkdir -p logs

# Run CIMA analysis
# Mode options: pseudobulk, singlecell, both
python scripts/01_cima_activity.py \
    --mode pseudobulk

# For full analysis including single-cell:
# python scripts/01_cima_activity.py --mode both

echo "========================================"
echo "End: $(date)"
echo "========================================"
