#!/bin/bash
#SBATCH --job-name=cima_singlecell
#SBATCH --output=logs/cima_sc_%j.out
#SBATCH --error=logs/cima_sc_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16

# Pan-Disease Cytokine Activity Atlas
# CIMA Single-Cell Activity Analysis
# Computes per-cell CytoSig and SecAct activities on 6.5M cells

echo "========================================"
echo "CIMA SINGLE-CELL ACTIVITY ANALYSIS"
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

cd /data/parks34/projects/2cytoatlas
mkdir -p logs

# Run CIMA single-cell analysis
python scripts/01_cima_activity.py \
    --mode singlecell

echo "========================================"
echo "End: $(date)"
echo "========================================"
