#!/bin/bash
#SBATCH --job-name=resampled_inflam
#SBATCH --output=logs/resampled_inflam_%j.out
#SBATCH --error=logs/resampled_inflam_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16

# Pan-Disease Cytokine Activity Atlas
# Resampled Bootstrap Validation — Inflammation Atlas
# Runs activity inference on resampled pseudobulk for main/val/ext cohorts

echo "========================================"
echo "RESAMPLED INFLAMMATION VALIDATION"
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

# Run resampled validation for all inflammation cohorts
python scripts/16_resampled_validation.py \
    --atlas inflammation_main inflammation_val inflammation_ext

echo "========================================"
echo "End: $(date)"
echo "========================================"
