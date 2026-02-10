#!/bin/bash
#SBATCH --job-name=inflam_singlecell
#SBATCH --output=logs/inflam_sc_%j.out
#SBATCH --error=logs/inflam_sc_%j.err
#SBATCH --time=72:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16

# Pan-Disease Cytokine Activity Atlas
# Inflammation Atlas Single-Cell Activity Analysis
# Computes per-cell activities on 6.3M cells (main + validation + external)

echo "========================================"
echo "INFLAMMATION SINGLE-CELL ACTIVITY ANALYSIS"
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

# Run Inflammation Atlas single-cell analysis
python scripts/02_inflam_activity.py \
    --mode singlecell

echo "========================================"
echo "End: $(date)"
echo "========================================"
