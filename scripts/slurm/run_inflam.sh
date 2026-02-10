#!/bin/bash
#SBATCH --job-name=inflam_activity
#SBATCH --output=logs/inflam_%j.out
#SBATCH --error=logs/inflam_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16

# Pan-Disease Cytokine Activity Atlas
# Inflammation Atlas Analysis Script
# Processes main (4.9M), validation (850K), external (573K) cohorts
# Includes treatment response prediction

echo "========================================"
echo "INFLAMMATION ATLAS ACTIVITY ANALYSIS"
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

# Run Inflammation Atlas analysis
# This includes:
# - Activity computation for all 3 cohorts
# - Disease differential analysis
# - Treatment response prediction (ML models)
# - Cross-cohort validation
python scripts/02_inflam_activity.py \
    --mode pseudobulk

# For full analysis including single-cell:
# python scripts/02_inflam_activity.py --mode both

echo "========================================"
echo "End: $(date)"
echo "========================================"
