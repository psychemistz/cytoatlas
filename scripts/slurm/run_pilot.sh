#!/bin/bash
#SBATCH --job-name=pilot_analysis
#SBATCH --output=logs/pilot_%j.out
#SBATCH --error=logs/pilot_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8

# Pan-Disease Cytokine Activity Atlas
# Pilot Analysis Script
# Runs on 100K cell subsets to validate pipeline

echo "========================================"
echo "PILOT ANALYSIS"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo "========================================"

# Load modules
module load CUDA/12.8.1
module load cuDNN/9.12.0/CUDA-12
source ~/bin/myconda
conda activate secactpy

# Set working directory
cd /data/parks34/projects/2cytoatlas

# Create logs directory if not exists
mkdir -p logs

# Run pilot analysis
python scripts/00_pilot_analysis.py \
    --n-cells 100000 \
    --seed 42

echo "========================================"
echo "End: $(date)"
echo "========================================"
