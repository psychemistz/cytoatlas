#!/bin/bash
#SBATCH --job-name=integrated_analysis
#SBATCH --output=logs/integrated_%j.out
#SBATCH --error=logs/integrated_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=norm
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

# Pan-Disease Cytokine Activity Atlas
# Integrated Cross-Atlas Analysis
# Harmonizes and compares results across all atlases

echo "========================================"
echo "INTEGRATED CROSS-ATLAS ANALYSIS"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo "========================================"

# Load modules
source ~/bin/myconda
conda activate secactpy

cd /data/parks34/projects/2cytoatlas
mkdir -p logs

# Run integrated analysis
python scripts/04_integrated.py

echo "========================================"
echo "End: $(date)"
echo "========================================"
