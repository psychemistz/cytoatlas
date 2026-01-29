#!/bin/bash
#SBATCH --job-name=figure_generation
#SBATCH --output=logs/figures_%j.out
#SBATCH --error=logs/figures_%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=norm
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4

# Pan-Disease Cytokine Activity Atlas
# Publication Figure Generation

echo "========================================"
echo "PUBLICATION FIGURE GENERATION"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo "========================================"

# Load modules
source ~/bin/myconda
conda activate secactpy

cd /data/parks34/projects/2secactpy
mkdir -p logs
mkdir -p results/figures

# Run figure generation
python scripts/05_figures.py --all

echo "========================================"
echo "Figures saved to: results/figures/"
echo "End: $(date)"
echo "========================================"
