#!/bin/bash
#SBATCH --job-name=atlas_validation
#SBATCH --output=logs/atlas_validation_%j.out
#SBATCH --error=logs/atlas_validation_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=norm
#SBATCH --cpus-per-task=8
#SBATCH --mem=128g
#SBATCH --gres=lscratch:50

# Atlas Validation: Expression-Activity Correlation Analysis
# Validates cytokine activity predictions by correlating with gene expression
#
# Usage:
#   sbatch run_atlas_validation.sh [atlas]
#   atlas: all (default), cima, inflammation, scatlas

set -e

# Setup
cd /data/parks34/projects/2secactpy
source ~/bin/myconda
conda activate secactpy

mkdir -p logs

echo "=================================================="
echo "Atlas Validation Analysis"
echo "Start time: $(date)"
echo "=================================================="

# Parse arguments
ATLAS=${1:-all}

echo "Atlas: $ATLAS"
echo ""

# Run validation
python scripts/09_atlas_validation.py --atlas $ATLAS

echo ""
echo "=================================================="
echo "Complete!"
echo "End time: $(date)"
echo "=================================================="
