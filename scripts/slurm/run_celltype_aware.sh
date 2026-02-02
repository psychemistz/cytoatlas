#!/bin/bash
#SBATCH --job-name=celltype_aware
#SBATCH --output=logs/celltype_aware_%j.out
#SBATCH --error=logs/celltype_aware_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=norm
#SBATCH --cpus-per-task=16
#SBATCH --mem=128g
#SBATCH --gres=lscratch:100

# Cell-Type-Aware Cytokine Activity Inference
# Runs on all atlases at both pseudo-bulk and single-cell levels

set -e

# Setup
cd /data/parks34/projects/2secactpy
source ~/bin/myconda
conda activate secactpy

mkdir -p logs

echo "=================================================="
echo "Cell-Type-Aware Cytokine Activity Inference"
echo "Start time: $(date)"
echo "=================================================="

# Parse arguments
MODE=${1:-pseudobulk}  # Default to pseudobulk
ATLAS=${2:-all}        # Default to all atlases

echo "Mode: $MODE"
echo "Atlas: $ATLAS"

# Run inference
python scripts/08_celltype_aware_activity.py --mode $MODE --atlas $ATLAS

echo ""
echo "=================================================="
echo "Complete!"
echo "End time: $(date)"
echo "=================================================="
