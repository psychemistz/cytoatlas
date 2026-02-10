#!/bin/bash
#SBATCH --job-name=atlas_val
#SBATCH --output=/data/parks34/projects/2cytoatlas/logs/atlas_validation_%j.out
#SBATCH --error=/data/parks34/projects/2cytoatlas/logs/atlas_validation_%j.err
#SBATCH --time=4:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4
#SBATCH --partition=norm

# Atlas-level validation: pseudobulk expression vs activity prediction
# This computes proper validation for the Atlas Validation panel

set -e

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "Working directory: $(pwd)"

# Activate conda environment
source ~/bin/myconda
conda activate secactpy

# Force unbuffered Python output
export PYTHONUNBUFFERED=1

cd /data/parks34/projects/2cytoatlas

# Create logs directory if needed
mkdir -p logs

# Parse arguments
ATLAS="${1:-all}"
SIG_TYPE="${2:-all}"

echo ""
echo "Atlas: $ATLAS"
echo "Signature type: $SIG_TYPE"
echo ""

# Run the script
python scripts/08_atlas_validation.py \
    --atlas "$ATLAS" \
    --signature-type "$SIG_TYPE"

echo ""
echo "Job completed: $(date)"
