#!/bin/bash
#SBATCH --job-name=gen_validation
#SBATCH --output=logs/gen_validation_%j.out
#SBATCH --error=logs/gen_validation_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=norm
#SBATCH --mem=200G
#SBATCH --cpus-per-task=8

# Pan-Disease Cytokine Activity Atlas
# Regenerate single-cell validation JSON files using ALL cells
# (CytoSig ~2.3GB, LinCytoSig ~5GB, SecAct ~65GB per atlas)
# Outputs: visualization/data/validation/{atlas}_validation.json

set -e

echo "========================================"
echo "GENERATE VALIDATION DATA (ALL CELLS)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo "========================================"

source ~/bin/myconda
conda activate secactpy

cd /data/parks34/projects/2secactpy
mkdir -p logs

echo "Memory available: $(free -h | grep Mem | awk '{print $7}')"

# Run for all atlases and all signature types (CytoSig + LinCytoSig + SecAct)
python cytoatlas-api/scripts/generate_validation_data.py \
    --atlas all \
    --signature-type all \
    --viz-data-path /vf/users/parks34/projects/2secactpy/visualization/data \
    --results-path /data/parks34/projects/2secactpy/results

echo "========================================"
echo "End: $(date)"
echo "========================================"
