#!/bin/bash
#SBATCH --job-name=gen_val_%a
#SBATCH --output=logs/gen_validation_%A_%a.out
#SBATCH --error=logs/gen_validation_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --partition=norm
#SBATCH --mem=200G
#SBATCH --cpus-per-task=8
#SBATCH --array=0-2

# Pan-Disease Cytokine Activity Atlas
# Regenerate single-cell validation JSON files using ALL cells
# One atlas per array task to stay within 200GB memory
# Usage: sbatch scripts/slurm/run_generate_validation.sh
# Outputs: visualization/data/validation/{atlas}_validation.json

set -e

ATLASES=(cima inflammation scatlas)
ATLAS=${ATLASES[$SLURM_ARRAY_TASK_ID]}

echo "========================================"
echo "GENERATE VALIDATION DATA (ALL CELLS)"
echo "Atlas: $ATLAS"
echo "Job ID: $SLURM_JOB_ID (array task $SLURM_ARRAY_TASK_ID)"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo "========================================"

source ~/bin/myconda
conda activate secactpy

cd /data/parks34/projects/2cytoatlas
mkdir -p logs

echo "Memory available: $(free -h | grep Mem | awk '{print $7}')"

python -u cytoatlas-api/scripts/generate_validation_data.py \
    --atlas "$ATLAS" \
    --signature-type all \
    --viz-data-path /vf/users/parks34/projects/2cytoatlas/visualization/data \
    --results-path /data/parks34/projects/2cytoatlas/results

echo "========================================"
echo "Atlas: $ATLAS"
echo "End: $(date)"
echo "========================================"
