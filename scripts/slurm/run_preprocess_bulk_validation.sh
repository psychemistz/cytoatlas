#!/bin/bash
#SBATCH --job-name=preprocess_bulk_val
#SBATCH --output=logs/preprocess_bulk_val_%j.out
#SBATCH --error=logs/preprocess_bulk_val_%j.err
#SBATCH --time=4:00:00
#SBATCH --partition=norm
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

# Pan-Disease Cytokine Activity Atlas
# Preprocess bulk validation data (regenerate bulk_donor_correlations.json)
# Depends on: run_resampled_inflammation.sh completion

echo "========================================"
echo "PREPROCESS BULK VALIDATION"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo "========================================"

source ~/bin/myconda
conda activate secactpy

cd /data/parks34/projects/2cytoatlas
mkdir -p logs

# Regenerate bulk_donor_correlations.json with inflammation resampled data
# Also outputs Parquet metadata files
python scripts/14_preprocess_bulk_validation.py

echo "========================================"
echo "End: $(date)"
echo "========================================"
