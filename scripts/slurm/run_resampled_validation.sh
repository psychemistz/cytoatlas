#!/bin/bash
#SBATCH --job-name=resamp_val
#SBATCH --time=24:00:00
#SBATCH --mem=200G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/resampled_validation/resamp_val_%j.out
#SBATCH --error=logs/resampled_validation/resamp_val_%j.err

# =============================================================================
# Resampled Pseudobulk Validation Pipeline
# =============================================================================
# Runs activity inference on bootstrap-resampled pseudobulk (100 iterations),
# computes expression-activity correlations with 95% confidence intervals,
# then regenerates visualization JSON.
#
# Input:  results/atlas_validation/{atlas}/pseudobulk/*_resampled.h5ad
# Output: results/cross_sample_validation/{atlas}/*_resampled_*.h5ad
#         results/cross_sample_validation/correlations/*_resampled_*_correlations.csv
#         visualization/data/bulk_donor_correlations.json (updated)
#
# Usage:
#   sbatch scripts/slurm/run_resampled_validation.sh
# =============================================================================

set -e

PROJECT_DIR="/vf/users/parks34/projects/2secactpy"

mkdir -p "${PROJECT_DIR}/logs/resampled_validation"

echo "=============================================================="
echo "Resampled Pseudobulk Validation Pipeline"
echo "=============================================================="
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "=============================================================="

source ~/bin/myconda
conda activate secactpy

cd "${PROJECT_DIR}"

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

# Step 1: Resampled activity inference + correlation analysis (all atlases)
echo ""
echo "=== Step 1: Resampled Activity Inference + Correlations ==="
python scripts/16_resampled_validation.py --atlas all --backend auto

# Step 2: Regenerate visualization JSON (incorporates resampled data)
echo ""
echo "=== Step 2: JSON Preprocessing ==="
python scripts/14_preprocess_bulk_validation.py

echo ""
echo "=============================================================="
echo "Complete: $(date)"
echo "=============================================================="
