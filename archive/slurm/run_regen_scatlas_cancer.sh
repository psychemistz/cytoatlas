#!/bin/bash
#SBATCH --job-name=regen_cancer
#SBATCH --time=24:00:00
#SBATCH --mem=200G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/regen_scatlas_cancer/regen_%j.out
#SBATCH --error=logs/regen_scatlas_cancer/regen_%j.err

# =============================================================================
# Regenerate scatlas_cancer Activity (zscore fix)
# =============================================================================
# Bug fix: activity was computed using result['beta'] instead of
# result['zscore'] from ridge(). This regenerates all scatlas_cancer
# donor-level activity and downstream data.
#
# Pipeline:
#   1. Script 12: Pseudobulk + activity (cross_sample_validation)
#   2. Script 13: Expression-activity correlations
#   3. Script 14: JSON preprocessing for visualization
#
# Usage:
#   sbatch scripts/slurm/run_regen_scatlas_cancer.sh
# =============================================================================

set -e

PROJECT_DIR="/vf/users/parks34/projects/2cytoatlas"

mkdir -p "${PROJECT_DIR}/logs/regen_scatlas_cancer"

echo "=============================================================="
echo "Regenerate scatlas_cancer Activity (zscore fix)"
echo "=============================================================="
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "=============================================================="

source ~/bin/myconda
conda activate secactpy

cd "${PROJECT_DIR}"

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

# Step 1: Regenerate pseudobulk + activity with --force
# Script 12 working copy already uses result['zscore']
echo ""
echo "=== Step 1: Cross-Sample Pseudobulk + Activity (script 12) ==="
python scripts/12_cross_sample_correlation.py --atlas scatlas_cancer --force

# Step 2: Recompute expression-activity correlations
echo ""
echo "=== Step 2: Correlation Analysis (script 13) ==="
python scripts/13_cross_sample_correlation_analysis.py --atlas scatlas_cancer

# Step 3: Regenerate visualization JSON
echo ""
echo "=== Step 3: JSON Preprocessing (script 14) ==="
python scripts/14_preprocess_bulk_validation.py

echo ""
echo "=============================================================="
echo "Complete: $(date)"
echo "=============================================================="
