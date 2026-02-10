#!/bin/bash
#SBATCH --job-name=bulk_val
#SBATCH --time=12:00:00
#SBATCH --mem=200G
#SBATCH --partition=norm
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/bulk_validation/bulk_val_%j.out
#SBATCH --error=logs/bulk_validation/bulk_val_%j.err

# =============================================================================
# Bulk RNA-seq Validation Pipeline (GTEx + TCGA)
# =============================================================================
# Runs activity inference on bulk RNA-seq data, then correlation analysis
# and JSON preprocessing.
#
# Usage:
#   sbatch scripts/slurm/run_bulk_validation.sh
# =============================================================================

set -e

PROJECT_DIR="/vf/users/parks34/projects/2cytoatlas"

mkdir -p "${PROJECT_DIR}/logs/bulk_validation"

echo "=============================================================="
echo "Bulk RNA-seq Validation Pipeline"
echo "=============================================================="
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "=============================================================="

source ~/bin/myconda
conda activate secactpy

cd "${PROJECT_DIR}"

# Step 1: Activity inference (GTEx + TCGA)
echo ""
echo "=== Step 1: Activity Inference ==="
python scripts/15_bulk_validation.py --dataset all --backend numpy --force

# Step 2: Correlation analysis
echo ""
echo "=== Step 2: Correlation Analysis ==="
python scripts/13_cross_sample_correlation_analysis.py --atlas gtex tcga

# Step 3: JSON preprocessing
echo ""
echo "=== Step 3: JSON Preprocessing ==="
python scripts/14_preprocess_bulk_validation.py

echo ""
echo "=============================================================="
echo "Complete: $(date)"
echo "=============================================================="
