#!/bin/bash
#SBATCH --job-name=val_summary
#SBATCH --time=2:00:00
#SBATCH --mem=64G
#SBATCH --partition=norm
#SBATCH --output=logs/validation/val_summary_%j.out
#SBATCH --error=logs/validation/val_summary_%j.err

# =============================================================================
# Validation Summary Preprocessing
# =============================================================================
# Computes per-entity Spearman correlations (expression vs activity) across
# 8 source categories for the Validation Summary boxplot tab.
#
# Input:  H5AD files in results/cross_sample_validation/{atlas}/
# Output: visualization/data/validation_corr_boxplot.json
#
# Usage:
#   sbatch scripts/slurm/validation/validation_summary.sh
# =============================================================================

set -e

PROJECT_DIR="/vf/users/parks34/projects/2cytoatlas"
LOG_DIR="${PROJECT_DIR}/logs/validation"

mkdir -p "${LOG_DIR}"

# Environment
echo "Setting up environment..."
source ~/bin/myconda
conda activate secactpy

cd /data/parks34/projects/2cytoatlas

echo "=============================================================="
echo "Validation Summary Preprocessing"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo "=============================================================="

python scripts/17_preprocess_validation_summary.py

echo ""
echo "=============================================================="
echo "Validation Summary Complete"
echo "End: $(date)"
echo "=============================================================="

# Show output
if [ -f visualization/data/validation_corr_boxplot.json ]; then
    echo "Output: visualization/data/validation_corr_boxplot.json"
    ls -lh visualization/data/validation_corr_boxplot.json
else
    echo "ERROR: Output file not generated!"
    exit 1
fi
