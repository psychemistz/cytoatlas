#!/bin/bash
#SBATCH --job-name=preprocess_bulk_val
#SBATCH --time=6:00:00
#SBATCH --mem=128G
#SBATCH --partition=norm
#SBATCH --output=logs/validation/preprocess_bulk_val_%j.out
#SBATCH --error=logs/validation/preprocess_bulk_val_%j.err

# =============================================================================
# Preprocess Bulk Validation Data
# =============================================================================
# Generates split JSON files + SQLite DB for donor/celltype/bulk validation tabs.
# Includes ALL targets (no top-N filtering for SecAct/LinCytoSig).
#
# Input:  H5AD files in results/cross_sample_validation/{atlas}/
#         Correlation CSVs in results/cross_sample_validation/correlations/
# Output: visualization/data/validation/{donor_scatter,celltype_scatter,bulk_rnaseq}/*.json
#         visualization/data/validation_scatter.db
#
# Usage:
#   sbatch scripts/slurm/validation/preprocess_bulk_validation.sh
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
echo "Preprocess Bulk Validation Data"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "Start: $(date)"
echo "=============================================================="

python scripts/14_preprocess_bulk_validation.py

echo ""
echo "=============================================================="
echo "Preprocess Bulk Validation Complete"
echo "End: $(date)"
echo "=============================================================="

# Show outputs
echo ""
echo "Split files:"
find visualization/data/validation -name "*.json" -type f | wc -l
echo "json files generated"

if [ -f visualization/data/validation_scatter.db ]; then
    echo "SQLite DB: visualization/data/validation_scatter.db"
    ls -lh visualization/data/validation_scatter.db
else
    echo "WARNING: SQLite DB not generated"
fi
