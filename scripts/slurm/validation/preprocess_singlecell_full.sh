#!/bin/bash
#SBATCH --job-name=preprocess_sc_full
#SBATCH --time=8:00:00
#SBATCH --mem=64G
#SBATCH --partition=norm
#SBATCH --output=logs/validation/preprocess_sc_full_%j.out
#SBATCH --error=logs/validation/preprocess_sc_full_%j.err

# =============================================================================
# Preprocess Single-Cell Validation (All Cells)
# =============================================================================
# For each atlas and signature type, reads single-cell activity H5ADs (backed
# mode) and computes:
#   - Exact statistics from ALL cells (Spearman rho, expressing fraction, etc.)
#   - 2D density bins (100x100 grid) from ALL cells
#   - 50K stratified sample for scatter overlay
#
# Input:  results/atlas_validation/{atlas}/singlecell/{atlas}_singlecell_{sigtype}.h5ad
#         Original expression H5ADs (CIMA, Inflammation, scAtlas)
# Output: visualization/data/singlecell_scatter.db
#
# Usage:
#   sbatch scripts/slurm/validation/preprocess_singlecell_full.sh
# =============================================================================

set -e

PROJECT_DIR="/vf/users/parks34/projects/2secactpy"
LOG_DIR="${PROJECT_DIR}/logs/validation"

mkdir -p "${LOG_DIR}"

# Environment
echo "Setting up environment..."
source ~/bin/myconda
conda activate secactpy

cd /data/parks34/projects/2secactpy

echo "=============================================================="
echo "Preprocess Single-Cell Validation (All Cells)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "Start: $(date)"
echo "=============================================================="

python scripts/18_preprocess_singlecell_validation.py

echo ""
echo "=============================================================="
echo "Preprocess Single-Cell Validation Complete"
echo "End: $(date)"
echo "=============================================================="

# Show output
if [ -f visualization/data/singlecell_scatter.db ]; then
    echo "SQLite DB: visualization/data/singlecell_scatter.db"
    ls -lh visualization/data/singlecell_scatter.db
else
    echo "ERROR: Output DB not generated!"
    exit 1
fi
