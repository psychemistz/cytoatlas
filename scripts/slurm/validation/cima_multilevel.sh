#!/bin/bash
#SBATCH --job-name=cima_multi
#SBATCH --time=4:00:00
#SBATCH --mem=200G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu
#SBATCH --output=logs/validation/cima_multi_%j.out
#SBATCH --error=logs/validation/cima_multi_%j.err

# =============================================================================
# CIMA Multi-Level Pseudobulk Generation (Single Pass)
# =============================================================================
# Generates ALL 4 levels (L1-L4) + bootstrap resamples in ONE pass.
# Output: 8 H5AD files in ~1 hour (vs ~4 hours with separate passes)
#
# Usage:
#   sbatch cima_multilevel.sh
#
# =============================================================================

set -e

# Paths
PROJECT_DIR="/vf/users/parks34/projects/2secactpy"
SCRIPT_DIR="${PROJECT_DIR}/scripts"
LOG_DIR="${PROJECT_DIR}/logs/validation"
OUTPUT_DIR="${PROJECT_DIR}/results/atlas_validation/cima/pseudobulk"

mkdir -p "${LOG_DIR}"
mkdir -p "${OUTPUT_DIR}"

# Environment
echo "Setting up environment..."
source ~/bin/myconda
conda activate secactpy

cd "${PROJECT_DIR}"

# Info
echo "=============================================================="
echo "CIMA Multi-Level Pseudobulk Generation"
echo "=============================================================="
echo "Start time: $(date)"
echo "Output: ${OUTPUT_DIR}"
echo "=============================================================="

# GPU check
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

# Run multi-level aggregation
python "${SCRIPT_DIR}/09_atlas_multilevel_pseudobulk.py" \
    --atlas cima \
    --output-dir "${OUTPUT_DIR}" \
    --n-bootstrap 100 \
    --skip-existing

echo ""
echo "=============================================================="
echo "Complete: $(date)"
echo "=============================================================="

# List outputs
echo ""
echo "Output files:"
ls -lh "${OUTPUT_DIR}/"*.h5ad 2>/dev/null || echo "No output files found"
