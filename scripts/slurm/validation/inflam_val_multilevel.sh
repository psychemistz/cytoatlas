#!/bin/bash
#SBATCH --job-name=inflam_val
#SBATCH --time=2:00:00
#SBATCH --mem=150G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu
#SBATCH --output=logs/validation/inflam_val_multi_%j.out
#SBATCH --error=logs/validation/inflam_val_multi_%j.err

# =============================================================================
# Inflammation Validation Multi-Level Pseudobulk Generation
# =============================================================================

set -e

PROJECT_DIR="/vf/users/parks34/projects/2secactpy"
SCRIPT_DIR="${PROJECT_DIR}/scripts"
LOG_DIR="${PROJECT_DIR}/logs/validation"
OUTPUT_DIR="${PROJECT_DIR}/results/atlas_validation/inflammation_val/pseudobulk"

mkdir -p "${LOG_DIR}"
mkdir -p "${OUTPUT_DIR}"

echo "Setting up environment..."
source ~/bin/myconda
conda activate secactpy

cd "${PROJECT_DIR}"

echo "=============================================================="
echo "Inflammation Validation Multi-Level Pseudobulk Generation"
echo "=============================================================="
echo "Start time: $(date)"
echo "Output: ${OUTPUT_DIR}"
echo "=============================================================="

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

python "${SCRIPT_DIR}/09_atlas_multilevel_pseudobulk.py" \
    --atlas inflammation_val \
    --output-dir "${OUTPUT_DIR}" \
    --n-bootstrap 100 \
    --skip-existing

echo ""
echo "=============================================================="
echo "Complete: $(date)"
echo "=============================================================="

ls -lh "${OUTPUT_DIR}/"*.h5ad 2>/dev/null || echo "No output files found"
