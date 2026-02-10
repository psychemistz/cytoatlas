#!/bin/bash
#SBATCH --job-name=scatlas_canc
#SBATCH --time=4:00:00
#SBATCH --mem=200G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu
#SBATCH --output=logs/validation/scatlas_canc_multi_%j.out
#SBATCH --error=logs/validation/scatlas_canc_multi_%j.err

# =============================================================================
# scAtlas Cancer Multi-Level Pseudobulk Generation
# =============================================================================

set -e

PROJECT_DIR="/vf/users/parks34/projects/2cytoatlas"
SCRIPT_DIR="${PROJECT_DIR}/scripts"
LOG_DIR="${PROJECT_DIR}/logs/validation"
OUTPUT_DIR="${PROJECT_DIR}/results/atlas_validation/scatlas_cancer/pseudobulk"

mkdir -p "${LOG_DIR}"
mkdir -p "${OUTPUT_DIR}"

echo "Setting up environment..."
source ~/bin/myconda
conda activate secactpy

cd "${PROJECT_DIR}"

echo "=============================================================="
echo "scAtlas Cancer Multi-Level Pseudobulk Generation"
echo "=============================================================="
echo "Start time: $(date)"
echo "Output: ${OUTPUT_DIR}"
echo "=============================================================="

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

python "${SCRIPT_DIR}/09_atlas_multilevel_pseudobulk.py" \
    --atlas scatlas_cancer \
    --output-dir "${OUTPUT_DIR}" \
    --n-bootstrap 100 \
    --skip-existing

echo ""
echo "=============================================================="
echo "Complete: $(date)"
echo "=============================================================="

ls -lh "${OUTPUT_DIR}/"*.h5ad 2>/dev/null || echo "No output files found"
