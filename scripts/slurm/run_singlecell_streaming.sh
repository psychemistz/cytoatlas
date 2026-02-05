#!/bin/bash
#SBATCH --job-name=sc_streaming
#SBATCH --time=48:00:00
#SBATCH --mem=200G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --output=logs/validation/sc_streaming_%A_%a.out
#SBATCH --error=logs/validation/sc_streaming_%A_%a.err
#SBATCH --array=0-17

# =============================================================================
# Single-Cell Activity Inference using H5AD Streaming
# =============================================================================
# Array job: 6 atlases × 3 signatures = 18 jobs
#
# Usage:
#   sbatch scripts/slurm/run_singlecell_streaming.sh
# =============================================================================

set -e

ATLASES=(
    "cima" "cima" "cima"
    "inflammation_main" "inflammation_main" "inflammation_main"
    "inflammation_val" "inflammation_val" "inflammation_val"
    "inflammation_ext" "inflammation_ext" "inflammation_ext"
    "scatlas_normal" "scatlas_normal" "scatlas_normal"
    "scatlas_cancer" "scatlas_cancer" "scatlas_cancer"
)
SIGNATURES=(
    "cytosig" "lincytosig" "secact"
    "cytosig" "lincytosig" "secact"
    "cytosig" "lincytosig" "secact"
    "cytosig" "lincytosig" "secact"
    "cytosig" "lincytosig" "secact"
    "cytosig" "lincytosig" "secact"
)

ATLAS="${ATLASES[$SLURM_ARRAY_TASK_ID]}"
SIGNATURE="${SIGNATURES[$SLURM_ARRAY_TASK_ID]}"

PROJECT_DIR="/vf/users/parks34/projects/2secactpy"
SCRIPT="${PROJECT_DIR}/scripts/run_singlecell_streaming.py"

mkdir -p "${PROJECT_DIR}/logs/validation"

echo "=============================================================="
echo "Single-Cell Activity Streaming"
echo "=============================================================="
echo "Atlas: ${ATLAS}"
echo "Signature: ${SIGNATURE}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "=============================================================="

source ~/bin/myconda
conda activate secactpy

cd "${PROJECT_DIR}"

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

python "${SCRIPT}" \
    --atlas "${ATLAS}" \
    --signature "${SIGNATURE}" \
    --skip-existing

echo ""
echo "=============================================================="
echo "Complete: $(date)"
echo "=============================================================="
