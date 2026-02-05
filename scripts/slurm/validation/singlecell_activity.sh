#!/bin/bash
#SBATCH --job-name=sc_activity
#SBATCH --time=24:00:00
#SBATCH --mem=200G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --output=logs/validation/sc_activity_%A_%a.out
#SBATCH --error=logs/validation/sc_activity_%A_%a.err
#SBATCH --array=0-17

# =============================================================================
# Single-Cell Activity Inference for All Atlases
# =============================================================================
# Array job configuration:
#   0-2:   CIMA (cytosig, lincytosig, secact)
#   3-5:   inflammation_main
#   6-8:   inflammation_val
#   9-11:  inflammation_ext
#   12-14: scatlas_normal
#   15-17: scatlas_cancer
#
# Total: 6 atlases × 3 signatures = 18 jobs
#
# Usage:
#   sbatch singlecell_activity.sh
#
# =============================================================================

set -e

# Config arrays
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

# Paths
PROJECT_DIR="/vf/users/parks34/projects/2secactpy"
SCRIPT="${PROJECT_DIR}/scripts/12_singlecell_activity.py"
LOG_DIR="${PROJECT_DIR}/logs/validation"

mkdir -p "${LOG_DIR}"

# Environment
echo "=============================================================="
echo "Single-Cell Activity Inference"
echo "=============================================================="
echo "Atlas: ${ATLAS}"
echo "Signature: ${SIGNATURE}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "=============================================================="

source ~/bin/myconda
conda activate secactpy

cd "${PROJECT_DIR}"

# GPU check
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

# Run
python "${SCRIPT}" \
    --atlas "${ATLAS}" \
    --signature "${SIGNATURE}" \
    --skip-existing

echo ""
echo "=============================================================="
echo "Complete: $(date)"
echo "=============================================================="
