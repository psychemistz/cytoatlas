#!/bin/bash
#SBATCH --job-name=scatlas_act
#SBATCH --time=4:00:00
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu
#SBATCH --output=logs/validation/scatlas_act_%A_%a.out
#SBATCH --error=logs/validation/scatlas_act_%A_%a.err
#SBATCH --array=0-3

# =============================================================================
# scAtlas Activity Inference
# =============================================================================
# Array job: processes both datasets × levels with all signatures
#
# Config mapping:
#   0: normal celltype
#   1: normal organ_celltype
#   2: cancer celltype
#   3: cancer organ_celltype
#
# Usage:
#   sbatch scatlas_activity.sh
#
# =============================================================================

set -e

# Config mapping (only existing levels)
DATASETS=("normal" "normal" "cancer" "cancer")
LEVELS=("celltype" "organ_celltype" "celltype" "organ_celltype")

DATASET="${DATASETS[$SLURM_ARRAY_TASK_ID]}"
LEVEL="${LEVELS[$SLURM_ARRAY_TASK_ID]}"
ATLAS="scatlas_${DATASET}"

# Paths
PROJECT_DIR="/vf/users/parks34/projects/2secactpy"
SCRIPT_DIR="${PROJECT_DIR}/scripts"
LOG_DIR="${PROJECT_DIR}/logs/validation"

mkdir -p "${LOG_DIR}"

# Environment
echo "Setting up environment..."
source ~/bin/myconda
conda activate secactpy

cd "${PROJECT_DIR}"

# Info
echo "=============================================================="
echo "scAtlas Activity Inference"
echo "=============================================================="
echo "Atlas: ${ATLAS}"
echo "Level: ${LEVEL}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Signatures: cytosig, lincytosig, secact"
echo "Start time: $(date)"
echo "=============================================================="

# GPU check
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

# Run with validation
python "${SCRIPT_DIR}/09_atlas_validation_activity.py" \
    --atlas "${ATLAS}" \
    --level "${LEVEL}" \
    --validate

echo ""
echo "Complete: $(date)"
