#!/bin/bash
#SBATCH --job-name=scatlas_pb
#SBATCH --time=8:00:00
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu
#SBATCH --output=logs/validation/scatlas_pb_%A_%a.out
#SBATCH --error=logs/validation/scatlas_pb_%A_%a.err
#SBATCH --array=0-5

# =============================================================================
# scAtlas Pseudobulk Generation
# =============================================================================
# Array job: processes both datasets × annotation levels
#
# Config mapping:
#   0: normal organ_celltype
#   1: normal celltype
#   2: normal organ
#   3: cancer organ_celltype
#   4: cancer celltype
#   5: cancer organ
#
# Usage:
#   sbatch scatlas_pseudobulk.sh
#
# =============================================================================

set -e

# Config mapping
DATASETS=("normal" "normal" "normal" "cancer" "cancer" "cancer")
LEVELS=("organ_celltype" "celltype" "organ" "organ_celltype" "celltype" "organ")

DATASET="${DATASETS[$SLURM_ARRAY_TASK_ID]}"
LEVEL="${LEVELS[$SLURM_ARRAY_TASK_ID]}"
ATLAS="scatlas_${DATASET}"

# Paths
PROJECT_DIR="/vf/users/parks34/projects/2cytoatlas"
SCRIPT_DIR="${PROJECT_DIR}/scripts"
LOG_DIR="${PROJECT_DIR}/logs/validation"
OUTPUT_DIR="${PROJECT_DIR}/results/atlas_validation"

mkdir -p "${LOG_DIR}"
mkdir -p "${OUTPUT_DIR}"

# Environment
echo "Setting up environment..."
source ~/bin/myconda
conda activate secactpy

cd "${PROJECT_DIR}"

# Info
echo "=============================================================="
echo "scAtlas Pseudobulk Generation"
echo "=============================================================="
echo "Atlas: ${ATLAS}"
echo "Level: ${LEVEL}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Start time: $(date)"
echo "=============================================================="

# GPU check
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

# Run
python "${SCRIPT_DIR}/09_atlas_validation_pseudobulk.py" \
    --atlas "${ATLAS}" \
    --level "${LEVEL}" \
    --output-dir "${OUTPUT_DIR}"

echo ""
echo "Complete: $(date)"
