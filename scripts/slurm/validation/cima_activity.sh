#!/bin/bash
#SBATCH --job-name=cima_act
#SBATCH --time=4:00:00
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu
#SBATCH --output=logs/validation/cima_act_%A_%a.out
#SBATCH --error=logs/validation/cima_act_%A_%a.err
#SBATCH --array=0-3

# =============================================================================
# CIMA Activity Inference (L1-L4 × 3 signatures)
# =============================================================================
# Array job: each task processes one annotation level with all signatures
#
# Usage:
#   sbatch cima_activity.sh
#
# =============================================================================

set -e

# Level mapping
LEVELS=("L1" "L2" "L3" "L4")
LEVEL="${LEVELS[$SLURM_ARRAY_TASK_ID]}"

# Paths
PROJECT_DIR="/vf/users/parks34/projects/2cytoatlas"
SCRIPT_DIR="${PROJECT_DIR}/scripts"
LOG_DIR="${PROJECT_DIR}/logs/validation"
OUTPUT_DIR="${PROJECT_DIR}/results/atlas_validation"

mkdir -p "${LOG_DIR}"

# Environment
echo "Setting up environment..."
source ~/bin/myconda
conda activate secactpy

cd "${PROJECT_DIR}"

# Info
echo "=============================================================="
echo "CIMA Activity Inference"
echo "=============================================================="
echo "Level: ${LEVEL}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Signatures: cytosig, lincytosig, secact"
echo "Start time: $(date)"
echo "=============================================================="

# GPU check
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

# Run with validation
python "${SCRIPT_DIR}/09_atlas_validation_activity.py" \
    --atlas cima \
    --level "${LEVEL}" \
    --validate

echo ""
echo "Complete: $(date)"
