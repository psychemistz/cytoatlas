#!/bin/bash
#SBATCH --job-name=cima_pb
#SBATCH --time=8:00:00
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu
#SBATCH --output=logs/validation/cima_pb_%A_%a.out
#SBATCH --error=logs/validation/cima_pb_%A_%a.err
#SBATCH --array=0-3

# =============================================================================
# CIMA Pseudobulk Generation (L1-L4)
# =============================================================================
# Array job: each task processes one annotation level
#
# Usage:
#   sbatch cima_pseudobulk.sh
#
# =============================================================================

set -e

# Level mapping
LEVELS=("L1" "L2" "L3" "L4")
LEVEL="${LEVELS[$SLURM_ARRAY_TASK_ID]}"

# Paths
PROJECT_DIR="/vf/users/parks34/projects/2secactpy"
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
echo "CIMA Pseudobulk Generation"
echo "=============================================================="
echo "Level: ${LEVEL}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Start time: $(date)"
echo "=============================================================="

# GPU check
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

# Run
python "${SCRIPT_DIR}/09_atlas_validation_pseudobulk.py" \
    --atlas cima \
    --level "${LEVEL}" \
    --output-dir "${OUTPUT_DIR}"

echo ""
echo "Complete: $(date)"
