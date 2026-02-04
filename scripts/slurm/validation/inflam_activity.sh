#!/bin/bash
#SBATCH --job-name=inflam_act
#SBATCH --time=4:00:00
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu
#SBATCH --output=logs/validation/inflam_act_%A_%a.out
#SBATCH --error=logs/validation/inflam_act_%A_%a.err
#SBATCH --array=0-5

# =============================================================================
# Inflammation Atlas Activity Inference
# =============================================================================
# Array job: processes all cohorts × levels with all signatures
#
# Config mapping:
#   0: main L1
#   1: main L2
#   2: val L1
#   3: val L2
#   4: ext L1
#   5: ext L2
#
# Usage:
#   sbatch inflam_activity.sh
#
# =============================================================================

set -e

# Config mapping
COHORTS=("main" "main" "val" "val" "ext" "ext")
LEVELS=("L1" "L2" "L1" "L2" "L1" "L2")

COHORT="${COHORTS[$SLURM_ARRAY_TASK_ID]}"
LEVEL="${LEVELS[$SLURM_ARRAY_TASK_ID]}"
ATLAS="inflammation_${COHORT}"

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
echo "Inflammation Atlas Activity Inference"
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
