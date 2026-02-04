#!/bin/bash
#SBATCH --job-name=inflam_pb
#SBATCH --time=6:00:00
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu
#SBATCH --output=logs/validation/inflam_pb_%A_%a.out
#SBATCH --error=logs/validation/inflam_pb_%A_%a.err
#SBATCH --array=0-5

# =============================================================================
# Inflammation Atlas Pseudobulk Generation
# =============================================================================
# Array job: processes all cohorts (main, val, ext) × levels (L1, L2)
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
#   sbatch inflam_pseudobulk.sh
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
echo "Inflammation Atlas Pseudobulk Generation"
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
