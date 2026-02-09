#!/bin/bash
#SBATCH --job-name=donor_level
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=180G
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --output=/data/parks34/projects/2secactpy/logs/donor_level_%A_%a.log

# Donor-Level Pseudobulk Pipeline
# ===============================
# Generates donor-agnostic pseudobulk and activity inference for all atlases
#
# Usage:
#   sbatch --array=0-5 scripts/slurm/run_donor_level_pipeline.sh
#   sbatch --array=0 scripts/slurm/run_donor_level_pipeline.sh  # CIMA only
#
# Array indices:
#   0: CIMA
#   1: Inflammation Main
#   2: Inflammation Validation
#   3: Inflammation External
#   4: scAtlas Normal
#   5: scAtlas Cancer

set -euo pipefail

# Environment setup
source ~/bin/myconda
conda activate secactpy

cd /data/parks34/projects/2secactpy

# Atlas array
ATLASES=(
    "cima"
    "inflammation_main"
    "inflammation_val"
    "inflammation_ext"
    "scatlas_normal"
    "scatlas_cancer"
)

# Get atlas for this array task
ATLAS="${ATLASES[$SLURM_ARRAY_TASK_ID]}"

echo "=============================================="
echo "Donor-Level Pipeline: ${ATLAS}"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "=============================================="

# Run pipeline
# Note: Single-cell activity is very expensive (millions of cells)
# Add --run-singlecell for specific atlases if needed
python scripts/11_donor_level_pipeline.py \
    --atlas "${ATLAS}" \
    --output-dir /vf/users/parks34/projects/2secactpy/results/donor_level

echo ""
echo "=============================================="
echo "Completed: ${ATLAS}"
echo "=============================================="
