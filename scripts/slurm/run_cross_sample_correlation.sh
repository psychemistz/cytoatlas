#!/bin/bash
#SBATCH --job-name=xsample
#SBATCH --time=24:00:00
#SBATCH --mem=180G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/cross_sample/xsample_%A_%a.out
#SBATCH --error=logs/cross_sample/xsample_%A_%a.err
#SBATCH --array=0-5

# =============================================================================
# Cross-Sample Correlation Validation Pipeline
# =============================================================================
# Array job: 6 atlases (0=cima, 1=inflam_main, 2=inflam_val,
#            3=inflam_ext, 4=scatlas_normal, 5=scatlas_cancer)
#
# Usage:
#   sbatch scripts/slurm/run_cross_sample_correlation.sh
#   sbatch --array=0 scripts/slurm/run_cross_sample_correlation.sh   # CIMA only
#   sbatch --array=1-3 scripts/slurm/run_cross_sample_correlation.sh # Inflammation only
# =============================================================================

set -e

ATLASES=(
    "cima"
    "inflammation_main"
    "inflammation_val"
    "inflammation_ext"
    "scatlas_normal"
    "scatlas_cancer"
)

ATLAS="${ATLASES[$SLURM_ARRAY_TASK_ID]}"
PROJECT_DIR="/vf/users/parks34/projects/2secactpy"
SCRIPT="${PROJECT_DIR}/scripts/12_cross_sample_correlation.py"

mkdir -p "${PROJECT_DIR}/logs/cross_sample"

echo "=============================================================="
echo "Cross-Sample Correlation Pipeline"
echo "=============================================================="
echo "Atlas: ${ATLAS}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "=============================================================="

source ~/bin/myconda
conda activate secactpy

cd "${PROJECT_DIR}"

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

python "${SCRIPT}" --atlas "${ATLAS}"

echo ""
echo "=============================================================="
echo "Complete: $(date)"
echo "=============================================================="
