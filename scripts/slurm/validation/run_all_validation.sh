#!/bin/bash
#SBATCH --job-name=atlas_validation
#SBATCH --time=48:00:00
#SBATCH --mem=128G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu
#SBATCH --output=logs/validation/master_%j.out
#SBATCH --error=logs/validation/master_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=seongyong.park@nih.gov

# =============================================================================
# Master Orchestrator: Atlas-Level Activity Inference Validation Pipeline
# =============================================================================
#
# This script orchestrates the full validation pipeline:
# 1. Generate pseudobulk for all atlases/levels
# 2. Run activity inference (CytoSig, LinCytoSig, SecAct)
# 3. Validate expression vs activity correlations
#
# Usage:
#   sbatch run_all_validation.sh           # Full pipeline
#   sbatch run_all_validation.sh --stage 1 # Pseudobulk only
#   sbatch run_all_validation.sh --stage 2 # Activity only
#
# =============================================================================

set -e

# Parse arguments
STAGE="${1:-all}"

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

# GPU check
echo "Checking GPU..."
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

# =============================================================================
# Stage 1: Pseudobulk Generation
# =============================================================================

run_pseudobulk() {
    echo ""
    echo "=============================================================="
    echo "STAGE 1: PSEUDOBULK GENERATION"
    echo "=============================================================="
    echo ""

    # CIMA (4 levels)
    for level in L1 L2 L3 L4; do
        echo "Processing CIMA ${level}..."
        python "${SCRIPT_DIR}/09_atlas_validation_pseudobulk.py" \
            --atlas cima \
            --level "${level}" \
            --output-dir "${OUTPUT_DIR}" \
            2>&1 | tee -a "${LOG_DIR}/cima_pseudobulk.log"
    done

    # Inflammation Atlas (main, validation, external - 2 levels each)
    for cohort in main val ext; do
        for level in L1 L2; do
            echo "Processing inflammation_${cohort} ${level}..."
            python "${SCRIPT_DIR}/09_atlas_validation_pseudobulk.py" \
                --atlas "inflammation_${cohort}" \
                --level "${level}" \
                --output-dir "${OUTPUT_DIR}" \
                2>&1 | tee -a "${LOG_DIR}/inflammation_pseudobulk.log"
        done
    done

    # scAtlas (normal and cancer - 3 levels each)
    for dataset in normal cancer; do
        for level in organ_celltype celltype organ; do
            echo "Processing scatlas_${dataset} ${level}..."
            python "${SCRIPT_DIR}/09_atlas_validation_pseudobulk.py" \
                --atlas "scatlas_${dataset}" \
                --level "${level}" \
                --output-dir "${OUTPUT_DIR}" \
                2>&1 | tee -a "${LOG_DIR}/scatlas_pseudobulk.log"
        done
    done

    echo "Pseudobulk generation complete."
}

# =============================================================================
# Stage 2: Activity Inference
# =============================================================================

run_activity() {
    echo ""
    echo "=============================================================="
    echo "STAGE 2: ACTIVITY INFERENCE"
    echo "=============================================================="
    echo ""

    # CIMA (4 levels × 3 signatures = 12 jobs)
    for level in L1 L2 L3 L4; do
        echo "Processing CIMA ${level} activity..."
        python "${SCRIPT_DIR}/09_atlas_validation_activity.py" \
            --atlas cima \
            --level "${level}" \
            --validate \
            2>&1 | tee -a "${LOG_DIR}/cima_activity.log"
    done

    # Inflammation Atlas
    for cohort in main val ext; do
        for level in L1 L2; do
            echo "Processing inflammation_${cohort} ${level} activity..."
            python "${SCRIPT_DIR}/09_atlas_validation_activity.py" \
                --atlas "inflammation_${cohort}" \
                --level "${level}" \
                --validate \
                2>&1 | tee -a "${LOG_DIR}/inflammation_activity.log"
        done
    done

    # scAtlas
    for dataset in normal cancer; do
        for level in organ_celltype celltype organ; do
            echo "Processing scatlas_${dataset} ${level} activity..."
            python "${SCRIPT_DIR}/09_atlas_validation_activity.py" \
                --atlas "scatlas_${dataset}" \
                --level "${level}" \
                --validate \
                2>&1 | tee -a "${LOG_DIR}/scatlas_activity.log"
        done
    done

    echo "Activity inference complete."
}

# =============================================================================
# Main Execution
# =============================================================================

echo "=============================================================="
echo "ATLAS-LEVEL ACTIVITY VALIDATION PIPELINE"
echo "=============================================================="
echo "Start time: $(date)"
echo "Stage: ${STAGE}"
echo "Output: ${OUTPUT_DIR}"
echo "=============================================================="

case "${STAGE}" in
    "1"|"pseudobulk")
        run_pseudobulk
        ;;
    "2"|"activity")
        run_activity
        ;;
    "all"|*)
        run_pseudobulk
        run_activity
        ;;
esac

echo ""
echo "=============================================================="
echo "PIPELINE COMPLETE"
echo "End time: $(date)"
echo "=============================================================="

# Summary
echo ""
echo "Output files:"
find "${OUTPUT_DIR}" -name "*.h5ad" -type f | head -20
echo ""
echo "Validation results:"
find "${OUTPUT_DIR}" -name "*_validation.csv" -type f | head -20
