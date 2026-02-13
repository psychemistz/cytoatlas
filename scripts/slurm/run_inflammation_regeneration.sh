#!/bin/bash
#SBATCH --job-name=inflam_regen
#SBATCH --time=12:00:00
#SBATCH --mem=200G
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu
#SBATCH --output=logs/validation/inflam_regeneration_%j.out
#SBATCH --error=logs/validation/inflam_regeneration_%j.err

# =============================================================================
# Inflammation Atlas Regeneration (all 3 cohorts)
# =============================================================================
# Regenerates pseudobulk, activity, correlations, and resampled validation
# with cell exclusion fix (Doublets + LowQuality_cells removed)
#
# Uses ridge_batch with GPU for activity inference.
# =============================================================================

set -e

PROJECT_DIR="/vf/users/parks34/projects/2cytoatlas"
SCRIPT_DIR="${PROJECT_DIR}/scripts"
LOG_DIR="${PROJECT_DIR}/logs/validation"

mkdir -p "${LOG_DIR}"

echo "Setting up environment..."
source ~/bin/myconda
conda activate secactpy

cd "${PROJECT_DIR}"

echo "=============================================================="
echo "Inflammation Atlas Regeneration"
echo "=============================================================="
echo "Start time: $(date)"
echo "=============================================================="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# =============================================================================
# Step 1: Regenerate pseudobulk + activity (12_)
# Uses ridge_batch with GPU, excludes Doublets/LowQuality_cells
# =============================================================================

echo ""
echo "=============================================================="
echo "STEP 1: Pseudobulk + Activity (12_cross_sample_correlation.py)"
echo "=============================================================="

for ATLAS in inflammation_main inflammation_val inflammation_ext; do
    echo ""
    echo "--- Processing: ${ATLAS} ---"
    echo "Start: $(date)"

    python "${SCRIPT_DIR}/12_cross_sample_correlation.py" \
        --atlas "${ATLAS}" \
        --force \
        --backend auto

    echo "Done: ${ATLAS} at $(date)"
done

# =============================================================================
# Step 2: Recompute correlations (13_)
# Uses min_samples=10 consistently
# =============================================================================

echo ""
echo "=============================================================="
echo "STEP 2: Correlations (13_cross_sample_correlation_analysis.py)"
echo "=============================================================="

# Run all 3 together so per-atlas CSVs are saved individually.
# NOTE: This also overwrites all_correlations.csv and correlation_summary.csv
# with only inflammation results. Rebuild the global combined files separately
# after verifying results (run 13_ with --atlas all).
python "${SCRIPT_DIR}/13_cross_sample_correlation_analysis.py" \
    --atlas inflammation_main inflammation_val inflammation_ext

echo "Done correlations at $(date)"

# =============================================================================
# Step 3: Resampled validation (16_)
# SKIPPED: Source resampled pseudobulk files (from 09_atlas_multilevel_pseudobulk.py)
# were generated before cell exclusion fix. Need to regenerate those first.
# TODO: Fix 09_ with cell exclusion, regenerate resampled pseudobulk, then run 16_.
# =============================================================================

echo ""
echo "=============================================================="
echo "STEP 3: Resampled Validation — SKIPPED"
echo "(Source resampled pseudobulk needs regeneration with cell exclusion)"
echo "=============================================================="

# =============================================================================
# Summary
# =============================================================================

echo ""
echo "=============================================================="
echo "ALL COMPLETE: $(date)"
echo "=============================================================="

echo ""
echo "Output files:"
for ATLAS in inflammation_main inflammation_val inflammation_ext; do
    echo ""
    echo "--- ${ATLAS} ---"
    ls -lh "${PROJECT_DIR}/results/cross_sample_validation/${ATLAS}/"*.h5ad 2>/dev/null || echo "No h5ad files"
done

echo ""
echo "--- Correlations ---"
ls -lh "${PROJECT_DIR}/results/cross_sample_validation/correlations/inflammation"* 2>/dev/null || echo "No correlation files"
