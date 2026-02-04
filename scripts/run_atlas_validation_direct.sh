#!/bin/bash
# =============================================================================
# Direct Execution: Atlas Validation Pipeline
# =============================================================================
# Runs the full validation pipeline directly on the current node (no SLURM).
# Designed for A100 GPU nodes with sufficient memory.
#
# Usage:
#   nohup bash scripts/run_atlas_validation_direct.sh > logs/validation_direct.log 2>&1 &
#
# To monitor:
#   tail -f logs/validation_direct.log
#
# ETA: ~10-12 hours total
# =============================================================================

set -e

# Paths
PROJECT_DIR="/vf/users/parks34/projects/2secactpy"
SCRIPT_DIR="${PROJECT_DIR}/scripts"
LOG_DIR="${PROJECT_DIR}/logs/validation"
OUTPUT_DIR="${PROJECT_DIR}/results/atlas_validation"

mkdir -p "${LOG_DIR}"
mkdir -p "${OUTPUT_DIR}"

cd "${PROJECT_DIR}"

# Environment
echo "=============================================================="
echo "ATLAS VALIDATION PIPELINE - DIRECT EXECUTION"
echo "=============================================================="
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo ""

source ~/bin/myconda
conda activate secactpy

# GPU check
echo "GPU Status:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# Track timing
TOTAL_START=$(date +%s)

# =============================================================================
# STAGE 1: PSEUDOBULK GENERATION
# =============================================================================

echo ""
echo "=============================================================="
echo "STAGE 1: PSEUDOBULK GENERATION"
echo "=============================================================="

STAGE1_START=$(date +%s)

# --- CIMA (4 levels) ---
echo ""
echo "[CIMA] Processing 4 levels (~6.5M cells)..."
for level in L1 L2 L3 L4; do
    echo "  [$(date +%H:%M:%S)] Starting CIMA ${level}..."
    python "${SCRIPT_DIR}/09_atlas_validation_pseudobulk.py" \
        --atlas cima --level "${level}" --output-dir "${OUTPUT_DIR}" \
        2>&1 | tee "${LOG_DIR}/cima_${level}_pseudobulk.log" | grep -E "^\[|Shape:|Saved:|Complete"
done

# --- Inflammation (3 cohorts × 2 levels) ---
echo ""
echo "[INFLAMMATION] Processing 3 cohorts × 2 levels (~6.3M cells)..."
for cohort in main val ext; do
    for level in L1 L2; do
        echo "  [$(date +%H:%M:%S)] Starting inflammation_${cohort} ${level}..."
        python "${SCRIPT_DIR}/09_atlas_validation_pseudobulk.py" \
            --atlas "inflammation_${cohort}" --level "${level}" --output-dir "${OUTPUT_DIR}" \
            2>&1 | tee "${LOG_DIR}/inflam_${cohort}_${level}_pseudobulk.log" | grep -E "^\[|Shape:|Saved:|Complete"
    done
done

# --- scAtlas (2 datasets × 3 levels) ---
echo ""
echo "[SCATLAS] Processing 2 datasets × 3 levels (~6.4M cells)..."
for dataset in normal cancer; do
    for level in organ_celltype celltype organ; do
        echo "  [$(date +%H:%M:%S)] Starting scatlas_${dataset} ${level}..."
        python "${SCRIPT_DIR}/09_atlas_validation_pseudobulk.py" \
            --atlas "scatlas_${dataset}" --level "${level}" --output-dir "${OUTPUT_DIR}" \
            2>&1 | tee "${LOG_DIR}/scatlas_${dataset}_${level}_pseudobulk.log" | grep -E "^\[|Shape:|Saved:|Complete"
    done
done

STAGE1_END=$(date +%s)
STAGE1_TIME=$((STAGE1_END - STAGE1_START))
echo ""
echo "Stage 1 complete in $((STAGE1_TIME / 3600))h $((STAGE1_TIME % 3600 / 60))m"

# =============================================================================
# STAGE 2: ACTIVITY INFERENCE
# =============================================================================

echo ""
echo "=============================================================="
echo "STAGE 2: ACTIVITY INFERENCE"
echo "=============================================================="

STAGE2_START=$(date +%s)

# --- CIMA ---
echo ""
echo "[CIMA] Running activity inference on 4 levels..."
for level in L1 L2 L3 L4; do
    echo "  [$(date +%H:%M:%S)] Starting CIMA ${level} activity..."
    python "${SCRIPT_DIR}/09_atlas_validation_activity.py" \
        --atlas cima --level "${level}" --validate \
        2>&1 | tee "${LOG_DIR}/cima_${level}_activity.log" | grep -E "^\[|Running|Saved:|Complete|Validated"
done

# --- Inflammation ---
echo ""
echo "[INFLAMMATION] Running activity inference on 6 configs..."
for cohort in main val ext; do
    for level in L1 L2; do
        echo "  [$(date +%H:%M:%S)] Starting inflammation_${cohort} ${level} activity..."
        python "${SCRIPT_DIR}/09_atlas_validation_activity.py" \
            --atlas "inflammation_${cohort}" --level "${level}" --validate \
            2>&1 | tee "${LOG_DIR}/inflam_${cohort}_${level}_activity.log" | grep -E "^\[|Running|Saved:|Complete|Validated"
    done
done

# --- scAtlas ---
echo ""
echo "[SCATLAS] Running activity inference on 6 configs..."
for dataset in normal cancer; do
    for level in organ_celltype celltype organ; do
        echo "  [$(date +%H:%M:%S)] Starting scatlas_${dataset} ${level} activity..."
        python "${SCRIPT_DIR}/09_atlas_validation_activity.py" \
            --atlas "scatlas_${dataset}" --level "${level}" --validate \
            2>&1 | tee "${LOG_DIR}/scatlas_${dataset}_${level}_activity.log" | grep -E "^\[|Running|Saved:|Complete|Validated"
    done
done

STAGE2_END=$(date +%s)
STAGE2_TIME=$((STAGE2_END - STAGE2_START))
echo ""
echo "Stage 2 complete in $((STAGE2_TIME / 3600))h $((STAGE2_TIME % 3600 / 60))m"

# =============================================================================
# STAGE 3: CROSS-ATLAS VALIDATION
# =============================================================================

echo ""
echo "=============================================================="
echo "STAGE 3: CROSS-ATLAS VALIDATION"
echo "=============================================================="

VALIDATION_DIR="${OUTPUT_DIR}/validation"
mkdir -p "${VALIDATION_DIR}"

python << 'EOF'
import json
from pathlib import Path
import pandas as pd
import numpy as np

OUTPUT_DIR = Path("/vf/users/parks34/projects/2secactpy/results/atlas_validation")
VALIDATION_DIR = OUTPUT_DIR / "validation"

# Collect all validation CSVs
validation_files = list(OUTPUT_DIR.rglob("*_validation.csv"))
print(f"Found {len(validation_files)} validation files")

all_results = []
for vf in validation_files:
    try:
        df = pd.read_csv(vf)
        parts = vf.stem.split('_')
        df['source_file'] = vf.name
        df['atlas'] = parts[0]
        all_results.append(df)
    except Exception as e:
        print(f"Error reading {vf}: {e}")

if all_results:
    combined_df = pd.concat(all_results, ignore_index=True)
    combined_df.to_csv(VALIDATION_DIR / "all_validation_results.csv", index=False)

    summary = {
        'total_validations': len(combined_df),
        'mean_pearson_r': float(combined_df['pearson_r'].mean()) if 'pearson_r' in combined_df.columns else None,
        'median_pearson_r': float(combined_df['pearson_r'].median()) if 'pearson_r' in combined_df.columns else None,
        'significant_count': int((combined_df['pearson_q'] < 0.05).sum()) if 'pearson_q' in combined_df.columns else 0,
    }

    with open(VALIDATION_DIR / "validation_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nValidation Summary:")
    print(f"  Total: {summary['total_validations']}")
    print(f"  Mean Pearson r: {summary['mean_pearson_r']:.3f}" if summary['mean_pearson_r'] else "  Mean Pearson r: N/A")
    print(f"  Significant (q<0.05): {summary['significant_count']}")

# Gene coverage
import anndata as ad
coverage_data = []
for h5ad_file in OUTPUT_DIR.rglob("**/activity/*.h5ad"):
    try:
        adata = ad.read_h5ad(h5ad_file)
        coverage_data.append({
            'file': h5ad_file.name,
            'signature': adata.uns.get('signature', 'unknown'),
            'gene_overlap': adata.uns.get('gene_overlap', 0),
            'n_samples': adata.n_obs,
        })
    except:
        pass

if coverage_data:
    coverage_df = pd.DataFrame(coverage_data)
    coverage_df.to_csv(VALIDATION_DIR / "gene_coverage.csv", index=False)
    print(f"\nGene Coverage by Signature:")
    print(coverage_df.groupby('signature')['gene_overlap'].agg(['mean', 'min', 'max']).round(3))

print("\nCross-atlas validation complete.")
EOF

# =============================================================================
# SUMMARY
# =============================================================================

TOTAL_END=$(date +%s)
TOTAL_TIME=$((TOTAL_END - TOTAL_START))

echo ""
echo "=============================================================="
echo "PIPELINE COMPLETE"
echo "=============================================================="
echo "End time: $(date)"
echo ""
echo "Timing:"
echo "  Stage 1 (Pseudobulk): $((STAGE1_TIME / 3600))h $((STAGE1_TIME % 3600 / 60))m $((STAGE1_TIME % 60))s"
echo "  Stage 2 (Activity):   $((STAGE2_TIME / 3600))h $((STAGE2_TIME % 3600 / 60))m $((STAGE2_TIME % 60))s"
echo "  TOTAL:                $((TOTAL_TIME / 3600))h $((TOTAL_TIME % 3600 / 60))m $((TOTAL_TIME % 60))s"
echo ""
echo "Output files:"
echo "  Pseudobulk: $(find ${OUTPUT_DIR} -name '*_pseudobulk_*.h5ad' | wc -l) files"
echo "  Activity:   $(find ${OUTPUT_DIR} -name '*.h5ad' -path '*/activity/*' | wc -l) files"
echo "  Validation: $(find ${OUTPUT_DIR} -name '*_validation.csv' | wc -l) files"
echo ""
echo "Results directory: ${OUTPUT_DIR}"
