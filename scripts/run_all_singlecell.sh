#!/bin/bash
# Run all single-cell activity inference on current node

set -e

cd /vf/users/parks34/projects/2secactpy
source ~/bin/myconda
conda activate secactpy

SCRIPT="scripts/12_singlecell_activity.py"
LOG_DIR="logs/validation"
mkdir -p "${LOG_DIR}"

echo "=========================================="
echo "Single-Cell Activity Inference - All Atlases"
echo "=========================================="
echo "Start: $(date)"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.free --format=csv
echo ""

ATLASES="cima inflammation_main inflammation_val inflammation_ext scatlas_normal scatlas_cancer"
SIGNATURES="cytosig lincytosig secact"

for atlas in $ATLASES; do
    for sig in $SIGNATURES; do
        echo ""
        echo "=========================================="
        echo "Processing: ${atlas} / ${sig}"
        echo "Time: $(date)"
        echo "=========================================="

        python "${SCRIPT}" --atlas "${atlas}" --signature "${sig}" --skip-existing 2>&1

        echo "Completed: ${atlas} / ${sig}"
    done
done

echo ""
echo "=========================================="
echo "ALL COMPLETE"
echo "End: $(date)"
echo "=========================================="
