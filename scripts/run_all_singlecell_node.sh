#!/bin/bash
# Run all single-cell activity jobs on current node

set -e
cd /vf/users/parks34/projects/2secactpy
source ~/bin/myconda
conda activate secactpy

LOG_DIR="logs/validation"
mkdir -p "$LOG_DIR"

SCRIPT="scripts/run_singlecell_streaming.py"

echo "==========================================="
echo "Single-Cell Activity - All Atlases"
echo "==========================================="
echo "Start: $(date)"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.free --format=csv
echo ""

ATLASES="cima inflammation_main inflammation_val inflammation_ext scatlas_normal scatlas_cancer"
SIGNATURES="cytosig lincytosig secact"

for atlas in $ATLASES; do
    for sig in $SIGNATURES; do
        echo ""
        echo "==========================================="
        echo "Processing: ${atlas} / ${sig}"
        echo "Time: $(date)"
        echo "==========================================="

        python "$SCRIPT" --atlas "$atlas" --signature "$sig" --skip-existing 2>&1 | tee "${LOG_DIR}/${atlas}_${sig}_streaming.log"

        echo "Completed: ${atlas} / ${sig}"
    done
done

echo ""
echo "==========================================="
echo "ALL COMPLETE: $(date)"
echo "==========================================="
