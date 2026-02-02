#!/bin/bash
#SBATCH --job-name=alphagenome
#SBATCH --output=/data/parks34/projects/2secactpy/logs/alphagenome_%j.out
#SBATCH --error=/data/parks34/projects/2secactpy/logs/alphagenome_%j.err
#SBATCH --time=168:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=norm

# Note: Stage 3 API calls take ~15 sec/variant
# Full run (29,816 variants) takes ~125 hours
# Use --resume flag to continue from checkpoint if job restarts

###############################################################################
# AlphaGenome eQTL Analysis Pipeline
#
# Prioritize regulatory variants from CIMA immune cell cis-eQTLs using
# AlphaGenome deep learning predictions.
#
# Usage:
#   sbatch run_alphagenome.sh               # Run all stages
#   sbatch run_alphagenome.sh --stage 1     # Run specific stage
#   sbatch run_alphagenome.sh --mock        # Use mock predictions (testing)
#   sbatch run_alphagenome.sh --resume      # Resume Stage 3 from checkpoint
#   sbatch run_alphagenome.sh --reset       # Reset checkpoint (start fresh)
###############################################################################

set -e

# Parse arguments
STAGE=""
MOCK=""
RESUME=""
RESET=""
MAX_VARIANTS=""
OUTPUT_SUFFIX=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --stage)
            STAGE="$2"
            shift 2
            ;;
        --mock)
            MOCK="--mock"
            shift
            ;;
        --resume)
            RESUME="--resume"
            shift
            ;;
        --reset)
            RESET="--reset"
            shift
            ;;
        --max-variants)
            MAX_VARIANTS="--max-variants $2"
            shift 2
            ;;
        --output-suffix)
            OUTPUT_SUFFIX="--output-suffix $2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Environment setup
source ~/bin/myconda
conda activate secactpy

# AlphaGenome API key (must be set in ~/.bashrc or environment)
if [ -z "$ALPHAGENOME_API_KEY" ]; then
    echo "ERROR: ALPHAGENOME_API_KEY environment variable not set"
    echo "Add to ~/.bashrc: export ALPHAGENOME_API_KEY='your-key'"
    exit 1
fi

# Working directory
cd /data/parks34/projects/2secactpy

# Create logs directory
mkdir -p logs

# Log start
echo "========================================"
echo "AlphaGenome eQTL Analysis Pipeline"
echo "========================================"
echo "Start time: $(date)"
echo "SLURM Job ID: ${SLURM_JOB_ID:-local}"
echo "Stage: ${STAGE:-all}"
echo "Mock mode: ${MOCK:-no}"
echo "Resume: ${RESUME:-no}"
echo "Reset: ${RESET:-no}"
echo ""

# Stage 1: Filter eQTLs
run_stage1() {
    echo "[$(date +%H:%M:%S)] === STAGE 1: Filter eQTLs ==="
    python scripts/08_alphagenome_stage1_filter.py
    echo ""
}

# Stage 2: Format for AlphaGenome
run_stage2() {
    echo "[$(date +%H:%M:%S)] === STAGE 2: Format variants ==="
    python scripts/08_alphagenome_stage2_format.py
    echo ""
}

# Stage 3: AlphaGenome predictions
run_stage3() {
    echo "[$(date +%H:%M:%S)] === STAGE 3: AlphaGenome predictions ==="
    python scripts/08_alphagenome_stage3_predict.py $MOCK $RESUME $RESET $MAX_VARIANTS $OUTPUT_SUFFIX
    echo ""
}

# Stage 4: Interpret predictions
run_stage4() {
    echo "[$(date +%H:%M:%S)] === STAGE 4: Interpret predictions ==="
    python scripts/08_alphagenome_stage4_interpret.py
    echo ""
}

# Stage 5: GTEx validation
run_stage5() {
    echo "[$(date +%H:%M:%S)] === STAGE 5: GTEx validation ==="
    python scripts/08_alphagenome_stage5_validate.py
    echo ""
}

# Run selected stage(s)
if [ -n "$STAGE" ]; then
    case $STAGE in
        1) run_stage1 ;;
        2) run_stage2 ;;
        3) run_stage3 ;;
        4) run_stage4 ;;
        5) run_stage5 ;;
        *)
            echo "Invalid stage: $STAGE (must be 1-5)"
            exit 1
            ;;
    esac
else
    # Run all stages
    run_stage1
    run_stage2
    run_stage3
    run_stage4
    run_stage5
fi

echo "========================================"
echo "Pipeline complete!"
echo "End time: $(date)"
echo "========================================"

# List output files
echo ""
echo "Output files:"
ls -lh results/alphagenome/

echo ""
echo "Summary:"
if [ -f results/alphagenome/stage1_summary.json ]; then
    echo "Stage 1: $(python -c "import json; d=json.load(open('results/alphagenome/stage1_summary.json')); print(f\"{d['output']['cytokine_eqtls']} cytokine eQTLs\")")"
fi
if [ -f results/alphagenome/stage2_summary.json ]; then
    echo "Stage 2: $(python -c "import json; d=json.load(open('results/alphagenome/stage2_summary.json')); print(f\"{d['output']['unique_variants']} unique variants\")")"
fi
if [ -f results/alphagenome/stage4_summary.json ]; then
    echo "Stage 4: $(python -c "import json; d=json.load(open('results/alphagenome/stage4_summary.json')); print(f\"{d['output']['prioritized_variants']} prioritized variants\")")"
fi
if [ -f results/alphagenome/stage5_validation_metrics.json ]; then
    echo "Stage 5: $(python -c "import json; d=json.load(open('results/alphagenome/stage5_validation_metrics.json')); print(f\"{d['concordance']['concordance']*100:.1f}% concordance\")")"
fi
