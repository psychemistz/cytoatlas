#!/bin/bash
# Pan-Disease Cytokine Activity Atlas
# Master submission script with job dependencies
#
# Usage:
#   ./run_all.sh           # Submit all jobs
#   ./run_all.sh --pilot   # Submit pilot only
#   ./run_all.sh --main    # Submit main analyses only (no pilot)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

# Create directories
mkdir -p logs
mkdir -p results/{pilot,cima,inflammation,scatlas,integrated,figures}

echo "========================================"
echo "Pan-Disease Cytokine Activity Atlas"
echo "Master Job Submission"
echo "========================================"
echo "Script directory: $SCRIPT_DIR"
echo "Working directory: $(pwd)"
echo ""

# Parse arguments
RUN_PILOT=true
RUN_MAIN=true

if [[ "$1" == "--pilot" ]]; then
    RUN_MAIN=false
fi
if [[ "$1" == "--main" ]]; then
    RUN_PILOT=false
fi

# Submit jobs
PILOT_JOB=""
CIMA_JOB=""
INFLAM_JOB=""
SCATLAS_JOB=""
INTEGRATED_JOB=""
FIGURES_JOB=""

# Phase 0: Pilot Analysis
if $RUN_PILOT; then
    echo "Submitting Phase 0: Pilot Analysis..."
    PILOT_JOB=$(sbatch --parsable "$SCRIPT_DIR/run_pilot.sh")
    echo "  Job ID: $PILOT_JOB"
fi

# Phase 1-3: Main Analyses (can run in parallel)
if $RUN_MAIN; then
    # Set dependency if pilot was run
    DEP=""
    if [[ -n "$PILOT_JOB" ]]; then
        DEP="--dependency=afterok:$PILOT_JOB"
    fi

    echo "Submitting Phase 1: CIMA Analysis..."
    CIMA_JOB=$(sbatch --parsable $DEP "$SCRIPT_DIR/run_cima.sh")
    echo "  Job ID: $CIMA_JOB"

    echo "Submitting Phase 2: Inflammation Atlas..."
    INFLAM_JOB=$(sbatch --parsable $DEP "$SCRIPT_DIR/run_inflam.sh")
    echo "  Job ID: $INFLAM_JOB"

    echo "Submitting Phase 3: scAtlas Analysis..."
    SCATLAS_JOB=$(sbatch --parsable $DEP "$SCRIPT_DIR/run_scatlas.sh")
    echo "  Job ID: $SCATLAS_JOB"

    # Phase 4: Integrated Analysis (depends on 1-3)
    echo "Submitting Phase 4: Integrated Analysis..."
    INTEGRATED_JOB=$(sbatch --parsable \
        --dependency=afterok:$CIMA_JOB:$INFLAM_JOB:$SCATLAS_JOB \
        "$SCRIPT_DIR/run_integrated.sh")
    echo "  Job ID: $INTEGRATED_JOB"

    # Phase 5: Figure Generation (depends on 4)
    echo "Submitting Phase 5: Figure Generation..."
    FIGURES_JOB=$(sbatch --parsable \
        --dependency=afterok:$INTEGRATED_JOB \
        "$SCRIPT_DIR/run_figures.sh")
    echo "  Job ID: $FIGURES_JOB"
fi

echo ""
echo "========================================"
echo "All jobs submitted!"
echo "========================================"
echo ""
echo "Monitor progress:"
echo "  squeue -u $USER"
echo ""
echo "View logs:"
echo "  tail -f logs/*.out"
echo ""
echo "Job summary:"
if [[ -n "$PILOT_JOB" ]]; then
    echo "  Pilot:      $PILOT_JOB"
fi
if [[ -n "$CIMA_JOB" ]]; then
    echo "  CIMA:       $CIMA_JOB"
    echo "  Inflam:     $INFLAM_JOB"
    echo "  scAtlas:    $SCATLAS_JOB"
    echo "  Integrated: $INTEGRATED_JOB"
    echo "  Figures:    $FIGURES_JOB"
fi
