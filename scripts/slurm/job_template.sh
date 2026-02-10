#!/bin/bash
#SBATCH --job-name={{JOB_NAME}}
#SBATCH --output=logs/{{LOG_PREFIX}}_%j.out
#SBATCH --error=logs/{{LOG_PREFIX}}_%j.err
#SBATCH --time={{TIME}}
#SBATCH --partition={{PARTITION}}
#SBATCH --mem={{MEM}}
#SBATCH --cpus-per-task={{CPUS}}
{{GRES_LINE}}
{{MAIL_LINE}}

# ============================================================
# {{DISPLAY_NAME}}
# Generated from jobs.yaml by submit_jobs.py
# ============================================================

set -e

echo "============================================================"
echo "{{DISPLAY_NAME}}"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo "============================================================"

# Environment setup
{{MODULE_LINES}}
source ~/bin/myconda
conda activate {{CONDA_ENV}}

cd {{PROJECT_DIR}}
mkdir -p logs

# GPU check (if applicable)
{{GPU_CHECK}}

# Run
python {{SCRIPT}} {{ARGS}}

echo "============================================================"
echo "Complete: $(date)"
echo "============================================================"
