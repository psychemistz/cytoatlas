#!/bin/bash
#SBATCH --job-name=inflam_symbol
#SBATCH --output=logs/validation/inflam_symbol_%j.out
#SBATCH --error=logs/validation/inflam_symbol_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=250G
#SBATCH --cpus-per-task=8
#SBATCH --partition=norm

# Create Symbol-Indexed Inflammation H5AD Files
# The main file is 17GB and expands to ~200GB in memory

set -e
cd /vf/users/parks34/projects/2cytoatlas
mkdir -p logs/validation

source ~/bin/myconda
conda activate secactpy

echo "=============================================================="
echo "Create Symbol-Indexed Inflammation H5AD Files"
echo "=============================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo ""

# Process all cohorts
python scripts/create_inflammation_symbol_h5ad.py --all

echo ""
echo "=============================================================="
echo "Complete: $(date)"
echo "=============================================================="
