#!/bin/bash
#SBATCH --job-name=validation_test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=/vf/users/parks34/projects/2secactpy/logs/validation_test_%j.out
#SBATCH --error=/vf/users/parks34/projects/2secactpy/logs/validation_test_%j.err

# Single test to verify pipeline works
# Usage: sbatch run_test_single.sh

set -e

source ~/bin/myconda
conda activate secactpy

cd /vf/users/parks34/projects/2secactpy
mkdir -p logs

echo "========================================"
echo "Validation Pipeline Test"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Date: $(date)"
echo "========================================"

# Run first test (CIMA / CytoSig / Pseudobulk / L1)
python scripts/validation/run_validation.py --config-index 0

echo "========================================"
echo "Test completed: $(date)"
echo "========================================"
