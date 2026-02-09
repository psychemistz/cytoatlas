#!/bin/bash
#SBATCH --job-name=inflam_val_sc
#SBATCH --output=/data/parks34/projects/2secactpy/logs/inflam_val_singlecell_%j.out
#SBATCH --error=/data/parks34/projects/2secactpy/logs/inflam_val_singlecell_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=64g
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=gpu

echo "========================================"
echo "INFLAMMATION ATLAS VALIDATION SINGLE-CELL ANALYSIS"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Start: $(date)"
echo "========================================"

# Load modules for GPU
module load CUDA/12.8.1
module load cuDNN/9.12.0/CUDA-12

# Activate conda environment
source ~/bin/myconda
conda activate secactpy

# Run analysis
cd /data/parks34/projects/2secactpy

python scripts/04_singlecell_batch.py \
    --dataset inflam_validation \
    --signature both \
    --batch-size 50000

echo "========================================"
echo "End: $(date)"
echo "========================================"
