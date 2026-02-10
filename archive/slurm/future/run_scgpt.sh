#!/bin/bash
#SBATCH --job-name=scgpt_analysis
#SBATCH --output=logs/scgpt_%j.out
#SBATCH --error=logs/scgpt_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=256G
#SBATCH --cpus-per-task=16

# Pan-Disease Cytokine Activity Atlas
# scGPT Cohort Analysis (~35M cells)

echo "========================================"
echo "SCGPT ANALYSIS"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start: $(date)"
echo "========================================"

# Load modules
module load CUDA/12.8.1
module load cuDNN/9.12.0/CUDA-12
source ~/bin/myconda
conda activate secactpy

cd /data/parks34/projects/2cytoatlas
mkdir -p logs

python scripts/18_scgpt_analysis.py --mode pseudobulk

echo "========================================"
echo "End: $(date)"
echo "========================================"
