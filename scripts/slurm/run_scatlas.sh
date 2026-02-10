#!/bin/bash
#SBATCH --job-name=scatlas_analysis
#SBATCH --output=logs/scatlas_%j.out
#SBATCH --error=logs/scatlas_%j.err
#SBATCH --time=08:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8

# Pan-Disease Cytokine Activity Atlas
# scAtlas Analysis Script
# Computes activities from raw counts using SecActpy (GPU-accelerated)

echo "========================================"
echo "scATLAS ANALYSIS"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo "========================================"

# Load modules
module load CUDA/12.8.1
module load cuDNN/9.12.0/CUDA-12
source ~/bin/myconda
conda activate secactpy

cd /data/parks34/projects/2cytoatlas
mkdir -p logs

# Run scAtlas analysis
# Mode options: normal, cancer, comparison, all
python scripts/03_scatlas_analysis.py \
    --mode all

echo "========================================"
echo "End: $(date)"
echo "========================================"
