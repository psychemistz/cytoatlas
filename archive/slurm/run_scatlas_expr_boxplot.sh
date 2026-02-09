#!/bin/bash
#SBATCH --job-name=scatlas_expr_bp
#SBATCH --output=/vf/users/parks34/projects/2secactpy/logs/scatlas_expr_boxplot_%j.out
#SBATCH --error=/vf/users/parks34/projects/2secactpy/logs/scatlas_expr_boxplot_%j.err
#SBATCH --time=4:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4
#SBATCH --partition=norm

# Extract scAtlas gene expression with boxplot statistics
# Uses cellType1 (433 cell types) and tissue (organ)

source ~/bin/myconda
conda activate secactpy

export PYTHONUNBUFFERED=1

cd /vf/users/parks34/projects/2secactpy

echo "Starting scAtlas gene expression extraction at $(date)"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"

python scripts/07e_scatlas_gene_expression_boxplot.py

echo "Finished at $(date)"
