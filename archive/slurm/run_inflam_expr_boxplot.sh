#!/bin/bash
#SBATCH --job-name=inflam_expr_bp
#SBATCH --output=/vf/users/parks34/projects/2cytoatlas/logs/inflam_expr_boxplot_%j.out
#SBATCH --error=/vf/users/parks34/projects/2cytoatlas/logs/inflam_expr_boxplot_%j.err
#SBATCH --time=2:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4
#SBATCH --partition=quick

# Extract Inflammation Atlas gene expression with 17 cell types and boxplot statistics

source ~/bin/myconda
conda activate secactpy

export PYTHONUNBUFFERED=1

cd /vf/users/parks34/projects/2cytoatlas

echo "Starting Inflammation Atlas gene expression extraction at $(date)"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"

python scripts/07e_inflammation_gene_expression_boxplot.py

echo "Finished at $(date)"
