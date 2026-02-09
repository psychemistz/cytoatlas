#!/bin/bash
#SBATCH --job-name=cima_expr_bp
#SBATCH --output=/vf/users/parks34/projects/2secactpy/logs/cima_expr_boxplot_%j.out
#SBATCH --error=/vf/users/parks34/projects/2secactpy/logs/cima_expr_boxplot_%j.err
#SBATCH --time=2:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=4
#SBATCH --partition=quick

# Extract CIMA gene expression with 27 cell types and boxplot statistics

source ~/bin/myconda
conda activate secactpy

export PYTHONUNBUFFERED=1

cd /vf/users/parks34/projects/2secactpy

echo "Starting CIMA gene expression extraction at $(date)"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"

python scripts/07e_cima_gene_expression_boxplot.py

echo "Finished at $(date)"
