#!/bin/bash
#SBATCH --job-name=gene_expr_v2
#SBATCH --output=/vf/users/parks34/projects/2secactpy/logs/gene_expr_v2_%j.out
#SBATCH --error=/vf/users/parks34/projects/2secactpy/logs/gene_expr_v2_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --mem=240G
#SBATCH --cpus-per-task=4
#SBATCH --partition=norm

source ~/bin/myconda
conda activate secactpy

cd /vf/users/parks34/projects/2secactpy

echo "Starting gene expression extraction v2 (with normalization fixes)"
echo "Start time: $(date)"

python scripts/07c_sampled_gene_expression.py

echo "End time: $(date)"
echo "Gene expression extraction complete"
