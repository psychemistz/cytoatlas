#!/bin/bash
#SBATCH --job-name=gene_expr
#SBATCH --output=/vf/users/parks34/projects/2secactpy/logs/gene_expr_%j.out
#SBATCH --error=/vf/users/parks34/projects/2secactpy/logs/gene_expr_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --partition=norm

# Load environment
source ~/bin/myconda
conda activate secactpy

# Change to project directory
cd /vf/users/parks34/projects/2secactpy

# Run preprocessing (sampled version for memory efficiency)
python scripts/07c_sampled_gene_expression.py

echo "Gene expression preprocessing complete"
