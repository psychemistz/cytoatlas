#!/bin/bash
#SBATCH --job-name=inflam_expr
#SBATCH --output=/vf/users/parks34/projects/2cytoatlas/logs/inflam_expr_%j.out
#SBATCH --error=/vf/users/parks34/projects/2cytoatlas/logs/inflam_expr_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --partition=norm

source ~/bin/myconda
conda activate secactpy

cd /vf/users/parks34/projects/2cytoatlas

python scripts/07d_inflammation_gene_expression.py

echo "Inflammation gene expression complete"
