#!/bin/bash
#SBATCH --job-name=all_gene_expr
#SBATCH --output=/vf/users/parks34/projects/2secactpy/logs/all_gene_expr_%j.out
#SBATCH --error=/vf/users/parks34/projects/2secactpy/logs/all_gene_expr_%j.err
#SBATCH --time=24:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=4
#SBATCH --partition=norm

# Extract ALL gene expression from all atlases
# Expected runtime: 8-12 hours
# Memory: ~100GB peak

source ~/bin/myconda
conda activate secactpy

cd /vf/users/parks34/projects/2secactpy

echo "Starting all gene expression extraction at $(date)"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"

python scripts/07d_all_gene_expression.py

echo "Finished at $(date)"
