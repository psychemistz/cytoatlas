#!/bin/bash
#SBATCH --job-name=boxplot_data
#SBATCH --output=/vf/users/parks34/projects/2secactpy/logs/boxplot_data_%j.out
#SBATCH --error=/vf/users/parks34/projects/2secactpy/logs/boxplot_data_%j.err
#SBATCH --time=8:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --partition=norm

# Generate box plot data (quartiles) for expression and activity

source ~/bin/myconda
conda activate secactpy

cd /vf/users/parks34/projects/2secactpy

echo "Starting box plot data generation at $(date)"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"

python scripts/07f_boxplot_data.py

echo "Finished at $(date)"
