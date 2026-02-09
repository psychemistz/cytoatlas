#!/bin/bash
#SBATCH --job-name=regen_pb
#SBATCH --output=/vf/users/parks34/projects/2secactpy/logs/regen_pseudobulk_%j.out
#SBATCH --error=/vf/users/parks34/projects/2secactpy/logs/regen_pseudobulk_%j.err
#SBATCH --time=2:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --partition=norm

source ~/bin/myconda
conda activate secactpy

cd /vf/users/parks34/projects/2secactpy

echo "Starting pseudobulk regeneration..."
python scripts/regen_scatlas_pseudobulk.py

echo "Regenerating boxplot data..."
python scripts/07f_boxplot_data.py

echo "Done!"
