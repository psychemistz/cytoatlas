#!/bin/bash
#SBATCH --job-name=sc_activity_json
#SBATCH --output=/vf/users/parks34/projects/2cytoatlas/logs/sc_activity_json_%j.out
#SBATCH --error=/vf/users/parks34/projects/2cytoatlas/logs/sc_activity_json_%j.err
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --partition=norm

# Process single-cell activity h5ad files into JSON for web visualization

source ~/bin/myconda
conda activate secactpy

cd /vf/users/parks34/projects/2cytoatlas

echo "Starting single-cell activity JSON extraction at $(date)"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"

python scripts/07e_singlecell_activity_json.py

echo "Finished at $(date)"
