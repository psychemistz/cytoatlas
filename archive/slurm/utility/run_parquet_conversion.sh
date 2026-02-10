#!/bin/bash
#SBATCH --job-name=parquet_convert
#SBATCH --output=logs/parquet_convert_%j.out
#SBATCH --error=logs/parquet_convert_%j.err
#SBATCH --time=2:00:00
#SBATCH --partition=norm
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

# Pan-Disease Cytokine Activity Atlas
# Convert visualization JSON files to Parquet format

echo "========================================"
echo "PARQUET CONVERSION"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start: $(date)"
echo "========================================"

source ~/bin/myconda
conda activate secactpy

cd /data/parks34/projects/2cytoatlas
mkdir -p logs

# Convert all JSON files to Parquet (snappy compression for fast reads)
python scripts/convert_json_to_parquet.py --all

echo "========================================"
echo "End: $(date)"
echo "========================================"
