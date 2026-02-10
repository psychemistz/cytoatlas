#!/bin/bash
#SBATCH --job-name=hf_download
#SBATCH --partition=norm
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=/vf/users/parks34/projects/2cytoatlas/logs/hf_download_%j.out
#SBATCH --error=/vf/users/parks34/projects/2cytoatlas/logs/hf_download_%j.err

# Download HuggingFace datasets: Genecorpus-30M and SpatialCorpus-110M
# Usage: sbatch run_download_hf_datasets.sh

set -e

source ~/bin/myconda
conda activate secactpy

cd /vf/users/parks34/projects/2cytoatlas
mkdir -p logs

OUTPUT_DIR="/data/Jiang_Lab/Data/Seongyong"

echo "========================================"
echo "HuggingFace Dataset Download"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "Output Directory: $OUTPUT_DIR"
echo "========================================"

# Download both datasets
python scripts/download_hf_datasets.py --dataset all --output-dir "$OUTPUT_DIR"

echo "========================================"
echo "Download completed: $(date)"
echo "========================================"

# List downloaded contents
echo ""
echo "Downloaded contents:"
ls -lh "$OUTPUT_DIR/Genecorpus-30M" 2>/dev/null || echo "Genecorpus-30M: not found"
echo ""
ls -lh "$OUTPUT_DIR/SpatialCorpus-110M" 2>/dev/null || echo "SpatialCorpus-110M: not found"
