#!/bin/bash
#SBATCH --job-name=sig_expr
#SBATCH --output=/data/parks34/projects/2secactpy/logs/signature_expression_%j.out
#SBATCH --error=/data/parks34/projects/2secactpy/logs/signature_expression_%j.err
#SBATCH --time=4:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --partition=norm

# Compute mean signature gene expression z-scores per cell type
# Used for validation panel (expression vs activity correlation)

set -e

echo "Job started: $(date)"
echo "Node: $(hostname)"
echo "Working directory: $(pwd)"

# Activate conda environment
source ~/bin/myconda
conda activate secactpy

cd /data/parks34/projects/2secactpy

# Create logs directory if needed
mkdir -p logs

# Parse arguments
ATLAS="${1:-all}"
SIG_TYPE="${2:-all}"

echo ""
echo "Atlas: $ATLAS"
echo "Signature type: $SIG_TYPE"
echo ""

# Run the script
python scripts/08_signature_expression.py \
    --atlas "$ATLAS" \
    --signature-type "$SIG_TYPE" \
    --max-cells 5000

echo ""
echo "Job completed: $(date)"
