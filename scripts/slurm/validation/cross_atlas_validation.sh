#!/bin/bash
#SBATCH --job-name=cross_atlas
#SBATCH --time=2:00:00
#SBATCH --mem=64G
#SBATCH --partition=norm
#SBATCH --output=logs/validation/cross_atlas_%j.out
#SBATCH --error=logs/validation/cross_atlas_%j.err

# =============================================================================
# Cross-Atlas Validation
# =============================================================================
# Aggregates validation results and computes cross-atlas consistency metrics.
#
# Usage:
#   sbatch cross_atlas_validation.sh
#
# Dependencies:
#   - All atlas-specific activity jobs must complete first
#
# =============================================================================

set -e

# Paths
PROJECT_DIR="/vf/users/parks34/projects/2cytoatlas"
SCRIPT_DIR="${PROJECT_DIR}/scripts"
LOG_DIR="${PROJECT_DIR}/logs/validation"
OUTPUT_DIR="${PROJECT_DIR}/results/atlas_validation"
VALIDATION_DIR="${OUTPUT_DIR}/validation"

mkdir -p "${LOG_DIR}"
mkdir -p "${VALIDATION_DIR}"

# Environment
echo "Setting up environment..."
source ~/bin/myconda
conda activate secactpy

cd "${PROJECT_DIR}"

# Info
echo "=============================================================="
echo "Cross-Atlas Validation"
echo "=============================================================="
echo "Start time: $(date)"
echo "Output: ${VALIDATION_DIR}"
echo "=============================================================="

# Aggregate all validation CSVs
echo "Aggregating validation results..."

python << 'EOF'
import os
import json
from pathlib import Path
import pandas as pd
import numpy as np

OUTPUT_DIR = Path("/vf/users/parks34/projects/2cytoatlas/results/atlas_validation")
VALIDATION_DIR = OUTPUT_DIR / "validation"
VALIDATION_DIR.mkdir(exist_ok=True)

# Collect all validation CSVs
validation_files = list(OUTPUT_DIR.rglob("*_validation.csv"))
print(f"Found {len(validation_files)} validation files")

all_results = []
for vf in validation_files:
    df = pd.read_csv(vf)
    # Extract atlas and signature from path
    parts = vf.stem.split('_')
    df['source_file'] = vf.name
    df['atlas'] = parts[0]
    all_results.append(df)

if all_results:
    combined_df = pd.concat(all_results, ignore_index=True)
    combined_df.to_csv(VALIDATION_DIR / "all_validation_results.csv", index=False)
    print(f"Combined {len(combined_df)} validation records")

    # Summary statistics
    summary = {
        'total_validations': len(combined_df),
        'atlases': combined_df['atlas'].nunique() if 'atlas' in combined_df.columns else 0,
        'mean_pearson_r': combined_df['pearson_r'].mean() if 'pearson_r' in combined_df.columns else None,
        'median_pearson_r': combined_df['pearson_r'].median() if 'pearson_r' in combined_df.columns else None,
        'mean_spearman_r': combined_df['spearman_r'].mean() if 'spearman_r' in combined_df.columns else None,
        'significant_count': (combined_df['pearson_q'] < 0.05).sum() if 'pearson_q' in combined_df.columns else 0,
        'files_processed': len(validation_files),
    }

    with open(VALIDATION_DIR / "validation_summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print("\nSummary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
else:
    print("No validation files found!")

# Gene coverage summary
coverage_data = []
for atlas_dir in OUTPUT_DIR.iterdir():
    if not atlas_dir.is_dir() or atlas_dir.name == 'validation':
        continue
    activity_dir = atlas_dir / 'activity'
    if not activity_dir.exists():
        continue

    for h5ad_file in activity_dir.glob("*.h5ad"):
        import anndata as ad
        try:
            adata = ad.read_h5ad(h5ad_file)
            coverage_data.append({
                'atlas': atlas_dir.name,
                'file': h5ad_file.name,
                'signature': adata.uns.get('signature', 'unknown'),
                'gene_overlap': adata.uns.get('gene_overlap', 0),
                'n_genes_used': adata.uns.get('n_genes_used', 0),
                'n_samples': adata.n_obs,
                'n_signatures': adata.n_vars,
            })
        except Exception as e:
            print(f"Error reading {h5ad_file}: {e}")

if coverage_data:
    coverage_df = pd.DataFrame(coverage_data)
    coverage_df.to_csv(VALIDATION_DIR / "gene_coverage.csv", index=False)
    print(f"\nGene coverage: {len(coverage_df)} activity files")
    print(coverage_df.groupby('signature')['gene_overlap'].agg(['mean', 'min', 'max']))

print("\nValidation complete.")
EOF

echo ""
echo "=============================================================="
echo "Cross-Atlas Validation Complete"
echo "End time: $(date)"
echo "=============================================================="

# List outputs
echo ""
echo "Output files:"
ls -la "${VALIDATION_DIR}/"
