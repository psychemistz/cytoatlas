#!/usr/bin/env python3
"""
Exploratory Pilot Analysis
===========================
Run CytoSig activity inference on subsets from each atlas to validate
the pipeline and identify promising research directions.

This script:
1. Samples 100K cells from CIMA and Inflammation Atlas
2. Loads pre-computed scAtlas activities
3. Validates biological sanity (cell type signatures)
4. Tests metadata linkage
5. Generates quick summary statistics

Run with: python 00_pilot_analysis.py [--n-cells 100000] [--seed 42]
"""

import os
import sys
import gc
import time
import warnings
from pathlib import Path
from typing import Optional, Dict, Tuple, List

import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
from scipy import stats

# Add SecActpy to path
sys.path.insert(0, '/data/parks34/projects/1ridgesig/SecActpy')
from secactpy import (
    load_cytosig, load_secact,
    ridge_batch, estimate_batch_size,
    CUPY_AVAILABLE
)

# Suppress warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# Configuration
# ==============================================================================

# Data paths
CIMA_H5AD = Path('/data/Jiang_Lab/Data/Seongyong/CIMA/Cell_Atlas/CIMA_RNA_6484974cells_36326genes_compressed.h5ad')
CIMA_BIOCHEM = Path('/data/Jiang_Lab/Data/Seongyong/CIMA/Cell_Atlas/CIMA_Sample_Blood_Biochemistry_Results.csv')
CIMA_METAB = Path('/data/Jiang_Lab/Data/Seongyong/CIMA/Cell_Atlas/CIMA_Sample_Plasma_Metabolites_and_Lipids_Results.csv')
CIMA_SAMPLE_META = Path('/data/Jiang_Lab/Data/Seongyong/CIMA/Metadata/CIMA_Sample_Information_Metadata.csv')

INFLAM_MAIN = Path('/data/Jiang_Lab/Data/Seongyong/Inflammation_Atlas/INFLAMMATION_ATLAS_main_afterQC.h5ad')
INFLAM_SAMPLE_META = Path('/data/Jiang_Lab/Data/Seongyong/Inflammation_Atlas/INFLAMMATION_ATLAS_afterQC_sampleMetadata.csv')

SCATLAS_CYTOSIG = Path('/data/Jiang_Lab/Data/Seongyong/scAtlas_2025/igt_s9_fine_CytoSig_activity.h5ad')
SCATLAS_CANCER_CYTOSIG = Path('/data/Jiang_Lab/Data/Seongyong/scAtlas_2025/PanCancer_igt_s9_fine_CytoSig_activity.h5ad')
SCATLAS_COUNTS = Path('/data/Jiang_Lab/Data/Seongyong/scAtlas_2025/igt_s9_fine_counts.h5ad')

# Output paths
OUTPUT_DIR = Path('/data/parks34/projects/2cytoatlas/results/pilot')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Default parameters
DEFAULT_N_CELLS = 100000
DEFAULT_SEED = 42
N_RAND = 100  # Reduced for pilot
BACKEND = 'cupy' if CUPY_AVAILABLE else 'numpy'

# ==============================================================================
# Utility Functions
# ==============================================================================

def log(msg: str):
    """Print timestamped log message."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


def subsample_adata(adata: ad.AnnData, n_cells: int, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Subsample cells from an AnnData object.

    Returns:
        indices: Array of cell indices
        obs_df: DataFrame of cell metadata for sampled cells
    """
    np.random.seed(seed)
    n_total = adata.shape[0]

    if n_cells >= n_total:
        indices = np.arange(n_total)
    else:
        indices = np.random.choice(n_total, size=n_cells, replace=False)
        indices = np.sort(indices)

    return indices


def aggregate_by_celltype(
    X: np.ndarray,
    cell_types: np.ndarray,
    gene_names: List[str]
) -> pd.DataFrame:
    """
    Aggregate expression by cell type (pseudo-bulk).

    Returns:
        DataFrame: genes x cell_types
    """
    unique_types = np.unique(cell_types)
    aggregated = {}

    for ct in unique_types:
        mask = cell_types == ct
        if sp.issparse(X):
            ct_sum = np.asarray(X[mask, :].sum(axis=0)).ravel()
        else:
            ct_sum = X[mask, :].sum(axis=0)
        aggregated[ct] = ct_sum

    return pd.DataFrame(aggregated, index=gene_names)


def normalize_and_differential(expr_df: pd.DataFrame) -> pd.DataFrame:
    """TPM normalize, log transform, and compute differential."""
    # TPM normalize
    col_sums = expr_df.sum(axis=0)
    col_sums = col_sums.replace(0, 1)
    expr_tpm = expr_df / col_sums * 1e6

    # Log transform
    expr_log = np.log2(expr_tpm + 1)

    # Differential (subtract row mean)
    row_means = expr_log.mean(axis=1)
    diff = expr_log.subtract(row_means, axis=0)

    return diff


def run_activity_inference(
    expr_diff: pd.DataFrame,
    signature: pd.DataFrame,
    sig_name: str
) -> Dict:
    """Run activity inference on differential expression data."""
    log(f"  Running {sig_name} activity inference...")

    # Find overlapping genes
    expr_genes = set(expr_diff.index.str.upper())
    sig_genes = set(signature.index.str.upper())
    common_genes = list(expr_genes & sig_genes)

    log(f"    Common genes: {len(common_genes)} / {len(sig_genes)}")

    if len(common_genes) < 10:
        log(f"    Warning: Too few common genes")
        return None

    # Align data (handle duplicate gene symbols by keeping first occurrence)
    expr_aligned = expr_diff.copy()
    expr_aligned.index = expr_aligned.index.str.upper()
    expr_aligned = expr_aligned[~expr_aligned.index.duplicated(keep='first')]
    expr_aligned = expr_aligned.loc[expr_aligned.index.isin(common_genes)]

    sig_aligned = signature.copy()
    sig_aligned.index = sig_aligned.index.str.upper()
    sig_aligned = sig_aligned[~sig_aligned.index.duplicated(keep='first')]

    # Re-compute common genes after dedup
    common_genes = list(set(expr_aligned.index) & set(sig_aligned.index))
    expr_aligned = expr_aligned.loc[common_genes]
    sig_aligned = sig_aligned.loc[common_genes]

    # Z-score normalize
    expr_scaled = (expr_aligned - expr_aligned.mean()) / expr_aligned.std(ddof=1)
    expr_scaled = expr_scaled.fillna(0)

    sig_scaled = (sig_aligned - sig_aligned.mean()) / sig_aligned.std(ddof=1)
    sig_scaled = sig_scaled.fillna(0)

    # Run ridge regression
    from secactpy import ridge
    result = ridge(
        X=sig_scaled.values,
        Y=expr_scaled.values,
        lambda_=5e5,
        n_rand=N_RAND,
        seed=DEFAULT_SEED,
        backend=BACKEND,
        verbose=False
    )

    # Convert to DataFrames
    feature_names = list(sig_scaled.columns)
    sample_names = list(expr_scaled.columns)

    result_df = {
        'beta': pd.DataFrame(result['beta'], index=feature_names, columns=sample_names),
        'zscore': pd.DataFrame(result['zscore'], index=feature_names, columns=sample_names),
        'pvalue': pd.DataFrame(result['pvalue'], index=feature_names, columns=sample_names),
    }

    log(f"    Activity matrix: {result_df['zscore'].shape}")
    log(f"    Time: {result['time']:.2f}s")

    return result_df


def validate_known_signatures(
    activity_df: pd.DataFrame,
    cell_type_col: str = None
) -> pd.DataFrame:
    """
    Validate known cytokine-cell type associations.

    Known associations to check:
    - IL17 in Th17 cells
    - IFNG in CD8 T cells and NK cells
    - IL10 in regulatory T cells (Tregs)
    - TNF in monocytes/macrophages
    - IL4 in Th2 cells
    """
    known_associations = [
        ('IL17', ['Th17', 'CD4', 'T17']),
        ('IFNG', ['CD8', 'NK', 'CTL']),
        ('IL10', ['Treg', 'regulatory']),
        ('TNF', ['Mono', 'Macro', 'Myeloid']),
        ('IL4', ['Th2', 'CD4']),
        ('IL6', ['Mono', 'Macro', 'Myeloid']),
        ('IL2', ['CD4', 'T cell']),
    ]

    results = []

    for cytokine, expected_celltypes in known_associations:
        # Find cytokine in activity matrix (case-insensitive)
        cyto_matches = [c for c in activity_df.index if cytokine.upper() in c.upper()]

        if not cyto_matches:
            continue

        cyto_name = cyto_matches[0]
        activities = activity_df.loc[cyto_name]

        # Check if expected cell types have high activity
        for celltype in activity_df.columns:
            is_expected = any(exp.lower() in celltype.lower() for exp in expected_celltypes)

            results.append({
                'cytokine': cyto_name,
                'cell_type': celltype,
                'activity': activities[celltype],
                'expected_high': is_expected,
            })

    result_df = pd.DataFrame(results)

    if len(result_df) > 0:
        # Rank within each cytokine
        result_df['rank'] = result_df.groupby('cytokine')['activity'].rank(ascending=False)

        # Check if expected cell types are in top 50%
        n_celltypes = result_df.groupby('cytokine').size().iloc[0] if len(result_df) > 0 else 1
        result_df['in_top_half'] = result_df['rank'] <= (n_celltypes / 2)

        # Summary: expected cell types should be in top half
        validation_summary = result_df[result_df['expected_high']].groupby('cytokine').agg({
            'in_top_half': 'mean',
            'rank': 'mean',
            'activity': 'mean'
        }).reset_index()
        validation_summary.columns = ['cytokine', 'pct_in_top_half', 'avg_rank', 'avg_activity']

        return validation_summary

    return pd.DataFrame()


# ==============================================================================
# Atlas-Specific Analysis Functions
# ==============================================================================

def pilot_cima(n_cells: int, seed: int) -> Dict:
    """Run pilot analysis on CIMA dataset."""
    log("\n" + "=" * 60)
    log("CIMA PILOT ANALYSIS")
    log("=" * 60)

    results = {}

    # Load signatures
    log("Loading CytoSig signatures...")
    cytosig = load_cytosig()
    log(f"  CytoSig shape: {cytosig.shape}")

    # Load h5ad
    log(f"Loading CIMA h5ad (backed mode)...")
    adata = ad.read_h5ad(CIMA_H5AD, backed='r')
    log(f"  Full shape: {adata.shape}")

    # Check available columns
    log(f"  obs columns: {list(adata.obs.columns)[:10]}...")

    # Get cell type column
    cell_type_col = 'cell_type_l2' if 'cell_type_l2' in adata.obs.columns else 'cell_type'
    log(f"  Using cell type column: {cell_type_col}")

    # Subsample
    log(f"Subsampling {n_cells:,} cells...")
    indices = subsample_adata(adata, n_cells, seed)
    log(f"  Sampled {len(indices):,} cells")

    # Get expression data
    log("Loading expression data...")
    if 'counts' in adata.layers:
        X = adata.layers['counts'][indices, :]
        log("  Using raw counts from layers['counts']")
    else:
        X = adata.X[indices, :]
        log("  Using .X")

    # Get cell types
    cell_types = adata.obs[cell_type_col].values[indices]
    gene_names = list(adata.var_names)

    # Get sample IDs for metadata linkage test
    if 'sample' in adata.obs.columns:
        samples = adata.obs['sample'].values[indices]
        unique_samples = np.unique(samples)
        log(f"  Unique samples in subset: {len(unique_samples)}")

    # Free backed adata
    del adata
    gc.collect()

    # Aggregate by cell type
    log("Aggregating by cell type...")
    expr_df = aggregate_by_celltype(X, cell_types, gene_names)
    log(f"  Pseudo-bulk shape: {expr_df.shape}")
    results['n_celltypes'] = expr_df.shape[1]

    # Normalize and compute differential
    log("Normalizing and computing differential...")
    expr_diff = normalize_and_differential(expr_df)

    # Run activity inference
    activity = run_activity_inference(expr_diff, cytosig, 'CytoSig')

    if activity is not None:
        results['activity'] = activity

        # Validate known signatures
        log("Validating known cytokine-cell type associations...")
        validation = validate_known_signatures(activity['zscore'])
        if len(validation) > 0:
            log("  Validation results:")
            for _, row in validation.iterrows():
                status = "PASS" if row['pct_in_top_half'] >= 0.5 else "FAIL"
                log(f"    {row['cytokine']}: {row['pct_in_top_half']*100:.0f}% in top half [{status}]")
            results['validation'] = validation

        # Save activity matrix
        activity['zscore'].to_csv(OUTPUT_DIR / 'cima_pilot_cytosig_activity.csv')
        log(f"  Saved: cima_pilot_cytosig_activity.csv")

    # Test metadata linkage
    log("Testing metadata linkage...")
    if CIMA_BIOCHEM.exists():
        biochem = pd.read_csv(CIMA_BIOCHEM)
        biochem_cols = [c for c in biochem.columns if c not in ['Sample', 'sample']]
        log(f"  Biochemistry markers: {len(biochem_cols)}")
        results['biochem_available'] = True
        results['n_biochem_markers'] = len(biochem_cols)

    if CIMA_METAB.exists():
        metab = pd.read_csv(CIMA_METAB)
        metab_cols = [c for c in metab.columns if c not in ['Sample', 'sample']]
        log(f"  Metabolite features: {len(metab_cols)}")
        results['metab_available'] = True
        results['n_metabolites'] = len(metab_cols)

    log("CIMA pilot complete!")
    return results


def pilot_inflammation(n_cells: int, seed: int) -> Dict:
    """Run pilot analysis on Inflammation Atlas."""
    log("\n" + "=" * 60)
    log("INFLAMMATION ATLAS PILOT ANALYSIS")
    log("=" * 60)

    results = {}

    # Load signatures
    log("Loading CytoSig signatures...")
    cytosig = load_cytosig()

    # Load sample metadata
    log("Loading sample metadata...")
    sample_meta = pd.read_csv(INFLAM_SAMPLE_META)
    log(f"  Total samples: {len(sample_meta)}")
    log(f"  Diseases: {sample_meta['disease'].nunique()}")
    log(f"  Disease list: {list(sample_meta['disease'].unique())[:10]}...")

    # Treatment response stats
    response_counts = sample_meta['therapyResponse'].value_counts()
    log(f"  Treatment response: R={response_counts.get('R', 0)}, NR={response_counts.get('NR', 0)}")
    results['n_responders'] = response_counts.get('R', 0)
    results['n_nonresponders'] = response_counts.get('NR', 0)

    # Load h5ad
    log(f"Loading Inflammation h5ad (backed mode)...")
    adata = ad.read_h5ad(INFLAM_MAIN, backed='r')
    log(f"  Full shape: {adata.shape}")
    log(f"  obs columns: {list(adata.obs.columns)}")

    # Get cell type column
    cell_type_col = 'Level2' if 'Level2' in adata.obs.columns else 'Level1'
    sample_col = 'sampleID' if 'sampleID' in adata.obs.columns else 'sample'
    log(f"  Using cell type column: {cell_type_col}")
    log(f"  Using sample column: {sample_col}")

    # Subsample
    log(f"Subsampling {n_cells:,} cells...")
    indices = subsample_adata(adata, n_cells, seed)
    log(f"  Sampled {len(indices):,} cells")

    # Get expression data (raw counts in .X for this dataset)
    log("Loading expression data...")
    X = adata.X[indices, :]

    # Get cell types and samples
    cell_types = adata.obs[cell_type_col].values[indices]
    samples = adata.obs[sample_col].values[indices]

    # Use gene symbols if available (Inflammation Atlas uses Ensembl IDs as index)
    if 'symbol' in adata.var.columns:
        gene_names = list(adata.var['symbol'].values)
        log("  Using gene symbols from var['symbol']")
    else:
        gene_names = list(adata.var_names)

    unique_samples = np.unique(samples)
    log(f"  Unique samples in subset: {len(unique_samples)}")

    # Free backed adata
    del adata
    gc.collect()

    # Aggregate by cell type
    log("Aggregating by cell type...")
    expr_df = aggregate_by_celltype(X, cell_types, gene_names)
    log(f"  Pseudo-bulk shape: {expr_df.shape}")
    results['n_celltypes'] = expr_df.shape[1]

    # Normalize and compute differential
    log("Normalizing and computing differential...")
    expr_diff = normalize_and_differential(expr_df)

    # Run activity inference
    activity = run_activity_inference(expr_diff, cytosig, 'CytoSig')

    if activity is not None:
        results['activity'] = activity

        # Save activity matrix
        activity['zscore'].to_csv(OUTPUT_DIR / 'inflam_pilot_cytosig_activity.csv')
        log(f"  Saved: inflam_pilot_cytosig_activity.csv")

        # Validate known signatures
        log("Validating known cytokine-cell type associations...")
        validation = validate_known_signatures(activity['zscore'])
        if len(validation) > 0:
            log("  Validation results:")
            for _, row in validation.iterrows():
                status = "PASS" if row['pct_in_top_half'] >= 0.5 else "FAIL"
                log(f"    {row['cytokine']}: {row['pct_in_top_half']*100:.0f}% in top half [{status}]")
            results['validation'] = validation

    # Test disease-activity association
    log("Testing disease-activity associations...")
    if activity is not None and 'sample' in dir():
        # Map samples to diseases
        sample_disease = sample_meta.set_index('sampleID')['disease'].to_dict()
        sample_diseases = [sample_disease.get(s, 'unknown') for s in unique_samples]
        disease_counts = pd.Series(sample_diseases).value_counts()
        log(f"  Diseases in subset: {dict(disease_counts.head(10))}")
        results['disease_counts'] = dict(disease_counts)

    log("Inflammation Atlas pilot complete!")
    return results


def pilot_scatlas() -> Dict:
    """Load and analyze pre-computed scAtlas activities."""
    log("\n" + "=" * 60)
    log("scATLAS PILOT ANALYSIS (Pre-computed)")
    log("=" * 60)

    results = {}

    # Load normal CytoSig activities
    if SCATLAS_CYTOSIG.exists():
        log(f"Loading normal organ CytoSig activities...")
        adata = ad.read_h5ad(SCATLAS_CYTOSIG, backed='r')
        log(f"  Shape: {adata.shape}")
        log(f"  Signatures: {adata.n_vars if hasattr(adata, 'n_vars') else adata.shape[1]}")

        # Check metadata
        log(f"  obs columns: {list(adata.obs.columns)[:10]}...")

        if 'tissue' in adata.obs.columns:
            tissues = adata.obs['tissue'].unique()
            log(f"  Organs/tissues: {len(tissues)}")
            results['normal_organs'] = len(tissues)

        results['normal_cells'] = adata.shape[0]
        del adata
        gc.collect()
    else:
        log("  Normal CytoSig activities not found")

    # Load cancer CytoSig activities
    if SCATLAS_CANCER_CYTOSIG.exists():
        log(f"Loading cancer CytoSig activities...")
        adata = ad.read_h5ad(SCATLAS_CANCER_CYTOSIG, backed='r')
        log(f"  Shape: {adata.shape}")

        if 'tissue' in adata.obs.columns:
            tissues = adata.obs['tissue'].unique()
            log(f"  Tissue types: {list(tissues)}")

        if 'cancerType' in adata.obs.columns:
            cancer_types = adata.obs['cancerType'].unique()
            log(f"  Cancer types: {len(cancer_types)}")
            results['cancer_types'] = len(cancer_types)

        results['cancer_cells'] = adata.shape[0]
        del adata
        gc.collect()
    else:
        log("  Cancer CytoSig activities not found")

    # Load sample of counts for verification
    if SCATLAS_COUNTS.exists():
        log(f"Loading normal counts (backed mode)...")
        adata = ad.read_h5ad(SCATLAS_COUNTS, backed='r')
        log(f"  Counts shape: {adata.shape}")
        log(f"  Genes: {adata.n_vars if hasattr(adata, 'n_vars') else adata.shape[1]}")
        results['n_genes'] = adata.shape[1]
        del adata
        gc.collect()

    log("scAtlas pilot complete!")
    return results


# ==============================================================================
# Summary and Decision Point
# ==============================================================================

def generate_pilot_summary(cima_results: Dict, inflam_results: Dict, scatlas_results: Dict):
    """Generate summary of pilot analysis results."""
    log("\n" + "=" * 60)
    log("PILOT ANALYSIS SUMMARY")
    log("=" * 60)

    summary = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'backend': BACKEND,
    }

    # CIMA summary
    log("\nCIMA Atlas:")
    if 'validation' in cima_results:
        val = cima_results['validation']
        avg_pass = (val['pct_in_top_half'] >= 0.5).mean() * 100
        log(f"  Biological validation: {avg_pass:.0f}% of known associations validated")
        summary['cima_validation_rate'] = avg_pass

    log(f"  Cell types analyzed: {cima_results.get('n_celltypes', 'N/A')}")
    log(f"  Biochemistry data: {'Available' if cima_results.get('biochem_available') else 'N/A'}")
    log(f"  Metabolomics data: {'Available' if cima_results.get('metab_available') else 'N/A'}")

    # Inflammation summary
    log("\nInflammation Atlas:")
    if 'validation' in inflam_results:
        val = inflam_results['validation']
        avg_pass = (val['pct_in_top_half'] >= 0.5).mean() * 100
        log(f"  Biological validation: {avg_pass:.0f}% of known associations validated")
        summary['inflam_validation_rate'] = avg_pass

    log(f"  Cell types analyzed: {inflam_results.get('n_celltypes', 'N/A')}")
    log(f"  Treatment response samples: R={inflam_results.get('n_responders', 0)}, NR={inflam_results.get('n_nonresponders', 0)}")
    log(f"  Diseases represented: {len(inflam_results.get('disease_counts', {}))}")

    # scAtlas summary
    log("\nscAtlas:")
    log(f"  Normal organs: {scatlas_results.get('normal_organs', 'N/A')}")
    log(f"  Normal cells: {scatlas_results.get('normal_cells', 'N/A'):,}")
    log(f"  Cancer types: {scatlas_results.get('cancer_types', 'N/A')}")
    log(f"  Cancer cells: {scatlas_results.get('cancer_cells', 'N/A'):,}")

    # Recommendations
    log("\n" + "=" * 60)
    log("RECOMMENDATIONS FOR FULL ANALYSIS")
    log("=" * 60)

    recommendations = []

    # Check biological validation
    cima_val = summary.get('cima_validation_rate', 0)
    inflam_val = summary.get('inflam_validation_rate', 0)

    if cima_val >= 50 and inflam_val >= 50:
        recommendations.append("PROCEED: Biological validation passed for both atlases")
    else:
        recommendations.append("WARNING: Review cell type annotations - some known associations not validated")

    # Check treatment response data
    n_response = inflam_results.get('n_responders', 0) + inflam_results.get('n_nonresponders', 0)
    if n_response >= 100:
        recommendations.append("PROCEED: Sufficient treatment response samples for prediction analysis")
    else:
        recommendations.append("WARNING: Limited treatment response samples - consider focusing on disease comparison")

    # Check metadata linkage
    if cima_results.get('biochem_available') and cima_results.get('metab_available'):
        recommendations.append("PROCEED: CIMA metadata linkage verified - metabolome analysis feasible")

    for rec in recommendations:
        log(f"  {rec}")

    summary['recommendations'] = recommendations

    # Save summary
    import json
    with open(OUTPUT_DIR / 'pilot_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    log(f"\nSaved summary to: {OUTPUT_DIR / 'pilot_summary.json'}")

    return summary


# ==============================================================================
# Entry Point
# ==============================================================================

def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Pilot Analysis for Cytokine Activity Project')
    parser.add_argument('--n-cells', type=int, default=DEFAULT_N_CELLS,
                        help=f'Number of cells to sample (default: {DEFAULT_N_CELLS})')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED,
                        help=f'Random seed (default: {DEFAULT_SEED})')
    parser.add_argument('--skip-cima', action='store_true',
                        help='Skip CIMA analysis')
    parser.add_argument('--skip-inflam', action='store_true',
                        help='Skip Inflammation analysis')
    parser.add_argument('--skip-scatlas', action='store_true',
                        help='Skip scAtlas analysis')
    args = parser.parse_args()

    log("=" * 60)
    log("CYTOKINE ACTIVITY PILOT ANALYSIS")
    log("=" * 60)
    log(f"N cells: {args.n_cells:,}")
    log(f"Seed: {args.seed}")
    log(f"Backend: {BACKEND}")
    log(f"Output: {OUTPUT_DIR}")

    start_time = time.time()

    # Run pilot analyses
    cima_results = {}
    inflam_results = {}
    scatlas_results = {}

    if not args.skip_cima:
        try:
            cima_results = pilot_cima(args.n_cells, args.seed)
        except Exception as e:
            log(f"CIMA pilot failed: {e}")
            import traceback
            traceback.print_exc()

    if not args.skip_inflam:
        try:
            inflam_results = pilot_inflammation(args.n_cells, args.seed)
        except Exception as e:
            log(f"Inflammation pilot failed: {e}")
            import traceback
            traceback.print_exc()

    if not args.skip_scatlas:
        try:
            scatlas_results = pilot_scatlas()
        except Exception as e:
            log(f"scAtlas pilot failed: {e}")
            import traceback
            traceback.print_exc()

    # Generate summary
    generate_pilot_summary(cima_results, inflam_results, scatlas_results)

    total_time = time.time() - start_time
    log(f"\nTotal time: {total_time/60:.1f} minutes")
    log("Pilot analysis complete!")


if __name__ == '__main__':
    main()
