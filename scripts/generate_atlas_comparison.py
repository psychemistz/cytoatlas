#!/usr/bin/env python3
"""
Generate Comprehensive Atlas Comparison Data.

Creates multiple comparison views for cross-atlas analysis:
1. Pseudobulk resampled comparison - bootstrap matched samples
2. Single-cell comparison - cell-level activity comparison
3. Cell type aggregated comparison - one point per cell type per condition
4. Prediction concordance - cross-atlas prediction validation

Outputs JSON data for visualization in the Atlas Comparison panel.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad
from scipy import stats
from scipy.spatial.distance import cosine
import warnings
warnings.filterwarnings('ignore')

# Add scripts directory to path for cell_type_mapping
import sys
sys.path.insert(0, str(Path(__file__).parent))
from cell_type_mapping import (
    COARSE_LINEAGES,
    CIMA_TO_COARSE, INFLAMMATION_TO_COARSE, SCATLAS_TO_COARSE,
    CIMA_TO_FINE, INFLAMMATION_TO_FINE, SCATLAS_TO_FINE,
    FINE_TYPES
)

# Paths
RESULTS_DIR = Path('/data/parks34/projects/2secactpy/results')
VIZ_OUTPUT_DIR = Path('/data/parks34/projects/2secactpy/visualization/data')

# Data files
DATA_FILES = {
    'cima': {
        'cytosig_pb': RESULTS_DIR / 'cima' / 'CIMA_CytoSig_pseudobulk.h5ad',
        'secact_pb': RESULTS_DIR / 'cima' / 'CIMA_SecAct_pseudobulk.h5ad',
        'cytosig_sc': RESULTS_DIR / 'cima' / 'CIMA_CytoSig_singlecell.h5ad',
        'secact_sc': RESULTS_DIR / 'cima' / 'CIMA_SecAct_singlecell.h5ad',
    },
    'inflammation': {
        'cytosig_pb': RESULTS_DIR / 'inflammation' / 'main_CytoSig_pseudobulk.h5ad',
        'secact_pb': RESULTS_DIR / 'inflammation' / 'main_SecAct_pseudobulk.h5ad',
        'cytosig_sc': RESULTS_DIR / 'inflammation' / 'main_CytoSig_singlecell.h5ad',
        'secact_sc': RESULTS_DIR / 'inflammation' / 'main_SecAct_singlecell.h5ad',
    },
    'scatlas': {
        'cytosig_sc': RESULTS_DIR / 'scatlas' / 'scatlas_normal_CytoSig_singlecell.h5ad',
        'secact_sc': RESULTS_DIR / 'scatlas' / 'scatlas_normal_SecAct_singlecell.h5ad',
    }
}


def get_coarse_mapping(atlas: str) -> dict:
    """Get coarse mapping for an atlas."""
    if atlas == 'cima':
        return CIMA_TO_COARSE
    elif atlas == 'inflammation':
        return INFLAMMATION_TO_COARSE
    elif atlas == 'scatlas':
        return SCATLAS_TO_COARSE
    return {}


def get_fine_mapping(atlas: str) -> dict:
    """Get fine mapping for an atlas."""
    if atlas == 'cima':
        return CIMA_TO_FINE
    elif atlas == 'inflammation':
        return INFLAMMATION_TO_FINE
    elif atlas == 'scatlas':
        return SCATLAS_TO_FINE
    return {}


def load_pseudobulk_data(atlas: str, sig_type: str) -> tuple:
    """
    Load pseudobulk data and return as DataFrame.

    Pseudobulk h5ad structure:
    - obs: signatures (index = signature names)
    - var: samples (with 'cell_type', 'sample', 'n_cells' columns)
    - X: activity matrix (signatures × samples)

    Returns:
        (activity_df, metadata_df) where activity_df has samples as rows, signatures as columns
    """
    key = f'{sig_type.lower()}_pb'
    path = DATA_FILES.get(atlas, {}).get(key)

    if not path or not path.exists():
        print(f"  [WARN] Pseudobulk file not found: {atlas}/{sig_type}")
        return None, None

    print(f"  Loading {path.name}...")
    adata = ad.read_h5ad(path)

    # Extract data
    X = adata.X
    if hasattr(X, 'toarray'):
        X = X.toarray()

    # Signatures are in obs_names, samples in var_names
    signatures = list(adata.obs_names)
    samples = list(adata.var_names)

    # Transpose: signatures × samples -> samples × signatures
    activity_df = pd.DataFrame(X.T, index=samples, columns=signatures)

    # Get metadata from var
    metadata_df = adata.var.copy()
    metadata_df.index = samples

    print(f"    Shape: {activity_df.shape} (samples × signatures)")
    print(f"    Cell types: {metadata_df['cell_type'].nunique()}")

    return activity_df, metadata_df


def load_singlecell_sample(atlas: str, sig_type: str, n_cells: int = 10000,
                           cell_type_filter: str = None) -> tuple:
    """
    Load a sample of single-cell data.

    Single-cell h5ad structure:
    - obs: cells (with metadata like cell_type, sample, etc.)
    - var: signatures (names in var['_index'] or var_names)
    - X: activity matrix (cells × signatures)

    Returns:
        (activity_df, metadata_df) where activity_df has cells as rows, signatures as columns
    """
    key = f'{sig_type.lower()}_sc'
    path = DATA_FILES.get(atlas, {}).get(key)

    if not path or not path.exists():
        print(f"  [WARN] Single-cell file not found: {atlas}/{sig_type}")
        return None, None

    print(f"  Loading sample from {path.name}...")
    adata = ad.read_h5ad(path, backed='r')

    # Get cell indices, optionally filtered by cell type
    if cell_type_filter:
        cell_type_col = 'cell_type' if 'cell_type' in adata.obs.columns else 'Level2'
        if cell_type_col not in adata.obs.columns:
            # Try subCluster for scAtlas
            cell_type_col = 'subCluster' if 'subCluster' in adata.obs.columns else None

        if cell_type_col:
            mask = adata.obs[cell_type_col] == cell_type_filter
            valid_idx = np.where(mask)[0]
        else:
            valid_idx = np.arange(adata.n_obs)
    else:
        valid_idx = np.arange(adata.n_obs)

    # Sample cells
    np.random.seed(42)
    n_sample = min(n_cells, len(valid_idx))
    sampled_idx = np.random.choice(valid_idx, n_sample, replace=False)
    sampled_idx = sorted(sampled_idx)

    # Load data for sampled cells
    X = adata.X[sampled_idx, :]
    if hasattr(X, 'toarray'):
        X = X.toarray()

    # Get signature names
    if '_index' in adata.var.columns:
        signatures = list(adata.var['_index'].values)
    else:
        signatures = list(adata.var_names)

    # Get metadata
    obs_df = adata.obs.iloc[sampled_idx].copy()

    activity_df = pd.DataFrame(X, columns=signatures)

    print(f"    Sampled: {n_sample} cells, {len(signatures)} signatures")

    return activity_df, obs_df


def compute_celltype_aggregated_comparison(atlas1: str, atlas2: str, sig_type: str,
                                            level: str = 'coarse') -> dict:
    """
    Compute cell type aggregated comparison.

    For each harmonized cell type, compute mean activity across all samples,
    then compare between atlases.

    Args:
        atlas1, atlas2: Atlas names
        sig_type: 'CytoSig' or 'SecAct'
        level: 'coarse' or 'fine'

    Returns:
        dict with comparison data
    """
    print(f"\n  Computing {level} cell type aggregated comparison: {atlas1} vs {atlas2}")

    # Load pseudobulk data
    act1, meta1 = load_pseudobulk_data(atlas1, sig_type)
    act2, meta2 = load_pseudobulk_data(atlas2, sig_type)

    # If pseudobulk not available, skip (single-cell lacks cell type metadata)
    if act1 is None:
        print(f"    [SKIP] No pseudobulk data for {atlas1}")
        return {'data': [], 'correlation': None, 'n': 0, 'note': f'No pseudobulk data for {atlas1}'}

    if act2 is None:
        print(f"    [SKIP] No pseudobulk data for {atlas2}")
        return {'data': [], 'correlation': None, 'n': 0, 'note': f'No pseudobulk data for {atlas2}'}

    if act1 is None or act2 is None:
        return {'data': [], 'correlation': None, 'n': 0}

    # Get mapping
    mapping1 = get_coarse_mapping(atlas1) if level == 'coarse' else get_fine_mapping(atlas1)
    mapping2 = get_coarse_mapping(atlas2) if level == 'coarse' else get_fine_mapping(atlas2)

    # Map cell types
    meta1['harmonized'] = meta1['cell_type'].map(mapping1)
    meta2['harmonized'] = meta2['cell_type'].map(mapping2)

    # Get common harmonized types and signatures
    common_types = set(meta1['harmonized'].dropna()) & set(meta2['harmonized'].dropna())
    common_sigs = set(act1.columns) & set(act2.columns)

    if sig_type == 'SecAct':
        # Limit to top 100 most variable signatures for SecAct
        combined_var = act1[list(common_sigs)].var() + act2[list(common_sigs)].var()
        common_sigs = set(combined_var.nlargest(100).index)

    common_sigs = sorted(common_sigs)
    common_types = sorted(common_types)

    print(f"    Common types: {len(common_types)}, Common signatures: {len(common_sigs)}")

    # Compute mean activity per cell type
    comparison_data = []

    for ct in common_types:
        mask1 = meta1['harmonized'] == ct
        mask2 = meta2['harmonized'] == ct

        if mask1.sum() == 0 or mask2.sum() == 0:
            continue

        mean1 = act1.loc[mask1, common_sigs].mean()
        mean2 = act2.loc[mask2, common_sigs].mean()

        for sig in common_sigs:
            comparison_data.append({
                'signature': sig,
                'cell_type': ct,
                'x': float(mean1[sig]),
                'y': float(mean2[sig]),
                'n_samples_x': int(mask1.sum()),
                'n_samples_y': int(mask2.sum())
            })

    # Compute correlation
    if comparison_data:
        x_vals = [d['x'] for d in comparison_data]
        y_vals = [d['y'] for d in comparison_data]

        # Remove NaN
        valid = [(x, y) for x, y in zip(x_vals, y_vals)
                 if not (np.isnan(x) or np.isnan(y))]

        if len(valid) > 2:
            x_clean, y_clean = zip(*valid)
            r, p = stats.spearmanr(x_clean, y_clean)
        else:
            r, p = np.nan, np.nan
    else:
        r, p = np.nan, np.nan

    return {
        'data': comparison_data,
        'correlation': float(r) if not np.isnan(r) else None,
        'pvalue': float(p) if not np.isnan(p) else None,
        'n': len(comparison_data),
        'n_celltypes': len(common_types),
        'n_signatures': len(common_sigs)
    }


def compute_pseudobulk_resampled_comparison(atlas1: str, atlas2: str, sig_type: str,
                                             level: str = 'coarse',
                                             n_bootstrap: int = 100) -> dict:
    """
    Compute pseudobulk comparison with bootstrap resampling.

    For each harmonized cell type:
    1. Get all samples from both atlases
    2. Bootstrap resample to get matched sample sizes
    3. Compute mean and CI for activity differences

    Returns:
        dict with resampled comparison statistics
    """
    print(f"\n  Computing resampled pseudobulk comparison: {atlas1} vs {atlas2}")

    # Load data
    act1, meta1 = load_pseudobulk_data(atlas1, sig_type)
    act2, meta2 = load_pseudobulk_data(atlas2, sig_type)

    if act1 is None or act2 is None:
        return {'data': [], 'n': 0}

    # Get mapping
    mapping1 = get_coarse_mapping(atlas1) if level == 'coarse' else get_fine_mapping(atlas1)
    mapping2 = get_coarse_mapping(atlas2) if level == 'coarse' else get_fine_mapping(atlas2)

    meta1['harmonized'] = meta1['cell_type'].map(mapping1)
    meta2['harmonized'] = meta2['cell_type'].map(mapping2)

    common_types = set(meta1['harmonized'].dropna()) & set(meta2['harmonized'].dropna())
    common_sigs = set(act1.columns) & set(act2.columns)

    if sig_type == 'SecAct':
        combined_var = act1[list(common_sigs)].var() + act2[list(common_sigs)].var()
        common_sigs = set(combined_var.nlargest(100).index)

    common_sigs = sorted(common_sigs)

    print(f"    Common types: {len(common_types)}, Running {n_bootstrap} bootstrap iterations...")

    resampled_data = []

    for ct in sorted(common_types):
        mask1 = meta1['harmonized'] == ct
        mask2 = meta2['harmonized'] == ct

        samples1 = act1.loc[mask1, common_sigs]
        samples2 = act2.loc[mask2, common_sigs]

        n1, n2 = len(samples1), len(samples2)
        if n1 < 2 or n2 < 2:
            continue

        # Bootstrap
        n_resample = min(n1, n2)

        for sig in common_sigs:
            boot_diffs = []

            for _ in range(n_bootstrap):
                idx1 = np.random.choice(n1, n_resample, replace=True)
                idx2 = np.random.choice(n2, n_resample, replace=True)

                mean1 = samples1.iloc[idx1][sig].mean()
                mean2 = samples2.iloc[idx2][sig].mean()
                boot_diffs.append(mean2 - mean1)

            boot_diffs = np.array(boot_diffs)

            resampled_data.append({
                'signature': sig,
                'cell_type': ct,
                'mean_diff': float(np.mean(boot_diffs)),
                'ci_low': float(np.percentile(boot_diffs, 2.5)),
                'ci_high': float(np.percentile(boot_diffs, 97.5)),
                'x_mean': float(samples1[sig].mean()),
                'y_mean': float(samples2[sig].mean()),
                'n_x': n1,
                'n_y': n2
            })

    return {
        'data': resampled_data,
        'n': len(resampled_data),
        'n_bootstrap': n_bootstrap
    }


def compute_singlecell_comparison(atlas1: str, atlas2: str, sig_type: str,
                                   level: str = 'coarse', n_cells: int = 5000) -> dict:
    """
    Compute single-cell level comparison.

    For each harmonized cell type, sample cells from both atlases
    and compare activity distributions.

    Returns:
        dict with per-celltype distribution comparisons
    """
    print(f"\n  Computing single-cell comparison: {atlas1} vs {atlas2}")

    # Get mapping
    mapping1 = get_coarse_mapping(atlas1) if level == 'coarse' else get_fine_mapping(atlas1)
    mapping2 = get_coarse_mapping(atlas2) if level == 'coarse' else get_fine_mapping(atlas2)

    # Get common cell types
    common_types = set(mapping1.values()) & set(mapping2.values())

    if sig_type == 'CytoSig':
        target_sigs = ['IFNG', 'TNF', 'IL6', 'IL10', 'IL17A', 'TGFB1', 'IL1B', 'IL4']
    else:
        target_sigs = None  # Will select top variable

    sc_data = []

    for ct in sorted(common_types)[:6]:  # Limit to 6 cell types for performance
        print(f"    Processing {ct}...")

        # Get original cell types that map to this harmonized type
        orig_types1 = [k for k, v in mapping1.items() if v == ct]
        orig_types2 = [k for k, v in mapping2.items() if v == ct]

        # Load samples
        act1, meta1 = load_singlecell_sample(atlas1, sig_type, n_cells=n_cells)
        act2, meta2 = load_singlecell_sample(atlas2, sig_type, n_cells=n_cells)

        if act1 is None or act2 is None:
            continue

        # Get cell type column
        ct_col1 = 'cell_type' if 'cell_type' in meta1.columns else ('Level2' if 'Level2' in meta1.columns else 'subCluster')
        ct_col2 = 'cell_type' if 'cell_type' in meta2.columns else ('Level2' if 'Level2' in meta2.columns else 'subCluster')

        # Filter to target cell types
        mask1 = meta1[ct_col1].isin(orig_types1)
        mask2 = meta2[ct_col2].isin(orig_types2)

        if mask1.sum() < 10 or mask2.sum() < 10:
            continue

        cells1 = act1.loc[mask1]
        cells2 = act2.loc[mask2]

        # Get common signatures
        common_sigs = set(cells1.columns) & set(cells2.columns)

        if target_sigs:
            common_sigs = common_sigs & set(target_sigs)
        else:
            # Top 20 most variable
            var = cells1[list(common_sigs)].var() + cells2[list(common_sigs)].var()
            common_sigs = set(var.nlargest(20).index)

        for sig in sorted(common_sigs):
            vals1 = cells1[sig].dropna().values
            vals2 = cells2[sig].dropna().values

            if len(vals1) < 10 or len(vals2) < 10:
                continue

            # KS test
            ks_stat, ks_p = stats.ks_2samp(vals1, vals2)

            # Effect size (Cohen's d)
            pooled_std = np.sqrt((vals1.std()**2 + vals2.std()**2) / 2)
            cohens_d = (vals1.mean() - vals2.mean()) / pooled_std if pooled_std > 0 else 0

            sc_data.append({
                'signature': sig,
                'cell_type': ct,
                'x_mean': float(vals1.mean()),
                'y_mean': float(vals2.mean()),
                'x_std': float(vals1.std()),
                'y_std': float(vals2.std()),
                'x_n': int(len(vals1)),
                'y_n': int(len(vals2)),
                'ks_stat': float(ks_stat),
                'ks_pvalue': float(ks_p),
                'cohens_d': float(cohens_d),
                # Distribution percentiles for violin plot
                'x_q25': float(np.percentile(vals1, 25)),
                'x_q50': float(np.percentile(vals1, 50)),
                'x_q75': float(np.percentile(vals1, 75)),
                'y_q25': float(np.percentile(vals2, 25)),
                'y_q50': float(np.percentile(vals2, 50)),
                'y_q75': float(np.percentile(vals2, 75)),
            })

    return {
        'data': sc_data,
        'n': len(sc_data)
    }


def compute_prediction_concordance(atlas1: str, atlas2: str, sig_type: str) -> dict:
    """
    Compute cross-atlas prediction concordance.

    Train a simple predictor on atlas1 cell types, test on atlas2.
    Measures how well cell type signatures generalize.

    Returns:
        dict with concordance metrics
    """
    print(f"\n  Computing prediction concordance: {atlas1} -> {atlas2}")

    # Load cell type aggregated data
    act1, meta1 = load_pseudobulk_data(atlas1, sig_type)
    act2, meta2 = load_pseudobulk_data(atlas2, sig_type)

    if act1 is None or act2 is None:
        # Single-cell files lack cell type metadata, can't do prediction
        print(f"    [SKIP] No pseudobulk data available")
        return {'overall_accuracy': None, 'n_samples': 0, 'note': 'Pseudobulk data required'}

    mapping1 = get_coarse_mapping(atlas1)
    mapping2 = get_coarse_mapping(atlas2)

    meta1['harmonized'] = meta1['cell_type'].map(mapping1)
    meta2['harmonized'] = meta2['cell_type'].map(mapping2)

    common_types = sorted(set(meta1['harmonized'].dropna()) & set(meta2['harmonized'].dropna()))
    common_sigs = sorted(set(act1.columns) & set(act2.columns))

    if sig_type == 'SecAct':
        var = act1[common_sigs].var() + act2[common_sigs].var()
        common_sigs = list(var.nlargest(50).index)

    print(f"    Common types: {len(common_types)}, Signatures: {len(common_sigs)}")

    # Compute centroid for each cell type in atlas1
    centroids1 = {}
    for ct in common_types:
        mask = meta1['harmonized'] == ct
        if mask.sum() > 0:
            centroids1[ct] = act1.loc[mask, common_sigs].mean().values

    # For each sample in atlas2, find nearest centroid
    concordance_details = []
    correct = 0
    total = 0

    for ct in common_types:
        mask2 = meta2['harmonized'] == ct
        if mask2.sum() == 0:
            continue

        samples2 = act2.loc[mask2, common_sigs]

        for idx, row in samples2.iterrows():
            sample_vec = row.values

            # Find nearest centroid
            min_dist = float('inf')
            predicted_ct = None

            for ref_ct, centroid in centroids1.items():
                dist = cosine(sample_vec, centroid)
                if dist < min_dist:
                    min_dist = dist
                    predicted_ct = ref_ct

            is_correct = predicted_ct == ct
            if is_correct:
                correct += 1
            total += 1

            concordance_details.append({
                'true_type': ct,
                'predicted_type': predicted_ct,
                'correct': is_correct,
                'distance': float(min_dist)
            })

    accuracy = correct / total if total > 0 else 0

    # Per-celltype accuracy
    ct_accuracy = {}
    for ct in common_types:
        ct_details = [d for d in concordance_details if d['true_type'] == ct]
        if ct_details:
            ct_accuracy[ct] = sum(d['correct'] for d in ct_details) / len(ct_details)

    return {
        'overall_accuracy': float(accuracy),
        'n_samples': total,
        'per_celltype_accuracy': ct_accuracy,
        'confusion_summary': concordance_details[:100]  # Limit for JSON size
    }


def main():
    print("=" * 70)
    print("Generating Comprehensive Atlas Comparison Data")
    print("=" * 70)

    results = {}

    # Define comparison pairs
    pairs = [
        ('cima', 'inflammation'),
        ('cima', 'scatlas'),
        ('inflammation', 'scatlas')
    ]

    for sig_type in ['CytoSig', 'SecAct']:
        print(f"\n{'='*70}")
        print(f"Processing {sig_type}")
        print("=" * 70)

        sig_key = sig_type.lower()
        results[sig_key] = {}

        for atlas1, atlas2 in pairs:
            pair_key = f"{atlas1}_vs_{atlas2}"
            print(f"\n{'-'*60}")
            print(f"Comparison: {atlas1.upper()} vs {atlas2.upper()}")
            print("-" * 60)

            results[sig_key][pair_key] = {}

            # 1. Cell type aggregated comparison (coarse)
            results[sig_key][pair_key]['celltype_aggregated_coarse'] = \
                compute_celltype_aggregated_comparison(atlas1, atlas2, sig_type, level='coarse')

            # 2. Cell type aggregated comparison (fine)
            results[sig_key][pair_key]['celltype_aggregated_fine'] = \
                compute_celltype_aggregated_comparison(atlas1, atlas2, sig_type, level='fine')

            # 3. Pseudobulk resampled comparison (if available)
            if DATA_FILES.get(atlas1, {}).get(f'{sig_key}_pb') and \
               DATA_FILES.get(atlas2, {}).get(f'{sig_key}_pb'):
                results[sig_key][pair_key]['pseudobulk_resampled'] = \
                    compute_pseudobulk_resampled_comparison(atlas1, atlas2, sig_type,
                                                            level='coarse', n_bootstrap=50)
            else:
                results[sig_key][pair_key]['pseudobulk_resampled'] = {'data': [], 'n': 0}

            # 4. Single-cell comparison (skip for now - metadata not in singlecell h5ad)
            # The singlecell h5ad files don't contain cell type metadata
            # Would need to join with original h5ad files to get cell type
            results[sig_key][pair_key]['singlecell'] = {'data': [], 'n': 0, 'note': 'Metadata not available in singlecell h5ad'}

            # 5. Prediction concordance
            results[sig_key][pair_key]['prediction_concordance'] = \
                compute_prediction_concordance(atlas1, atlas2, sig_type)

    # Save results
    print("\n" + "=" * 70)
    print("Saving results...")
    print("=" * 70)

    # Update cross_atlas.json
    cross_atlas_path = VIZ_OUTPUT_DIR / 'cross_atlas.json'
    if cross_atlas_path.exists():
        with open(cross_atlas_path, 'r') as f:
            cross_atlas = json.load(f)
    else:
        cross_atlas = {}

    # Store under 'atlas_comparison' key
    cross_atlas['atlas_comparison'] = results

    with open(cross_atlas_path, 'w') as f:
        json.dump(cross_atlas, f)

    print(f"Updated: {cross_atlas_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    for sig_key, sig_results in results.items():
        print(f"\n{sig_key.upper()}:")
        for pair_key, pair_results in sig_results.items():
            print(f"\n  {pair_key}:")
            for comp_type, comp_data in pair_results.items():
                if isinstance(comp_data, dict):
                    n = comp_data.get('n', len(comp_data.get('data', [])))
                    corr = comp_data.get('correlation', comp_data.get('overall_accuracy'))
                    if corr is not None:
                        print(f"    {comp_type}: n={n}, r/acc={corr:.3f}")
                    else:
                        print(f"    {comp_type}: n={n}")

    print("\nDone!")


if __name__ == '__main__':
    main()
