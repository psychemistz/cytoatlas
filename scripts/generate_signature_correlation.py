#!/usr/bin/env python3
"""
Generate Signature Correlation Data for Cross-Atlas Panel.

Computes pairwise correlations between signatures across atlases and identifies
co-regulated signature modules using hierarchical clustering.

Generates data for both CytoSig (43) and SecAct (1170) signatures.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import warnings
warnings.filterwarnings('ignore')

# Paths
RESULTS_DIR = Path('/data/parks34/projects/2secactpy/results')
VIZ_OUTPUT_DIR = Path('/data/parks34/projects/2secactpy/visualization/data')

def load_atlas_activity(atlas: str, sig_type: str) -> pd.DataFrame:
    """Load pseudobulk activity data for an atlas.

    Returns DataFrame with signatures as columns and samples as rows.

    Data structure:
    - Pseudobulk: obs=signatures, var=samples, X[i,j]=activity of sig i in sample j
      -> Transpose to get samples as rows, signatures as columns
    - Singlecell: obs=cells, var=signatures (names in var['_index'])
      -> Already has signatures as columns
    """
    if atlas == 'cima':
        if sig_type == 'CytoSig':
            path = RESULTS_DIR / 'cima' / 'CIMA_CytoSig_pseudobulk.h5ad'
        else:
            path = RESULTS_DIR / 'cima' / 'CIMA_SecAct_pseudobulk.h5ad'
    elif atlas == 'inflammation':
        if sig_type == 'CytoSig':
            path = RESULTS_DIR / 'inflammation' / 'main_CytoSig_pseudobulk.h5ad'
        else:
            path = RESULTS_DIR / 'inflammation' / 'main_SecAct_pseudobulk.h5ad'
    elif atlas == 'scatlas':
        if sig_type == 'CytoSig':
            path = RESULTS_DIR / 'scatlas' / 'scatlas_normal_CytoSig_singlecell.h5ad'
        else:
            path = RESULTS_DIR / 'scatlas' / 'scatlas_normal_SecAct_singlecell.h5ad'
    else:
        raise ValueError(f"Unknown atlas: {atlas}")

    if not path.exists():
        print(f"  File not found: {path}")
        return pd.DataFrame()

    print(f"  Loading {path.name}...")

    if 'singlecell' in path.name:
        # Single-cell: obs=cells, var=signatures
        # Sample 50K cells for efficiency
        adata = ad.read_h5ad(path, backed='r')
        n_cells = adata.n_obs
        np.random.seed(42)
        idx = np.random.choice(n_cells, min(50000, n_cells), replace=False)
        X = adata.X[sorted(idx), :]
        if hasattr(X, 'toarray'):
            X = X.toarray()

        # Get signature names from var['_index']
        if '_index' in adata.var.columns:
            sig_names = list(adata.var['_index'].values)
        else:
            sig_names = list(adata.var_names)

        df = pd.DataFrame(X, columns=sig_names)
        print(f"    Signatures: {sig_names[:3]}...")
    else:
        # Pseudobulk: obs=signatures, var=samples
        # X[i,j] = activity of signature i in sample j
        # Need to transpose: rows=samples, columns=signatures
        adata = ad.read_h5ad(path)
        X = adata.X
        if hasattr(X, 'toarray'):
            X = X.toarray()

        # X is (n_signatures, n_samples), transpose to (n_samples, n_signatures)
        sig_names = list(adata.obs_names)
        sample_names = list(adata.var_names)

        df = pd.DataFrame(X.T, index=sample_names, columns=sig_names)
        print(f"    Signatures: {sig_names[:3]}...")

    print(f"    Shape: {df.shape} (samples/cells × signatures)")
    return df


def compute_correlation_matrix(dfs: list, method: str = 'spearman') -> tuple:
    """Compute correlation matrix across multiple atlases.

    Args:
        dfs: List of DataFrames with same columns (signatures)
        method: 'spearman' or 'pearson'

    Returns:
        (correlation_matrix, signatures list)
    """
    # Concatenate all data
    combined = pd.concat(dfs, axis=0, ignore_index=True)

    # Get common signatures
    signatures = list(combined.columns)
    print(f"  Computing {method} correlation for {len(signatures)} signatures...")

    # Compute correlation
    if method == 'spearman':
        corr_matrix = combined.corr(method='spearman')
    else:
        corr_matrix = combined.corr(method='pearson')

    return corr_matrix.values, signatures


def identify_modules(corr_matrix: np.ndarray, signatures: list, n_modules: int = None) -> list:
    """Identify co-regulated signature modules using hierarchical clustering.

    Args:
        corr_matrix: Correlation matrix
        signatures: List of signature names
        n_modules: Number of modules (auto-detect if None)

    Returns:
        List of module dicts with name, members, color
    """
    print("  Identifying modules via hierarchical clustering...")

    # Convert correlation to distance
    # Use 1 - |correlation| as distance
    dist_matrix = 1 - np.abs(corr_matrix)
    np.fill_diagonal(dist_matrix, 0)

    # Ensure symmetric and valid
    dist_matrix = (dist_matrix + dist_matrix.T) / 2
    dist_matrix = np.clip(dist_matrix, 0, 2)

    # Convert to condensed form for linkage
    dist_condensed = squareform(dist_matrix, checks=False)

    # Hierarchical clustering
    Z = linkage(dist_condensed, method='ward')

    # Determine number of clusters
    if n_modules is None:
        # Use silhouette or fixed number based on size
        n_modules = min(8, max(3, len(signatures) // 10))

    # Cut tree to get clusters
    clusters = fcluster(Z, n_modules, criterion='maxclust')

    # Group signatures by cluster
    module_colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3',
                     '#ff7f00', '#a65628', '#f781bf', '#999999']

    modules = []
    for i in range(1, n_modules + 1):
        members = [signatures[j] for j in range(len(signatures)) if clusters[j] == i]
        if members:
            # Name module by most representative signature (highest mean |corr| within module)
            member_idx = [signatures.index(m) for m in members]
            sub_corr = corr_matrix[np.ix_(member_idx, member_idx)]
            mean_corr = np.abs(sub_corr).mean(axis=1)
            top_member = members[np.argmax(mean_corr)]

            modules.append({
                'name': f'Module {i} ({top_member})',
                'members': members,
                'color': module_colors[(i-1) % len(module_colors)],
                'n_members': len(members),
                'mean_correlation': float(np.mean(sub_corr[np.triu_indices(len(members), k=1)]))
            })

    # Sort by size
    modules.sort(key=lambda x: -x['n_members'])

    return modules


def main():
    print("=" * 60)
    print("Generating Signature Correlation Data")
    print("=" * 60)

    results = {}

    for sig_type in ['CytoSig', 'SecAct']:
        print(f"\n{'='*60}")
        print(f"Processing {sig_type}")
        print("=" * 60)

        # Load data from all atlases
        dfs = []
        for atlas in ['cima', 'inflammation', 'scatlas']:
            print(f"\n{atlas.upper()}:")
            df = load_atlas_activity(atlas, sig_type)
            if not df.empty:
                dfs.append(df)

        if not dfs:
            print(f"No data available for {sig_type}")
            continue

        # Get common signatures across atlases
        common_sigs = set(dfs[0].columns)
        for df in dfs[1:]:
            common_sigs &= set(df.columns)
        common_sigs = sorted(common_sigs)

        print(f"\nCommon signatures across atlases: {len(common_sigs)}")

        # Filter to common signatures
        dfs = [df[common_sigs] for df in dfs]

        # Compute correlation matrix
        corr_matrix, signatures = compute_correlation_matrix(dfs, method='spearman')

        # For SecAct, limit to top variable signatures for visualization
        if sig_type == 'SecAct' and len(signatures) > 200:
            print(f"  Selecting top 200 most variable signatures for visualization...")
            combined = pd.concat(dfs, axis=0, ignore_index=True)
            variances = combined.var()
            top_sigs = variances.nlargest(200).index.tolist()

            # Recompute for top signatures
            dfs_filtered = [df[top_sigs] for df in dfs]
            corr_matrix, signatures = compute_correlation_matrix(dfs_filtered, method='spearman')

        # Identify modules
        n_modules = 4 if sig_type == 'CytoSig' else 8
        modules = identify_modules(corr_matrix, signatures, n_modules)

        print(f"\nIdentified {len(modules)} modules:")
        for mod in modules:
            print(f"  {mod['name']}: {mod['n_members']} signatures, mean r={mod['mean_correlation']:.3f}")

        # Store results
        results[sig_type.lower()] = {
            'signatures': signatures,
            'matrix': corr_matrix.tolist(),
            'modules': modules,
            'n_signatures': len(signatures),
            'n_samples': sum(len(df) for df in dfs)
        }

    # Update cross_atlas.json
    print("\n" + "=" * 60)
    print("Saving results...")
    print("=" * 60)

    cross_atlas_path = VIZ_OUTPUT_DIR / 'cross_atlas.json'
    if cross_atlas_path.exists():
        with open(cross_atlas_path, 'r') as f:
            cross_atlas = json.load(f)
    else:
        cross_atlas = {}

    # Update correlation section with both types
    cross_atlas['correlation'] = {
        'cytosig': results.get('cytosig', {}),
        'secact': results.get('secact', {}),
        # Keep backward compatibility
        'signatures': results.get('cytosig', {}).get('signatures', []),
        'matrix': results.get('cytosig', {}).get('matrix', []),
        'modules': results.get('cytosig', {}).get('modules', [])
    }

    with open(cross_atlas_path, 'w') as f:
        json.dump(cross_atlas, f)

    print(f"Updated: {cross_atlas_path}")

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for sig_type, data in results.items():
        print(f"\n{sig_type.upper()}:")
        print(f"  Signatures: {data['n_signatures']}")
        print(f"  Samples: {data['n_samples']}")
        print(f"  Modules: {len(data['modules'])}")

    print("\nDone!")


if __name__ == '__main__':
    main()
