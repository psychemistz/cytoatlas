#!/usr/bin/env python3
"""
Regenerate cross-atlas conserved signatures data from actual activity results.

This script computes:
1. Mean activity per signature across each atlas (CIMA, Inflammation, scAtlas)
2. Determines "active" status based on |mean activity| > threshold
3. Generates overlap counts for all CytoSig (43) + SecAct (1170) signatures
4. Saves to results/integrated/signature_overlap.csv

Rationale for the Conserved Tab:
- Shows which cytokine/secreted protein signatures are consistently active across atlases
- "Conserved" signatures represent fundamental immune programs
- "Atlas-specific" signatures reveal context-dependent biology
"""

import pandas as pd
import numpy as np
import anndata as ad
from pathlib import Path
import json

# Paths
RESULTS_DIR = Path('/data/parks34/projects/2cytoatlas/results')
CIMA_DIR = RESULTS_DIR / 'cima'
INFLAM_DIR = RESULTS_DIR / 'inflammation'
SCATLAS_DIR = RESULTS_DIR / 'scatlas'
OUTPUT_DIR = RESULTS_DIR / 'integrated'
VIZ_OUTPUT_DIR = Path('/data/parks34/projects/2cytoatlas/visualization/data')

# Activity threshold: consider a signature "active" if |mean| > this value
ACTIVITY_THRESHOLD = 0.3

def load_pseudobulk_means(h5ad_path: Path) -> pd.Series:
    """Load pseudobulk h5ad and compute mean activity per signature.

    Pseudobulk files have:
    - obs = signatures (e.g., 43 CytoSig or 1170 SecAct)
    - var = cell_type × sample combinations
    - X[i,j] = activity of signature i in sample-celltype j

    We compute mean across all samples (axis=1) to get mean activity per signature.
    """
    if not h5ad_path.exists():
        print(f"  File not found: {h5ad_path}")
        return pd.Series(dtype=float)

    print(f"  Loading {h5ad_path.name}...")
    adata = ad.read_h5ad(h5ad_path)

    # Compute mean across all samples (axis=1) per signature (obs)
    X = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
    mean_activity = pd.Series(
        X.mean(axis=1),
        index=adata.obs_names
    )
    print(f"    {len(mean_activity)} signatures, mean range: [{mean_activity.min():.2f}, {mean_activity.max():.2f}]")
    return mean_activity


def load_singlecell_means(h5ad_path: Path) -> pd.Series:
    """Load single-cell h5ad in backed mode and compute mean activity.

    Single-cell files have:
    - obs = cells
    - var = signatures (names stored in var['_index'])
    - X[i,j] = activity of signature j in cell i

    We compute mean across all cells (axis=0) to get mean activity per signature.
    """
    if not h5ad_path.exists():
        print(f"  File not found: {h5ad_path}")
        return pd.Series(dtype=float)

    print(f"  Loading {h5ad_path.name} (backed)...")
    adata = ad.read_h5ad(h5ad_path, backed='r')

    # Get signature names from var['_index'] if available
    if '_index' in adata.var.columns:
        sig_names = adata.var['_index'].values
    else:
        sig_names = adata.var_names

    # Sample for large files
    n_cells = adata.n_obs
    if n_cells > 100000:
        np.random.seed(42)
        idx = np.random.choice(n_cells, 100000, replace=False)
        X_sample = adata.X[sorted(idx), :]
    else:
        X_sample = adata.X[:]

    # Convert to dense if needed
    if hasattr(X_sample, 'toarray'):
        X_sample = X_sample.toarray()
    elif hasattr(X_sample, 'A1'):
        pass  # Already handled below

    mean_activity = pd.Series(
        X_sample.mean(axis=0).flatten() if hasattr(X_sample.mean(axis=0), 'flatten') else X_sample.mean(axis=0),
        index=sig_names
    )
    print(f"    {len(mean_activity)} signatures from {n_cells} cells")
    return mean_activity


def main():
    print("=" * 60)
    print("Regenerating Cross-Atlas Conserved Signatures")
    print("=" * 60)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def combine_means(cytosig_means, secact_means):
        """Combine CytoSig and SecAct means, handling overlapping signatures.

        For overlapping signatures, we prefer CytoSig values (cytokine-specific).
        """
        combined = pd.concat([cytosig_means, secact_means])
        # For duplicates, keep first occurrence (CytoSig preferred)
        combined = combined[~combined.index.duplicated(keep='first')]
        return combined

    # Load mean activity from each atlas
    print("\n1. Loading CIMA activity...")
    cima_cytosig = load_pseudobulk_means(CIMA_DIR / 'CIMA_CytoSig_pseudobulk.h5ad')
    cima_secact = load_pseudobulk_means(CIMA_DIR / 'CIMA_SecAct_pseudobulk.h5ad')
    cima_means = combine_means(cima_cytosig, cima_secact)
    print(f"  CIMA total: {len(cima_means)} unique signatures")

    print("\n2. Loading Inflammation activity...")
    inflam_cytosig = load_pseudobulk_means(INFLAM_DIR / 'main_CytoSig_pseudobulk.h5ad')
    inflam_secact = load_pseudobulk_means(INFLAM_DIR / 'main_SecAct_pseudobulk.h5ad')
    inflam_means = combine_means(inflam_cytosig, inflam_secact)
    print(f"  Inflammation total: {len(inflam_means)} unique signatures")

    print("\n3. Loading scAtlas activity...")
    # Use single-cell files for scAtlas (no pseudobulk available)
    scatlas_cytosig = load_singlecell_means(SCATLAS_DIR / 'scatlas_normal_CytoSig_singlecell.h5ad')
    scatlas_secact = load_singlecell_means(SCATLAS_DIR / 'scatlas_normal_SecAct_singlecell.h5ad')
    scatlas_means = combine_means(scatlas_cytosig, scatlas_secact)
    print(f"  scAtlas total: {len(scatlas_means)} unique signatures")

    # Get all unique signatures
    all_signatures = sorted(set(cima_means.index) | set(inflam_means.index) | set(scatlas_means.index))
    print(f"\n4. Total unique signatures: {len(all_signatures)}")

    # Determine signature type (CytoSig vs SecAct)
    cytosig_sigs = set(cima_cytosig.index) | set(inflam_cytosig.index) | set(scatlas_cytosig.index)

    def safe_get(series, key, default=np.nan):
        """Safely get a scalar value from a series."""
        if key in series.index:
            val = series.loc[key]
            # If multiple values (shouldn't happen after dedup), take first
            if hasattr(val, '__len__') and not isinstance(val, str):
                return float(val.iloc[0]) if len(val) > 0 else default
            return float(val)
        return default

    def is_active(mean_val, threshold=ACTIVITY_THRESHOLD):
        """Check if a signature is active based on |mean| > threshold."""
        if np.isnan(mean_val):
            return False
        return abs(mean_val) > threshold

    # Build results table
    results = []
    for sig in all_signatures:
        cima_mean = safe_get(cima_means, sig)
        inflam_mean = safe_get(inflam_means, sig)
        scatlas_mean = safe_get(scatlas_means, sig)

        # Determine if "active" in each atlas based on absolute mean activity
        in_cima = is_active(cima_mean)
        in_inflam = is_active(inflam_mean)
        in_scatlas = is_active(scatlas_mean)

        n_atlases = sum([in_cima, in_inflam, in_scatlas])

        results.append({
            'signature': sig,
            'signature_type': 'CytoSig' if sig in cytosig_sigs else 'SecAct',
            'cima': in_cima,
            'inflammation': in_inflam,
            'scatlas': in_scatlas,
            'n_atlases': n_atlases,
            'cima_mean': cima_mean if not np.isnan(cima_mean) else None,
            'inflammation_mean': inflam_mean if not np.isnan(inflam_mean) else None,
            'scatlas_mean': scatlas_mean if not np.isnan(scatlas_mean) else None
        })

    df = pd.DataFrame(results)
    df = df.sort_values(['n_atlases', 'signature'], ascending=[False, True])

    # Save to CSV
    df.to_csv(OUTPUT_DIR / 'signature_overlap.csv', index=False)
    print(f"\n5. Saved signature_overlap.csv: {len(df)} signatures")

    # Compute overlap counts
    all_three = len(df[(df['cima']) & (df['inflammation']) & (df['scatlas'])])
    cima_inflam = len(df[(df['cima']) & (df['inflammation']) & (~df['scatlas'])])
    cima_scatlas = len(df[(df['cima']) & (~df['inflammation']) & (df['scatlas'])])
    inflam_scatlas = len(df[(~df['cima']) & (df['inflammation']) & (df['scatlas'])])
    cima_only = len(df[(df['cima']) & (~df['inflammation']) & (~df['scatlas'])])
    inflam_only = len(df[(~df['cima']) & (df['inflammation']) & (~df['scatlas'])])
    scatlas_only = len(df[(~df['cima']) & (~df['inflammation']) & (df['scatlas'])])
    none_active = len(df[(~df['cima']) & (~df['inflammation']) & (~df['scatlas'])])

    print("\n6. Overlap counts:")
    print(f"   All 3 atlases: {all_three}")
    print(f"   CIMA + Inflammation: {cima_inflam}")
    print(f"   CIMA + scAtlas: {cima_scatlas}")
    print(f"   Inflammation + scAtlas: {inflam_scatlas}")
    print(f"   CIMA only: {cima_only}")
    print(f"   Inflammation only: {inflam_only}")
    print(f"   scAtlas only: {scatlas_only}")
    print(f"   Not active in any: {none_active}")

    # By signature type
    cytosig_df = df[df['signature_type'] == 'CytoSig']
    secact_df = df[df['signature_type'] == 'SecAct']

    print(f"\n7. By signature type:")
    print(f"   CytoSig: {len(cytosig_df)} signatures")
    print(f"     Active in >=1 atlas: {len(cytosig_df[cytosig_df['n_atlases'] >= 1])}")
    print(f"     Active in >=2 atlases: {len(cytosig_df[cytosig_df['n_atlases'] >= 2])}")
    print(f"     Active in all 3: {len(cytosig_df[cytosig_df['n_atlases'] == 3])}")
    print(f"   SecAct: {len(secact_df)} signatures")
    print(f"     Active in >=1 atlas: {len(secact_df[secact_df['n_atlases'] >= 1])}")
    print(f"     Active in >=2 atlases: {len(secact_df[secact_df['n_atlases'] >= 2])}")
    print(f"     Active in all 3: {len(secact_df[secact_df['n_atlases'] == 3])}")

    # Update cross_atlas.json
    print("\n8. Updating cross_atlas.json...")
    cross_atlas_path = VIZ_OUTPUT_DIR / 'cross_atlas.json'

    if cross_atlas_path.exists():
        with open(cross_atlas_path, 'r') as f:
            cross_atlas = json.load(f)
    else:
        cross_atlas = {}

    # Update conserved section
    cross_atlas['conserved'] = {
        'signatures': df.to_dict('records'),
        'overlap_counts': {
            'all_three': all_three,
            'cima_inflam': cima_inflam,
            'cima_scatlas': cima_scatlas,
            'inflam_scatlas': inflam_scatlas,
            'cima_only': cima_only,
            'inflam_only': inflam_only,
            'scatlas_only': scatlas_only
        }
    }
    cross_atlas['atlas_specific_signatures'] = cross_atlas['conserved']['overlap_counts']

    with open(cross_atlas_path, 'w') as f:
        json.dump(cross_atlas, f)

    print(f"   Updated {cross_atlas_path}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == '__main__':
    main()
