#!/usr/bin/env python3
"""
Surgical patch: compute missing CIMA SecAct single-cell comparisons
and update cross_atlas.json.

Uses h5py for direct, memory-efficient access to the large h5ad files
instead of anndata backed mode (which was OOM-prone).
"""

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import stats

# Add scripts dir for cell_type_mapping
sys.path.insert(0, str(Path(__file__).parent))
from cell_type_mapping import (
    CIMA_TO_COARSE, INFLAMMATION_TO_COARSE,
    CIMA_TO_FINE, INFLAMMATION_TO_FINE,
)
from generate_atlas_comparison import (
    SCATLAS_CELLTYPE1_TO_COARSE, SCATLAS_CELLTYPE1_TO_FINE,
)

# Single-cell activity files (atlas_validation copies)
SC_FILES = {
    'cima': '/vf/users/parks34/projects/2secactpy/results/atlas_validation/cima/singlecell/cima_singlecell_secact.h5ad',
    'inflammation': '/vf/users/parks34/projects/2secactpy/results/atlas_validation/inflammation_main/singlecell/inflammation_main_singlecell_secact.h5ad',
    'scatlas': '/vf/users/parks34/projects/2secactpy/results/atlas_validation/scatlas_normal/singlecell/scatlas_normal_singlecell_secact.h5ad',
}

# Original h5ad files for cell type metadata
ORIGINAL_H5AD = {
    'cima': '/data/Jiang_Lab/Data/Seongyong/CIMA/Cell_Atlas/CIMA_RNA_6484974cells_36326genes_compressed.h5ad',
    'inflammation': '/data/Jiang_Lab/Data/Seongyong/Inflammation_Atlas/INFLAMMATION_ATLAS_main_afterQC.h5ad',
    'scatlas': '/data/Jiang_Lab/Data/Seongyong/scAtlas_2025/igt_s9_fine_counts.h5ad',
}

CELLTYPE_COLUMNS = {
    'cima': 'cell_type_l2',
    'inflammation': 'Level2',
    'scatlas': 'cellType1',
}

VIZ_DATA = Path('/data/parks34/projects/2secactpy/visualization/data')
N_SAMPLE = 50000


def decode(val):
    """Decode bytes to string if needed."""
    if isinstance(val, bytes):
        return val.decode('utf-8')
    return str(val)


def get_mapping(atlas: str, level: str) -> dict:
    if level == 'coarse':
        return {'cima': CIMA_TO_COARSE, 'inflammation': INFLAMMATION_TO_COARSE,
                'scatlas': SCATLAS_CELLTYPE1_TO_COARSE}.get(atlas, {})
    else:
        return {'cima': CIMA_TO_FINE, 'inflammation': INFLAMMATION_TO_FINE,
                'scatlas': SCATLAS_CELLTYPE1_TO_FINE}.get(atlas, {})


def read_h5_categorical(h5_obs, col_name):
    """Read a categorical column from h5ad obs group."""
    col = h5_obs[col_name]
    if isinstance(col, h5py.Dataset):
        # Simple array
        return np.array([decode(v) for v in col[:]])
    elif isinstance(col, h5py.Group):
        # Categorical encoding: codes + categories
        codes = col['codes'][:]
        categories = np.array([decode(c) for c in col['categories'][:]])
        return categories[codes]
    return None


def load_sc_with_celltypes(atlas: str):
    """Load sampled single-cell activity + cell types using h5py directly."""
    act_path = SC_FILES[atlas]
    orig_path = ORIGINAL_H5AD[atlas]
    ct_col = CELLTYPE_COLUMNS[atlas]

    # Read activity file - get shape and signature names
    print(f"  [{atlas}] Opening activity file...", flush=True)
    with h5py.File(act_path, 'r') as f:
        n_total, n_sigs = f['X'].shape
        # Signature names from var/_index
        sig_names = [decode(s) for s in f['var']['_index'][:]]

        # Sample cell indices (same seed as original script)
        np.random.seed(42)
        sample_idx = np.sort(np.random.choice(n_total, min(N_SAMPLE, n_total), replace=False))

        # Read activity for sampled cells in chunks to avoid huge memory allocation
        print(f"  [{atlas}] Reading {len(sample_idx)} cells x {n_sigs} signatures...", flush=True)
        X = f['X'][sample_idx, :]

    act_df = pd.DataFrame(X, columns=sig_names)
    print(f"  [{atlas}] Activity loaded: {act_df.shape}", flush=True)

    # Read cell types from original h5ad
    print(f"  [{atlas}] Reading cell types from original h5ad ({ct_col})...", flush=True)
    with h5py.File(orig_path, 'r') as f:
        ct_all = read_h5_categorical(f['obs'], ct_col)
        cell_types = ct_all[sample_idx]

    meta_df = pd.DataFrame({'cell_type': cell_types})
    print(f"  [{atlas}] Done: {len(meta_df)} cells, {len(np.unique(cell_types))} cell types", flush=True)

    return act_df, meta_df


def compute_comparison(act1, meta1, act2, meta2,
                       atlas1: str, atlas2: str, level: str):
    """Compute single-cell mean activity comparison."""
    mapping1 = get_mapping(atlas1, level)
    mapping2 = get_mapping(atlas2, level)

    m1 = meta1.copy()
    m2 = meta2.copy()
    m1['harmonized'] = m1['cell_type'].map(mapping1)
    m2['harmonized'] = m2['cell_type'].map(mapping2)

    common_types = sorted(
        set(m1['harmonized'].dropna().unique()) &
        set(m2['harmonized'].dropna().unique())
    )
    common_sigs = sorted(set(act1.columns) & set(act2.columns))

    print(f"    Common types: {len(common_types)}, Common signatures: {len(common_sigs)}", flush=True)

    comparison_data = []
    all_x, all_y = [], []

    for ct in common_types:
        mask1 = m1['harmonized'] == ct
        mask2 = m2['harmonized'] == ct
        n1, n2 = mask1.sum(), mask2.sum()
        if n1 < 10 or n2 < 10:
            continue

        mean1 = act1.loc[mask1, common_sigs].mean()
        mean2 = act2.loc[mask2, common_sigs].mean()

        for sig in common_sigs:
            x_val = float(mean1[sig])
            y_val = float(mean2[sig])
            if np.isnan(x_val) or np.isnan(y_val):
                continue

            comparison_data.append({
                'signature': sig,
                'cell_type': ct,
                'x': x_val,
                'y': y_val,
                'n_cells_x': int(n1),
                'n_cells_y': int(n2),
            })
            all_x.append(x_val)
            all_y.append(y_val)

    if len(all_x) > 2:
        r, p = stats.spearmanr(all_x, all_y)
    else:
        r, p = np.nan, np.nan

    return {
        'data': comparison_data,
        'n': len(comparison_data),
        'n_celltypes': len(common_types),
        'n_signatures': len(common_sigs),
        'overall_correlation': float(r) if not np.isnan(r) else None,
        'overall_pvalue': float(p) if not np.isnan(p) else None,
        'n_cells_atlas1': int(m1['harmonized'].notna().sum()),
        'n_cells_atlas2': int(m2['harmonized'].notna().sum()),
    }


def main():
    print("=" * 60, flush=True)
    print("Patching cross_atlas.json: CIMA SecAct single-cell data", flush=True)
    print("=" * 60, flush=True)

    # Load single-cell data for all 3 atlases
    print("\n1. Loading single-cell data...", flush=True)
    cima_act, cima_meta = load_sc_with_celltypes('cima')
    inflam_act, inflam_meta = load_sc_with_celltypes('inflammation')
    scatlas_act, scatlas_meta = load_sc_with_celltypes('scatlas')

    # Compute the 4 missing comparisons
    print("\n2. Computing CIMA vs Inflammation (coarse)...", flush=True)
    ci_coarse = compute_comparison(
        cima_act, cima_meta, inflam_act, inflam_meta,
        'cima', 'inflammation', 'coarse')
    print(f"   n={ci_coarse['n']}, r={ci_coarse['overall_correlation']}", flush=True)

    print("\n3. Computing CIMA vs Inflammation (fine)...", flush=True)
    ci_fine = compute_comparison(
        cima_act, cima_meta, inflam_act, inflam_meta,
        'cima', 'inflammation', 'fine')
    print(f"   n={ci_fine['n']}, r={ci_fine['overall_correlation']}", flush=True)

    print("\n4. Computing CIMA vs scAtlas (coarse)...", flush=True)
    cs_coarse = compute_comparison(
        cima_act, cima_meta, scatlas_act, scatlas_meta,
        'cima', 'scatlas', 'coarse')
    print(f"   n={cs_coarse['n']}, r={cs_coarse['overall_correlation']}", flush=True)

    print("\n5. Computing CIMA vs scAtlas (fine)...", flush=True)
    cs_fine = compute_comparison(
        cima_act, cima_meta, scatlas_act, scatlas_meta,
        'cima', 'scatlas', 'fine')
    print(f"   n={cs_fine['n']}, r={cs_fine['overall_correlation']}", flush=True)

    # Load and patch cross_atlas.json
    print("\n6. Patching cross_atlas.json...", flush=True)
    cross_atlas_path = VIZ_DATA / 'cross_atlas.json'
    with open(cross_atlas_path, 'r') as f:
        cross_atlas = json.load(f)

    secact = cross_atlas['atlas_comparison']['secact']
    secact['cima_vs_inflammation']['singlecell_mean_coarse'] = ci_coarse
    secact['cima_vs_inflammation']['singlecell_mean_fine'] = ci_fine
    secact['cima_vs_scatlas']['singlecell_mean_coarse'] = cs_coarse
    secact['cima_vs_scatlas']['singlecell_mean_fine'] = cs_fine

    with open(cross_atlas_path, 'w') as f:
        json.dump(cross_atlas, f)

    print(f"   Saved to {cross_atlas_path}", flush=True)

    # Summary
    print("\n" + "=" * 60, flush=True)
    print("Patch Summary", flush=True)
    print("=" * 60, flush=True)
    for label, result in [
        ("cima_vs_inflammation coarse", ci_coarse),
        ("cima_vs_inflammation fine  ", ci_fine),
        ("cima_vs_scatlas coarse     ", cs_coarse),
        ("cima_vs_scatlas fine       ", cs_fine),
    ]:
        r = result['overall_correlation']
        r_str = f"{r:.3f}" if r is not None else "N/A"
        print(f"  {label}: n={result['n']}, r={r_str}", flush=True)
    print("\nDone!", flush=True)


if __name__ == '__main__':
    main()
