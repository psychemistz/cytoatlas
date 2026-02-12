#!/usr/bin/env python3
"""
Compute gene-filtered LinCytoSig activity and validation correlations.

Gene filter = restrict LinCytoSig signature genes to CytoSig gene space (~4,881 genes).
Compares:
  - lincyto_orig: full LinCytoSig (~20K genes)
  - lincyto_filt: gene-filtered LinCytoSig (~4,881 genes, CytoSig overlap)

For each atlas, loads donor-level pseudobulk, aggregates to donor-only
(or donor×organ for scAtlas), runs ridge regression, and computes
Spearman correlations between predicted activity and target gene expression.

Usage:
    python scripts/compute_lincytosig_gene_filter.py
"""

import json
import gc
import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import stats

import secactpy

warnings.filterwarnings('ignore')

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE = Path('/data/parks34/projects/2cytoatlas')
VALIDATION_DIR = BASE / 'results' / 'atlas_validation'
CORR_DIR = BASE / 'results' / 'cross_sample_validation' / 'correlations'
VIZ_DIR = BASE / 'visualization' / 'data' / 'validation'
OUTPUT_DIR = BASE / 'results' / 'lincytosig_gene_filter'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Gene name mapping (CytoSig target → HGNC symbol)
MAPPING_PATH = BASE / 'cytoatlas-api' / 'static' / 'data' / 'signature_gene_mapping.json'

# ─── Atlas configurations ─────────────────────────────────────────────────────
ATLAS_CONFIGS = {
    'cima': {
        'pseudobulk': VALIDATION_DIR / 'cima' / 'pseudobulk' / 'cima_pseudobulk_l1_donor.h5ad',
        'donor_col': 'donor',
        'level': 'donor_only',
    },
    'inflammation_main': {
        'pseudobulk': VALIDATION_DIR / 'inflammation_main' / 'pseudobulk' / 'inflammation_main_pseudobulk_l1_donor.h5ad',
        'donor_col': 'donor',
        'level': 'donor_only',
    },
    'inflammation_val': {
        'pseudobulk': VALIDATION_DIR / 'inflammation_val' / 'pseudobulk' / 'inflammation_val_pseudobulk_l1_donor.h5ad',
        'donor_col': 'donor',
        'level': 'donor_only',
    },
    'inflammation_ext': {
        'pseudobulk': VALIDATION_DIR / 'inflammation_ext' / 'pseudobulk' / 'inflammation_ext_pseudobulk_l1_donor.h5ad',
        'donor_col': 'donor',
        'level': 'donor_only',
    },
    'scatlas_normal': {
        'pseudobulk': VALIDATION_DIR / 'scatlas_normal' / 'pseudobulk' / 'scatlas_normal_pseudobulk_donor_celltype.h5ad',
        'donor_col': 'donor',
        'level': 'donor_organ',
    },
    'scatlas_cancer': {
        'pseudobulk': VALIDATION_DIR / 'scatlas_cancer' / 'pseudobulk' / 'scatlas_cancer_pseudobulk_donor_celltype.h5ad',
        'donor_col': 'donor',
        'level': 'donor_organ',
    },
}


def load_gene_mapping():
    """Load CytoSig target → gene symbol mapping."""
    if MAPPING_PATH.exists():
        with open(MAPPING_PATH) as f:
            data = json.load(f)
        mapping = {}
        if 'cytosig_mapping' in data:
            for k, v in data['cytosig_mapping'].items():
                if isinstance(v, dict) and 'hgnc_symbol' in v:
                    mapping[k] = v['hgnc_symbol']
                elif isinstance(v, str):
                    mapping[k] = v
        return mapping
    return {}


def resolve_gene_name(target, gene_set, mapping):
    """Resolve CytoSig/LinCytoSig target name to gene symbol in expression data."""
    # Direct match
    if target in gene_set:
        return target

    # Via mapping
    if target in mapping:
        mapped = mapping[target]
        if mapped in gene_set:
            return mapped

    # LinCytoSig: CellType__Cytokine → extract cytokine
    if '__' in target:
        cytokine = target.split('__')[-1]
        if cytokine in gene_set:
            return cytokine
        if cytokine in mapping:
            mapped = mapping[cytokine]
            if mapped in gene_set:
                return mapped

    # Common aliases
    aliases = {
        'TNFA': 'TNF', 'IFNA': 'IFNA1', 'IFNB': 'IFNB1',
        'IFN1': 'IFNA1', 'IFNL': 'IFNL1', 'Activin A': 'INHBA',
        'NO': 'NOS2', 'TRAIL': 'TNFSF10', 'CD40L': 'CD40LG',
        'GMCSF': 'CSF2', 'GCSF': 'CSF3', 'MCSF': 'CSF1',
        'LIF': 'LIF', 'OSM': 'OSM', 'BMP2': 'BMP2',
    }
    # Try alias for the base name
    base = target.split('__')[-1] if '__' in target else target
    if base in aliases:
        alias = aliases[base]
        if alias in gene_set:
            return alias

    return None


def convert_ensembl_to_symbol(adata):
    """Convert Ensembl ID var_names to gene symbols using var['symbol'] column.

    Handles duplicates by keeping the first occurrence.
    """
    if 'symbol' not in adata.var.columns:
        return adata
    if not adata.var_names[0].startswith('ENSG'):
        return adata

    print('    Converting Ensembl IDs to gene symbols...')
    symbols = adata.var['symbol'].values
    # Filter out empty/NaN symbols
    valid = pd.notna(symbols) & (symbols != '') & (symbols != 'nan')
    # Keep only valid symbols, remove duplicates (keep first)
    seen = set()
    keep_idx = []
    new_names = []
    for i, (is_valid, sym) in enumerate(zip(valid, symbols)):
        if is_valid and sym not in seen:
            seen.add(sym)
            keep_idx.append(i)
            new_names.append(sym)

    adata_sub = adata[:, keep_idx].copy()
    adata_sub.var_names = pd.Index(new_names)
    print(f'    Converted {adata.shape[1]} Ensembl IDs → {adata_sub.shape[1]} gene symbols')
    return adata_sub


def aggregate_to_donor(adata):
    """Aggregate celltype × donor pseudobulk to donor-only level.

    Reverses log1p(CPM), sums counts weighted by n_cells per donor,
    then re-normalizes to log1p(CPM).
    """
    # Convert Ensembl to symbols if needed
    adata = convert_ensembl_to_symbol(adata)

    donors = adata.obs['donor'].values
    n_cells = adata.obs['n_cells'].values.astype(np.float64)
    unique_donors = np.unique(donors)

    n_donors = len(unique_donors)
    n_genes = adata.shape[1]
    gene_names = adata.var_names.tolist()

    print(f'    Aggregating {adata.shape[0]} celltype×donor → {n_donors} donors...')

    # Load full matrix into memory
    X = adata.X
    if hasattr(X, 'toarray'):
        X = X.toarray()
    X = X.astype(np.float64)

    # Reverse log1p(CPM) to approximate counts
    # log1p(CPM) → CPM → counts_approx
    X_cpm = np.expm1(X)

    # Weight by n_cells: approximate raw counts = CPM * n_cells / 1e6
    X_counts = X_cpm * (n_cells[:, None] / 1e6)

    # Sum per donor
    donor_expr = np.zeros((n_donors, n_genes), dtype=np.float64)
    donor_ncells = np.zeros(n_donors, dtype=np.float64)

    donor_to_idx = {d: i for i, d in enumerate(unique_donors)}
    for i, donor in enumerate(donors):
        idx = donor_to_idx[donor]
        donor_expr[idx] += X_counts[i]
        donor_ncells[idx] += n_cells[i]

    # Re-normalize to CPM + log1p
    row_sums = donor_expr.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    donor_cpm = donor_expr / row_sums * 1e6
    donor_log = np.log1p(donor_cpm).astype(np.float32)

    # Create AnnData
    obs = pd.DataFrame({
        'donor': unique_donors,
        'n_cells': donor_ncells.astype(int),
    }, index=unique_donors)

    result = ad.AnnData(
        X=donor_log,
        obs=obs,
        var=pd.DataFrame(index=gene_names),
    )
    return result


def run_ridge(expr_matrix, sig_matrix, lambda_=5e5, n_rand=1000, seed=42):
    """Run ridge regression: activity = ridge(signature, expression).

    Parameters
    ----------
    expr_matrix : ndarray, shape (n_genes, n_samples)
        Normalized expression matrix (genes in rows, donors in columns).
    sig_matrix : ndarray, shape (n_genes, n_features)
        Signature matrix (genes in rows, signatures in columns).

    Returns
    -------
    dict with 'zscore', 'pvalue', 'beta' arrays, shape (n_features, n_samples).
    """
    # Standardize expression per gene (across samples)
    Y = expr_matrix.copy()
    Y_mean = Y.mean(axis=1, keepdims=True)
    Y_std = Y.std(axis=1, ddof=1, keepdims=True)
    Y_std[Y_std < 1e-10] = 1.0
    Y = (Y - Y_mean) / Y_std

    # Ridge regression
    result = secactpy.ridge(
        X=sig_matrix,
        Y=Y,
        lambda_=lambda_,
        n_rand=n_rand,
        seed=seed,
    )
    return result


def compute_correlations(activity, gene_expr, target_names, gene_names_for_targets):
    """Compute Spearman correlation between activity and target gene expression.

    Parameters
    ----------
    activity : ndarray, shape (n_samples, n_targets)
    gene_expr : ndarray, shape (n_samples, n_genes)
    target_names : list of str
    gene_names_for_targets : dict mapping target → gene column index

    Returns
    -------
    list of dicts with target, gene, spearman_rho, spearman_pval, n_samples
    """
    rows = []
    for i, target in enumerate(target_names):
        if target not in gene_names_for_targets:
            continue
        gene_idx = gene_names_for_targets[target]
        act = activity[:, i]
        expr = gene_expr[:, gene_idx]

        # Remove NaN/inf
        mask = np.isfinite(act) & np.isfinite(expr)
        act_clean = act[mask]
        expr_clean = expr[mask]

        if len(act_clean) < 5:
            continue

        rho, pval = stats.spearmanr(act_clean, expr_clean)
        rows.append({
            'target': target,
            'gene': gene_names_for_targets.get(f'{target}_name', target),
            'spearman_rho': rho,
            'spearman_pval': pval,
            'n_samples': len(act_clean),
            'mean_expr': float(np.mean(expr_clean)),
            'std_expr': float(np.std(expr_clean)),
            'mean_activity': float(np.mean(act_clean)),
            'std_activity': float(np.std(act_clean)),
        })
    return rows


def process_atlas(atlas_name, config):
    """Process one atlas: compute gene-filtered LinCytoSig correlations."""
    print(f'\n{"="*60}')
    print(f'Processing: {atlas_name}')
    print(f'{"="*60}')

    pb_path = config['pseudobulk']
    if not pb_path.exists():
        print(f'  ERROR: Pseudobulk not found: {pb_path}')
        return None

    # Load pseudobulk
    print(f'  Loading pseudobulk: {pb_path.name}')
    adata = ad.read_h5ad(pb_path)
    print(f'  Shape: {adata.shape}')

    # Aggregate to donor level
    donor_adata = aggregate_to_donor(adata)
    del adata
    gc.collect()
    print(f'  Donor-level shape: {donor_adata.shape}')

    # Load signature matrices
    print('  Loading signature matrices...')
    cytosig = secactpy.load_cytosig()
    lincytosig_orig = secactpy.load_lincytosig()

    # Gene-filtered LinCytoSig: restrict to CytoSig gene set
    cytosig_genes = set(cytosig.index)
    lincytosig_genes = set(lincytosig_orig.index)
    overlap_genes = sorted(cytosig_genes & lincytosig_genes)
    print(f'  CytoSig genes: {len(cytosig_genes)}, LinCytoSig genes: {len(lincytosig_genes)}')
    print(f'  Gene overlap (filter set): {len(overlap_genes)}')

    lincytosig_filt = lincytosig_orig.loc[lincytosig_orig.index.isin(overlap_genes)]
    print(f'  LinCytoSig orig: {lincytosig_orig.shape}, filt: {lincytosig_filt.shape}')

    # Gene name mapping
    gene_mapping = load_gene_mapping()
    expr_gene_set = set(donor_adata.var_names)

    # Common genes between expression and each signature
    expr_genes = donor_adata.var_names.tolist()

    results = {}
    for sig_name, sig_df in [('lincyto_orig', lincytosig_orig), ('lincyto_filt', lincytosig_filt)]:
        print(f'\n  --- {sig_name} ---')
        sig_genes = set(sig_df.index)
        common = sorted(sig_genes & expr_gene_set)
        print(f'  Common genes with expression: {len(common)}')

        if len(common) < 100:
            print(f'  WARNING: Too few common genes ({len(common)}), skipping')
            continue

        # Prepare matrices
        gene_idx = [expr_genes.index(g) for g in common]
        expr_matrix = donor_adata.X[:, gene_idx].T  # (n_genes, n_samples)
        if hasattr(expr_matrix, 'toarray'):
            expr_matrix = expr_matrix.toarray()
        expr_matrix = expr_matrix.astype(np.float64)

        sig_matrix = sig_df.loc[common].values.astype(np.float64)
        # Fill NaN in signature (LinCytoSig has some NaN)
        sig_matrix = np.nan_to_num(sig_matrix, nan=0.0)

        print(f'  Expression: {expr_matrix.shape}, Signature: {sig_matrix.shape}')

        # Run ridge regression
        print(f'  Running ridge regression (lambda=5e5, n_rand=1000)...')
        result = run_ridge(expr_matrix, sig_matrix, lambda_=5e5, n_rand=1000)
        activity = result['zscore'].T  # (n_samples, n_targets)
        print(f'  Activity shape: {activity.shape}')

        # Map targets to gene expression columns
        target_names = sig_df.columns.tolist()
        target_gene_map = {}
        for target in target_names:
            gene = resolve_gene_name(target, expr_gene_set, gene_mapping)
            if gene is not None:
                gene_col_idx = expr_genes.index(gene)
                target_gene_map[target] = gene_col_idx
                target_gene_map[f'{target}_name'] = gene

        print(f'  Mapped {len([k for k in target_gene_map if not k.endswith("_name")])} targets to genes')

        # Compute correlations
        corr_rows = compute_correlations(
            activity,
            donor_adata.X.toarray() if hasattr(donor_adata.X, 'toarray') else donor_adata.X,
            target_names,
            target_gene_map,
        )

        for row in corr_rows:
            row['atlas'] = atlas_name
            row['level'] = config['level']
            row['signature'] = sig_name

        results[sig_name] = pd.DataFrame(corr_rows)
        print(f'  Computed {len(corr_rows)} correlations')

        # Summary
        if corr_rows:
            rhos = [r['spearman_rho'] for r in corr_rows]
            print(f'  Median rho: {np.median(rhos):.4f}, Mean: {np.mean(rhos):.4f}')
            print(f'  % positive: {100*sum(1 for r in rhos if r > 0)/len(rhos):.1f}%')

        del expr_matrix, sig_matrix, result, activity
        gc.collect()

    # Save per-atlas results
    for sig_name, df in results.items():
        out_path = OUTPUT_DIR / f'{atlas_name}_{sig_name}_correlations.csv'
        df.to_csv(out_path, index=False)
        print(f'  Saved: {out_path.name}')

    del donor_adata
    gc.collect()

    return results


def build_6way_comparison(all_results):
    """Build the 6-way comparison JSON from computed correlations + existing data."""
    print('\n' + '='*60)
    print('Building 6-way comparison')
    print('='*60)

    # Load existing data for CytoSig, SecAct, and best selections
    with open(VIZ_DIR / 'best_lincytosig_selection.json') as f:
        best = json.load(f)

    atlas_name_map = {
        'cima': 'CIMA',
        'inflammation_main': 'Inflammation Main',
        'inflammation_val': 'Inflammation Val',
        'inflammation_ext': 'Inflammation Ext',
        'scatlas_normal': 'scAtlas Normal',
        'scatlas_cancer': 'scAtlas Cancer',
    }

    results_6way = {}

    for atlas_key, atlas_label in atlas_name_map.items():
        config = ATLAS_CONFIGS[atlas_key]
        level = config['level']

        # Load existing correlation CSV for CytoSig and SecAct
        corr_path = CORR_DIR / f'{atlas_key}_correlations.csv'
        if not corr_path.exists():
            print(f'  Skipping {atlas_label}: no correlation CSV')
            continue

        df_corr = pd.read_csv(corr_path, low_memory=False)
        dl = df_corr[df_corr['level'] == level]

        # CytoSig rhos at donor level
        cs = dl[dl['signature'] == 'cytosig'].drop_duplicates('target').set_index('target')['spearman_rho']

        # SecAct rhos at donor level
        sa = dl[dl['signature'] == 'secact'].drop_duplicates('target').set_index('target')['spearman_rho']

        # LinCytoSig orig and filt: from newly computed data
        lcs_orig_df = all_results.get(atlas_key, {}).get('lincyto_orig', pd.DataFrame())
        lcs_filt_df = all_results.get(atlas_key, {}).get('lincyto_filt', pd.DataFrame())

        # Extract cytokine from LinCytoSig target names
        def extract_cytokine_rhos(lcs_df):
            if lcs_df.empty:
                return {}
            lcs_df = lcs_df.copy()
            lcs_df['cytokine'] = lcs_df['target'].str.split('__').str[-1]
            # Map IFN variants
            lcs_df.loc[lcs_df['cytokine'].isin(['IFNA', 'IFNB']), 'cytokine'] = 'IFN1'
            return lcs_df.groupby('cytokine')['spearman_rho'].median().to_dict()

        lcs_orig_rhos = extract_cytokine_rhos(lcs_orig_df)
        lcs_filt_rhos = extract_cytokine_rhos(lcs_filt_df)

        # Best selections: look up in newly computed data
        def get_best_rhos(lcs_df, best_selection):
            if lcs_df.empty:
                return {}
            lookup = lcs_df.set_index('target')['spearman_rho'].to_dict()
            result = {}
            for cytokine, best_target in best_selection.items():
                if best_target in lookup:
                    result[cytokine] = lookup[best_target]
            return result

        lcs_best_orig_rhos = get_best_rhos(lcs_orig_df, best['best_orig'])
        lcs_best_filt_rhos = get_best_rhos(lcs_filt_df, best['best_filt'])

        # Common cytokines across all 6 methods
        common = sorted(
            set(cs.index) &
            set(lcs_orig_rhos.keys()) &
            set(lcs_filt_rhos.keys()) &
            set(lcs_best_orig_rhos.keys()) &
            set(sa.index)
        )

        if not common:
            print(f'  {atlas_label}: no common cytokines found')
            continue

        r = {
            'cytokines': common,
            'cytosig': [round(float(cs.get(c, np.nan)), 4) for c in common],
            'lincyto_orig': [round(float(lcs_orig_rhos.get(c, np.nan)), 4) for c in common],
            'lincyto_filt': [round(float(lcs_filt_rhos.get(c, np.nan)), 4) for c in common],
            'lincyto_best_orig': [round(float(lcs_best_orig_rhos.get(c, np.nan)), 4) for c in common],
            'lincyto_best_filt': [round(float(lcs_best_filt_rhos.get(c, np.nan)), 4) for c in common],
            'secact': [round(float(sa.get(c, np.nan)), 4) for c in common],
        }
        results_6way[atlas_label] = r

        print(f'\n  {atlas_label}: {len(common)} cytokines')
        for m in ['cytosig', 'lincyto_orig', 'lincyto_filt', 'lincyto_best_orig', 'lincyto_best_filt', 'secact']:
            vals = [v for v in r[m] if not np.isnan(v)]
            if vals:
                print(f'    {m}: median={np.median(vals):.4f}, mean={np.mean(vals):.4f}')

    # Save
    out_path = VIZ_DIR / 'method_comparison_6way_all.json'
    with open(out_path, 'w') as f:
        json.dump(results_6way, f, indent=2)
    print(f'\nSaved: {out_path}')

    return results_6way


def main():
    print('Gene-Filtered LinCytoSig Computation')
    print('='*60)
    print(f'Output directory: {OUTPUT_DIR}')

    # Process all atlases
    all_results = {}
    for atlas_name, config in ATLAS_CONFIGS.items():
        try:
            results = process_atlas(atlas_name, config)
            if results:
                all_results[atlas_name] = results
        except Exception as e:
            print(f'\n  ERROR processing {atlas_name}: {e}')
            import traceback
            traceback.print_exc()

    # Build 6-way comparison
    build_6way_comparison(all_results)

    print('\n' + '='*60)
    print('DONE')
    print('='*60)


if __name__ == '__main__':
    main()
