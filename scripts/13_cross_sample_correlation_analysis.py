#!/usr/bin/env python3
"""
Cross-Sample Correlation Analysis.

For each cytokine/secreted protein, correlate its gene expression with
its predicted activity across samples (donors). This validates whether
activity predictions track with actual gene expression levels.

Computes:
  1. Self-correlation: expression(gene_X) vs activity(target_X) across donors
  2. Per-celltype correlations for donor×celltype aggregation levels
  3. Summary statistics and classification

Usage:
    python scripts/13_cross_sample_correlation_analysis.py
    python scripts/13_cross_sample_correlation_analysis.py --atlas cima
    python scripts/13_cross_sample_correlation_analysis.py --atlas cima --levels donor_only
"""

import argparse
import json
import re
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import anndata as ad
import numpy as np
import pandas as pd
from scipy import stats

sys.stdout.reconfigure(line_buffering=True)

BASE_DIR = Path('/data/parks34/projects/2cytoatlas/results/cross_sample_validation')
OUTPUT_DIR = BASE_DIR / 'correlations'
MAPPING_PATH = Path('/vf/users/parks34/projects/2cytoatlas/cytoatlas-api/static/data/signature_gene_mapping.json')
CELLTYPE_MAPPING_PATH = Path('/vf/users/parks34/projects/2cytoatlas/data/lincytosig_celltype_mapping.json')


def log(msg: str) -> None:
    ts = time.strftime('%H:%M:%S')
    print(f"[{ts}] {msg}", flush=True)


# =============================================================================
# Gene Name Mapping
# =============================================================================

def load_target_to_gene_mapping() -> Dict[str, str]:
    """
    Load CytoSig target name -> HGNC gene symbol mapping.

    Returns dict like {'TNFA': 'TNF', 'GMCSF': 'CSF2', 'IFNG': 'IFNG', ...}
    """
    with open(MAPPING_PATH) as f:
        data = json.load(f)

    mapping = {}
    for target_name, info in data.get('cytosig_mapping', {}).items():
        mapping[target_name] = info['hgnc_symbol']

    return mapping


def resolve_gene_name(target: str, gene_set: set, cytosig_map: Dict[str, str]) -> Optional[str]:
    """
    Resolve a signature target name to a gene name in the expression matrix.

    Strategy:
      1. Direct match (target name == gene name)
      2. CytoSig alias mapping (TNFA -> TNF)
      3. For LinCytoSig format "CellType__Cytokine", extract cytokine and resolve
    """
    # Direct match
    if target in gene_set:
        return target

    # CytoSig alias
    if target in cytosig_map:
        mapped = cytosig_map[target]
        if mapped in gene_set:
            return mapped

    # LinCytoSig format: CellType__Cytokine
    if '__' in target:
        cytokine = target.split('__')[-1]
        if cytokine in gene_set:
            return cytokine
        if cytokine in cytosig_map:
            mapped = cytosig_map[cytokine]
            if mapped in gene_set:
                return mapped

    return None


# =============================================================================
# LinCytoSig Cell Type Mapping
# =============================================================================

def load_celltype_mapping() -> Optional[dict]:
    """Load LinCytoSig cell type mapping from JSON."""
    if not CELLTYPE_MAPPING_PATH.exists():
        log(f"WARNING: celltype mapping not found at {CELLTYPE_MAPPING_PATH}")
        return None
    with open(CELLTYPE_MAPPING_PATH) as f:
        return json.load(f)


def extract_lincytosig_celltype(target: str) -> Optional[str]:
    """Extract cell type from LinCytoSig target name (e.g., 'Monocyte__TNF' -> 'Monocyte')."""
    if '__' not in target:
        return None
    return target.split('__')[0]


def get_matching_celltypes(
    lincytosig_ct: str,
    atlas_key: str,
    celltype_col: str,
    mapping: dict,
    unique_celltypes: Optional[set] = None,
) -> Optional[List[str]]:
    """Get atlas cell types matching a LinCytoSig cell type.

    Returns list of matching cell type values, or None if unmappable.
    """
    atlas_section = mapping.get(atlas_key)
    if atlas_section is None:
        return None

    # Check unmappable list
    if lincytosig_ct in atlas_section.get('unmappable', []):
        return None

    ct_mapping = atlas_section.get('mapping', {}).get(lincytosig_ct)
    if ct_mapping is None:
        return None

    # scatlas_normal: uses regex patterns against cellType1
    if atlas_key == 'scatlas_normal':
        patterns = ct_mapping.get('cellType1_patterns', [])
        if patterns and unique_celltypes is not None:
            compiled = [re.compile(p) for p in patterns]
            matched = []
            for ct_val in unique_celltypes:
                for pat in compiled:
                    if pat.search(str(ct_val)):
                        matched.append(ct_val)
                        break
            return matched if matched else None
        # Fall through to direct column lookup if no patterns
        direct = ct_mapping.get(celltype_col, [])
        return direct if direct else None

    # scatlas_cancer: uses prefix extraction + exact matches
    if atlas_key == 'scatlas_cancer':
        matched = set()
        # Exact matches for this celltype_col
        exact = ct_mapping.get('cellType1_exact', [])
        if unique_celltypes is not None:
            for ct_val in unique_celltypes:
                ct_str = str(ct_val)
                if ct_str in exact:
                    matched.add(ct_str)

            # Prefix-based matching: extract prefix from cellType1 values
            # and check if prefix maps to this LinCytoSig celltype
            prefix_map = atlas_section.get('prefix_extraction', {}).get('prefix_to_lincytosig', {})
            # Build reverse: which prefixes map to this lincytosig_ct?
            target_prefixes = [p for p, lcs in prefix_map.items() if lcs == lincytosig_ct]
            if target_prefixes:
                for ct_val in unique_celltypes:
                    ct_str = str(ct_val)
                    # Extract prefix before _NN_ (underscore + digits + underscore)
                    m = re.match(r'^([A-Za-z]+)_\d+', ct_str)
                    if m and m.group(1) in target_prefixes:
                        matched.add(ct_str)

        return sorted(matched) if matched else None

    # CIMA and inflammation: direct column lookup
    # Try exact column name first, then strip 'pred' suffix (Level1pred -> Level1)
    direct = ct_mapping.get(celltype_col, [])
    if not direct and celltype_col.endswith('pred'):
        direct = ct_mapping.get(celltype_col[:-4], [])
    return direct if direct else None


# =============================================================================
# Correlation Computation
# =============================================================================

def compute_correlations_for_level(
    pb_path: Path,
    act_path: Path,
    sig_name: str,
    cytosig_map: Dict[str, str],
    gene_col: Optional[str] = None,
    atlas_key: Optional[str] = None,
    celltype_mapping: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Compute Spearman correlations between expression and activity for all targets.

    For donor_only: one correlation per target across all donors.
    For donor_celltype: overall + per-celltype correlations.
    For LinCytoSig targets: also compute celltype-restricted ('matched') correlations.

    Returns DataFrame with columns:
        target, gene, celltype, spearman_rho, spearman_pval, n_samples,
        mean_expr, std_expr, mean_activity, std_activity,
        lincytosig_celltype, matched_atlas_celltypes
    """
    # Load data
    pb = ad.read_h5ad(pb_path)
    act = ad.read_h5ad(act_path)

    # Get gene names for matching
    if gene_col and gene_col in pb.var.columns:
        gene_names = pb.var[gene_col].tolist()
    else:
        gene_names = list(pb.var_names)

    gene_set = set(gene_names)
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}

    # Get expression matrix
    expr = pb.X
    if hasattr(expr, 'toarray'):
        expr = expr.toarray()
    expr = np.asarray(expr, dtype=np.float64)

    # Get activity matrix
    activity = act.X
    if hasattr(activity, 'toarray'):
        activity = activity.toarray()
    activity = np.asarray(activity, dtype=np.float64)

    targets = list(act.var_names)

    # Determine if we have celltype information
    has_celltype = False
    celltype_col = None
    for col in ['cell_type_l1', 'cell_type_l2', 'cell_type_l3', 'cell_type_l4',
                'Level1', 'Level2', 'Level1pred', 'Level2pred',
                'cellType1', 'cellType2',
                'cancerType', 'tissue', 'tissue_type', 'cancer_type']:
        if col in act.obs.columns:
            has_celltype = True
            celltype_col = col
            break

    results = []

    for t_idx, target in enumerate(targets):
        gene = resolve_gene_name(target, gene_set, cytosig_map)
        if gene is None:
            continue

        g_idx = gene_to_idx[gene]
        expr_vals = expr[:, g_idx]
        act_vals = activity[:, t_idx]

        # Skip if no variance
        if np.std(expr_vals) < 1e-10 or np.std(act_vals) < 1e-10:
            continue

        # Overall correlation
        mask = np.isfinite(expr_vals) & np.isfinite(act_vals)
        if mask.sum() < 10:
            continue

        rho, pval = stats.spearmanr(expr_vals[mask], act_vals[mask])

        # Extract LinCytoSig cell type info for new columns
        lcs_ct = extract_lincytosig_celltype(target) if '__' in target else None

        results.append({
            'target': target,
            'gene': gene,
            'celltype': 'all',
            'spearman_rho': rho,
            'spearman_pval': pval,
            'n_samples': int(mask.sum()),
            'mean_expr': float(np.mean(expr_vals[mask])),
            'std_expr': float(np.std(expr_vals[mask])),
            'mean_activity': float(np.mean(act_vals[mask])),
            'std_activity': float(np.std(act_vals[mask])),
            'lincytosig_celltype': lcs_ct or '',
            'matched_atlas_celltypes': '',
        })

        # Celltype-restricted ("matched") correlation for LinCytoSig targets
        if has_celltype and lcs_ct is not None and celltype_mapping is not None and atlas_key is not None:
            celltypes = act.obs[celltype_col].values
            unique_cts = set(str(c) for c in celltypes)
            matched_cts = get_matching_celltypes(
                lcs_ct, atlas_key, celltype_col, celltype_mapping,
                unique_celltypes=unique_cts,
            )
            if matched_cts is None:
                # Unmappable
                results.append({
                    'target': target,
                    'gene': gene,
                    'celltype': 'unmappable',
                    'spearman_rho': None,
                    'spearman_pval': None,
                    'n_samples': 0,
                    'mean_expr': None,
                    'std_expr': None,
                    'mean_activity': None,
                    'std_activity': None,
                    'lincytosig_celltype': lcs_ct,
                    'matched_atlas_celltypes': '',
                })
            elif len(matched_cts) > 0:
                ct_set = set(matched_cts)
                ct_mask = np.array([str(c) in ct_set for c in celltypes]) & mask
                if ct_mask.sum() >= 10:
                    m_expr = expr_vals[ct_mask]
                    m_act = act_vals[ct_mask]
                    if np.std(m_expr) > 1e-10 and np.std(m_act) > 1e-10:
                        m_rho, m_pval = stats.spearmanr(m_expr, m_act)
                        results.append({
                            'target': target,
                            'gene': gene,
                            'celltype': 'matched',
                            'spearman_rho': m_rho,
                            'spearman_pval': m_pval,
                            'n_samples': int(ct_mask.sum()),
                            'mean_expr': float(np.mean(m_expr)),
                            'std_expr': float(np.std(m_expr)),
                            'mean_activity': float(np.mean(m_act)),
                            'std_activity': float(np.std(m_act)),
                            'lincytosig_celltype': lcs_ct,
                            'matched_atlas_celltypes': ','.join(str(c) for c in matched_cts),
                        })

        # Per-celltype correlations
        if has_celltype:
            celltypes = act.obs[celltype_col].values
            for ct in sorted(set(celltypes)):
                ct_mask = (celltypes == ct) & mask
                if ct_mask.sum() < 10:
                    continue

                ct_expr = expr_vals[ct_mask]
                ct_act = act_vals[ct_mask]

                if np.std(ct_expr) < 1e-10 or np.std(ct_act) < 1e-10:
                    continue

                ct_rho, ct_pval = stats.spearmanr(ct_expr, ct_act)
                results.append({
                    'target': target,
                    'gene': gene,
                    'celltype': str(ct),
                    'spearman_rho': ct_rho,
                    'spearman_pval': ct_pval,
                    'n_samples': int(ct_mask.sum()),
                    'mean_expr': float(np.mean(ct_expr)),
                    'std_expr': float(np.std(ct_expr)),
                    'mean_activity': float(np.mean(ct_act)),
                    'std_activity': float(np.std(ct_act)),
                    'lincytosig_celltype': lcs_ct or '',
                    'matched_atlas_celltypes': '',
                })

    return pd.DataFrame(results)


# =============================================================================
# Atlas Configuration
# =============================================================================

ATLAS_CONFIGS = OrderedDict([
    ('cima', {
        'levels': ['donor_only', 'donor_l1', 'donor_l2', 'donor_l3', 'donor_l4'],
        'signatures': ['cytosig', 'lincytosig', 'secact'],
        'gene_col': None,
    }),
    ('inflammation_main', {
        'levels': ['donor_only', 'donor_l1', 'donor_l2'],
        'signatures': ['cytosig', 'lincytosig', 'secact'],
        'gene_col': 'symbol',
    }),
    ('inflammation_val', {
        'levels': ['donor_only', 'donor_l1', 'donor_l2'],
        'signatures': ['cytosig', 'lincytosig', 'secact'],
        'gene_col': 'symbol',
    }),
    ('inflammation_ext', {
        'levels': ['donor_only', 'donor_l1', 'donor_l2'],
        'signatures': ['cytosig', 'lincytosig', 'secact'],
        'gene_col': 'symbol',
    }),
    ('inflammation_combined', {
        'levels': ['donor_only', 'donor_l1', 'donor_l2'],
        'signatures': ['cytosig', 'lincytosig', 'secact'],
        'gene_col': 'symbol',
    }),
    ('scatlas_normal', {
        'levels': ['donor_organ', 'donor_organ_celltype1', 'donor_organ_celltype2'],
        'signatures': ['cytosig', 'lincytosig', 'secact'],
        'gene_col': None,
    }),
    ('scatlas_cancer', {
        'levels': ['donor_only', 'tumor_only', 'tumor_by_cancer', 'tumor_by_cancer_celltype1'],
        'signatures': ['cytosig', 'lincytosig', 'secact'],
        'gene_col': None,
    }),
    ('gtex', {
        'levels': ['donor_only', 'by_tissue'],
        'signatures': ['cytosig', 'lincytosig', 'secact'],
        'gene_col': None,
        'expr_suffix': 'expression',
    }),
    ('tcga', {
        'levels': ['donor_only', 'by_cancer', 'primary_only', 'primary_by_cancer'],
        'signatures': ['cytosig', 'lincytosig', 'secact'],
        'gene_col': None,
        'expr_suffix': 'expression',
    }),
])


def run_atlas_correlations(
    atlas_name: str,
    levels: Optional[List[str]] = None,
    cytosig_map: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Run correlation analysis for one atlas, all levels and signatures."""
    if cytosig_map is None:
        cytosig_map = load_target_to_gene_mapping()

    config = ATLAS_CONFIGS[atlas_name]
    gene_col = config['gene_col']
    expr_suffix = config.get('expr_suffix', 'pseudobulk')
    process_levels = levels if levels else config['levels']

    # Derive atlas key for celltype mapping lookup
    # inflammation_main/val/ext all use the 'inflammation' section
    if atlas_name.startswith('inflammation'):
        mapping_key = 'inflammation'
    else:
        mapping_key = atlas_name

    # Load celltype mapping once
    celltype_mapping = load_celltype_mapping()

    all_results = []

    for level in process_levels:
        for sig in config['signatures']:
            pb_path = BASE_DIR / atlas_name / f"{atlas_name}_{level}_{expr_suffix}.h5ad"
            act_path = BASE_DIR / atlas_name / f"{atlas_name}_{level}_{sig}.h5ad"

            if not pb_path.exists() or not act_path.exists():
                log(f"  SKIP {level}/{sig}: files not found")
                continue

            log(f"  {level} / {sig}...")
            df = compute_correlations_for_level(
                pb_path=pb_path,
                act_path=act_path,
                sig_name=sig,
                cytosig_map=cytosig_map,
                gene_col=gene_col,
                atlas_key=mapping_key,
                celltype_mapping=celltype_mapping,
            )

            if len(df) > 0:
                df['atlas'] = atlas_name
                df['level'] = level
                df['signature'] = sig
                all_results.append(df)

                # Quick summary for overall correlations
                overall = df[df['celltype'] == 'all']
                if len(overall) > 0:
                    median_rho = overall['spearman_rho'].median()
                    n_sig = (overall['spearman_pval'] < 0.05).sum()
                    n_pos = (overall['spearman_rho'] > 0).sum()
                    log(f"    {len(overall)} targets matched, median rho={median_rho:.3f}, "
                        f"{n_sig}/{len(overall)} significant (p<0.05), "
                        f"{n_pos}/{len(overall)} positive")

    if all_results:
        return pd.concat(all_results, ignore_index=True)
    return pd.DataFrame()


def generate_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Generate summary statistics per atlas/level/signature."""
    if len(df) == 0:
        return pd.DataFrame()

    overall = df[df['celltype'] == 'all'].copy()

    summary_rows = []
    for (atlas, level, sig), group in overall.groupby(['atlas', 'level', 'signature']):
        n_targets = len(group)
        n_matched = n_targets
        n_significant = (group['spearman_pval'] < 0.05).sum()
        n_positive = (group['spearman_rho'] > 0).sum()
        n_strong = (group['spearman_rho'].abs() > 0.3).sum()

        summary_rows.append({
            'atlas': atlas,
            'level': level,
            'signature': sig,
            'n_targets_matched': n_matched,
            'n_significant_p05': int(n_significant),
            'n_positive_corr': int(n_positive),
            'n_strong_corr_03': int(n_strong),
            'pct_significant': n_significant / n_matched * 100 if n_matched > 0 else 0,
            'pct_positive': n_positive / n_matched * 100 if n_matched > 0 else 0,
            'median_rho': group['spearman_rho'].median(),
            'mean_rho': group['spearman_rho'].mean(),
            'std_rho': group['spearman_rho'].std(),
            'min_rho': group['spearman_rho'].min(),
            'max_rho': group['spearman_rho'].max(),
            'median_n_samples': group['n_samples'].median(),
        })

    return pd.DataFrame(summary_rows)


def main():
    parser = argparse.ArgumentParser(description="Cross-Sample Correlation Analysis")
    parser.add_argument('--atlas', nargs='+', default=None,
                        help='Atlas name(s) or "all" (default: all)')
    parser.add_argument('--levels', nargs='+', default=None,
                        help='Specific levels to analyze')

    args = parser.parse_args()

    # Determine atlases
    if args.atlas is None or 'all' in args.atlas:
        atlas_list = list(ATLAS_CONFIGS.keys())
    else:
        atlas_list = args.atlas

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cytosig_map = load_target_to_gene_mapping()

    all_results = []

    for atlas_name in atlas_list:
        log(f"\n{'=' * 60}")
        log(f"ATLAS: {atlas_name}")
        log(f"{'=' * 60}")

        df = run_atlas_correlations(
            atlas_name=atlas_name,
            levels=args.levels,
            cytosig_map=cytosig_map,
        )

        if len(df) > 0:
            # Save per-atlas results
            atlas_path = OUTPUT_DIR / f"{atlas_name}_correlations.csv"
            df.to_csv(atlas_path, index=False)
            log(f"\nSaved: {atlas_path.name} ({len(df)} rows)")
            all_results.append(df)

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)

        # Save combined results
        combined_path = OUTPUT_DIR / "all_correlations.csv"
        combined.to_csv(combined_path, index=False)
        log(f"\nSaved: {combined_path.name} ({len(combined)} rows)")

        # Generate and save summary
        summary = generate_summary(combined)
        summary_path = OUTPUT_DIR / "correlation_summary.csv"
        summary.to_csv(summary_path, index=False)
        log(f"Saved: {summary_path.name}")

        # Print summary table
        log(f"\n{'=' * 90}")
        log("SUMMARY: Overall correlations (celltype='all')")
        log(f"{'=' * 90}")
        log(f"{'Atlas':<20} {'Level':<25} {'Sig':<12} {'N':>4} "
            f"{'Med rho':>8} {'%Sig':>6} {'%Pos':>6} {'%Strong':>7}")
        log("-" * 90)
        for _, row in summary.iterrows():
            log(f"{row['atlas']:<20} {row['level']:<25} {row['signature']:<12} "
                f"{row['n_targets_matched']:>4} {row['median_rho']:>8.3f} "
                f"{row['pct_significant']:>5.1f}% {row['pct_positive']:>5.1f}% "
                f"{row['n_strong_corr_03']/row['n_targets_matched']*100:>6.1f}%")

    log(f"\n{'=' * 60}")
    log("DONE")
    log(f"{'=' * 60}")


if __name__ == '__main__':
    main()
