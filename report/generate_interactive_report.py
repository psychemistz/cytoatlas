#!/usr/bin/env python3
"""
CytoAtlas PI Report — Interactive HTML Generation
===================================================
Generates PI_REPORT.html with embedded Plotly.js interactive figures.

Changes from previous version:
  0) Remove floating TOC
  1) Update dataset references
  2) Fix parse_10M description (not ground truth)
  3) Update SecAct reference (Ru et al.)
  4) Replace DNN-based → DL-based
  5) Embed summary table as HTML
  6) Interactive Plotly boxplot (all methods × all atlases)
  7) Add TWEAK duplicate explanation
  8) Add gene mapping verification note
  9) Consistent atlas ordering (bulk, CIMA, Inflammation, scAtlas)
 10) Interactive consistency plot
 11) Add LinCytoSig/SecAct to aggregation levels
 12) Interactive heatmap with tabs
 13) Interactive bulk validation with dropdown
 14) Interactive cell-type scatter with dropdown
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE = Path('/data/parks34/projects/2cytoatlas')
CORR_DIR = BASE / 'results' / 'cross_sample_validation' / 'correlations'
VIZ_DIR = BASE / 'visualization' / 'data' / 'validation'
SCATTER_DONOR = VIZ_DIR / 'donor_scatter'
SCATTER_CT = VIZ_DIR / 'celltype_scatter'
REPORT_DIR = BASE / 'report'

# ─── Atlas ordering (item 9): bulk, CIMA, Inflammation, scAtlas ───────────────
ATLAS_ORDER = [
    'gtex', 'tcga', 'cima',
    'inflammation_main', 'inflammation_val', 'inflammation_ext',
    'scatlas_normal', 'scatlas_cancer',
]
ATLAS_LABELS = [
    'GTEx', 'TCGA', 'CIMA',
    'Inflam (Main)', 'Inflam (Val)', 'Inflam (Ext)',
    'scAtlas (Normal)', 'scAtlas (Cancer)',
]

LEVEL_MAP = {
    'gtex': 'donor_only', 'tcga': 'donor_only',
    'cima': 'donor_only',
    'inflammation_main': 'donor_only', 'inflammation_val': 'donor_only',
    'inflammation_ext': 'donor_only',
    'scatlas_normal': 'donor_organ', 'scatlas_cancer': 'donor_organ',
}

SIG_COLORS = {
    'cytosig': '#2563EB',
    'lincytosig': '#D97706',
    'secact': '#059669',
}

BIO_FAMILIES = {
    'Interferon': ['IFNG', 'IFN1', 'IFNL'],
    'TGF-beta': ['TGFB1', 'TGFB2', 'TGFB3', 'BMP2', 'BMP4', 'BMP6', 'GDF11'],
    'Interleukin': ['IL1A', 'IL1B', 'IL2', 'IL4', 'IL6', 'IL10', 'IL12', 'IL13',
                    'IL15', 'IL17A', 'IL21', 'IL22', 'IL27', 'IL33'],
    'TNF': ['TNFA', 'LTA', 'TRAIL', 'TWEAK', 'CD40L'],
    'Growth Factor': ['EGF', 'FGF2', 'HGF', 'VEGFA', 'PDGFB', 'IGF1'],
    'Chemokine': ['CXCL12'],
    'Colony-Stimulating': ['GMCSF', 'GCSF', 'MCSF'],
}
TARGET_TO_FAMILY = {}
for fam, targets in BIO_FAMILIES.items():
    for t in targets:
        TARGET_TO_FAMILY[t] = fam

FAMILY_COLORS = {
    'Interferon': '#DC2626', 'TGF-beta': '#2563EB', 'Interleukin': '#059669',
    'TNF': '#D97706', 'Growth Factor': '#8B5CF6', 'Chemokine': '#EC4899',
    'Colony-Stimulating': '#6366F1', 'Other': '#9CA3AF',
}


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_all_correlations():
    dfs = []
    for f in CORR_DIR.glob('*_correlations.csv'):
        if f.name == 'all_correlations.csv' or 'resampled' in f.name or 'summary' in f.name:
            continue
        dfs.append(pd.read_csv(f, low_memory=False))
    bulk = pd.read_csv(CORR_DIR / 'all_correlations.csv', low_memory=False)
    dfs.append(bulk)
    merged = pd.concat(dfs, ignore_index=True)
    merged = merged.drop_duplicates(subset=['target', 'celltype', 'atlas', 'level', 'signature'])
    return merged


def load_method_comparison():
    with open(VIZ_DIR / 'method_comparison.json') as f:
        return json.load(f)


def load_scatter(directory, filename):
    fp = directory / filename
    if fp.exists():
        with open(fp) as f:
            return json.load(f)
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# DATA PREPARATION FOR INTERACTIVE FIGURES
# ═══════════════════════════════════════════════════════════════════════════════

def prepare_summary_table(df):
    """Prepare summary statistics as list of dicts for HTML table.
    All CytoSig rows first, then all SecAct rows.
    """
    rows = []
    for sig_type in ['cytosig', 'secact']:
        for atlas in ATLAS_ORDER:
            level = LEVEL_MAP[atlas]
            label = ATLAS_LABELS[ATLAS_ORDER.index(atlas)]
            sub = df[(df['atlas'] == atlas) & (df['level'] == level) & (df['signature'] == sig_type)]
            if len(sub) == 0:
                continue
            rhos = sub['spearman_rho'].dropna()
            n_sig = (sub['spearman_pval'] < 0.05).sum() if 'spearman_pval' in sub.columns else 0
            n_pos = (rhos > 0).sum()
            rows.append({
                'atlas': label,
                'signature': sig_type.upper(),
                'n_targets': int(len(rhos)),
                'median_rho': round(float(rhos.median()), 3),
                'mean_rho': round(float(rhos.mean()), 3),
                'std_rho': round(float(rhos.std()), 3),
                'min_rho': round(float(rhos.min()), 3),
                'max_rho': round(float(rhos.max()), 3),
                'pct_sig': round(100 * n_sig / len(sub), 1) if len(sub) > 0 else 0,
                'pct_pos': round(100 * n_pos / len(rhos), 1) if len(rhos) > 0 else 0,
            })
    return rows


MATCHED_TARGETS = [
    'BDNF', 'BMP2', 'BMP4', 'BMP6', 'CXCL12', 'FGF2', 'GDF11', 'HGF',
    'IFNG', 'IL10', 'IL15', 'IL1A', 'IL1B', 'IL21', 'IL27', 'IL6',
    'LIF', 'LTA', 'OSM', 'TGFB1', 'TGFB3', 'VEGFA',
]


def prepare_boxplot_data(df):
    """Prepare rho arrays for section 4.2 boxplot (CytoSig + SecAct + SecAct matched)."""
    result = {}
    for atlas in ATLAS_ORDER:
        level = LEVEL_MAP[atlas]
        label = ATLAS_LABELS[ATLAS_ORDER.index(atlas)]
        result[label] = {}
        for sig_type in ['cytosig', 'secact']:
            sub = df[(df['atlas'] == atlas) & (df['level'] == level) & (df['signature'] == sig_type)]
            rhos = sub['spearman_rho'].dropna().tolist()
            result[label][sig_type] = [round(r, 4) for r in rhos]
        # SecAct matched (only targets shared with CytoSig)
        secact_sub = df[(df['atlas'] == atlas) & (df['level'] == level) & (df['signature'] == 'secact')]
        matched_sub = secact_sub[secact_sub['target'].isin(MATCHED_TARGETS)]
        matched_rhos = matched_sub['spearman_rho'].dropna().tolist()
        result[label]['secact_matched'] = [round(r, 4) for r in matched_rhos]
    return result


def prepare_method_comparison_boxplot(df):
    """Prepare rho arrays for section 5.1 boxplot — 6-way method comparison.

    Loads pre-computed 6-way comparison data (ridge regression on pseudobulk)
    for all 6 atlases: CIMA, Inflammation Main/Val/Ext, scAtlas Normal/Cancer.

    Six methods:
      1. CytoSig — 43 cytokines, 4,881 curated genes (cell-type agnostic)
      2. LinCytoSig (orig) — cell-type-matched signatures, all ~20K genes
      3. LinCytoSig (gene-filtered) — cell-type-matched, restricted to CytoSig 4,881 genes
      4. LinCytoSig (best-bulk, orig) — best signature per cytokine selected by
         GTEx+TCGA bulk correlation, all ~20K genes
      5. LinCytoSig (best-bulk, filtered) — best bulk signature, CytoSig genes only
      6. SecAct — 1,249 secreted proteins
    """
    sixway_path = VIZ_DIR / 'method_comparison_6way_all.json'
    if not sixway_path.exists():
        # Fall back to old file
        sixway_path = VIZ_DIR / 'method_comparison_6way.json'
    if not sixway_path.exists():
        print(f'  WARNING: no 6-way JSON found — falling back to empty')
        return {}

    with open(sixway_path) as f:
        sixway = json.load(f)

    # Map 6-way atlas keys to display labels
    atlas_key_to_label = {
        'CIMA': 'CIMA',
        'Inflammation Main': 'Inflam (Main)',
        'Inflammation Val': 'Inflam (Val)',
        'Inflammation Ext': 'Inflam (Ext)',
        'scAtlas Normal': 'scAtlas (Normal)',
        'scAtlas Cancer': 'scAtlas (Cancer)',
    }

    # Method keys match the JSON keys directly
    method_keys = ['cytosig', 'lincyto_orig', 'lincyto_filt',
                   'lincyto_best_orig', 'lincyto_best_filt', 'secact']

    result = {}
    for src_key, label in atlas_key_to_label.items():
        if src_key not in sixway:
            continue
        atlas_data = sixway[src_key]
        entry = {}
        for key in method_keys:
            entry[key] = [round(v, 4) for v in atlas_data.get(key, [])]
        result[label] = entry
    return result


def prepare_consistency_data(df):
    """Prepare data for cross-atlas consistency line chart."""
    key_targets = ['IFNG', 'IL1B', 'TNFA', 'TGFB1', 'IL6', 'IL10', 'IL17A',
                   'IL4', 'BMP2', 'EGF', 'HGF', 'VEGFA', 'CXCL12', 'GMCSF']
    cytosig = df[df['signature'] == 'cytosig']
    result = {}
    for target in key_targets:
        rhos = []
        for atlas in ATLAS_ORDER:
            level = LEVEL_MAP[atlas]
            match = cytosig[(cytosig['target'] == target) & (cytosig['atlas'] == atlas)
                            & (cytosig['level'] == level)]
            if len(match) > 0:
                rhos.append(round(float(match['spearman_rho'].values[0]), 4))
            else:
                rhos.append(None)
        family = TARGET_TO_FAMILY.get(target, 'Other')
        result[target] = {'rhos': rhos, 'family': family, 'color': FAMILY_COLORS.get(family, '#9CA3AF')}
    return result


def prepare_heatmap_data(df):
    """Prepare heatmap matrices for CytoSig and SecAct (section 4.7)."""
    result = {}
    for sig_type in ['cytosig', 'secact']:
        sub = df[df['signature'] == sig_type].copy()
        sig_atlas_order = ATLAS_ORDER
        sig_atlas_labels = ATLAS_LABELS

        # Collect rows at donor level
        rows = []
        for atlas in sig_atlas_order:
            level = LEVEL_MAP[atlas]
            atlas_sub = sub[(sub['atlas'] == atlas) & (sub['level'] == level)]
            for _, row in atlas_sub.iterrows():
                rows.append(row)
        if not rows:
            result[sig_type] = {'targets': [], 'matrix': [], 'atlases': sig_atlas_labels}
            continue
        sub_df = pd.DataFrame(rows)

        if sig_type == 'cytosig':
            targets = sorted(sub_df['target'].unique())
        else:  # secact — show matched CytoSig targets + top additional by median rho
            # First: all matched targets
            matched_set = set(MATCHED_TARGETS)
            matched = sorted([t for t in sub_df['target'].unique() if t in matched_set])
            # Then: top additional targets by median rho across atlases
            remaining = sub_df[~sub_df['target'].isin(matched_set)]
            if len(remaining) > 0:
                median_rhos = remaining.groupby('target')['spearman_rho'].median().sort_values(ascending=False)
                top_extra = list(median_rhos.head(25).index)
            else:
                top_extra = []
            targets = matched + sorted(top_extra)

        matrix = []
        for target in targets:
            row_vals = []
            for atlas in sig_atlas_order:
                level = LEVEL_MAP[atlas]
                match = sub_df[(sub_df['target'] == target) & (sub_df['atlas'] == atlas)
                               & (sub_df['level'] == level)]
                if len(match) > 0:
                    row_vals.append(round(float(match['spearman_rho'].values[0]), 3))
                else:
                    row_vals.append(None)
            matrix.append(row_vals)

        result[sig_type] = {'targets': targets, 'matrix': matrix, 'atlases': sig_atlas_labels}
    return result


def prepare_levels_data(df):
    """Prepare aggregation level comparison data for CytoSig and SecAct (section 4.5)."""
    configs = [
        ('cima', ['donor_only', 'donor_l1', 'donor_l2', 'donor_l3', 'donor_l4'], 'CIMA'),
        ('inflammation_main', ['donor_only', 'donor_l1', 'donor_l2'], 'Inflammation (Main)'),
        ('scatlas_normal', ['donor_organ', 'donor_organ_celltype1', 'donor_organ_celltype2'], 'scAtlas Normal'),
    ]
    level_labels = {
        'donor_only': 'Donor Only', 'donor_l1': 'Donor x L1', 'donor_l2': 'Donor x L2',
        'donor_l3': 'Donor x L3', 'donor_l4': 'Donor x L4',
        'donor_organ': 'Donor x Organ', 'donor_organ_celltype1': 'Donor x Organ x CT1',
        'donor_organ_celltype2': 'Donor x Organ x CT2',
    }
    result = {}
    for atlas, levels, title in configs:
        result[title] = {}
        for sig_type in ['cytosig', 'secact']:
            sub = df[(df['atlas'] == atlas) & (df['signature'] == sig_type)]
            level_data = {}
            for level in levels:
                rhos = sub[sub['level'] == level]['spearman_rho'].dropna()
                if len(rhos) > 0:
                    level_data[level_labels.get(level, level)] = {
                        'rhos': [round(r, 4) for r in rhos.tolist()],
                        'median': round(float(rhos.median()), 4),
                        'n': int(len(rhos)),
                    }
            result[title][sig_type] = level_data
        # SecAct matched (only targets shared with CytoSig)
        secact_sub = df[(df['atlas'] == atlas) & (df['signature'] == 'secact')]
        matched_sub = secact_sub[secact_sub['target'].isin(MATCHED_TARGETS)]
        level_data = {}
        for level in levels:
            rhos = matched_sub[matched_sub['level'] == level]['spearman_rho'].dropna()
            if len(rhos) > 0:
                level_data[level_labels.get(level, level)] = {
                    'rhos': [round(r, 4) for r in rhos.tolist()],
                    'median': round(float(rhos.median()), 4),
                    'n': int(len(rhos)),
                }
        result[title]['secact_matched'] = level_data
    return result


def prepare_bulk_validation_data(df):
    """Prepare bulk validation data for GTEx/TCGA — CytoSig and SecAct only.
    For SecAct: show matched CytoSig targets + top additional targets to match count.
    """
    result = {}
    for atlas in ['gtex', 'tcga']:
        label = 'GTEx' if atlas == 'gtex' else 'TCGA'
        result[label] = {}

        # CytoSig
        cyto_sub = df[(df['atlas'] == atlas) & (df['signature'] == 'cytosig') & (df['level'] == 'donor_only')]
        if len(cyto_sub) > 0:
            sorted_sub = cyto_sub.sort_values('spearman_rho', ascending=False)
            cyto_targets = set(sorted_sub['target'].tolist())
            n_cyto = len(cyto_targets)
            result[label]['cytosig'] = {
                'targets': sorted_sub['target'].tolist(),
                'rhos': [round(r, 4) for r in sorted_sub['spearman_rho'].tolist()],
                'median': round(float(cyto_sub['spearman_rho'].median()), 3),
                'n': int(len(sorted_sub)),
            }
        else:
            cyto_targets = set()
            n_cyto = 44

        # SecAct — matched targets first, then top additional to match CytoSig count
        secact_sub = df[(df['atlas'] == atlas) & (df['signature'] == 'secact') & (df['level'] == 'donor_only')]
        if len(secact_sub) > 0:
            secact_sub = secact_sub.drop_duplicates(subset=['target'], keep='first')
            # Split into matched (shared with CytoSig) and extra
            matched = secact_sub[secact_sub['target'].isin(cyto_targets)].sort_values('spearman_rho', ascending=False)
            extra = secact_sub[~secact_sub['target'].isin(cyto_targets)].sort_values('spearman_rho', ascending=False)
            # Take matched + enough extra to reach n_cyto total
            n_extra = max(0, n_cyto - len(matched))
            selected = pd.concat([matched, extra.head(n_extra)], ignore_index=True)
            selected = selected.sort_values('spearman_rho', ascending=False)
            # Mark which are matched vs additional
            is_matched = [t in cyto_targets for t in selected['target']]
            result[label]['secact'] = {
                'targets': selected['target'].tolist(),
                'rhos': [round(r, 4) for r in selected['spearman_rho'].tolist()],
                'matched': is_matched,
                'median': round(float(selected['spearman_rho'].median()), 3),
                'n': int(len(selected)),
                'total_secact': int(len(secact_sub)),
            }
    return result


def prepare_scatter_data():
    """Prepare donor scatter data for key targets."""
    key_targets = ['IFNG', 'IL1B', 'TNFA', 'TGFB1', 'IL6', 'IL10', 'VEGFA', 'CD40L', 'TRAIL', 'HGF']
    atlas_files = {
        'CIMA': 'cima_cytosig.json',
        'Inflam (Main)': 'inflammation_main_cytosig.json',
        'scAtlas (Normal)': 'scatlas_normal_cytosig.json',
        'scAtlas (Cancer)': 'scatlas_cancer_cytosig.json',
        'GTEx': 'gtex_cytosig.json',
        'TCGA': 'tcga_cytosig.json',
    }
    result = {}
    for atlas_label, filename in atlas_files.items():
        data = load_scatter(SCATTER_DONOR, filename)
        if data is None:
            continue
        result[atlas_label] = {}
        for target in key_targets:
            if target in data:
                d = data[target]
                pts = d.get('points', [])
                result[atlas_label][target] = {
                    'x': [round(p[0], 3) for p in pts],
                    'y': [round(p[1], 3) for p in pts],
                    'rho': round(d.get('rho', 0), 4),
                    'pval': d.get('pval', 1),
                    'n': d.get('n', len(pts)),
                }
    return result


def prepare_lincytosig_vs_cytosig(df):
    """Prepare matched target data for LinCytoSig vs CytoSig scatter (section 5.2).
    Uses method_comparison.json which has celltype-level matched comparisons.
    """
    mc = load_method_comparison()
    if not mc:
        return {}
    categories = mc.get('categories', [])
    matched = mc.get('matched_targets', {})
    cyto_rhos = mc.get('cytosig', {}).get('rhos', {})
    lin_rhos = mc.get('lincytosig', {}).get('rhos', {})

    result = {}
    for cat in categories:
        key = cat['key']
        label = cat['label']
        points = []
        for lin_target, mapping in matched.items():
            cyto_target = mapping.get('cytosig')
            if not cyto_target:
                continue
            cyto_val = cyto_rhos.get(cyto_target, {}).get(key)
            lin_val = lin_rhos.get(lin_target, {}).get(key)
            if cyto_val is not None and lin_val is not None:
                points.append({
                    'target': lin_target,
                    'cytosig': round(float(cyto_val), 4),
                    'lincytosig': round(float(lin_val), 4),
                })
        n_lin_win = sum(1 for p in points if p['lincytosig'] > p['cytosig'])
        n_cyto_win = sum(1 for p in points if p['cytosig'] > p['lincytosig'])
        n_tie = len(points) - n_lin_win - n_cyto_win
        result[label] = {
            'points': points,
            'n_lin_win': n_lin_win,
            'n_cyto_win': n_cyto_win,
            'n_tie': n_tie,
        }
    return result


def prepare_good_bad_data(df):
    """Prepare top/bottom correlated targets per atlas for CytoSig and SecAct."""
    atlas_configs = [
        ('cima', 'donor_only', 'CIMA'),
        ('inflammation_main', 'donor_only', 'Inflam (Main)'),
        ('scatlas_normal', 'donor_organ', 'scAtlas (Normal)'),
        ('gtex', 'donor_only', 'GTEx'),
        ('tcga', 'donor_only', 'TCGA'),
    ]
    result = {}
    for sig_type in ['cytosig', 'secact']:
        result[sig_type] = {}
        for atlas, level, label in atlas_configs:
            sub = df[(df['atlas'] == atlas) & (df['level'] == level) & (df['signature'] == sig_type)]
            # Drop duplicates by target (keep first = highest level match)
            sub = sub.drop_duplicates(subset=['target'], keep='first')
            sub = sub.sort_values('spearman_rho', ascending=False).reset_index(drop=True)
            top = sub.head(15)
            bottom = sub.tail(15).sort_values('spearman_rho', ascending=True).reset_index(drop=True)
            result[sig_type][label] = {
                'top': [{'target': r['target'], 'rho': round(float(r['spearman_rho']), 4)}
                        for _, r in top.iterrows()],
                'bottom': [{'target': r['target'], 'rho': round(float(r['spearman_rho']), 4)}
                           for _, r in bottom.iterrows()],
            }
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# HTML GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def generate_html(summary_table, boxplot_data, consistency_data, heatmap_data,
                  levels_data, bulk_data, scatter_data, good_bad_data,
                  method_boxplot_data, lincytosig_vs_cytosig_data):

    # Serialize all data as JSON for embedding
    data_json = json.dumps({
        'summary': summary_table,
        'boxplot': boxplot_data,
        'consistency': consistency_data,
        'heatmap': heatmap_data,
        'levels': levels_data,
        'bulk': bulk_data,
        'scatter': scatter_data,
        'goodBad': good_bad_data,
        'atlasLabels': ATLAS_LABELS,
        'sigColors': SIG_COLORS,
        'familyColors': FAMILY_COLORS,
        'methodBoxplot': method_boxplot_data,
        'linVsCyto': lincytosig_vs_cytosig_data,
    }, separators=(',', ':'))

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CytoAtlas: PI Report &mdash; Peng Jiang</title>
<script src="https://cdn.plot.ly/plotly-2.35.0.min.js" charset="utf-8"></script>
<style>
  :root {{
    --blue: #2563EB; --amber: #D97706; --emerald: #059669; --red: #DC2626;
    --gray-50: #F9FAFB; --gray-100: #F3F4F6; --gray-200: #E5E7EB;
    --gray-300: #D1D5DB; --gray-500: #6B7280; --gray-700: #374151; --gray-900: #111827;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    color: var(--gray-900); background: var(--gray-50);
    line-height: 1.7; font-size: 15px;
  }}
  .container {{ max-width: 1200px; margin: 0 auto; background: white; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  .header {{
    background: linear-gradient(135deg, #1E3A5F 0%, #2563EB 50%, #059669 100%);
    color: white; padding: 50px 60px 40px;
  }}
  .header h1 {{ font-size: 32px; font-weight: 700; margin-bottom: 8px; }}
  .header .subtitle {{ font-size: 18px; opacity: 0.9; margin-bottom: 24px; }}
  .header .meta {{ font-size: 14px; opacity: 0.8; line-height: 1.8; }}
  .header .meta strong {{ opacity: 1; }}
  .content {{ padding: 40px 60px 60px; }}
  h2 {{
    font-size: 24px; font-weight: 700; color: var(--gray-900);
    border-bottom: 3px solid var(--blue); padding-bottom: 8px;
    margin: 48px 0 20px;
  }}
  h2:first-child {{ margin-top: 0; }}
  h3 {{ font-size: 18px; font-weight: 600; color: var(--gray-700); margin: 32px 0 12px; }}
  p {{ margin: 0 0 14px; }}
  ul, ol {{ margin: 0 0 14px 24px; }}
  li {{ margin-bottom: 6px; }}
  table {{ width: 100%; border-collapse: collapse; margin: 16px 0 20px; font-size: 13.5px; }}
  th {{
    background: var(--gray-700); color: white; padding: 10px 12px;
    text-align: left; font-weight: 600; font-size: 12.5px;
    text-transform: uppercase; letter-spacing: 0.3px;
  }}
  td {{ padding: 8px 12px; border-bottom: 1px solid var(--gray-200); vertical-align: top; }}
  tr:nth-child(even) td {{ background: var(--gray-50); }}
  tr:hover td {{ background: #EFF6FF; }}
  .figure {{
    margin: 28px 0; background: var(--gray-50);
    border: 1px solid var(--gray-200); border-radius: 8px; overflow: hidden;
  }}
  .figure img {{
    width: 100%; display: block; border-bottom: 1px solid var(--gray-200);
    cursor: zoom-in; transition: filter 0.2s;
  }}
  .figure img:hover {{ filter: brightness(0.93); }}
  .figure .caption {{ padding: 12px 16px; font-size: 13px; color: var(--gray-500); }}
  .figure .caption strong {{ color: var(--gray-700); }}
  .callout {{
    background: #EFF6FF; border-left: 4px solid var(--blue);
    padding: 16px 20px; margin: 20px 0; border-radius: 0 6px 6px 0;
  }}
  .callout.green {{ background: #ECFDF5; border-color: var(--emerald); }}
  .callout.amber {{ background: #FFFBEB; border-color: var(--amber); }}
  .callout.red {{ background: #FEF2F2; border-color: var(--red); }}
  .callout p:last-child {{ margin-bottom: 0; }}
  .stats-row {{ display: flex; gap: 16px; margin: 24px 0; flex-wrap: wrap; }}
  .stat-card {{
    flex: 1; min-width: 140px; background: white;
    border: 1px solid var(--gray-200); border-radius: 8px;
    padding: 16px; text-align: center;
  }}
  .stat-card .number {{ font-size: 28px; font-weight: 700; color: var(--blue); display: block; }}
  .stat-card .label {{
    font-size: 12px; color: var(--gray-500); text-transform: uppercase;
    letter-spacing: 0.5px; margin-top: 4px;
  }}
  hr {{ border: none; border-top: 2px solid var(--gray-200); margin: 40px 0; }}
  code {{
    background: var(--gray-100); padding: 2px 6px; border-radius: 4px;
    font-family: 'Consolas', 'Monaco', monospace; font-size: 13px;
  }}
  .badge {{
    display: inline-block; padding: 2px 8px; border-radius: 12px;
    font-size: 11px; font-weight: 600; text-transform: uppercase;
  }}
  .badge.blue {{ background: #DBEAFE; color: var(--blue); }}
  .badge.amber {{ background: #FEF3C7; color: var(--amber); }}
  .badge.green {{ background: #D1FAE5; color: var(--emerald); }}
  .win {{ color: var(--emerald); font-weight: 600; }}
  .lose {{ color: var(--red); font-weight: 600; }}
  .toc {{
    background: var(--gray-50); border: 1px solid var(--gray-200);
    border-radius: 8px; padding: 24px 28px; margin: 24px 0;
  }}
  .toc h3 {{ margin: 0 0 12px; font-size: 16px; }}
  .toc ol {{ margin: 0 0 0 20px; }}
  .toc li {{ margin-bottom: 4px; }}
  .toc a {{ color: var(--blue); text-decoration: none; }}
  .toc a:hover {{ text-decoration: underline; }}
  /* Plotly containers */
  .plotly-container {{
    margin: 20px 0; border: 1px solid var(--gray-200); border-radius: 8px;
    overflow: hidden; background: white;
  }}
  .plotly-container .caption {{
    padding: 10px 16px; font-size: 13px; color: var(--gray-500);
    border-top: 1px solid var(--gray-200); background: var(--gray-50);
  }}
  /* Tabs for heatmap/levels */
  .tab-bar {{
    display: flex; gap: 0; border-bottom: 2px solid var(--gray-200); margin-bottom: 0;
  }}
  .tab-btn {{
    padding: 10px 20px; border: none; background: var(--gray-50);
    cursor: pointer; font-size: 13px; font-weight: 600;
    color: var(--gray-500); border-bottom: 3px solid transparent;
    transition: all 0.2s;
  }}
  .tab-btn:hover {{ background: var(--gray-100); }}
  .tab-btn.active {{ color: var(--blue); border-bottom-color: var(--blue); background: white; }}
  .tab-btn.cytosig {{ color: var(--blue); }}
  .tab-btn.lincytosig {{ color: var(--amber); }}
  .tab-btn.secact {{ color: var(--emerald); }}
  /* Dropdown controls */
  .controls {{ padding: 12px 16px; display: flex; gap: 16px; align-items: center; flex-wrap: wrap; }}
  .controls label {{ font-size: 13px; font-weight: 600; color: var(--gray-700); }}
  .controls select {{
    padding: 6px 12px; border: 1px solid var(--gray-300); border-radius: 6px;
    font-size: 13px; background: white; cursor: pointer;
  }}
  /* Lightbox */
  .lightbox-overlay {{
    position: fixed; top: 0; left: 0; width: 100%; height: 100%;
    background: rgba(0,0,0,0.92); z-index: 1000;
    display: flex; align-items: center; justify-content: center;
    opacity: 0; pointer-events: none; transition: opacity 0.3s ease;
  }}
  .lightbox-overlay.active {{ opacity: 1; pointer-events: auto; }}
  .lightbox-overlay img {{
    max-width: 92vw; max-height: 88vh; object-fit: contain;
    border-radius: 4px; box-shadow: 0 8px 40px rgba(0,0,0,0.6);
  }}
  .lightbox-close {{
    position: absolute; top: 16px; right: 20px; color: #fff; font-size: 32px;
    cursor: pointer; background: rgba(255,255,255,0.1); border: none; border-radius: 50%;
    width: 44px; height: 44px; display: flex; align-items: center; justify-content: center;
  }}
  .lightbox-close:hover {{ background: rgba(255,255,255,0.25); }}
  @media print {{
    body {{ background: white; }}
    .container {{ box-shadow: none; }}
    .header {{ -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
    .figure {{ break-inside: avoid; }}
    h2 {{ break-after: avoid; }}
    .lightbox-overlay {{ display: none !important; }}
    .plotly-container {{ break-inside: avoid; }}
  }}
</style>
</head>
<body>

<div class="container">

<!-- HEADER -->
<div class="header">
  <h1>CytoAtlas</h1>
  <div class="subtitle">Pan-Disease Single-Cell Cytokine Activity Atlas</div>
  <div class="meta">
    <strong>Report for:</strong> Peng Jiang, Ph.D. &mdash; CDSL, National Cancer Institute<br>
    <strong>Prepared by:</strong> Seongyong Park<br>
    <strong>Date:</strong> February 2026
  </div>
</div>

<div class="content">

<!-- EXECUTIVE SUMMARY -->
<h2>Executive Summary</h2>

<p>CytoAtlas is a comprehensive computational resource that maps cytokine and secreted protein signaling activity across <strong>240 million human cells</strong> from six independent datasets spanning healthy donors, inflammatory diseases, cancers, drug perturbations, and spatial transcriptomics. The system uses <strong>linear ridge regression</strong> against experimentally derived signature matrices to infer activity &mdash; producing fully interpretable, conditional z-scores rather than black-box predictions.</p>

<div class="stats-row">
  <div class="stat-card"><span class="number">240M</span><span class="label">Total Cells</span></div>
  <div class="stat-card"><span class="number">6</span><span class="label">Datasets</span></div>
  <div class="stat-card"><span class="number">1,293</span><span class="label">Signatures</span></div>
  <div class="stat-card"><span class="number">8</span><span class="label">Validation Atlases</span></div>
  <div class="stat-card"><span class="number">262</span><span class="label">API Endpoints</span></div>
  <div class="stat-card"><span class="number">12</span><span class="label">Web Pages</span></div>
</div>

<div class="callout green">
<p><strong>Key results:</strong></p>
<ul>
  <li>1,293 signatures (44 CytoSig + 178 LinCytoSig + 1,249 SecAct) validated across 8 independent atlases</li>
  <li>Spearman correlations reach &rho;=0.6&ndash;0.9 for well-characterized cytokines (IL1B, TNFA, VEGFA, TGFB family)</li>
  <li>Cross-atlas consistency demonstrates signatures generalize across CIMA, Inflammation Atlas, scAtlas, GTEx, and TCGA</li>
  <li>LinCytoSig improves prediction for select immune cell types (Basophil, NK, DC: +0.18&ndash;0.21 &Delta;&rho;)</li>
  <li>SecAct achieves the highest correlations in bulk &amp; organ-level analyses (median &rho;=0.40 in GTEx/TCGA)</li>
</ul>
</div>

<!-- TOC (inline only, no floating TOC — item 0) -->
<div class="toc">
  <h3>Table of Contents</h3>
  <ol>
    <li><a href="#sec1">System Architecture and Design Rationale</a></li>
    <li><a href="#sec2">Dataset Catalog</a></li>
    <li><a href="#sec3">Scientific Value Proposition</a></li>
    <li><a href="#sec4">Validation Results</a></li>
    <li><a href="#sec5">CytoSig vs LinCytoSig vs SecAct Comparison</a></li>
    <li><a href="#sec6">Key Takeaways for Scientific Discovery</a></li>
    <li><a href="#sec7">Appendix: Technical Specifications</a></li>
  </ol>
</div>

<hr>

<!-- SECTION 1 -->
<h2 id="sec1">1. System Architecture and Design Rationale</h2>

<h3>1.1 Why This Architecture?</h3>

<p>CytoAtlas was designed around three principles that distinguish it from typical bioinformatics databases:</p>

<p><strong>Principle 1: Linear interpretability over complex models.</strong><br>
Ridge regression (L2-regularized linear regression) was chosen deliberately over methods like autoencoders, graph neural networks, or foundation models. The resulting activity z-scores are <strong>conditional on the specific genes in the signature matrix</strong>, meaning every prediction can be traced to a weighted combination of known gene responses.</p>

<p><strong>Principle 2: Multi-level validation at every aggregation.</strong><br>
CytoAtlas validates at five levels: donor-level pseudobulk, donor &times; cell-type pseudobulk, single-cell, bulk RNA-seq (GTEx/TCGA), and bootstrap resampled with confidence intervals.</p>

<p><strong>Principle 3: Reproducibility through separation of concerns.</strong></p>

<table>
  <tr><th>Component</th><th>Technology</th><th>Purpose</th></tr>
  <tr><td><strong>Pipeline</strong></td><td>Python + CuPy (GPU)</td><td>Activity inference, 10&ndash;34x speedup</td></tr>
  <tr><td><strong>Storage</strong></td><td>DuckDB (3 databases, 68 tables)</td><td>Columnar analytics, no server needed</td></tr>
  <tr><td><strong>API</strong></td><td>FastAPI (262 endpoints)</td><td>RESTful data access, caching, auth</td></tr>
  <tr><td><strong>Frontend</strong></td><td>React 19 + TypeScript</td><td>Interactive exploration (12 pages)</td></tr>
</table>

<h3>1.2 Processing Scale</h3>

<table>
  <tr><th>Dataset</th><th>Cells</th><th>Samples</th><th>Time</th><th>GPU</th></tr>
  <tr><td>CIMA</td><td>6.5M</td><td>421 donors</td><td>~2h</td><td>A100</td></tr>
  <tr><td>Inflammation Atlas</td><td>6.3M</td><td>1,047 samples</td><td>~2h</td><td>A100</td></tr>
  <tr><td>scAtlas</td><td>6.4M</td><td>781 donors</td><td>~2h</td><td>A100</td></tr>
  <tr><td>parse_10M</td><td>9.7M</td><td>1,092 conditions</td><td>~3h</td><td>A100</td></tr>
  <tr><td>Tahoe-100M</td><td>100.6M</td><td>14 plates</td><td>~12h</td><td>A100</td></tr>
  <tr><td>SpatialCorpus-110M</td><td>~110M</td><td>251 datasets</td><td>~12h</td><td>A100</td></tr>
</table>

<div class="figure">
  <img src="figures/fig1_dataset_overview.png" alt="Figure 1: Dataset Overview">
  <div class="caption"><strong>Figure 1.</strong> CytoAtlas overview. (A) Cell counts across 6 datasets totaling 240M cells. (B) Three signature matrices. (C) Multi-level validation strategy.</div>
</div>

<hr>

<!-- SECTION 2: Dataset Catalog -->
<h2 id="sec2">2. Dataset Catalog</h2>

<h3>2.1 Datasets and Scale</h3>

<!-- Item 1: Updated references -->
<table>
  <tr><th>#</th><th>Dataset</th><th>Cells</th><th>Donors/Samples</th><th>Cell Types</th><th>Reference</th></tr>
  <tr><td>1</td><td><strong>CIMA</strong></td><td>6,484,974</td><td>421 donors</td><td>27 L2 / 100+ L3</td><td>J. Yin et al., <em>Science</em>, 2026</td></tr>
  <tr><td>2</td><td><strong>Inflammation Atlas</strong></td><td>6,340,934</td><td>1,047 samples</td><td>66+</td><td>Jimenez-Gracia et al., <em>Nature Medicine</em>, 2026</td></tr>
  <tr><td>3</td><td><strong>scAtlas</strong></td><td>6,440,926</td><td>781 donors</td><td>100+</td><td>Q. Shi et al., <em>Nature</em>, 2025</td></tr>
  <tr><td>4</td><td><strong>parse_10M</strong></td><td>9,697,974</td><td>12 donors &times; 91 cytokines</td><td>18 PBMC types</td><td>Oesinghaus et al., <em>bioRxiv</em>, 2026</td></tr>
  <tr><td>5</td><td><strong>Tahoe-100M</strong></td><td>~100,600,000</td><td>50 cell lines &times; 95 drugs</td><td>50 cell lines</td><td>Zhang et al., <em>bioRxiv</em>, 2026</td></tr>
  <tr><td>6</td><td><strong>SpatialCorpus-110M</strong></td><td>~110,000,000</td><td>251 spatial datasets</td><td>Variable</td><td>Tejada-Lapuerta et al., <em>Nature Methods</em>, 2025</td></tr>
</table>

<h3>2.2 Disease and Condition Categories</h3>

<p><strong>Inflammation Atlas (20 diseases):</strong> RA, SLE, Sjogren's, PSA, Crohn's, UC, COVID-19, Sepsis, HIV, HBV, BRCA, CRC, HNSCC, NPC, COPD, Cirrhosis, MS, Asthma, Atopic Dermatitis</p>
<p><strong>scAtlas:</strong> Normal (35+ organs) + Cancer (15+ types: LUAD, CRC, BRCA, LIHC, PAAD, KIRC, OV, SKCM, GBM, etc.)</p>
<!-- Item 2: parse_10M is NOT ground truth -->
<p><strong>parse_10M:</strong> 90 cytokines &times; 12 donors &mdash; independent <em>in vitro</em> perturbation dataset for comparison. A considerable portion of cytokines (~58%) are produced in <em>E. coli</em>, with the remainder from insect (Sf21, 12%) and mammalian (CHO, NS0, HEK293, ~30%) expression systems. Because exogenous perturbagens may induce effects differing from endogenously produced cytokines, parse_10M serves as an independent comparison rather than strict ground truth. CytoSig/SecAct has a potential advantage in this regard, as it infers relationships directly from physiologically relevant samples.</p>
<p><strong>Tahoe-100M:</strong> 95 drugs across 50 cancer cell lines</p>
<p><strong>SpatialCorpus:</strong> Visium, Xenium, MERFISH, MERSCOPE, CosMx, ISS, Slide-seq &mdash; 30+ tissue types</p>

<h3>2.3 Signature Matrices</h3>

<!-- Item 3: Updated SecAct reference -->
<table>
  <tr><th>Matrix</th><th>Targets</th><th>Construction</th><th>Reference</th></tr>
  <tr><td><span class="badge blue">CytoSig</span></td><td>44 cytokines</td><td>Median log2FC across all experimental bulk RNA-seq</td><td>Jiang et al., <em>Nature Methods</em>, 2021</td></tr>
  <tr><td><span class="badge amber">LinCytoSig</span></td><td>178 (45 cell types &times; 1&ndash;13 cytokines)</td><td>Cell-type-stratified median from CytoSig database (<a href="LINCYTOSIG_METHODOLOGY.html">methodology</a>)</td><td>This work</td></tr>
  <tr><td><span class="badge green">SecAct</span></td><td>1,249 secreted proteins</td><td>Median global Moran's I across 1,000 Visium datasets</td><td>Ru et al., <em>Nature Methods</em>, 2026 (in press)</td></tr>
</table>

<hr>

<!-- SECTION 3: Scientific Value -->
<h2 id="sec3">3. Scientific Value Proposition</h2>

<h3>3.1 What Makes CytoAtlas Different from Deep Learning Approaches?</h3>

<p>Most single-cell analysis tools use complex models (VAEs, GNNs, transformers) that produce <strong>aggregated, non-linear representations</strong> difficult to interpret biologically. CytoAtlas takes the opposite approach:</p>

<table>
  <tr><th>Property</th><th>CytoAtlas (Ridge Regression)</th><th>Typical DL Approach</th></tr>
  <tr><td><strong>Model</strong></td><td>Linear (z = X&beta; + &epsilon;)</td><td>Non-linear (multi-layer NN)</td></tr>
  <tr><td><strong>Interpretability</strong></td><td>Every gene's contribution is a coefficient</td><td>Feature importance approximated post-hoc</td></tr>
  <tr><td><strong>Conditionality</strong></td><td>Activity conditional on specific gene set</td><td>Latent space mixes all features</td></tr>
  <tr><td><strong>Confidence</strong></td><td>Permutation-based z-scores with CI</td><td>Often point estimates only</td></tr>
  <tr><td><strong>Generalization</strong></td><td>Tested across 8 independent cohorts</td><td>Often held-out splits of same cohort</td></tr>
  <tr><td><strong>Bias</strong></td><td>Transparent &mdash; limited by signature matrix genes</td><td>Hidden in architecture and training data</td></tr>
</table>

<!-- Item 4: DNN-based → DL-based -->
<div class="callout">
<p><strong>The key insight:</strong> CytoAtlas is not trying to replace DL-based tools. It provides an <strong>orthogonal, complementary signal</strong> that a human scientist can directly inspect. When CytoAtlas says "IFNG activity is elevated in CD8+ T cells from RA patients," you can verify this by checking the IFNG signature genes in those cells.</p>
</div>

<h3>3.2 What Scientific Questions Does CytoAtlas Answer?</h3>
<ol>
  <li><strong>Which cytokines are active in which cell types across diseases?</strong></li>
  <li><strong>Are cytokine activities consistent across independent cohorts?</strong></li>
  <li><strong>Does cell-type-specific biology matter for cytokine inference?</strong></li>
  <li><strong>Which secreted proteins beyond cytokines show validated activity?</strong></li>
  <li><strong>How do drugs alter cytokine activity in cancer cells?</strong></li>
  <li><strong>What is the spatial organization of cytokine signaling?</strong></li>
  <li><strong>Can we predict treatment response from cytokine activity?</strong></li>
</ol>

<h3>3.3 Validation Philosophy</h3>
<p>CytoAtlas validates against a simple but powerful principle: <strong>if CytoSig predicts high IFNG activity for a sample, that sample should have high IFNG gene expression.</strong> This expression-activity correlation is computed via Spearman rank correlation across donors/samples.</p>
<p>This is a <em>conservative</em> validation &mdash; it only captures signatures where the target gene itself is expressed. Signatures that act through downstream effectors would not be captured, meaning our validation <strong>underestimates</strong> true accuracy.</p>

<hr>

<!-- SECTION 4: Validation Results -->
<h2 id="sec4">4. Validation Results</h2>

<!-- Item 5: Embedded summary table instead of figure -->
<h3>4.1 Overall Performance Summary</h3>

<div id="summary-table-container"></div>

<div class="callout">
<p><strong>How &ldquo;N Targets&rdquo; is determined:</strong> A target is included in the validation for a given atlas only if (1) the target&rsquo;s signature genes overlap sufficiently with the atlas gene expression matrix, and (2) the target gene itself is expressed in enough samples to compute a meaningful Spearman correlation. Targets whose gene is absent or not detected in a dataset are excluded.</p>
<p><strong>Donor-only atlases</strong> (CIMA, Inflammation, GTEx, TCGA): N = number of unique targets with valid correlations. CytoSig defines 43 cytokines and SecAct defines 1,170 secreted proteins. The Inflammation Atlas (main/validation cohorts) retains only 33 of 43 CytoSig targets and 805 of 1,170 SecAct targets because 10 cytokine genes (BDNF, BMP4, CXCL12, GCSF, IFN1, IL13, IL17A, IL36, IL4, WNT3A) are not sufficiently expressed in these blood/PBMC samples. CIMA, GTEx, and similar multi-organ datasets retain nearly all targets (&ge;97%).</p>
<p><strong>Donor-organ atlases</strong> (scAtlas Normal, scAtlas Cancer): N = <strong>target &times; organ pairs</strong>, because validation is stratified by organ/tissue context. For scAtlas Normal, each target is validated independently across 25 organs (Bladder, Blood, Breast, Colon, Heart, Kidney, Liver, Lung, etc.), yielding up to 43 &times; 25 = 1,075 CytoSig entries (actual: 1,013 after filtering) and 1,140 &times; 25 = 28,500 SecAct entries (actual: 27,154). For scAtlas Cancer, validation spans 7 tissue contexts (Tumor, Adjacent, Blood, Metastasis, Pleural Fluids, Pre-Lesion, All), yielding 43 &times; 7 = 301 CytoSig entries (actual: 295) and 1,140 &times; 7 = 7,980 SecAct entries (actual: 7,809). Some target-organ pairs are excluded when the target gene lacks sufficient expression in that organ.</p>
<p><strong>Note on scAtlas duplicate entries:</strong> At finer aggregation levels (e.g., donor_organ_celltype1 vs donor_organ_celltype2), the same target can appear multiple times with different correlation values. This is expected &mdash; finer cell-type annotation changes the composition of each pseudobulk sample, yielding different expression-activity relationships. The summary table above uses the donor_organ level for scAtlas.</p>
</div>

<!-- Item 6: Interactive boxplot -->
<h3>4.2 Correlation Distributions</h3>
<div class="plotly-container">
  <div id="boxplot-chart" style="height:500px;"></div>
  <div class="caption"><strong>Figure 2.</strong> Spearman &rho; distributions across atlases for CytoSig (44 targets), SecAct (1,249 targets), and SecAct restricted to CytoSig-matched targets (22 shared targets). Donor-level pseudobulk. Hover for details.</div>
</div>

<div class="callout">
<p><strong>Why does SecAct appear to underperform CytoSig in the Inflammation Atlas?</strong></p>
<p>This is a <strong>composition effect</strong>, not a genuine performance gap. CytoSig tests only 43 curated, high-signal cytokines, while SecAct tests 1,249 secreted proteins &mdash; including many tissue-expressed targets (collagens, metalloproteinases, apolipoproteins, complement factors) with minimal expression variation in blood/PBMC samples. On the <strong>22 matched targets</strong> shared between both methods, SecAct consistently outperforms CytoSig across all atlases (e.g., median &rho; = 0.51 vs 0.32 in Inflammation Main).</p>
<p>The Inflammation Atlas is largely blood-derived, so many SecAct targets that perform well in multi-organ contexts (scAtlas, GTEx, TCGA) contribute near-zero or negative correlations here. In fact, 99 SecAct targets are negative <em>only</em> in inflammation but positive in all other atlases, reflecting tissue-specific expression limitations rather than inference failure. The &ldquo;SecAct (CytoSig-matched)&rdquo; boxplot above demonstrates the fair comparison on equal footing.</p>
</div>

<h3>4.3 Best and Worst Correlated Targets</h3>
<div class="plotly-container">
  <div class="controls">
    <label>Signature:</label>
    <select id="goodbad-sig-select" onchange="updateGoodBad()">
      <option value="cytosig">CytoSig</option>
      <option value="secact">SecAct</option>
    </select>
    <label>Atlas:</label>
    <select id="goodbad-atlas-select" onchange="updateGoodBad()">
    </select>
  </div>
  <div id="goodbad-chart" style="height:700px;"></div>
  <div class="caption"><strong>Figure 3.</strong> Top 15 (best) and bottom 15 (worst) correlated targets. Select signature type and atlas from dropdowns.</div>
</div>

<p><strong>Consistently well-correlated targets (&rho; &gt; 0.3 across multiple atlases):</strong></p>
<ul>
  <li><strong>IL1B</strong> (&rho; = 0.67 CIMA, 0.68 Inflammation) &mdash; canonical inflammatory cytokine</li>
  <li><strong>TNFA</strong> (&rho; = 0.63 CIMA, 0.60 Inflammation) &mdash; master inflammatory regulator</li>
  <li><strong>VEGFA</strong> (&rho; = 0.79 Inflammation, 0.92 scAtlas) &mdash; angiogenesis factor</li>
  <li><strong>TGFB1/2/3</strong> (&rho; = 0.35&ndash;0.55 across atlases)</li>
  <li><strong>BMP2/4</strong> (&rho; = 0.26&ndash;0.92 depending on atlas)</li>
</ul>

<p><strong>Consistently poorly correlated targets (&rho; &lt; 0 in multiple atlases):</strong></p>
<ul>
  <li><strong>CD40L</strong> (&rho; = &minus;0.48 CIMA, &minus;0.56 Inflammation) &mdash; membrane-bound, not secreted</li>
  <li><strong>TRAIL</strong> (&rho; = &minus;0.46 CIMA, &minus;0.55 Inflammation) &mdash; apoptosis inducer</li>
  <li><strong>LTA</strong> (&rho; = &minus;0.33 CIMA), <strong>HGF</strong> (&rho; = &minus;0.25 CIMA)</li>
</ul>

<!-- Gene mapping verification and mechanistic analysis -->
<div class="callout amber">
<p><strong>Gene mapping verified:</strong> All four targets are correctly mapped (CD40L&rarr;CD40LG, TRAIL&rarr;TNFSF10, LTA&rarr;LTA, HGF&rarr;HGF). No gene ID confusion exists. The poor correlations reflect specific molecular mechanisms:</p>
</div>

<table>
  <tr><th>Target</th><th>Gene</th><th>Dominant Mechanism</th><th>Contributing Factors</th></tr>
  <tr>
    <td><strong>CD40L</strong></td><td>CD40LG</td>
    <td>Platelet-derived sCD40L invisible to scRNA-seq (~95% of circulating CD40L); ADAM10-mediated membrane shedding</td>
    <td>Unstable mRNA (3&prime;-UTR destabilizing element); transient expression kinetics (peak 6&ndash;8h post-activation); paracrine disconnect (T cell &rarr; B cell/DC)</td>
  </tr>
  <tr>
    <td><strong>TRAIL</strong></td><td>TNFSF10</td>
    <td>Three decoy receptors (DcR1/TNFRSF10C, DcR2/TNFRSF10D, OPG/TNFRSF11B) competitively sequester ligand without signaling</td>
    <td>Non-functional splice variants (TRAIL-beta, TRAIL-gamma lack exon 3) inflate mRNA counts; cathepsin E-mediated shedding; apoptosis-induced survival bias in scRNA-seq data</td>
  </tr>
  <tr>
    <td><strong>LTA</strong></td><td>LTA</td>
    <td>Obligate heteromeric complex with LTB: the dominant form (LT&alpha;1&beta;2) requires <em>LTB</em> co-expression and signals through LTBR, not TNFR1/2</td>
    <td>Mathematical collinearity with TNFA in ridge regression (LTA3 homotrimer binds the same TNFR1/2 receptors as TNF-&alpha;); 7 known splice variants; low/transient expression</td>
  </tr>
  <tr>
    <td><strong>HGF</strong></td><td>HGF</td>
    <td>Obligate mesenchymal-to-epithelial paracrine topology: HGF produced by fibroblasts/stellate cells, MET receptor on epithelial cells</td>
    <td>Secreted as inactive pro-HGF requiring proteolytic cleavage by HGFAC/uPA (post-translational activation is rate-limiting); ECM/heparin sequestration creates stored protein pool invisible to transcriptomics</td>
  </tr>
</table>

<div class="callout">
<p><strong>Key insight:</strong> None of these targets have isoforms or subunits mapping to different gene IDs that would cause gene ID confusion. The poor correlations are driven by <strong>post-translational regulation</strong> (membrane shedding, proteolytic activation, decoy receptor sequestration), <strong>paracrine signaling topology</strong> (producer and responder cells are different cell types), and <strong>heteromeric complex dependence</strong> (LTA requires LTB). These represent fundamental limitations of using ligand mRNA abundance to predict downstream signaling activity &mdash; the CytoSig activity scores themselves remain valid readouts of pathway activation in the measured cells.</p>
</div>

<!-- Item 10: Interactive consistency plot -->
<h3>4.4 Cross-Atlas Consistency</h3>
<div class="plotly-container">
  <div id="consistency-chart" style="height:550px;"></div>
  <div class="caption"><strong>Figure 4.</strong> Key cytokine target correlations tracked across 8 independent atlases (CytoSig, donor-level). Lines are colored by cytokine family: <span style="color:#DC2626;font-weight:600">Interferon</span> (red), <span style="color:#2563EB;font-weight:600">TGF-&beta;</span> (blue), <span style="color:#059669;font-weight:600">Interleukin</span> (green), <span style="color:#D97706;font-weight:600">TNF</span> (amber), <span style="color:#8B5CF6;font-weight:600">Growth Factor</span> (purple), <span style="color:#EC4899;font-weight:600">Chemokine</span> (pink), <span style="color:#6366F1;font-weight:600">Colony-Stimulating</span> (indigo). Click legend entries to show/hide targets.</div>
</div>

<!-- Item 11: Aggregation levels for all 3 methods side by side -->
<h3>4.5 Effect of Aggregation Level</h3>
<div class="plotly-container">
  <div class="controls">
    <label>Atlas:</label>
    <select id="levels-atlas-select" onchange="renderLevels()">
    </select>
  </div>
  <div id="levels-chart" style="height:500px;"></div>
  <div class="caption"><strong>Figure 5.</strong> Effect of cell-type annotation granularity on validation correlations. CytoSig (43 targets), SecAct (1,249 targets), and SecAct restricted to CytoSig-matched targets (22 shared targets) shown side by side. Select atlas from dropdown.</div>
</div>

<div class="callout">
<p><strong>Aggregation levels explained:</strong> Pseudobulk profiles are aggregated at increasingly fine cell-type resolution. At coarser levels, each pseudobulk profile averages more cells, yielding smoother expression estimates but masking cell-type-specific signals. At finer levels, each profile is more cell-type-specific but based on fewer cells.</p>
</div>

<table>
  <tr><th>Atlas</th><th>Level</th><th>Description</th><th>N Cell Types</th></tr>
  <tr><td rowspan="5"><strong>CIMA</strong></td>
    <td>Donor Only</td><td>Whole-sample pseudobulk per donor</td><td>1 (all)</td></tr>
  <tr><td>Donor &times; L1</td><td>Broad lineages (B, CD4_T, CD8_T, Myeloid, NK, etc.)</td><td>7</td></tr>
  <tr><td>Donor &times; L2</td><td>Intermediate (CD4_memory, CD8_naive, DC, Mono, etc.)</td><td>28</td></tr>
  <tr><td>Donor &times; L3</td><td>Fine-grained (CD4_Tcm, cMono, Switched_Bm, etc.)</td><td>39</td></tr>
  <tr><td>Donor &times; L4</td><td>Finest marker-annotated (CD4_Th17-like_RORC, cMono_IL1B, etc.)</td><td>73</td></tr>
  <tr><td rowspan="3"><strong>Inflammation</strong></td>
    <td>Donor Only</td><td>Whole-sample pseudobulk per donor</td><td>1 (all)</td></tr>
  <tr><td>Donor &times; L1</td><td>Broad categories (B, DC, Mono, T_CD4/CD8 subsets, etc.)</td><td>18</td></tr>
  <tr><td>Donor &times; L2</td><td>Fine-grained (Th1, Th2, Tregs, NK_adaptive, etc.)</td><td>65</td></tr>
  <tr><td rowspan="3"><strong>scAtlas Normal</strong></td>
    <td>Donor &times; Organ</td><td>Per-organ pseudobulk (Bladder, Blood, Breast, Lung, etc.)</td><td>25 organs</td></tr>
  <tr><td>Donor &times; Organ &times; CT1</td><td>Broad cell types within each organ</td><td>191</td></tr>
  <tr><td>Donor &times; Organ &times; CT2</td><td>Fine cell types within each organ</td><td>356</td></tr>
</table>

<h3>4.6 Representative Scatter Plots</h3>
<div class="plotly-container">
  <div class="controls">
    <label>Target:</label>
    <select id="scatter-target-select" onchange="updateScatter()">
    </select>
    <label>Atlas:</label>
    <select id="scatter-atlas-select" onchange="updateScatter()">
    </select>
  </div>
  <div id="scatter-chart" style="height:500px;"></div>
  <div class="caption"><strong>Figure 6.</strong> Donor-level expression vs CytoSig predicted activity. Select target and atlas from dropdowns.</div>
</div>

<!-- Item 12: Interactive heatmap with tabs -->
<h3>4.7 Biologically Important Targets Heatmap</h3>
<div class="plotly-container">
  <div class="tab-bar" id="heatmap-tabs">
    <button class="tab-btn cytosig active" onclick="switchHeatmapTab('cytosig')">CytoSig</button>
    <button class="tab-btn secact" onclick="switchHeatmapTab('secact')">SecAct</button>
  </div>
  <div id="heatmap-chart" style="min-height:500px;"></div>
  <div class="caption"><strong>Figure 7.</strong> Spearman &rho; heatmap for biologically important targets across all atlases. Switch between signature types. Hover over cells for details.</div>
</div>

<div class="callout">
<p><strong>How each correlation value is computed:</strong> For each (target, atlas) cell, we compute Spearman rank correlation between <em>predicted cytokine activity</em> (ridge regression z-score) and <em>target gene expression</em> across all donor-level pseudobulk samples. Specifically:</p>
<ol>
  <li><strong>Pseudobulk aggregation:</strong> For each atlas, gene expression is aggregated to the donor level (one profile per donor or donor &times; cell type).</li>
  <li><strong>Activity inference:</strong> Ridge regression (<code>secactpy.ridge</code>, &lambda;=5&times;10<sup>5</sup>) is applied using the signature matrix (CytoSig: 4,881 genes &times; 43 cytokines; SecAct: 7,919 genes &times; 1,249 targets) to predict activity z-scores for each pseudobulk sample.</li>
  <li><strong>Correlation:</strong> Spearman &rho; is computed between the predicted activity z-score and the original expression of the target gene across all donor-level samples within that atlas. A positive &rho; means higher predicted activity tracks with higher target gene expression.</li>
</ol>
<p>GTEx/TCGA use donor-only pseudobulk; CIMA uses donor-only; Inflammation uses donor-only; scAtlas uses donor &times; organ.</p>
</div>

<!-- Item 13: Interactive bulk validation -->
<h3>4.8 Bulk RNA-seq Validation (GTEx &amp; TCGA)</h3>
<div class="plotly-container">
  <div class="controls">
    <label>Dataset:</label>
    <select id="bulk-dataset-select" onchange="updateBulk()">
      <option value="GTEx">GTEx</option>
      <option value="TCGA">TCGA</option>
    </select>
    <label>Signature:</label>
    <select id="bulk-sig-select" onchange="updateBulk()">
      <option value="cytosig">CytoSig</option>
      <option value="secact">SecAct (matched + top additional)</option>
    </select>
  </div>
  <div id="bulk-chart" style="height:500px;"></div>
  <div class="caption"><strong>Figure 8.</strong> Bulk RNA-seq validation: targets ranked by Spearman &rho;. Select dataset and signature type from dropdowns.</div>
</div>

<hr>

<!-- SECTION 5 -->
<h2 id="sec5">5. CytoSig vs LinCytoSig vs SecAct Comparison</h2>

<h3>5.1 Method Overview</h3>

<table>
  <tr>
    <th>Method</th><th>Targets</th><th>Genes</th><th>Specificity</th><th>Selection</th>
  </tr>
  <tr>
    <td><span class="badge blue">CytoSig</span></td>
    <td>43 cytokines</td><td>4,881 curated</td><td>Cell-type agnostic</td><td>&mdash;</td>
  </tr>
  <tr>
    <td><span class="badge amber">LinCytoSig (orig)</span></td>
    <td>178 (45 CT &times; cytokines)</td><td>All ~20K</td><td>Cell-type specific</td><td>Matched cell type</td>
  </tr>
  <tr>
    <td><span style="background:#F59E0B;color:#fff;padding:1px 8px;border-radius:3px;font-size:0.85em">LinCytoSig (gene-filtered)</span></td>
    <td>178</td><td>4,881 (CytoSig overlap)</td><td>Cell-type specific</td><td>Matched cell type</td>
  </tr>
  <tr>
    <td><span style="background:#B45309;color:#fff;padding:1px 8px;border-radius:3px;font-size:0.85em">LinCytoSig Best (orig)</span></td>
    <td>43 (1 per cytokine)</td><td>All ~20K</td><td>Best CT per cytokine</td><td>Max bulk (GTEx+TCGA) &rho;</td>
  </tr>
  <tr>
    <td><span style="background:#92400E;color:#fff;padding:1px 8px;border-radius:3px;font-size:0.85em">LinCytoSig Best (gene-filtered)</span></td>
    <td>43 (1 per cytokine)</td><td>4,881 (CytoSig overlap)</td><td>Best CT per cytokine</td><td>Max bulk &rho; (filtered)</td>
  </tr>
  <tr>
    <td><span class="badge green">SecAct</span></td>
    <td>1,249 secreted proteins</td><td>Spatial Moran&rsquo;s I</td><td>Cell-type agnostic</td><td>&mdash;</td>
  </tr>
</table>
<p style="margin-top:0.5em;font-size:0.9em;color:#6B7280;">
  <strong>Gene filter:</strong> LinCytoSig signatures restricted from ~20K to CytoSig&rsquo;s 4,881 curated genes.
  <strong>Best selection:</strong> For each cytokine, test all cell-type-specific LinCytoSig signatures and select the one with highest GTEx+TCGA bulk RNA-seq correlation as the representative.
  See <a href="LINCYTOSIG_METHODOLOGY.html">LinCytoSig Methodology</a> for details.
</p>

<div class="plotly-container">
  <div class="controls">
    <label>View:</label>
    <select id="method-boxplot-view" onchange="updateMethodBoxplot()">
      <option value="all">All Atlases</option>
    </select>
  </div>
  <div id="method-boxplot-chart" style="height:550px;"></div>
  <div class="caption"><strong>Figure 9.</strong> Six-way signature method comparison at matched (cell type, cytokine) pair level across all 6 atlases. All 6 methods are evaluated on the <em>same set</em> of matched pairs per atlas (identical n). Use dropdown to view individual atlas boxplots. For LinCytoSig construction, see <a href="LINCYTOSIG_METHODOLOGY.html">LinCytoSig Methodology</a>.</div>
</div>

<div class="callout">
<p><strong>Six methods compared on identical matched pairs across all 6 atlases:</strong></p>
<ol>
  <li><strong><span style="color:#2563EB">CytoSig</span></strong> &mdash; 43 cytokines, 4,881 curated genes, cell-type agnostic</li>
  <li><strong><span style="color:#D97706">LinCytoSig (orig)</span></strong> &mdash; cell-type-matched signatures, all ~20K genes</li>
  <li><strong><span style="color:#F59E0B">LinCytoSig (gene-filtered)</span></strong> &mdash; cell-type-matched signatures, restricted to CytoSig&rsquo;s 4,881 genes</li>
  <li><strong><span style="color:#B45309">LinCytoSig Best (orig)</span></strong> &mdash; best cell-type signature per cytokine (selected by GTEx+TCGA bulk &rho;), all ~20K genes</li>
  <li><strong><span style="color:#92400E">LinCytoSig Best (gene-filtered)</span></strong> &mdash; best cell-type signature per cytokine (selected by bulk &rho; on filtered genes), restricted to 4,881 genes</li>
  <li><strong><span style="color:#059669">SecAct</span></strong> &mdash; 1,249 secreted proteins (Moran&rsquo;s I), subset matching CytoSig targets</li>
</ol>
<p><strong>Key findings:</strong> SecAct achieves the highest median &rho; across all 6 atlases.
CytoSig consistently outperforms the cell-type-matched LinCytoSig (orig), largely because LinCytoSig signatures use all ~20K genes (amplifying noise) versus CytoSig&rsquo;s curated 4,881.
Gene filtering improves LinCytoSig in 5 of 6 atlases (CIMA +102%, Inflam Ext +114%), confirming noise reduction from restricting the gene space.
The &ldquo;best&rdquo; selection strategy (one representative cell-type signature per cytokine) further improves performance, with &ldquo;Best (gene-filtered)&rdquo; approaching or exceeding CytoSig in Inflammation atlases.
Consistent ranking: SecAct &gt; CytoSig &gt; LinCytoSig Best (filt) &ge; LinCytoSig Best (orig) &gt; LinCytoSig (filt) &gt; LinCytoSig (orig).</p>
</div>

<h3>5.2 When Does LinCytoSig Outperform CytoSig?</h3>

<div class="plotly-container">
  <div class="controls">
    <label>Atlas:</label>
    <select id="linvscyto-atlas-select" onchange="updateLinVsCyto()">
    </select>
  </div>
  <div id="linvscyto-chart" style="height:550px;"></div>
  <div class="caption"><strong>Figure 10.</strong> Matched target correlation comparison (celltype-level). Points above diagonal = LinCytoSig outperforms CytoSig. Select atlas from dropdown.</div>
</div>

<p><strong>LinCytoSig wins:</strong> Basophil (<span class="win">+0.21</span>), NK Cell (<span class="win">+0.19</span>), Dendritic Cell (<span class="win">+0.18</span>)</p>
<p><strong>CytoSig wins:</strong> Lymphatic Endothelial (<span class="lose">&minus;0.73</span>), Adipocyte (<span class="lose">&minus;0.44</span>), Osteocyte (<span class="lose">&minus;0.40</span>), PBMC (<span class="lose">&minus;0.38</span>)</p>

<p><strong>Recommendation:</strong> Use LinCytoSig for <strong>cell-type-resolved</strong> questions and CytoSig for <strong>donor-level</strong> questions.</p>

<h3>5.3 SecAct: Breadth Over Depth</h3>

<ul>
  <li><strong>Highest median &rho;</strong> in organ-level analyses (scAtlas normal: 0.307, cancer: 0.363)</li>
  <li><strong>Highest median &rho;</strong> in bulk RNA-seq (GTEx: 0.395, TCGA: 0.415)</li>
  <li><strong>97.1% positive correlation</strong> in TCGA</li>
</ul>

<div class="figure">
  <img src="figures/fig11_secact_novel_signatures.png" alt="Figure: SecAct Novel">
  <div class="caption"><strong>Figure 12.</strong> Top 30 novel SecAct targets with consistent positive correlation. Distribution of all SecAct mean &rho; values.</div>
</div>

<h3>5.4 LinCytoSig Specificity Deep Dive</h3>
<div class="figure">
  <img src="figures/fig10_lincytosig_advantage_by_celltype.png" alt="Figure: LinCytoSig Advantage">
  <div class="caption"><strong>Figure 11.</strong> LinCytoSig advantage by cell type. Basophil, NK Cell, and Dendritic Cell benefit most.</div>
</div>
<div class="figure">
  <img src="figures/fig13_lincytosig_specificity.png" alt="Figure: LinCytoSig Specificity">
  <div class="caption"><strong>Figure 13.</strong> Top 20 cases where LinCytoSig outperforms vs underperforms CytoSig.</div>
</div>

<hr>

<!-- SECTION 6 -->
<h2 id="sec6">6. Key Takeaways for Scientific Discovery</h2>

<h3>6.1 What CytoAtlas Enables</h3>
<ol>
  <li><strong>Quantitative cytokine activity per cell type per disease</strong></li>
  <li><strong>Cross-disease comparison</strong> &mdash; same 44 CytoSig signatures across 20 diseases, 35 organs, 15 cancer types</li>
  <li><strong>Independent perturbation comparison</strong> &mdash; parse_10M provides 90 cytokine perturbations &times; 12 donors &times; 18 cell types for independent comparison with CytoSig predictions</li>
  <li><strong>Drug-cytokine interaction</strong> &mdash; Tahoe-100M maps 95 drugs &times; 50 cancer cell lines</li>
  <li><strong>Spatial context</strong> &mdash; SpatialCorpus-110M maps cytokine activity to spatial neighborhoods</li>
</ol>

<h3>6.2 Limitations</h3>
<ol>
  <li><strong>Linear model:</strong> Cannot capture non-linear cytokine interactions</li>
  <li><strong>Transcriptomics-only:</strong> Post-translational regulation invisible</li>
  <li><strong>Signature matrix bias:</strong> Underrepresented cell types have weaker signatures</li>
  <li><strong>Validation metric:</strong> Expression-activity correlation underestimates true accuracy</li>
</ol>

<h3>6.3 Future Directions</h3>
<ol>
  <li>scGPT cohort integration (~35M cells)</li>
  <li>cellxgene Census integration</li>
  <li>Drug response prediction models</li>
  <li>Spatial cytokine niches</li>
  <li>Treatment response biomarkers</li>
</ol>

<hr>

<!-- SECTION 7 -->
<h2 id="sec7">7. Appendix: Technical Specifications</h2>

<h3>A. Computational Infrastructure</h3>
<ul>
  <li><strong>GPU:</strong> NVIDIA A100 80GB (SLURM gpu partition)</li>
  <li><strong>Memory:</strong> 256&ndash;512GB host RAM per node</li>
  <li><strong>Pipeline:</strong> 24 Python scripts, 18 pipeline subpackages (~18.7K lines)</li>
  <li><strong>API:</strong> 262 REST endpoints across 17 routers</li>
  <li><strong>Frontend:</strong> 12 pages, 122 source files, 11.4K LOC</li>
</ul>

<h3>B. Statistical Methods</h3>
<ul>
  <li><strong>Activity inference:</strong> Ridge regression (&lambda;=5&times;10<sup>5</sup>, z-score normalization, permutation-based significance)</li>
  <li><strong>Correlation:</strong> Spearman rank correlation</li>
  <li><strong>Multiple testing:</strong> Benjamini-Hochberg FDR (q &lt; 0.05)</li>
  <li><strong>Bootstrap:</strong> 100&ndash;1000 resampling iterations</li>
  <li><strong>Differential:</strong> Wilcoxon rank-sum test with effect size</li>
</ul>

</div><!-- content -->
</div><!-- container -->

<!-- Lightbox for static images -->
<div class="lightbox-overlay" id="lightbox">
  <button class="lightbox-close" onclick="closeLightbox()">&times;</button>
  <img src="" alt="">
</div>

<script>
// ═══════════════════════════════════════════════════════════════════════════
// DATA
// ═══════════════════════════════════════════════════════════════════════════
var DATA = {data_json};

var PLOTLY_CONFIG = {{responsive: true, displayModeBar: true, modeBarButtonsToRemove: ['lasso2d','select2d']}};
var SIG_NAMES = {{'cytosig':'CytoSig','lincytosig':'LinCytoSig','secact':'SecAct'}};

// ═══════════════════════════════════════════════════════════════════════════
// 4.1 SUMMARY TABLE (embedded HTML)
// ═══════════════════════════════════════════════════════════════════════════
(function() {{
  var rows = DATA.summary;
  var sigBg = {{'CYTOSIG':'#DBEAFE','LINCYTOSIG':'#FEF3C7','SECACT':'#D1FAE5'}};
  var html = '<table><tr>';
  var cols = ['Atlas','Signature','N Targets','Median \\u03c1','Mean \\u03c1','Std \\u03c1','Min \\u03c1','Max \\u03c1','% Significant','% Positive'];
  cols.forEach(function(c) {{ html += '<th>' + c + '</th>'; }});
  html += '</tr>';
  rows.forEach(function(r) {{
    var bg = sigBg[r.signature] || '';
    html += '<tr>';
    html += '<td style="background:'+bg+'">' + r.atlas + '</td>';
    html += '<td style="background:'+bg+';font-weight:600">' + r.signature + '</td>';
    html += '<td style="background:'+bg+'">' + r.n_targets + '</td>';
    html += '<td style="background:'+bg+';font-weight:600">' + r.median_rho + '</td>';
    html += '<td style="background:'+bg+'">' + r.mean_rho + '</td>';
    html += '<td style="background:'+bg+'">' + r.std_rho + '</td>';
    html += '<td style="background:'+bg+'">' + r.min_rho + '</td>';
    html += '<td style="background:'+bg+'">' + r.max_rho + '</td>';
    html += '<td style="background:'+bg+'">' + r.pct_sig + '%</td>';
    html += '<td style="background:'+bg+'">' + r.pct_pos + '%</td>';
    html += '</tr>';
  }});
  html += '</table>';
  document.getElementById('summary-table-container').innerHTML = html;
}})();

// ═══════════════════════════════════════════════════════════════════════════
// 4.2 BOXPLOT (interactive Plotly — CytoSig, SecAct, SecAct matched)
// ═══════════════════════════════════════════════════════════════════════════
(function() {{
  var bd = DATA.boxplot;
  var atlases = DATA.atlasLabels;
  var sigConfigs = [
    {{key:'cytosig', name:'CytoSig (n=43)', color:'#2563EB'}},
    {{key:'secact', name:'SecAct (n=1,249)', color:'#059669'}},
    {{key:'secact_matched', name:'SecAct (CytoSig-matched, n=22)', color:'#10B981', dash:'dot'}},
  ];
  var traces = [];
  sigConfigs.forEach(function(cfg) {{
    var y = [], x = [];
    atlases.forEach(function(a) {{
      if (bd[a] && bd[a][cfg.key]) {{
        bd[a][cfg.key].forEach(function(v) {{
          y.push(v); x.push(a);
        }});
      }}
    }});
    traces.push({{
      type: 'box', y: y, x: x, name: cfg.name,
      marker: {{color: cfg.color}},
      boxpoints: false, line: {{width: 1.5}},
    }});
  }});
  Plotly.newPlot('boxplot-chart', traces, {{
    title: 'Cross-Sample Validation: Spearman Correlation Distributions (Donor-Level)',
    yaxis: {{title: 'Spearman \\u03c1', zeroline: true, zerolinecolor: '#ccc'}},
    boxmode: 'group', legend: {{orientation:'h', y:1.15, x:0.5, xanchor:'center'}},
    margin: {{t:100, b:80}},
    annotations: [{{
      text: 'CytoSig: 43 cytokines | SecAct: 1,249 secreted proteins | SecAct matched: 22 targets shared with CytoSig',
      xref:'paper', yref:'paper', x:0.5, y:-0.18, showarrow:false,
      font:{{size:11, color:'#6B7280'}}, xanchor:'center',
    }}],
  }}, PLOTLY_CONFIG);
}})();

// ═══════════════════════════════════════════════════════════════════════════
// 4.3 GOOD/BAD CORRELATIONS (interactive with signature + atlas dropdown)
// ═══════════════════════════════════════════════════════════════════════════
(function() {{
  var gb = DATA.goodBad;
  var atlasSel = document.getElementById('goodbad-atlas-select');
  var sigSel = document.getElementById('goodbad-sig-select');
  // Populate atlas dropdown from first sig type
  var firstSig = Object.keys(gb)[0];
  Object.keys(gb[firstSig]).forEach(function(a) {{
    var opt = document.createElement('option');
    opt.value = a; opt.textContent = a;
    atlasSel.appendChild(opt);
  }});
  window.updateGoodBad = function() {{
    var sig = sigSel.value;
    var atlas = atlasSel.value;
    var sigData = gb[sig];
    if (!sigData || !sigData[atlas]) return;
    var d = sigData[atlas];
    // Top 15 (reverse for horizontal bar layout)
    var topTargets = d.top.map(function(r){{return r.target;}}).reverse();
    var topRhos = d.top.map(function(r){{return r.rho;}}).reverse();
    // Bottom 15
    var botTargets = d.bottom.map(function(r){{return r.target;}}).reverse();
    var botRhos = d.bottom.map(function(r){{return r.rho;}}).reverse();
    var sigColor = DATA.sigColors[sig] || '#6B7280';
    var traces = [
      {{type:'bar', y:topTargets, x:topRhos, orientation:'h',
        marker:{{color:'#DC2626'}},
        xaxis:'x', yaxis:'y', text:topRhos.map(function(r){{return r.toFixed(3);}}), textposition:'outside',
        hovertemplate:'<b>%{{y}}</b><br>\\u03c1 = %{{x:.4f}}<extra></extra>'}},
      {{type:'bar', y:botTargets, x:botRhos, orientation:'h',
        marker:{{color:'#2563EB'}},
        xaxis:'x2', yaxis:'y2', text:botRhos.map(function(r){{return r.toFixed(3);}}), textposition:'outside',
        hovertemplate:'<b>%{{y}}</b><br>\\u03c1 = %{{x:.4f}}<extra></extra>'}},
    ];
    Plotly.newPlot('goodbad-chart', traces, {{
      title: atlas + ' — ' + SIG_NAMES[sig] + ': Best & Worst Correlated Targets',
      grid: {{rows:1, columns:2, pattern:'independent'}},
      xaxis: {{title:'Spearman \\u03c1', domain:[0, 0.45]}},
      yaxis: {{automargin: true}},
      xaxis2: {{title:'Spearman \\u03c1', domain:[0.55, 1]}},
      yaxis2: {{automargin: true}},
      annotations: [
        {{text:'<b>Top 15 (Best)</b>', xref:'paper', yref:'paper', x:0.22, y:1.06, showarrow:false, font:{{color:'#DC2626',size:14}}}},
        {{text:'<b>Bottom 15 (Worst)</b>', xref:'paper', yref:'paper', x:0.78, y:1.06, showarrow:false, font:{{color:'#2563EB',size:14}}}},
      ],
      showlegend: false, margin: {{t:80, l:120, r:20}},
    }}, PLOTLY_CONFIG);
  }};
  updateGoodBad();
}})();

// ═══════════════════════════════════════════════════════════════════════════
// 4.4 CROSS-ATLAS CONSISTENCY (interactive Plotly line chart)
// ═══════════════════════════════════════════════════════════════════════════
(function() {{
  var cd = DATA.consistency;
  var atlases = DATA.atlasLabels;
  var traces = [];
  Object.keys(cd).forEach(function(target) {{
    var d = cd[target];
    traces.push({{
      type: 'scatter', mode: 'lines+markers',
      x: atlases, y: d.rhos,
      name: target, line: {{color: d.color, width: 2}},
      marker: {{size: 8}},
      hovertemplate: '<b>' + target + '</b><br>%{{x}}: \\u03c1=%{{y:.3f}}<extra></extra>',
    }});
  }});
  Plotly.newPlot('consistency-chart', traces, {{
    title: 'CytoSig: Cross-Atlas Consistency of Key Cytokine Targets (Donor-Level)',
    yaxis: {{title: 'Spearman \\u03c1', zeroline: true, zerolinecolor: '#ccc'}},
    legend: {{font: {{size: 11}}}},
    hovermode: 'closest',
    margin: {{t:60, b:80, r:20}},
  }}, PLOTLY_CONFIG);
}})();

// ═══════════════════════════════════════════════════════════════════════════
// 4.5 AGGREGATION LEVELS (CytoSig + SecAct + SecAct matched, side-by-side per atlas)
// ═══════════════════════════════════════════════════════════════════════════
(function() {{
  var ld = DATA.levels;
  var atlasNames = Object.keys(ld);
  var atlasSel = document.getElementById('levels-atlas-select');
  atlasNames.forEach(function(a) {{
    var opt = document.createElement('option');
    opt.value = a; opt.textContent = a;
    atlasSel.appendChild(opt);
  }});
  var sigConfigs = [
    {{key:'cytosig', name:'CytoSig (n=43)', color:'#2563EB'}},
    {{key:'secact', name:'SecAct (n=1,249)', color:'#059669'}},
    {{key:'secact_matched', name:'SecAct (CytoSig-matched, n=22)', color:'#10B981'}},
  ];
  window.renderLevels = function() {{
    var atlasName = atlasSel.value;
    var atlasData = ld[atlasName];
    if (!atlasData) return;
    // Collect all levels across all sigs
    var allLevels = [];
    sigConfigs.forEach(function(cfg) {{
      if (atlasData[cfg.key]) {{
        Object.keys(atlasData[cfg.key]).forEach(function(lv) {{
          if (allLevels.indexOf(lv) === -1) allLevels.push(lv);
        }});
      }}
    }});
    var traces = [];
    sigConfigs.forEach(function(cfg) {{
      var sigData = atlasData[cfg.key];
      if (!sigData) return;
      var y = [], x = [];
      allLevels.forEach(function(level) {{
        if (sigData[level]) {{
          sigData[level].rhos.forEach(function(v) {{
            y.push(v);
            x.push(level);
          }});
        }}
      }});
      traces.push({{
        type: 'box', y: y, x: x,
        name: cfg.name,
        marker: {{color: cfg.color}},
        boxpoints: false, line: {{width: 1.5}},
      }});
    }});
    Plotly.newPlot('levels-chart', traces, {{
      title: atlasName + ' — Effect of Aggregation Level',
      yaxis: {{title: 'Spearman \\u03c1', zeroline: true, zerolinecolor: '#ccc'}},
      boxmode: 'group',
      legend: {{orientation:'h', y:1.12, x:0.5, xanchor:'center'}},
      margin: {{t:90, b:100}},
      xaxis: {{tickangle: -20}},
    }}, PLOTLY_CONFIG);
  }};
  renderLevels();
}})();

// ═══════════════════════════════════════════════════════════════════════════
// 4.6 SCATTER PLOTS (interactive with dropdown)
// ═══════════════════════════════════════════════════════════════════════════
(function() {{
  var sd = DATA.scatter;
  var targetSel = document.getElementById('scatter-target-select');
  var atlasSel = document.getElementById('scatter-atlas-select');
  // Populate dropdowns
  var allTargets = new Set();
  var allAtlases = Object.keys(sd);
  allAtlases.forEach(function(a) {{
    Object.keys(sd[a]).forEach(function(t) {{ allTargets.add(t); }});
    var opt = document.createElement('option');
    opt.value = a; opt.textContent = a;
    atlasSel.appendChild(opt);
  }});
  ['IFNG','IL1B','TNFA','TGFB1','IL6','IL10','VEGFA','CD40L','TRAIL','HGF'].forEach(function(t) {{
    if (allTargets.has(t)) {{
      var opt = document.createElement('option');
      opt.value = t; opt.textContent = t;
      targetSel.appendChild(opt);
    }}
  }});
  window.updateScatter = function() {{
    var target = targetSel.value;
    var atlas = atlasSel.value;
    if (!sd[atlas] || !sd[atlas][target]) {{
      Plotly.newPlot('scatter-chart', [], {{title: 'No data for ' + target + ' in ' + atlas, margin:{{t:40}}}}, PLOTLY_CONFIG);
      return;
    }}
    var d = sd[atlas][target];
    var color = d.rho > 0.2 ? '#059669' : (d.rho < 0 ? '#DC2626' : '#6B7280');
    var traces = [{{
      type: 'scatter', mode: 'markers',
      x: d.x, y: d.y,
      marker: {{color: color, size: 6, opacity: 0.5}},
      hovertemplate: 'Expr: %{{x:.2f}}<br>Activity: %{{y:.2f}}<extra></extra>',
    }}];
    // Regression line
    if (d.x.length > 2) {{
      var sorted = d.x.map(function(v,i) {{ return [v, d.y[i]]; }}).sort(function(a,b) {{ return a[0]-b[0]; }});
      var n = sorted.length, sx=0, sy=0, sxx=0, sxy=0;
      sorted.forEach(function(p) {{ sx+=p[0]; sy+=p[1]; sxx+=p[0]*p[0]; sxy+=p[0]*p[1]; }});
      var slope = (n*sxy - sx*sy) / (n*sxx - sx*sx);
      var intercept = (sy - slope*sx) / n;
      var x0 = sorted[0][0], x1 = sorted[sorted.length-1][0];
      traces.push({{type:'scatter', mode:'lines', x:[x0,x1], y:[slope*x0+intercept, slope*x1+intercept],
        line:{{color:'black', dash:'dash', width:2}}, showlegend:false}});
    }}
    var pstr = d.pval < 0.001 ? d.pval.toExponential(1) : d.pval.toFixed(4);
    Plotly.newPlot('scatter-chart', traces, {{
      title: target + ' — ' + atlas + '  (\\u03c1=' + d.rho.toFixed(3) + ', p=' + pstr + ', n=' + d.n + ')',
      xaxis: {{title: 'Mean Expression'}}, yaxis: {{title: 'Predicted Activity'}},
      showlegend: false, margin: {{t:60}},
    }}, PLOTLY_CONFIG);
  }};
  updateScatter();
}})();

// ═══════════════════════════════════════════════════════════════════════════
// 4.7 HEATMAP (tabbed Plotly)
// ═══════════════════════════════════════════════════════════════════════════
var currentHeatmapTab = 'cytosig';
window.switchHeatmapTab = function(sig) {{
  currentHeatmapTab = sig;
  document.querySelectorAll('#heatmap-tabs .tab-btn').forEach(function(b){{b.classList.remove('active');}});
  document.querySelector('#heatmap-tabs .tab-btn.' + sig).classList.add('active');
  renderHeatmap(sig);
}};

function renderHeatmap(sig) {{
  var hd = DATA.heatmap[sig];
  if (!hd || hd.targets.length === 0) {{
    Plotly.newPlot('heatmap-chart', [], {{title: 'No data for ' + SIG_NAMES[sig]}}, PLOTLY_CONFIG);
    return;
  }}
  var h = Math.max(500, hd.targets.length * 18 + 100);
  document.getElementById('heatmap-chart').style.height = h + 'px';
  // Replace nulls with NaN for plotly
  var z = hd.matrix.map(function(row) {{
    return row.map(function(v) {{ return v === null ? NaN : v; }});
  }});
  var hovertext = hd.targets.map(function(t, i) {{
    return hd.atlases.map(function(a, j) {{
      var v = hd.matrix[i][j];
      return t + '<br>' + a + '<br>\\u03c1 = ' + (v !== null ? v.toFixed(3) : 'N/A');
    }});
  }});
  // Red-white-blue: high=red, middle=white, low=blue
  var rwb = [[0,'#2563EB'],[0.25,'#93C5FD'],[0.5,'#FFFFFF'],[0.75,'#FCA5A5'],[1,'#DC2626']];
  Plotly.newPlot('heatmap-chart', [{{
    type: 'heatmap', z: z, x: hd.atlases, y: hd.targets,
    colorscale: rwb, zmin: -0.5, zmax: 0.8,
    hovertext: hovertext, hoverinfo: 'text',
    colorbar: {{title: 'Spearman \\u03c1', len: 0.5}},
  }}], {{
    title: SIG_NAMES[sig] + ' — Biologically Important Targets',
    xaxis: {{side: 'bottom', tickangle: -30}},
    yaxis: {{autorange: 'reversed', tickfont: {{size: 9}}}},
    margin: {{t:50, l:160, r:60, b:100}},
  }}, PLOTLY_CONFIG);
}}
renderHeatmap('cytosig');

// ═══════════════════════════════════════════════════════════════════════════
// 4.8 BULK VALIDATION (CytoSig + SecAct, always bar chart)
// ═══════════════════════════════════════════════════════════════════════════
window.updateBulk = function() {{
  var dataset = document.getElementById('bulk-dataset-select').value;
  var sig = document.getElementById('bulk-sig-select').value;
  var bd = DATA.bulk;
  if (!bd[dataset] || !bd[dataset][sig]) {{
    Plotly.newPlot('bulk-chart', [], {{title: 'No data', margin:{{t:40}}}}, PLOTLY_CONFIG);
    return;
  }}
  var d = bd[dataset][sig];
  var n = d.targets.length;
  var colors;
  if (sig === 'secact' && d.matched) {{
    // Color matched targets with SecAct green, additional targets with gray
    colors = d.matched.map(function(m, i) {{
      return m ? '#059669' : '#94A3B8';
    }});
  }} else {{
    colors = d.rhos.map(function(r) {{
      return r > 0.2 ? DATA.sigColors[sig] : (r < -0.1 ? '#DC2626' : '#6B7280');
    }});
  }}
  var totalNote = (sig === 'secact' && d.total_secact) ?
    '  [showing ' + n + ' of ' + d.total_secact + ' total SecAct targets: matched CytoSig targets + top additional]' : '';
  var traces = [{{
    type: 'bar', x: d.targets, y: d.rhos,
    marker: {{color: colors}},
    hovertemplate: '<b>%{{x}}</b><br>\\u03c1 = %{{y:.3f}}<extra></extra>',
  }}];
  var title = dataset + ' — ' + SIG_NAMES[sig] + '  (n=' + n + ', median \\u03c1=' + d.median + ')';
  var layout = {{
    title: title,
    xaxis: {{title: 'Target (ranked by \\u03c1)', tickangle: -90, tickfont: {{size: 7}}}},
    yaxis: {{title: 'Spearman \\u03c1'}},
    showlegend: false, margin: {{t:60, b:120}},
  }};
  if (sig === 'secact' && d.matched) {{
    layout.annotations = [{{
      text: 'Green = matched CytoSig target, Gray = additional SecAct target',
      xref:'paper', yref:'paper', x:0.5, y:1.06, showarrow:false,
      font:{{size:11, color:'#6B7280'}},
    }}];
  }}
  Plotly.newPlot('bulk-chart', traces, layout, PLOTLY_CONFIG);
}};
updateBulk();

// ═══════════════════════════════════════════════════════════════════════════
// 5.1 METHOD COMPARISON BOXPLOT (6-way with dropdown for per-atlas view)
// ═══════════════════════════════════════════════════════════════════════════
(function() {{
  var mb = DATA.methodBoxplot;
  if (!mb) return;
  var allAtlases = Object.keys(mb);
  var sigConfigs = [
    {{key:'cytosig',           name:'CytoSig',                     color:'#2563EB'}},
    {{key:'lincyto_orig',      name:'LinCytoSig (orig)',            color:'#D97706'}},
    {{key:'lincyto_filt',      name:'LinCytoSig (gene-filtered)',   color:'#F59E0B'}},
    {{key:'lincyto_best_orig', name:'LinCytoSig (best-bulk)',       color:'#B45309'}},
    {{key:'lincyto_best_filt', name:'LinCytoSig (best-bulk+filt)',  color:'#92400E'}},
    {{key:'secact',            name:'SecAct',                       color:'#059669'}},
  ];

  // Populate dropdown
  var sel = document.getElementById('method-boxplot-view');
  allAtlases.forEach(function(a) {{
    var opt = document.createElement('option');
    opt.value = a; opt.textContent = a;
    sel.appendChild(opt);
  }});

  window.updateMethodBoxplot = function() {{
    var view = sel.value;
    var atlasesToShow = (view === 'all') ? allAtlases : [view];
    var isSingle = (view !== 'all');
    var traces = [];

    sigConfigs.forEach(function(cfg) {{
      var y = [], x = [];
      atlasesToShow.forEach(function(a) {{
        if (mb[a] && mb[a][cfg.key] && mb[a][cfg.key].length > 0) {{
          mb[a][cfg.key].forEach(function(v) {{
            y.push(v);
            x.push(isSingle ? cfg.name : a);
          }});
        }}
      }});
      if (y.length > 0) {{
        var n = mb[atlasesToShow[0]][cfg.key] ? mb[atlasesToShow[0]][cfg.key].length : 0;
        traces.push({{
          type: 'box', y: y, x: x,
          name: cfg.name + (isSingle ? '' : ' (n=' + n + ')'),
          marker: {{color: cfg.color}},
          boxpoints: isSingle ? 'all' : false,
          jitter: 0.3, pointpos: 0,
          line: {{width: 1.5}},
          showlegend: !isSingle,
        }});
      }}
    }});

    var title = isSingle
      ? view + ' \u2014 6-Way Method Comparison (n=' + (mb[view].cytosig ? mb[view].cytosig.length : 0) + ' matched pairs)'
      : '6-Way Signature Method Comparison (Expression\u2013Activity Correlation)';

    Plotly.newPlot('method-boxplot-chart', traces, {{
      title: title,
      yaxis: {{title: 'Spearman \\u03c1', zeroline: true, zerolinecolor: '#ccc'}},
      boxmode: 'group',
      legend: {{orientation:'h', y:1.25, x:0.5, xanchor:'center', font:{{size:9}}}},
      margin: {{t:isSingle ? 80 : 140, b:isSingle ? 120 : 80}},
    }}, PLOTLY_CONFIG);
  }};
  updateMethodBoxplot();
}})();

// ═══════════════════════════════════════════════════════════════════════════
// 5.2 LINCYTOSIG VS CYTOSIG SCATTER (matched targets, per atlas)
// ═══════════════════════════════════════════════════════════════════════════
(function() {{
  var lvc = DATA.linVsCyto;
  if (!lvc) return;
  var atlasNames = Object.keys(lvc);
  var sel = document.getElementById('linvscyto-atlas-select');
  atlasNames.forEach(function(a) {{
    var opt = document.createElement('option');
    opt.value = a; opt.textContent = a;
    sel.appendChild(opt);
  }});
  window.updateLinVsCyto = function() {{
    var atlas = sel.value;
    var d = lvc[atlas];
    if (!d || !d.points || d.points.length === 0) {{
      Plotly.newPlot('linvscyto-chart', [], {{title: 'No matched data for ' + atlas}}, PLOTLY_CONFIG);
      return;
    }}
    var pts = d.points;
    var xv = pts.map(function(p) {{ return p.cytosig; }});
    var yv = pts.map(function(p) {{ return p.lincytosig; }});
    var texts = pts.map(function(p) {{ return p.target; }});
    // Color by who wins
    var colors = pts.map(function(p) {{
      if (p.lincytosig > p.cytosig) return '#D97706';  // amber = LinCytoSig wins
      if (p.cytosig > p.lincytosig) return '#2563EB';   // blue = CytoSig wins
      return '#6B7280';
    }});
    // Diagonal line
    var mn = Math.min(Math.min.apply(null, xv), Math.min.apply(null, yv)) - 0.1;
    var mx = Math.max(Math.max.apply(null, xv), Math.max.apply(null, yv)) + 0.1;
    var traces = [
      {{type:'scatter', mode:'lines', x:[mn,mx], y:[mn,mx],
        line:{{color:'#D1D5DB', dash:'dash', width:1}}, showlegend:false, hoverinfo:'skip'}},
      {{type:'scatter', mode:'markers', x:xv, y:yv, text:texts,
        marker:{{color:colors, size:8, opacity:0.7, line:{{width:0.5, color:'white'}}}},
        hovertemplate:'<b>%{{text}}</b><br>CytoSig \\u03c1: %{{x:.3f}}<br>LinCytoSig \\u03c1: %{{y:.3f}}<extra></extra>',
        showlegend:false}},
    ];
    var title = atlas + '  (LinCytoSig wins: ' + d.n_lin_win +
      ', CytoSig wins: ' + d.n_cyto_win + ', Tie: ' + d.n_tie + ')';
    Plotly.newPlot('linvscyto-chart', traces, {{
      title: title,
      xaxis: {{title: 'CytoSig Spearman \\u03c1', zeroline:true, zerolinecolor:'#ccc'}},
      yaxis: {{title: 'LinCytoSig Spearman \\u03c1', zeroline:true, zerolinecolor:'#ccc'}},
      margin: {{t:60, b:60}},
      annotations: [
        {{text:'Above diagonal = LinCytoSig better', xref:'paper', yref:'paper',
          x:0.02, y:0.98, showarrow:false, font:{{size:11, color:'#D97706'}}}},
        {{text:'Below diagonal = CytoSig better', xref:'paper', yref:'paper',
          x:0.98, y:0.02, showarrow:false, font:{{size:11, color:'#2563EB'}}, xanchor:'right'}},
      ],
    }}, PLOTLY_CONFIG);
  }};
  updateLinVsCyto();
}})();

// ═══════════════════════════════════════════════════════════════════════════
// LIGHTBOX for static images
// ═══════════════════════════════════════════════════════════════════════════
document.querySelectorAll('.figure img').forEach(function(img) {{
  img.addEventListener('click', function() {{
    var lb = document.getElementById('lightbox');
    lb.querySelector('img').src = img.src;
    lb.classList.add('active');
    document.body.style.overflow = 'hidden';
  }});
}});
function closeLightbox() {{
  var lb = document.getElementById('lightbox');
  lb.classList.remove('active');
  document.body.style.overflow = '';
}}
document.getElementById('lightbox').addEventListener('click', function(e) {{
  if (e.target === this) closeLightbox();
}});
document.addEventListener('keydown', function(e) {{
  if (e.key === 'Escape') closeLightbox();
}});

</script>

</body>
</html>"""

    return html


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print('CytoAtlas PI Report — Interactive HTML Generation')
    print('=' * 60)

    print('\nLoading correlation data...')
    df = load_all_correlations()
    print(f'  Loaded {len(df)} correlation records')

    print('\nPreparing data for interactive figures...')
    summary_table = prepare_summary_table(df)
    print(f'  Summary table: {len(summary_table)} rows')

    boxplot_data = prepare_boxplot_data(df)
    print(f'  Boxplot data: {len(boxplot_data)} atlases')

    consistency_data = prepare_consistency_data(df)
    print(f'  Consistency data: {len(consistency_data)} targets')

    heatmap_data = prepare_heatmap_data(df)
    for sig, d in heatmap_data.items():
        print(f'  Heatmap {sig}: {len(d["targets"])} targets')

    levels_data = prepare_levels_data(df)
    print(f'  Levels data: {len(levels_data)} atlases x 3 signatures')

    bulk_data = prepare_bulk_validation_data(df)
    print(f'  Bulk validation: {len(bulk_data)} datasets')

    scatter_data = prepare_scatter_data()
    print(f'  Scatter data: {len(scatter_data)} atlases')

    good_bad_data = prepare_good_bad_data(df)
    print(f'  Good/bad data: {len(good_bad_data)} signature types')

    method_boxplot_data = prepare_method_comparison_boxplot(df)
    print(f'  Method comparison boxplot: {len(method_boxplot_data)} atlases')

    lincytosig_vs_cytosig_data = prepare_lincytosig_vs_cytosig(df)
    print(f'  LinCytoSig vs CytoSig: {len(lincytosig_vs_cytosig_data)} atlases')

    print('\nGenerating HTML...')
    html = generate_html(summary_table, boxplot_data, consistency_data,
                         heatmap_data, levels_data, bulk_data, scatter_data,
                         good_bad_data, method_boxplot_data, lincytosig_vs_cytosig_data)

    output_path = REPORT_DIR / 'PI_REPORT.html'
    output_path.write_text(html, encoding='utf-8')
    size_kb = len(html) / 1024
    print(f'  Written to: {output_path}')
    print(f'  Size: {size_kb:.0f} KB')

    print('\n' + '=' * 60)
    print('Done!')


if __name__ == '__main__':
    main()
