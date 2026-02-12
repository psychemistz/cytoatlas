#!/usr/bin/env python3
"""
CytoAtlas PI Report — Static Figure Generation
================================================
Generates all figures for the comprehensive report to PI (Peng Jiang).

Output: report/figures/
"""

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE = Path('/data/parks34/projects/2cytoatlas')
CORR_DIR = BASE / 'results' / 'cross_sample_validation' / 'correlations'
VIZ_DIR = BASE / 'visualization' / 'data' / 'validation'
SCATTER_DONOR = VIZ_DIR / 'donor_scatter'
SCATTER_CT = VIZ_DIR / 'celltype_scatter'
SCATTER_RESAMP = VIZ_DIR / 'resampled_scatter'
CELLTYPE_SIG = BASE / 'results' / 'celltype_signatures'
FIG_DIR = BASE / 'report' / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ─── Style ───────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Color palette
COLORS = {
    'cytosig': '#2563EB',      # blue
    'lincytosig': '#D97706',   # amber
    'secact': '#059669',       # emerald
    'good': '#059669',         # green
    'bad': '#DC2626',          # red
    'neutral': '#6B7280',      # gray
}

ATLAS_COLORS = {
    'cima': '#3B82F6',
    'inflammation_main': '#EF4444',
    'inflammation_val': '#F97316',
    'inflammation_ext': '#F59E0B',
    'scatlas_normal': '#10B981',
    'scatlas_cancer': '#8B5CF6',
    'gtex': '#6366F1',
    'tcga': '#EC4899',
}

# Biologically important target families
BIO_FAMILIES = {
    'Interferon': ['IFNG', 'IFN1', 'IFNL'],
    'TGF-β': ['TGFB1', 'TGFB2', 'TGFB3', 'BMP2', 'BMP4', 'BMP6', 'GDF11'],
    'Interleukin': ['IL1A', 'IL1B', 'IL2', 'IL4', 'IL6', 'IL10', 'IL12', 'IL13',
                    'IL15', 'IL17A', 'IL21', 'IL22', 'IL27', 'IL33'],
    'TNF': ['TNFA', 'LTA', 'TRAIL', 'TWEAK', 'CD40L'],
    'Growth Factor': ['EGF', 'FGF2', 'HGF', 'VEGFA', 'PDGFB', 'IGF1'],
    'Chemokine': ['CXCL12'],
    'Colony-Stimulating': ['GMCSF', 'GCSF', 'MCSF'],
}

# Flatten for lookup
TARGET_TO_FAMILY = {}
for fam, targets in BIO_FAMILIES.items():
    for t in targets:
        TARGET_TO_FAMILY[t] = fam

FAMILY_COLORS = {
    'Interferon': '#DC2626',
    'TGF-β': '#2563EB',
    'Interleukin': '#059669',
    'TNF': '#D97706',
    'Growth Factor': '#8B5CF6',
    'Chemokine': '#EC4899',
    'Colony-Stimulating': '#6366F1',
    'Other': '#9CA3AF',
}


def load_all_correlations():
    """Load and merge all per-atlas + bulk correlations."""
    dfs = []
    for f in CORR_DIR.glob('*_correlations.csv'):
        if f.name == 'all_correlations.csv' or 'resampled' in f.name or 'summary' in f.name:
            continue
        df = pd.read_csv(f, low_memory=False)
        dfs.append(df)
    # Also load bulk
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
# FIGURE 1: Dataset Overview
# ═══════════════════════════════════════════════════════════════════════════════
def fig1_dataset_overview():
    """Dataset scale, signature matrices, and validation layers."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel A: Dataset cell counts
    ax = axes[0]
    datasets = ['CIMA', 'Inflammation\nAtlas', 'scAtlas', 'parse_10M', 'Tahoe-100M', 'SpatialCorpus\n-110M']
    cells_millions = [6.5, 6.3, 6.4, 9.7, 100.6, 110.0]
    colors = ['#3B82F6', '#EF4444', '#10B981', '#F59E0B', '#8B5CF6', '#EC4899']
    bars = ax.barh(datasets, cells_millions, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_xlabel('Cells (millions)')
    ax.set_title('A. Dataset Scale (240M total)')
    for bar, v in zip(bars, cells_millions):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f'{v:.1f}M', va='center', fontsize=9, fontweight='bold')
    ax.set_xlim(0, 130)

    # Panel B: Signature coverage
    ax = axes[1]
    sig_types = ['CytoSig\n(44 cytokines)', 'LinCytoSig\n(178 cell-type\nspecific)', 'SecAct\n(1,249 secreted\nproteins)']
    counts = [44, 178, 1249]
    sig_colors = [COLORS['cytosig'], COLORS['lincytosig'], COLORS['secact']]
    bars = ax.bar(sig_types, counts, color=sig_colors, edgecolor='white', linewidth=0.5)
    ax.set_ylabel('Number of Signatures')
    ax.set_title('B. Signature Matrices (1,293 total)')
    for bar, v in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                str(v), ha='center', fontsize=10, fontweight='bold')

    # Panel C: Analysis layers
    ax = axes[2]
    layers = ['Pseudobulk\n(donor)', 'Pseudobulk\n(donor × celltype)', 'Single-cell\n(per cell)', 'Bulk RNA-seq\n(GTEx/TCGA)']
    layer_counts = [8, 8, 6, 2]  # Number of atlases at each layer
    layer_colors = ['#3B82F6', '#10B981', '#F59E0B', '#8B5CF6']
    bars = ax.barh(layers, layer_counts, color=layer_colors, edgecolor='white')
    ax.set_xlabel('Number of Atlas × Level Combinations')
    ax.set_title('C. Validation Layers')
    ax.set_xlim(0, 12)

    fig.suptitle('CytoAtlas: Pan-Disease Single-Cell Cytokine Activity Atlas', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(FIG_DIR / 'fig1_dataset_overview.png')
    fig.savefig(FIG_DIR / 'fig1_dataset_overview.pdf')
    plt.close(fig)
    print('  ✓ Figure 1: Dataset overview')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 2: Correlation Summary Across All Atlases (Boxplot)
# ═══════════════════════════════════════════════════════════════════════════════
def fig2_correlation_summary(df):
    """Boxplot of Spearman rho across all atlas × level × signature combinations."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 7), sharey=True)

    for idx, sig_type in enumerate(['cytosig', 'lincytosig', 'secact']):
        ax = axes[idx]
        sub = df[df['signature'] == sig_type].copy()

        # Get donor-level only for consistent comparison
        donor_levels = sub[sub['level'].str.contains('donor', case=False)]

        # Group by atlas
        atlas_order = ['cima', 'inflammation_main', 'inflammation_val', 'inflammation_ext',
                       'scatlas_normal', 'scatlas_cancer', 'gtex', 'tcga']
        atlas_labels = ['CIMA', 'Inflam\n(Main)', 'Inflam\n(Val)', 'Inflam\n(Ext)',
                        'scAtlas\n(Normal)', 'scAtlas\n(Cancer)', 'GTEx', 'TCGA']

        data_boxes = []
        positions = []
        box_colors = []
        used_labels = []
        for i, (atlas, label) in enumerate(zip(atlas_order, atlas_labels)):
            subset = donor_levels[donor_levels['atlas'] == atlas]
            if len(subset) > 0:
                data_boxes.append(subset['spearman_rho'].dropna().values)
                positions.append(i)
                box_colors.append(ATLAS_COLORS.get(atlas, '#6B7280'))
                used_labels.append(label)

        if data_boxes:
            bp = ax.boxplot(data_boxes, positions=positions, widths=0.6, patch_artist=True,
                           showfliers=False, medianprops=dict(color='black', linewidth=2))
            for patch, color in zip(bp['boxes'], box_colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            ax.set_xticks(positions)
            ax.set_xticklabels(used_labels, fontsize=8, rotation=0)

        ax.axhline(0, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
        ax.set_title(f'{sig_type.upper()}', fontsize=13, fontweight='bold',
                     color=COLORS[sig_type])
        if idx == 0:
            ax.set_ylabel('Spearman ρ (expression vs activity)')

    fig.suptitle('Cross-Sample Validation: Spearman Correlation Distributions\n(Donor-Level Pseudobulk)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(FIG_DIR / 'fig2_correlation_summary_boxplot.png')
    fig.savefig(FIG_DIR / 'fig2_correlation_summary_boxplot.pdf')
    plt.close(fig)
    print('  ✓ Figure 2: Correlation summary boxplot')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 3: Top / Bottom Targets — Good vs Bad Correlations
# ═══════════════════════════════════════════════════════════════════════════════
def fig3_good_bad_correlations(df):
    """Bar charts of best and worst correlated targets for CytoSig across atlases."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    atlas_configs = [
        ('cima', 'donor_only', 'CIMA (Donor)'),
        ('inflammation_main', 'donor_only', 'Inflammation Main (Donor)'),
        ('scatlas_normal', 'donor_organ', 'scAtlas Normal (Organ)'),
    ]

    for col, (atlas, level, title) in enumerate(atlas_configs):
        sub = df[(df['atlas'] == atlas) & (df['level'] == level) & (df['signature'] == 'cytosig')]
        sub = sub.sort_values('spearman_rho', ascending=False)

        # Top 15 (good)
        ax = axes[0, col]
        top = sub.head(15)
        colors_top = [FAMILY_COLORS.get(TARGET_TO_FAMILY.get(t, 'Other'), '#9CA3AF')
                      for t in top['target']]
        bars = ax.barh(range(len(top)), top['spearman_rho'].values, color=colors_top)
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(top['target'].values, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel('Spearman ρ')
        ax.set_title(f'{title}\nTop 15 (Best Correlation)', fontweight='bold')
        ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
        # Annotate rho values
        for i, (_, row) in enumerate(top.iterrows()):
            ax.text(row['spearman_rho'] + 0.01, i, f'{row["spearman_rho"]:.3f}',
                    va='center', fontsize=8, color='black')

        # Bottom 15 (bad)
        ax = axes[1, col]
        bottom = sub.tail(15).iloc[::-1]
        colors_bot = [FAMILY_COLORS.get(TARGET_TO_FAMILY.get(t, 'Other'), '#9CA3AF')
                      for t in bottom['target']]
        bars = ax.barh(range(len(bottom)), bottom['spearman_rho'].values, color=colors_bot)
        ax.set_yticks(range(len(bottom)))
        ax.set_yticklabels(bottom['target'].values, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel('Spearman ρ')
        ax.set_title(f'{title}\nBottom 15 (Worst Correlation)', fontweight='bold')
        ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
        for i, (_, row) in enumerate(bottom.iterrows()):
            ax.text(max(row['spearman_rho'] - 0.15, -0.5), i, f'{row["spearman_rho"]:.3f}',
                    va='center', fontsize=8, color='black')

    # Legend for families
    handles = [plt.Rectangle((0, 0), 1, 1, fc=c) for fam, c in FAMILY_COLORS.items()]
    fig.legend(handles, list(FAMILY_COLORS.keys()), loc='lower center', ncol=4, fontsize=9,
               frameon=True, title='Cytokine Family')

    fig.suptitle('CytoSig Validation: Best & Worst Correlated Targets by Atlas',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    fig.savefig(FIG_DIR / 'fig3_good_bad_correlations_cytosig.png')
    fig.savefig(FIG_DIR / 'fig3_good_bad_correlations_cytosig.pdf')
    plt.close(fig)
    print('  ✓ Figure 3: Good/bad correlations (CytoSig)')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 4: Biologically Important Targets Heatmap
# ═══════════════════════════════════════════════════════════════════════════════
def fig4_bio_targets_heatmap(df):
    """Heatmap of Spearman rho for biologically important targets across atlases."""
    fig, axes = plt.subplots(1, 3, figsize=(22, 10))

    for idx, sig_type in enumerate(['cytosig', 'lincytosig', 'secact']):
        ax = axes[idx]

        # Filter to donor-level (simplest aggregation)
        sub = df[df['signature'] == sig_type].copy()
        # Use lowest aggregation level per atlas
        level_map = {
            'cima': 'donor_only', 'inflammation_main': 'donor_only',
            'inflammation_val': 'donor_only', 'inflammation_ext': 'donor_only',
            'scatlas_normal': 'donor_organ', 'scatlas_cancer': 'donor_organ',
            'gtex': 'donor_only', 'tcga': 'donor_only',
        }
        rows = []
        for atlas, level in level_map.items():
            atlas_sub = sub[(sub['atlas'] == atlas) & (sub['level'] == level)]
            for _, row in atlas_sub.iterrows():
                rows.append(row)
        sub = pd.DataFrame(rows)

        if sig_type == 'cytosig':
            # All cytosig targets are biologically important
            targets = sorted(sub['target'].unique())
        elif sig_type == 'lincytosig':
            # Filter to immune-relevant cell types and key cytokines
            immune_cts = ['Macrophage', 'Monocyte', 'T_CD4', 'T_CD8', 'B_Cell',
                          'NK_Cell', 'Dendritic_Cell', 'Neutrophil']
            key_cyto = ['IFNG', 'TNFA', 'IL1B', 'IL4', 'IL6', 'IL10', 'IL12',
                        'IL2', 'IL13', 'IL15', 'TGFB1', 'GMCSF']
            targets = []
            for t in sub['target'].unique():
                parts = t.rsplit('__', 1)
                if len(parts) == 2:
                    ct, cyto = parts
                    if ct in immune_cts and cyto in key_cyto:
                        targets.append(t)
            targets = sorted(targets)[:40]  # Limit for readability
        else:
            # SecAct: use same cytokine names as CytoSig for comparison
            bio_targets = list(TARGET_TO_FAMILY.keys())
            targets = [t for t in sub['target'].unique() if t in bio_targets]
            targets = sorted(targets)

        if not targets:
            ax.text(0.5, 0.5, 'No matching targets', ha='center', va='center', transform=ax.transAxes)
            continue

        atlas_order = ['cima', 'inflammation_main', 'inflammation_val', 'inflammation_ext',
                       'scatlas_normal', 'scatlas_cancer', 'gtex', 'tcga']
        atlas_labels = ['CIMA', 'Inflam (M)', 'Inflam (V)', 'Inflam (E)',
                        'scAtlas (N)', 'scAtlas (C)', 'GTEx', 'TCGA']

        # Build heatmap matrix
        matrix = np.full((len(targets), len(atlas_order)), np.nan)
        for i, target in enumerate(targets):
            for j, atlas in enumerate(atlas_order):
                level = level_map[atlas]
                match = sub[(sub['target'] == target) & (sub['atlas'] == atlas) & (sub['level'] == level)]
                if len(match) > 0:
                    matrix[i, j] = match['spearman_rho'].values[0]

        im = ax.imshow(matrix, aspect='auto', cmap='RdBu_r', vmin=-0.5, vmax=0.8,
                       interpolation='nearest')
        ax.set_xticks(range(len(atlas_labels)))
        ax.set_xticklabels(atlas_labels, rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(len(targets)))
        ax.set_yticklabels(targets, fontsize=7)
        ax.set_title(sig_type.upper(), fontsize=13, fontweight='bold', color=COLORS[sig_type])

        # Add value annotations for significant ones
        for i in range(len(targets)):
            for j in range(len(atlas_order)):
                if not np.isnan(matrix[i, j]):
                    color = 'white' if abs(matrix[i, j]) > 0.4 else 'black'
                    ax.text(j, i, f'{matrix[i, j]:.2f}', ha='center', va='center',
                            fontsize=5, color=color)

    plt.colorbar(im, ax=axes, shrink=0.6, label='Spearman ρ')
    fig.suptitle('Cross-Sample Validation: Biologically Important Targets\n(Donor-Level Spearman Correlation)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 0.95, 0.95])
    fig.savefig(FIG_DIR / 'fig4_bio_targets_heatmap.png')
    fig.savefig(FIG_DIR / 'fig4_bio_targets_heatmap.pdf')
    plt.close(fig)
    print('  ✓ Figure 4: Biologically important targets heatmap')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 5: Scatter Plots — Representative Good & Bad Correlations
# ═══════════════════════════════════════════════════════════════════════════════
def fig5_representative_scatter(df):
    """Representative scatter plots showing good and bad correlations."""
    # Find best/worst CytoSig targets in CIMA
    cima_donor = df[(df['atlas'] == 'cima') & (df['level'] == 'donor_only') & (df['signature'] == 'cytosig')]
    cima_donor_sorted = cima_donor.sort_values('spearman_rho', ascending=False)

    # Key biologically important targets: pick good and bad ones
    good_targets = cima_donor_sorted[cima_donor_sorted['spearman_rho'] > 0.3].head(4)['target'].tolist()
    # Pick the MOST negative targets (sort ascending, take first 4)
    bad_targets = cima_donor_sorted.nsmallest(4, 'spearman_rho')['target'].tolist()

    targets_to_plot = good_targets + bad_targets
    n_targets = len(targets_to_plot)

    if n_targets == 0:
        print('  ⚠ No scatter targets found for figure 5')
        return

    # Load CIMA donor scatter data
    scatter_data = load_scatter(SCATTER_DONOR, 'cima_cytosig.json')
    if scatter_data is None:
        print('  ⚠ cima_cytosig.json not found')
        return

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    for idx, target in enumerate(targets_to_plot[:8]):
        row = idx // 4
        col = idx % 4
        ax = axes[row, col]

        if target in scatter_data:
            data = scatter_data[target]
            points = data.get('points', [])
            if points:
                x = [p[0] for p in points]
                y = [p[1] for p in points]
                rho = data.get('rho', np.nan)
                pval = data.get('pval', np.nan)
                n = data.get('n', len(points))

                color = COLORS['good'] if rho > 0.2 else (COLORS['bad'] if rho < 0 else COLORS['neutral'])
                ax.scatter(x, y, alpha=0.4, s=15, c=color, edgecolors='none')

                # Add regression line
                if len(x) > 2:
                    slope, intercept, _, _, _ = stats.linregress(x, y)
                    x_line = np.linspace(min(x), max(x), 100)
                    ax.plot(x_line, slope * x_line + intercept, 'k--', linewidth=1.5, alpha=0.7)

                pval_str = f'{pval:.1e}' if pval < 0.001 else f'{pval:.4f}'
                ax.set_title(f'{target}\nρ={rho:.3f}, p={pval_str}, n={n}', fontsize=10,
                            fontweight='bold', color=color)
            else:
                ax.text(0.5, 0.5, 'No points', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(target, fontsize=10)
        else:
            ax.text(0.5, 0.5, 'Not found', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(target, fontsize=10)

        ax.set_xlabel('Mean Expression')
        ax.set_ylabel('Predicted Activity')

    axes[0, 0].annotate('GOOD CORRELATIONS', xy=(0, 1.15), xycoords='axes fraction',
                        fontsize=12, fontweight='bold', color=COLORS['good'])
    axes[1, 0].annotate('POOR CORRELATIONS', xy=(0, 1.15), xycoords='axes fraction',
                        fontsize=12, fontweight='bold', color=COLORS['bad'])

    fig.suptitle('CIMA: Donor-Level Expression vs CytoSig Predicted Activity',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIG_DIR / 'fig5_representative_scatter_cima.png')
    fig.savefig(FIG_DIR / 'fig5_representative_scatter_cima.pdf')
    plt.close(fig)
    print('  ✓ Figure 5: Representative scatter plots')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 6: Cross-Atlas Consistency for Key Targets
# ═══════════════════════════════════════════════════════════════════════════════
def fig6_cross_atlas_consistency(df):
    """Show how key target correlations are consistent across atlases."""
    key_targets = ['IFNG', 'IL1B', 'TNFA', 'TGFB1', 'IL6', 'IL10', 'IL17A',
                   'IL4', 'BMP2', 'EGF', 'HGF', 'VEGFA', 'CXCL12', 'GMCSF']

    cytosig = df[(df['signature'] == 'cytosig')].copy()

    # Use donor-level only
    level_map = {
        'cima': 'donor_only', 'inflammation_main': 'donor_only',
        'inflammation_val': 'donor_only', 'inflammation_ext': 'donor_only',
        'scatlas_normal': 'donor_organ', 'scatlas_cancer': 'donor_organ',
        'gtex': 'donor_only', 'tcga': 'donor_only',
    }

    fig, ax = plt.subplots(figsize=(14, 8))

    atlas_order = list(level_map.keys())
    atlas_labels = ['CIMA', 'Inflam (M)', 'Inflam (V)', 'Inflam (E)',
                    'scAtlas (N)', 'scAtlas (C)', 'GTEx', 'TCGA']

    for i, target in enumerate(key_targets):
        rhos = []
        for atlas in atlas_order:
            level = level_map[atlas]
            match = cytosig[(cytosig['target'] == target) & (cytosig['atlas'] == atlas) & (cytosig['level'] == level)]
            if len(match) > 0:
                rhos.append(match['spearman_rho'].values[0])
            else:
                rhos.append(np.nan)

        color = FAMILY_COLORS.get(TARGET_TO_FAMILY.get(target, 'Other'), '#9CA3AF')
        ax.plot(range(len(atlas_order)), rhos, 'o-', label=target, color=color,
                markersize=6, linewidth=1.5, alpha=0.8)

    ax.set_xticks(range(len(atlas_labels)))
    ax.set_xticklabels(atlas_labels, rotation=30, ha='right')
    ax.set_ylabel('Spearman ρ')
    ax.set_title('CytoSig: Cross-Atlas Consistency of Key Cytokine Targets\n(Donor-Level Correlation)',
                 fontsize=13, fontweight='bold')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, frameon=True)

    plt.tight_layout()
    fig.savefig(FIG_DIR / 'fig6_cross_atlas_consistency.png')
    fig.savefig(FIG_DIR / 'fig6_cross_atlas_consistency.pdf')
    plt.close(fig)
    print('  ✓ Figure 6: Cross-atlas consistency')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 7: Validation Levels Comparison (Donor vs Celltype vs Bulk)
# ═══════════════════════════════════════════════════════════════════════════════
def fig7_validation_levels(df):
    """Compare correlations across different aggregation levels for all atlases.

    GTEx/TCGA: donor-only level.
    CIMA, Inflammation: donor-level pseudobulk with cell-type stratification.
    scAtlas: donor × organ level pseudobulk with cell-type stratification.
    """
    fig, axes = plt.subplots(2, 3, figsize=(22, 12))
    axes_flat = axes.flatten()

    atlas_configs = [
        ('cima', ['donor_only', 'donor_l1', 'donor_l2', 'donor_l3', 'donor_l4'], 'CIMA'),
        ('inflammation_main', ['donor_only', 'donor_l1', 'donor_l2'], 'Inflammation (Main)'),
        ('inflammation_val', ['donor_only', 'donor_l1', 'donor_l2'], 'Inflammation (Val)'),
        ('inflammation_ext', ['donor_only', 'donor_l1', 'donor_l2'], 'Inflammation (Ext)'),
        ('scatlas_normal', ['donor_organ', 'donor_organ_celltype1', 'donor_organ_celltype2'], 'scAtlas Normal'),
        ('scatlas_cancer', ['donor_organ', 'donor_organ_celltype1', 'donor_organ_celltype2'], 'scAtlas Cancer'),
    ]

    level_labels = {
        'donor_only': 'Donor\nOnly',
        'donor_l1': 'Donor\n× L1',
        'donor_l2': 'Donor\n× L2',
        'donor_l3': 'Donor\n× L3',
        'donor_l4': 'Donor\n× L4',
        'donor_organ': 'Donor\n× Organ',
        'donor_organ_celltype1': 'Donor × Organ\n× Celltype1',
        'donor_organ_celltype2': 'Donor × Organ\n× Celltype2',
    }

    for idx, (atlas, levels, title) in enumerate(atlas_configs):
        ax = axes_flat[idx]
        sub = df[(df['atlas'] == atlas) & (df['signature'] == 'cytosig')]

        data_boxes = []
        labels = []
        for level in levels:
            level_data = sub[sub['level'] == level]['spearman_rho'].dropna()
            if len(level_data) > 0:
                data_boxes.append(level_data.values)
                labels.append(level_labels.get(level, level))

        if data_boxes:
            bp = ax.boxplot(data_boxes, patch_artist=True, showfliers=False,
                           medianprops=dict(color='black', linewidth=2))
            color = ATLAS_COLORS.get(atlas.replace('inflammation_', 'inflammation_'), '#3B82F6')
            cmap_colors = plt.cm.Blues(np.linspace(0.3, 0.8, len(data_boxes)))
            for patch, c in zip(bp['boxes'], cmap_colors):
                patch.set_facecolor(c)
                patch.set_alpha(0.8)
            ax.set_xticklabels(labels, fontsize=8)

        ax.set_ylabel('Spearman ρ' if idx % 3 == 0 else '')
        ax.set_title(f'{title} — CytoSig', fontweight='bold', fontsize=11)
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)

        if data_boxes:
            for i, d in enumerate(data_boxes):
                med = np.median(d)
                ax.text(i + 1, med + 0.02, f'{med:.3f}', ha='center', fontsize=7, fontweight='bold')

    fig.suptitle('Effect of Aggregation Level on Validation Correlations\n'
                 'CIMA/Inflammation: donor-level pseudobulk; scAtlas: donor × organ pseudobulk\n'
                 '(Finer cell-type stratification → more data points but lower per-target correlations)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(FIG_DIR / 'fig7_validation_levels.png')
    fig.savefig(FIG_DIR / 'fig7_validation_levels.pdf')
    plt.close(fig)
    print('  ✓ Figure 7: Validation levels comparison (all 6 atlases)')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 8: 6-Way Method Comparison (CytoSig, 4× LinCytoSig, SecAct)
# ═══════════════════════════════════════════════════════════════════════════════
def fig8_method_comparison():
    """6-way comparison: CytoSig, LinCytoSig (orig/filt/best-orig/best-filt), SecAct.

    Uses donor-level (CIMA/Inflammation) or donor×organ (scAtlas) pseudobulk
    correlations for matched cytokines across all 6 atlases.
    """
    with open(VIZ_DIR / 'method_comparison_6way_all.json') as f:
        data = json.load(f)

    methods = [
        ('cytosig', 'CytoSig', '#2563EB'),
        ('lincyto_orig', 'LinCytoSig\n(no filter)', '#D97706'),
        ('lincyto_filt', 'LinCytoSig\n(gene filter)', '#B45309'),
        ('lincyto_best_orig', 'LinCytoSig\nBest (no filter)', '#DC2626'),
        ('lincyto_best_filt', 'LinCytoSig\nBest (gene filter)', '#991B1B'),
        ('secact', 'SecAct', '#059669'),
    ]
    atlas_order = [
        'CIMA', 'Inflammation Main', 'Inflammation Val',
        'Inflammation Ext', 'scAtlas Normal', 'scAtlas Cancer',
    ]

    # Panel A: 6 subplots, one per atlas, each with 6 boxplots (one per method)
    fig, axes = plt.subplots(2, 3, figsize=(24, 14))
    axes_flat = axes.flatten()

    for ax_idx, atlas_name in enumerate(atlas_order):
        ax = axes_flat[ax_idx]
        if atlas_name not in data:
            ax.set_visible(False)
            continue

        atlas_data = data[atlas_name]
        n_cytokines = len(atlas_data['cytokines'])

        box_data = []
        box_labels = []
        box_colors = []
        for method_key, method_label, method_color in methods:
            vals = atlas_data.get(method_key, [])
            vals = [v for v in vals if v is not None]
            box_data.append(vals)
            box_labels.append(method_label)
            box_colors.append(method_color)

        bp = ax.boxplot(box_data, patch_artist=True, showfliers=True,
                       flierprops=dict(marker='o', markersize=3, alpha=0.3),
                       medianprops=dict(color='black', linewidth=2),
                       widths=0.6)
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.set_xticklabels(box_labels, fontsize=7, rotation=30, ha='right')
        ax.set_ylabel('Spearman ρ')
        ax.set_title(f'{atlas_name} (n={n_cytokines} cytokines)', fontweight='bold', fontsize=11)
        ax.axhline(0, color='gray', linestyle='--', alpha=0.5)

        # Median labels
        for i, vals in enumerate(box_data):
            if vals:
                med = np.median(vals)
                ax.text(i + 1, med + 0.02, f'{med:.3f}', ha='center', fontsize=7,
                        fontweight='bold', color=box_colors[i])

    fig.suptitle('6-Way Method Comparison: CytoSig vs LinCytoSig Variants vs SecAct\n'
                 '(Donor-level pseudobulk validation per matched cytokine)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(FIG_DIR / 'fig8_method_comparison.png')
    fig.savefig(FIG_DIR / 'fig8_method_comparison.pdf')
    plt.close(fig)
    print('  ✓ Figure 8: 6-way method comparison (all atlases)')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 9: LinCytoSig vs CytoSig — Matched Target Comparison
# ═══════════════════════════════════════════════════════════════════════════════
def fig9_lincytosig_vs_cytosig():
    """Scatter: LinCytoSig vs CytoSig matched correlations, identify winners."""
    mc = load_method_comparison()
    matched = mc['matched_targets']  # dict: lincytosig_target -> {cytosig: str, secact: str}

    categories = mc['categories']
    cat_keys = [c['key'] for c in categories]
    cat_labels = [c['label'] for c in categories]

    n_cats = len(cat_keys)
    n_cols = 3
    n_rows = (n_cats + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
    axes = axes.flatten()

    for cat_idx, (cat_key, cat_label) in enumerate(zip(cat_keys, cat_labels)):
        ax = axes[cat_idx]

        cytosig_rhos = []
        lincytosig_rhos = []
        labels = []
        celltype_labels = []

        for lin_target, match_info in matched.items():
            cyto_target = match_info.get('cytosig') if isinstance(match_info, dict) else match_info
            if cyto_target is None:
                continue
            lin_rho = mc['lincytosig']['rhos'].get(lin_target, {}).get(cat_key)
            cyto_rho = mc['cytosig']['rhos'].get(cyto_target, {}).get(cat_key)

            if lin_rho is not None and cyto_rho is not None:
                if not (isinstance(lin_rho, float) and np.isnan(lin_rho)):
                    if not (isinstance(cyto_rho, float) and np.isnan(cyto_rho)):
                        cytosig_rhos.append(cyto_rho)
                        lincytosig_rhos.append(lin_rho)
                        labels.append(lin_target)
                        # Extract celltype
                        parts = lin_target.rsplit('__', 1)
                        celltype_labels.append(parts[0] if len(parts) == 2 else 'Unknown')

        if not cytosig_rhos:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(cat_label)
            continue

        cytosig_arr = np.array(cytosig_rhos)
        lincytosig_arr = np.array(lincytosig_rhos)

        # Color by who wins
        colors = []
        for c, l in zip(cytosig_arr, lincytosig_arr):
            if l > c + 0.05:
                colors.append(COLORS['lincytosig'])  # LinCytoSig wins
            elif c > l + 0.05:
                colors.append(COLORS['cytosig'])  # CytoSig wins
            else:
                colors.append(COLORS['neutral'])  # Tie

        ax.scatter(cytosig_arr, lincytosig_arr, c=colors, alpha=0.6, s=40, edgecolors='white', linewidths=0.5)

        # Diagonal
        lim = [-0.5, 1.0]
        ax.plot(lim, lim, 'k--', alpha=0.4, linewidth=1)
        ax.set_xlim(lim)
        ax.set_ylim(lim)

        # Count winners
        n_lin_wins = np.sum(lincytosig_arr > cytosig_arr + 0.05)
        n_cyto_wins = np.sum(cytosig_arr > lincytosig_arr + 0.05)
        n_tie = len(cytosig_arr) - n_lin_wins - n_cyto_wins

        ax.set_xlabel('CytoSig ρ')
        ax.set_ylabel('LinCytoSig ρ')
        ax.set_title(f'{cat_label}\nLinCytoSig↑: {n_lin_wins}  CytoSig↑: {n_cyto_wins}  Tie: {n_tie}',
                     fontsize=10, fontweight='bold')

        # Label notable points (biggest winners for lincytosig)
        diffs = lincytosig_arr - cytosig_arr
        top_indices = np.argsort(diffs)[-3:]  # Top 3 LinCytoSig winners
        bottom_indices = np.argsort(diffs)[:3]  # Top 3 CytoSig winners
        for i in list(top_indices) + list(bottom_indices):
            if abs(diffs[i]) > 0.1:
                ax.annotate(labels[i].replace('__', '\n'), (cytosig_arr[i], lincytosig_arr[i]),
                           fontsize=6, alpha=0.8, textcoords='offset points', xytext=(5, 5))

    # Hide unused axes
    for i in range(cat_idx + 1, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle('LinCytoSig vs CytoSig: Matched Target Correlation Comparison\n(Points above diagonal = LinCytoSig better)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIG_DIR / 'fig9_lincytosig_vs_cytosig_scatter.png')
    fig.savefig(FIG_DIR / 'fig9_lincytosig_vs_cytosig_scatter.pdf')
    plt.close(fig)
    print('  ✓ Figure 9: LinCytoSig vs CytoSig matched comparison')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 10: LinCytoSig Advantage Analysis — Which cell types benefit?
# ═══════════════════════════════════════════════════════════════════════════════
def fig10_lincytosig_advantage():
    """Analyze when and why LinCytoSig outperforms CytoSig by cell type."""
    mc = load_method_comparison()
    matched = mc['matched_targets']

    categories = mc['categories']
    cat_keys = [c['key'] for c in categories]

    # Collect per-celltype differences across all categories
    celltype_advantages = {}
    for lin_target, match_info in matched.items():
        cyto_target = match_info.get('cytosig') if isinstance(match_info, dict) else match_info
        if cyto_target is None:
            continue
        parts = lin_target.rsplit('__', 1)
        if len(parts) != 2:
            continue
        celltype, cytokine = parts

        if celltype not in celltype_advantages:
            celltype_advantages[celltype] = {'diffs': [], 'n_better': 0, 'n_worse': 0, 'n_total': 0}

        for cat_key in cat_keys:
            lin_rho = mc['lincytosig']['rhos'].get(lin_target, {}).get(cat_key)
            cyto_rho = mc['cytosig']['rhos'].get(cyto_target, {}).get(cat_key)
            if lin_rho is not None and cyto_rho is not None:
                if not (isinstance(lin_rho, float) and np.isnan(lin_rho)):
                    if not (isinstance(cyto_rho, float) and np.isnan(cyto_rho)):
                        diff = lin_rho - cyto_rho
                        celltype_advantages[celltype]['diffs'].append(diff)
                        celltype_advantages[celltype]['n_total'] += 1
                        if diff > 0.05:
                            celltype_advantages[celltype]['n_better'] += 1
                        elif diff < -0.05:
                            celltype_advantages[celltype]['n_worse'] += 1

    # Compute mean advantage per celltype
    for ct in celltype_advantages:
        diffs = celltype_advantages[ct]['diffs']
        celltype_advantages[ct]['mean_diff'] = np.mean(diffs) if diffs else 0
        celltype_advantages[ct]['median_diff'] = np.median(diffs) if diffs else 0

    # Sort by mean diff
    sorted_cts = sorted(celltype_advantages.items(), key=lambda x: x[1]['mean_diff'], reverse=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 10))

    # Panel A: Mean difference by celltype
    ax = axes[0]
    ct_names = [ct for ct, _ in sorted_cts]
    mean_diffs = [v['mean_diff'] for _, v in sorted_cts]
    colors = [COLORS['lincytosig'] if d > 0 else COLORS['cytosig'] for d in mean_diffs]

    bars = ax.barh(range(len(ct_names)), mean_diffs, color=colors, alpha=0.8)
    ax.set_yticks(range(len(ct_names)))
    ax.set_yticklabels(ct_names, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color='black', linewidth=1)
    ax.set_xlabel('Mean Δρ (LinCytoSig − CytoSig)')
    ax.set_title('A. Mean Advantage by Cell Type\n(positive = LinCytoSig better)', fontweight='bold')

    # Annotate values
    for i, (ct, v) in enumerate(sorted_cts):
        ax.text(v['mean_diff'] + 0.005 * np.sign(v['mean_diff']),
                i, f'{v["mean_diff"]:+.3f}', va='center', fontsize=7)

    # Panel B: Win rate by celltype
    ax = axes[1]
    win_rates = []
    for ct, v in sorted_cts:
        if v['n_total'] > 0:
            win_rates.append(100 * v['n_better'] / v['n_total'])
        else:
            win_rates.append(0)

    lose_rates = []
    for ct, v in sorted_cts:
        if v['n_total'] > 0:
            lose_rates.append(100 * v['n_worse'] / v['n_total'])
        else:
            lose_rates.append(0)

    ax.barh(range(len(ct_names)), win_rates, color=COLORS['lincytosig'], alpha=0.7, label='LinCytoSig wins')
    ax.barh(range(len(ct_names)), [-r for r in lose_rates], color=COLORS['cytosig'], alpha=0.7, label='CytoSig wins')
    ax.set_yticks(range(len(ct_names)))
    ax.set_yticklabels(ct_names, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color='black', linewidth=1)
    ax.set_xlabel('% Comparisons Won')
    ax.set_title('B. Win Rate by Cell Type', fontweight='bold')
    ax.legend(fontsize=9, loc='lower right')

    fig.suptitle('LinCytoSig Advantage Analysis:\nWhen Does Cell-Type Specificity Help?',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(FIG_DIR / 'fig10_lincytosig_advantage_by_celltype.png')
    fig.savefig(FIG_DIR / 'fig10_lincytosig_advantage_by_celltype.pdf')
    plt.close(fig)
    print('  ✓ Figure 10: LinCytoSig advantage by cell type')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 11: SecAct — Novel High-Correlation Signatures
# ═══════════════════════════════════════════════════════════════════════════════
def fig11_secact_novel(df):
    """Identify novel SecAct signatures with consistently high correlations."""
    secact = df[(df['signature'] == 'secact')].copy()

    # Use donor-level
    level_map = {
        'cima': 'donor_only', 'inflammation_main': 'donor_only',
        'scatlas_normal': 'donor_organ', 'gtex': 'donor_only', 'tcga': 'donor_only',
    }

    # Compute mean rho across atlases per target
    target_stats = {}
    for target in secact['target'].unique():
        rhos = []
        for atlas, level in level_map.items():
            match = secact[(secact['target'] == target) & (secact['atlas'] == atlas) & (secact['level'] == level)]
            if len(match) > 0:
                rhos.append(match['spearman_rho'].values[0])
        if len(rhos) >= 3:
            target_stats[target] = {
                'mean_rho': np.mean(rhos),
                'min_rho': np.min(rhos),
                'max_rho': np.max(rhos),
                'std_rho': np.std(rhos),
                'n_atlases': len(rhos),
                'consistent': np.min(rhos) > 0,
            }

    if not target_stats:
        print('  ⚠ No SecAct targets found for figure 11')
        return

    stats_df = pd.DataFrame(target_stats).T
    stats_df.index.name = 'target'
    stats_df = stats_df.reset_index()

    # Top novel: high mean, consistent positive, NOT a known CytoSig target
    cytosig_targets = set(TARGET_TO_FAMILY.keys())
    novel = stats_df[~stats_df['target'].isin(cytosig_targets)]
    novel = novel[novel['consistent'] == True].sort_values('mean_rho', ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(16, 10))

    # Panel A: Top 30 novel SecAct targets
    ax = axes[0]
    top30 = novel.head(30)
    bars = ax.barh(range(len(top30)), top30['mean_rho'].values, color=COLORS['secact'], alpha=0.8)
    # Error bars
    for i, (_, row) in enumerate(top30.iterrows()):
        ax.plot([row['min_rho'], row['max_rho']], [i, i], 'k-', linewidth=1, alpha=0.5)
    ax.set_yticks(range(len(top30)))
    ax.set_yticklabels(top30['target'].values, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('Mean Spearman ρ (across atlases)')
    ax.set_title('A. Top 30 Novel SecAct Targets\n(Consistent Positive Correlation)', fontweight='bold')
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    for i, (_, row) in enumerate(top30.iterrows()):
        ax.text(row['mean_rho'] + 0.01, i, f'{row["mean_rho"]:.3f}', va='center', fontsize=7)

    # Panel B: Distribution of all SecAct mean_rhos
    ax = axes[1]
    ax.hist(stats_df['mean_rho'].values, bins=50, color=COLORS['secact'], alpha=0.7, edgecolor='white')
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Mean Spearman ρ (across atlases)')
    ax.set_ylabel('Number of Targets')
    ax.set_title(f'B. Distribution of {len(stats_df)} SecAct Targets\n(Mean ρ across ≥3 atlases)', fontweight='bold')

    # Annotate stats
    n_positive = (stats_df['mean_rho'] > 0).sum()
    n_strong = (stats_df['mean_rho'] > 0.3).sum()
    ax.text(0.95, 0.95, f'Total: {len(stats_df)}\nPositive (ρ>0): {n_positive} ({100*n_positive/len(stats_df):.0f}%)\nStrong (ρ>0.3): {n_strong} ({100*n_strong/len(stats_df):.0f}%)',
            transform=ax.transAxes, va='top', ha='right', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    fig.suptitle('SecAct: Novel Secreted Protein Activity Signatures\n(Beyond the 44 CytoSig Cytokines)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(FIG_DIR / 'fig11_secact_novel_signatures.png')
    fig.savefig(FIG_DIR / 'fig11_secact_novel_signatures.pdf')
    plt.close(fig)
    print('  ✓ Figure 11: SecAct novel signatures')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 12: Bulk RNA-seq Validation (GTEx & TCGA)
# ═══════════════════════════════════════════════════════════════════════════════
def fig12_bulk_validation(df):
    """GTEx and TCGA bulk RNA-seq validation results."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for row_idx, (atlas, title) in enumerate([('gtex', 'GTEx'), ('tcga', 'TCGA')]):
        for col_idx, sig_type in enumerate(['cytosig', 'lincytosig', 'secact']):
            ax = axes[row_idx, col_idx]
            # Use donor_only level for consistent comparison
            sub = df[(df['atlas'] == atlas) & (df['signature'] == sig_type) & (df['level'] == 'donor_only')]
            rhos = sub['spearman_rho'].dropna().values

            if len(rhos) == 0:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{title} — {sig_type.upper()}', fontweight='bold')
                continue

            # Ranked bar chart for all
            sorted_rhos = np.sort(rhos)[::-1]
            n = len(sorted_rhos)
            bar_colors = [COLORS['good'] if r > 0.2 else (COLORS['bad'] if r < -0.1 else COLORS['neutral'])
                          for r in sorted_rhos]

            if n <= 60:
                ax.bar(range(n), sorted_rhos, color=bar_colors, alpha=0.8, width=0.8)
                ax.set_xlabel('Target (ranked by ρ)')
            else:
                # For large sets use histogram
                ax.hist(rhos, bins=40, color=COLORS[sig_type], alpha=0.7, edgecolor='white')
                ax.set_xlabel('Spearman ρ')
                ax.set_ylabel('Count')
                # Add vertical lines for median
                med = np.median(rhos)
                ax.axvline(med, color='black', linewidth=2, linestyle='-', label=f'median={med:.3f}')
                ax.legend(fontsize=9)

            ax.axhline(0, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
            n_sig = np.sum(np.abs(rhos) > 0)  # just for display
            med = np.median(rhos)
            ax.set_title(f'{title} — {sig_type.upper()}\n(n={n}, median ρ={med:.3f})',
                         fontweight='bold', color=COLORS[sig_type])
            if col_idx == 0 and n <= 60:
                ax.set_ylabel('Spearman ρ')

    fig.suptitle('Bulk RNA-seq Validation: Donor-Level Spearman Correlation\n(CytoSig/LinCytoSig/SecAct predicted activity vs target gene expression)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(FIG_DIR / 'fig12_bulk_validation.png')
    fig.savefig(FIG_DIR / 'fig12_bulk_validation.pdf')
    plt.close(fig)
    print('  ✓ Figure 12: Bulk RNA-seq validation')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 13: LinCytoSig Cell-Type Specificity Deep Dive
# ═══════════════════════════════════════════════════════════════════════════════
def fig13_lincytosig_specificity():
    """Show that LinCytoSig captures cell-type-specific biology CytoSig misses."""
    mc = load_method_comparison()
    matched = mc['matched_targets']

    # Find cases where LinCytoSig dramatically outperforms CytoSig
    examples = []
    for lin_target, match_info in matched.items():
        cyto_target = match_info.get('cytosig') if isinstance(match_info, dict) else match_info
        if cyto_target is None:
            continue
        parts = lin_target.rsplit('__', 1)
        if len(parts) != 2:
            continue
        celltype, cytokine = parts

        for cat_key in [c['key'] for c in mc['categories']]:
            lin_rho = mc['lincytosig']['rhos'].get(lin_target, {}).get(cat_key)
            cyto_rho = mc['cytosig']['rhos'].get(cyto_target, {}).get(cat_key)
            if lin_rho is not None and cyto_rho is not None:
                if not (isinstance(lin_rho, float) and np.isnan(lin_rho)):
                    if not (isinstance(cyto_rho, float) and np.isnan(cyto_rho)):
                        examples.append({
                            'lin_target': lin_target,
                            'cyto_target': cyto_target,
                            'celltype': celltype,
                            'cytokine': cytokine,
                            'category': cat_key,
                            'lin_rho': lin_rho,
                            'cyto_rho': cyto_rho,
                            'diff': lin_rho - cyto_rho,
                        })

    if not examples:
        print('  ⚠ No examples for figure 13')
        return

    ex_df = pd.DataFrame(examples)

    # Top LinCytoSig winners
    top_lin = ex_df.nlargest(20, 'diff')
    # Top CytoSig winners (LinCytoSig worst)
    top_cyto = ex_df.nsmallest(20, 'diff')

    fig, axes = plt.subplots(1, 2, figsize=(18, 10))

    # Panel A: LinCytoSig wins
    ax = axes[0]
    labels = [f'{r["celltype"]}__{r["cytokine"]}\n({r["category"].split("_")[0]})' for _, r in top_lin.iterrows()]
    x_pos = range(len(top_lin))
    ax.barh(x_pos, top_lin['lin_rho'].values, color=COLORS['lincytosig'], alpha=0.8, label='LinCytoSig')
    ax.barh(x_pos, top_lin['cyto_rho'].values, color=COLORS['cytosig'], alpha=0.5, label='CytoSig')
    ax.set_yticks(x_pos)
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel('Spearman ρ')
    ax.set_title('A. LinCytoSig Best Advantages\n(Δρ up to +1.6)', fontweight='bold', color=COLORS['lincytosig'])
    ax.legend(fontsize=9)
    for i, (_, row) in enumerate(top_lin.iterrows()):
        ax.text(max(row['lin_rho'], row['cyto_rho']) + 0.02, i,
                f'Δ={row["diff"]:+.3f}', va='center', fontsize=7, fontweight='bold')

    # Panel B: CytoSig wins
    ax = axes[1]
    labels = [f'{r["celltype"]}__{r["cytokine"]}\n({r["category"].split("_")[0]})' for _, r in top_cyto.iterrows()]
    x_pos = range(len(top_cyto))
    ax.barh(x_pos, top_cyto['cyto_rho'].values, color=COLORS['cytosig'], alpha=0.8, label='CytoSig')
    ax.barh(x_pos, top_cyto['lin_rho'].values, color=COLORS['lincytosig'], alpha=0.5, label='LinCytoSig')
    ax.set_yticks(x_pos)
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel('Spearman ρ')
    ax.set_title('B. CytoSig Best Advantages\n(Δρ up to -1.5)', fontweight='bold', color=COLORS['cytosig'])
    ax.legend(fontsize=9)
    for i, (_, row) in enumerate(top_cyto.iterrows()):
        ax.text(max(row['lin_rho'], row['cyto_rho']) + 0.02, i,
                f'Δ={row["diff"]:+.3f}', va='center', fontsize=7, fontweight='bold')

    fig.suptitle('Cell-Type Specificity: When LinCytoSig Wins vs CytoSig Wins',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(FIG_DIR / 'fig13_lincytosig_specificity.png')
    fig.savefig(FIG_DIR / 'fig13_lincytosig_specificity.pdf')
    plt.close(fig)
    print('  ✓ Figure 13: LinCytoSig specificity deep dive')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 14: Celltype-Level Scatter Examples (Good vs Bad)
# ═══════════════════════════════════════════════════════════════════════════════
def fig14_celltype_scatter_examples(df):
    """Show celltype-level scatter plots for selected targets."""
    # Pick a few key targets
    key_targets = ['IFNG', 'TGFB1', 'IL1B', 'IL6']

    fig, axes = plt.subplots(len(key_targets), 3, figsize=(18, 5 * len(key_targets)))

    for row_idx, target in enumerate(key_targets):
        for col_idx, (atlas, level, label) in enumerate([
            ('cima', 'donor_l1', 'CIMA (L1)'),
            ('inflammation_main', 'donor_l1', 'Inflammation Main (L1)'),
            ('scatlas_normal', 'donor_organ_celltype1', 'scAtlas Normal (Celltype1)'),
        ]):
            ax = axes[row_idx, col_idx] if len(key_targets) > 1 else axes[col_idx]

            # Load scatter data
            filename = f'{atlas}_{level}_cytosig.json'
            scatter_data = load_scatter(SCATTER_CT, filename)

            if scatter_data is None:
                # Try alternate filename patterns
                alt_names = [f'{atlas}_l1_cytosig.json', f'{atlas}_celltype_cytosig.json']
                for alt in alt_names:
                    scatter_data = load_scatter(SCATTER_CT, alt)
                    if scatter_data is not None:
                        break

            if scatter_data and target in scatter_data:
                data = scatter_data[target]
                points = data.get('points', [])
                rho = data.get('rho', np.nan)

                if points:
                    x = [p[0] for p in points]
                    y = [p[1] for p in points]

                    # Color by celltype if available
                    celltypes = data.get('celltypes', [])
                    if celltypes and len(points) > 0 and len(points[0]) > 2:
                        ct_indices = [p[2] if len(p) > 2 else 0 for p in points]
                        unique_cts = sorted(set(ct_indices))
                        cmap = plt.cm.Set3(np.linspace(0, 1, max(len(unique_cts), 1)))
                        for ct_idx in unique_cts:
                            mask = [i for i, c in enumerate(ct_indices) if c == ct_idx]
                            ct_name = celltypes[int(ct_idx)] if int(ct_idx) < len(celltypes) else f'CT{ct_idx}'
                            ax.scatter([x[i] for i in mask], [y[i] for i in mask],
                                      alpha=0.5, s=15, label=ct_name[:15], color=cmap[int(ct_idx) % len(cmap)])
                    else:
                        color = COLORS['good'] if rho > 0.2 else COLORS['bad'] if rho < 0 else COLORS['neutral']
                        ax.scatter(x, y, alpha=0.4, s=15, c=color)

                ax.set_title(f'{target} — {label}\nρ = {rho:.3f}', fontweight='bold', fontsize=10)
            else:
                ax.text(0.5, 0.5, f'{target}\nNo data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{target} — {label}', fontsize=10)

            ax.set_xlabel('Mean Expression')
            ax.set_ylabel('Predicted Activity')

    fig.suptitle('Cell-Type-Level Validation: Key Cytokine Targets\n(Each point = celltype × donor pseudobulk)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIG_DIR / 'fig14_celltype_scatter_examples.png')
    fig.savefig(FIG_DIR / 'fig14_celltype_scatter_examples.pdf')
    plt.close(fig)
    print('  ✓ Figure 14: Celltype-level scatter examples')


# ═══════════════════════════════════════════════════════════════════════════════
# FIGURE 15: Summary Statistics Table
# ═══════════════════════════════════════════════════════════════════════════════
def fig15_summary_table(df):
    """Publication-ready summary statistics table."""
    level_map = {
        'cima': 'donor_only', 'inflammation_main': 'donor_only',
        'inflammation_val': 'donor_only', 'inflammation_ext': 'donor_only',
        'scatlas_normal': 'donor_organ', 'scatlas_cancer': 'donor_organ',
        'gtex': 'donor_only', 'tcga': 'donor_only',
    }

    rows = []
    for atlas in level_map:
        level = level_map[atlas]
        for sig_type in ['cytosig', 'lincytosig', 'secact']:
            sub = df[(df['atlas'] == atlas) & (df['level'] == level) & (df['signature'] == sig_type)]
            if len(sub) == 0:
                continue
            rhos = sub['spearman_rho'].dropna()
            n_sig = (sub['spearman_pval'] < 0.05).sum()
            n_pos = (rhos > 0).sum()
            rows.append({
                'Atlas': atlas.replace('_', ' ').title(),
                'Signature': sig_type.upper(),
                'N Targets': len(rhos),
                'Median ρ': f'{rhos.median():.3f}',
                'Mean ρ': f'{rhos.mean():.3f}',
                'Std ρ': f'{rhos.std():.3f}',
                'Min ρ': f'{rhos.min():.3f}',
                'Max ρ': f'{rhos.max():.3f}',
                '% Significant': f'{100 * n_sig / len(sub):.1f}%',
                '% Positive': f'{100 * n_pos / len(rhos):.1f}%',
            })

    table_df = pd.DataFrame(rows)

    # Tight figure height: header + rows + padding
    row_height = 0.3
    fig_height = (len(rows) + 1) * row_height + 1.5
    fig, ax = plt.subplots(figsize=(18, fig_height))
    ax.axis('off')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    table = ax.table(cellText=table_df.values, colLabels=table_df.columns,
                     cellLoc='center', loc='upper center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)  # Make rows taller for readability
    table.auto_set_column_width(col=list(range(len(table_df.columns))))

    # Color header
    for j in range(len(table_df.columns)):
        table[0, j].set_facecolor('#374151')
        table[0, j].set_text_props(color='white', fontweight='bold')

    # Color rows by signature type
    for i in range(len(rows)):
        sig = rows[i]['Signature']
        if sig == 'CYTOSIG':
            bg = '#DBEAFE'
        elif sig == 'LINCYTOSIG':
            bg = '#FEF3C7'
        else:
            bg = '#D1FAE5'
        for j in range(len(table_df.columns)):
            table[i + 1, j].set_facecolor(bg)

    ax.set_title('Cross-Sample Validation Summary Statistics\n(Donor-Level Pseudobulk Spearman Correlation)',
                 fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    fig.savefig(FIG_DIR / 'fig15_summary_table.png')
    fig.savefig(FIG_DIR / 'fig15_summary_table.pdf')
    plt.close(fig)

    # Also save as CSV
    table_df.to_csv(FIG_DIR / 'summary_statistics.csv', index=False)
    print('  ✓ Figure 15: Summary statistics table')


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    print('CytoAtlas PI Report — Figure Generation')
    print('=' * 60)

    print('\nLoading correlation data...')
    df = load_all_correlations()
    print(f'  Loaded {len(df)} correlation records')
    print(f'  Atlases: {df["atlas"].nunique()} | Signatures: {df["signature"].nunique()} | Targets: {df["target"].nunique()}')

    print('\nGenerating figures...')
    fig1_dataset_overview()
    fig2_correlation_summary(df)
    fig3_good_bad_correlations(df)
    fig4_bio_targets_heatmap(df)
    fig5_representative_scatter(df)
    fig6_cross_atlas_consistency(df)
    fig7_validation_levels(df)
    fig8_method_comparison()
    fig9_lincytosig_vs_cytosig()
    fig10_lincytosig_advantage()
    fig11_secact_novel(df)
    fig12_bulk_validation(df)
    fig13_lincytosig_specificity()
    fig14_celltype_scatter_examples(df)
    fig15_summary_table(df)

    print('\n' + '=' * 60)
    print(f'All figures saved to: {FIG_DIR}/')
    print('Done!')


if __name__ == '__main__':
    main()
