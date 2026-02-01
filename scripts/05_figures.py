#!/usr/bin/env python3
"""
Publication Figure Generation
==============================
Generate publication-quality figures for the Pan-Disease Cytokine Activity Atlas.

Figures:
1. Overview schema with data summary
2. Cytokine activity heatmaps (cell types × cytokines × diseases)
3. Cross-disease comparison (shared vs unique programs)
4. Treatment response prediction (ROC curves, feature importance)
5. Cytokine-metabolome correlations (CIMA)
6. Volcano plots for differential analyses

Requirements:
- matplotlib >= 3.5
- seaborn >= 0.12
- scipy
- pandas
- numpy
"""

import os
import sys
import json
import warnings
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from scipy import stats
from scipy.cluster import hierarchy

# Suppress warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# Configuration
# ==============================================================================

# Input paths
RESULTS_DIR = Path('/data/parks34/projects/2secactpy/results')
CIMA_DIR = RESULTS_DIR / 'cima'
INFLAM_DIR = RESULTS_DIR / 'inflammation'
SCATLAS_DIR = RESULTS_DIR / 'scatlas'
INTEGRATED_DIR = RESULTS_DIR / 'integrated'

# Output paths
FIGURES_DIR = RESULTS_DIR / 'figures'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Figure settings
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

# Color palettes
DISEASE_COLORS = {
    'healthy': '#4DAF4A',
    'RA': '#E41A1C',
    'PSA': '#FF7F00',
    'CD': '#377EB8',
    'UC': '#984EA3',
    'PS': '#FFFF33',
    'COVID': '#A65628',
    'COPD': '#F781BF',
    'SLE': '#999999',
    'MS': '#66C2A5',
}

CELLTYPE_COLORS = {
    'B cells': '#1f77b4',
    'CD4 T': '#ff7f0e',
    'CD8 T': '#2ca02c',
    'NK': '#d62728',
    'Monocytes': '#9467bd',
    'DC': '#8c564b',
    'Tregs': '#e377c2',
}


def log(msg: str):
    """Print timestamped log message."""
    import time
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


# ==============================================================================
# Figure 1: Overview Schema
# ==============================================================================

def figure1_overview_schema():
    """
    Create overview schema showing multi-atlas integration.
    """
    log("Generating Figure 1: Overview Schema...")

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Data summary boxes
    boxes = [
        {'x': 1, 'y': 6, 'w': 3, 'h': 1.5, 'label': 'CIMA Atlas\n6.5M cells\n428 healthy donors',
         'color': '#4DAF4A', 'extra': 'Metabolomics\nBiochemistry'},
        {'x': 5, 'y': 6, 'w': 3, 'h': 1.5, 'label': 'Inflammation Atlas\n6.3M cells\n20 diseases',
         'color': '#E41A1C', 'extra': 'Treatment Response\n(n=208)'},
        {'x': 9, 'y': 6, 'w': 3, 'h': 1.5, 'label': 'scAtlas\n6.4M cells\n35 organs + cancer',
         'color': '#377EB8', 'extra': 'Pre-computed\nactivities'},
    ]

    for box in boxes:
        rect = mpatches.FancyBboxPatch(
            (box['x'], box['y']), box['w'], box['h'],
            boxstyle="round,pad=0.1", facecolor=box['color'], alpha=0.3,
            edgecolor=box['color'], linewidth=2
        )
        ax.add_patch(rect)
        ax.text(box['x'] + box['w']/2, box['y'] + box['h']/2 + 0.2,
                box['label'], ha='center', va='center', fontsize=10, fontweight='bold')
        ax.text(box['x'] + box['w']/2, box['y'] - 0.3,
                box['extra'], ha='center', va='top', fontsize=8, style='italic')

    # Arrow to SecActpy
    ax.annotate('', xy=(6, 4.5), xytext=(6, 5.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    # SecActpy box
    rect = mpatches.FancyBboxPatch(
        (4, 3.5), 4, 1, boxstyle="round,pad=0.1",
        facecolor='#FFD700', alpha=0.5, edgecolor='#B8860B', linewidth=2
    )
    ax.add_patch(rect)
    ax.text(6, 4, 'SecActpy Ridge Regression\nCytoSig (44) + SecAct (1,249)',
            ha='center', va='center', fontsize=10, fontweight='bold')

    # Output arrows
    outputs = [
        {'x': 2, 'y': 2, 'label': 'Cytokine Activity\nAtlas'},
        {'x': 5, 'y': 2, 'label': 'Disease-Specific\nSignatures'},
        {'x': 8, 'y': 2, 'label': 'Treatment Response\nPrediction'},
        {'x': 11, 'y': 2, 'label': 'Metabolome\nCorrelations'},
    ]

    for out in outputs:
        ax.annotate('', xy=(out['x'], out['y'] + 0.8), xytext=(6, 3.5),
                    arrowprops=dict(arrowstyle='->', lw=1.5, color='gray'))
        rect = mpatches.FancyBboxPatch(
            (out['x'] - 1.2, out['y'] - 0.5), 2.4, 1,
            boxstyle="round,pad=0.1", facecolor='lightgray', alpha=0.5,
            edgecolor='gray', linewidth=1
        )
        ax.add_patch(rect)
        ax.text(out['x'], out['y'], out['label'],
                ha='center', va='center', fontsize=9)

    # Title
    ax.text(6, 7.8, 'Pan-Disease Single-Cell Cytokine Activity Atlas',
            ha='center', va='center', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig1_overview_schema.pdf')
    plt.savefig(FIGURES_DIR / 'fig1_overview_schema.png')
    plt.close()
    log("  Saved: fig1_overview_schema.pdf/png")


# ==============================================================================
# Figure 2: Cytokine Activity Heatmap
# ==============================================================================

def figure2_activity_heatmap():
    """
    Create heatmap of cytokine activities across cell types and diseases.
    """
    log("Generating Figure 2: Activity Heatmap...")

    # Try to load disease differential data
    disease_diff_path = INFLAM_DIR / 'disease_differential.csv'
    if not disease_diff_path.exists():
        log("  Warning: disease_differential.csv not found, skipping")
        return

    diff_df = pd.read_csv(disease_diff_path)

    # Filter to CytoSig and significant results
    cyto_df = diff_df[
        (diff_df['signature'] == 'CytoSig') &
        (diff_df['qvalue'] < 0.05)
    ].copy()

    if len(cyto_df) == 0:
        log("  Warning: No significant results, skipping")
        return

    # Pivot to matrix
    pivot_df = cyto_df.pivot_table(
        index='protein',
        columns='disease',
        values='activity_diff',
        aggfunc='mean'
    ).fillna(0)

    # Cluster rows and columns
    if pivot_df.shape[0] > 2 and pivot_df.shape[1] > 2:
        row_linkage = hierarchy.linkage(pivot_df.values, method='ward')
        col_linkage = hierarchy.linkage(pivot_df.values.T, method='ward')

        row_order = hierarchy.leaves_list(row_linkage)
        col_order = hierarchy.leaves_list(col_linkage)

        pivot_df = pivot_df.iloc[row_order, col_order]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Custom colormap (blue-white-red)
    cmap = LinearSegmentedColormap.from_list('bwr', ['#2166AC', 'white', '#B2182B'])

    # Heatmap
    vmax = max(abs(pivot_df.values.min()), abs(pivot_df.values.max()))
    vmax = min(vmax, 3)  # Cap at +/- 3

    sns.heatmap(
        pivot_df, ax=ax, cmap=cmap, center=0, vmin=-vmax, vmax=vmax,
        cbar_kws={'label': 'log2 Fold Change vs Healthy', 'shrink': 0.6},
        xticklabels=True, yticklabels=True
    )

    ax.set_xlabel('Disease', fontsize=12)
    ax.set_ylabel('Cytokine', fontsize=12)
    ax.set_title('Cytokine Activity Changes Across Diseases\n(FDR < 0.05)', fontsize=14)

    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig2_activity_heatmap.pdf')
    plt.savefig(FIGURES_DIR / 'fig2_activity_heatmap.png')
    plt.close()
    log("  Saved: fig2_activity_heatmap.pdf/png")


# ==============================================================================
# Figure 3: Treatment Response ROC Curves
# ==============================================================================

def figure3_treatment_response():
    """
    Create ROC curves for treatment response prediction.
    """
    log("Generating Figure 3: Treatment Response Prediction...")

    # Load prediction summary
    pred_path = INFLAM_DIR / 'treatment_prediction_summary.csv'
    if not pred_path.exists():
        log("  Warning: treatment_prediction_summary.csv not found, skipping")
        return

    pred_df = pd.read_csv(pred_path)

    # Filter to diseases with good AUC
    good_pred = pred_df[pred_df['best_auc'] >= 0.6].copy()

    if len(good_pred) == 0:
        log("  Warning: No good predictions found, skipping")
        return

    # Create bar plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Sort by AUC
    good_pred = good_pred.sort_values('best_auc', ascending=True)

    diseases = good_pred['disease'].tolist()
    aucs = good_pred['best_auc'].tolist()
    colors = ['#E41A1C' if d == 'all' else '#377EB8' for d in diseases]

    bars = ax.barh(range(len(diseases)), aucs, color=colors, alpha=0.7, edgecolor='black')

    ax.set_yticks(range(len(diseases)))
    ax.set_yticklabels(diseases)
    ax.set_xlabel('AUC-ROC', fontsize=12)
    ax.set_title('Treatment Response Prediction Performance', fontsize=14)
    ax.axvline(x=0.7, color='green', linestyle='--', label='AUC = 0.7', alpha=0.7)
    ax.axvline(x=0.5, color='gray', linestyle=':', label='Random', alpha=0.7)
    ax.set_xlim(0.4, 1.0)
    ax.legend(loc='lower right')

    # Add value labels
    for i, (bar, auc) in enumerate(zip(bars, aucs)):
        ax.text(auc + 0.02, i, f'{auc:.2f}', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig3_treatment_response.pdf')
    plt.savefig(FIGURES_DIR / 'fig3_treatment_response.png')
    plt.close()
    log("  Saved: fig3_treatment_response.pdf/png")


# ==============================================================================
# Figure 4: Volcano Plots
# ==============================================================================

def figure4_volcano_plots():
    """
    Create volcano plots for disease differential analysis.
    """
    log("Generating Figure 4: Volcano Plots...")

    disease_diff_path = INFLAM_DIR / 'disease_differential.csv'
    if not disease_diff_path.exists():
        log("  Warning: disease_differential.csv not found, skipping")
        return

    diff_df = pd.read_csv(disease_diff_path)

    # Get top 4 diseases by number of significant hits
    cyto_df = diff_df[diff_df['signature'] == 'CytoSig'].copy()
    disease_counts = cyto_df[cyto_df['qvalue'] < 0.05].groupby('disease').size()
    top_diseases = disease_counts.nlargest(4).index.tolist()

    if len(top_diseases) == 0:
        log("  Warning: No significant diseases, skipping")
        return

    # Create subplot grid
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, disease in enumerate(top_diseases):
        ax = axes[i]
        disease_data = cyto_df[cyto_df['disease'] == disease].copy()

        if len(disease_data) == 0:
            continue

        # Calculate -log10(pvalue)
        disease_data['neg_log_p'] = -np.log10(disease_data['pvalue'].clip(lower=1e-50))

        # Color by significance
        colors = []
        for _, row in disease_data.iterrows():
            if row['qvalue'] < 0.05:
                if row['activity_diff'] > 0.5:
                    colors.append('#E41A1C')  # Up, red
                elif row['activity_diff'] < -0.5:
                    colors.append('#377EB8')  # Down, blue
                else:
                    colors.append('gray')
            else:
                colors.append('lightgray')

        ax.scatter(disease_data['activity_diff'], disease_data['neg_log_p'],
                   c=colors, alpha=0.7, s=50, edgecolors='black', linewidths=0.5)

        # Add significance lines
        ax.axhline(y=-np.log10(0.05), color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=-0.5, color='gray', linestyle='--', alpha=0.5)

        # Label top hits
        top_hits = disease_data.nlargest(5, 'neg_log_p')
        for _, row in top_hits.iterrows():
            ax.annotate(row['protein'], (row['activity_diff'], row['neg_log_p']),
                        fontsize=8, ha='center', va='bottom')

        ax.set_xlabel('log2 Fold Change', fontsize=10)
        ax.set_ylabel('-log10(p-value)', fontsize=10)
        ax.set_title(f'{disease}', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig4_volcano_plots.pdf')
    plt.savefig(FIGURES_DIR / 'fig4_volcano_plots.png')
    plt.close()
    log("  Saved: fig4_volcano_plots.pdf/png")


# ==============================================================================
# Figure 5: Metabolome Correlations (CIMA)
# ==============================================================================

def figure5_metabolome_correlations():
    """
    Create correlation network/heatmap for cytokine-metabolome associations.
    """
    log("Generating Figure 5: Metabolome Correlations...")

    corr_path = CIMA_DIR / 'CIMA_correlation_metabolites.csv'
    if not corr_path.exists():
        log("  Warning: CIMA_correlation_metabolites.csv not found, skipping")
        return

    corr_df = pd.read_csv(corr_path)

    # Filter to significant correlations
    sig_corr = corr_df[
        (corr_df['qvalue'] < 0.05) &
        (corr_df['signature'] == 'CytoSig')
    ].copy()

    if len(sig_corr) < 10:
        log("  Warning: Too few significant correlations, skipping")
        return

    # Get top correlations (by absolute rho)
    sig_corr['abs_rho'] = sig_corr['rho'].abs()
    top_corr = sig_corr.nlargest(50, 'abs_rho')

    # Create pivot for heatmap
    pivot = top_corr.pivot_table(
        index='protein', columns='feature', values='rho', aggfunc='mean'
    ).fillna(0)

    # Select top proteins and metabolites
    top_proteins = pivot.abs().mean(axis=1).nlargest(15).index.tolist()
    top_features = pivot.abs().mean(axis=0).nlargest(20).index.tolist()
    pivot_sub = pivot.loc[top_proteins, top_features]

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))

    cmap = LinearSegmentedColormap.from_list('correlation', ['#2166AC', 'white', '#B2182B'])

    sns.heatmap(
        pivot_sub, ax=ax, cmap=cmap, center=0, vmin=-0.5, vmax=0.5,
        cbar_kws={'label': 'Spearman ρ', 'shrink': 0.6},
        xticklabels=True, yticklabels=True, annot=False
    )

    ax.set_xlabel('Metabolite', fontsize=12)
    ax.set_ylabel('Cytokine', fontsize=12)
    ax.set_title('Cytokine-Metabolome Correlations (CIMA)\n(FDR < 0.05)', fontsize=14)

    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=9)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig5_metabolome_correlations.pdf')
    plt.savefig(FIGURES_DIR / 'fig5_metabolome_correlations.png')
    plt.close()
    log("  Saved: fig5_metabolome_correlations.pdf/png")


# ==============================================================================
# Figure 6: Cross-Disease Comparison (Shared vs Unique)
# ==============================================================================

def figure6_cross_disease_comparison():
    """
    Create UpSet-style plot showing shared vs disease-specific signatures.
    """
    log("Generating Figure 6: Cross-Disease Comparison...")

    disease_diff_path = INFLAM_DIR / 'disease_differential.csv'
    if not disease_diff_path.exists():
        log("  Warning: disease_differential.csv not found, skipping")
        return

    diff_df = pd.read_csv(disease_diff_path)

    # Get significant signatures per disease
    sig_df = diff_df[
        (diff_df['qvalue'] < 0.05) &
        (diff_df['signature'] == 'CytoSig') &
        (diff_df['activity_diff'].abs() > 0.5)
    ].copy()

    if len(sig_df) == 0:
        log("  Warning: No significant signatures, skipping")
        return

    # Count diseases per protein
    protein_disease_counts = sig_df.groupby('protein')['disease'].nunique()

    # Categorize
    unique_proteins = protein_disease_counts[protein_disease_counts == 1].index.tolist()
    shared_2_3 = protein_disease_counts[(protein_disease_counts >= 2) & (protein_disease_counts <= 3)].index.tolist()
    shared_4_plus = protein_disease_counts[protein_disease_counts >= 4].index.tolist()

    # Create bar plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Category counts
    ax1 = axes[0]
    categories = ['Disease-Specific\n(1 disease)', 'Shared\n(2-3 diseases)', 'Core Inflammatory\n(≥4 diseases)']
    counts = [len(unique_proteins), len(shared_2_3), len(shared_4_plus)]
    colors = ['#4DAF4A', '#377EB8', '#E41A1C']

    bars = ax1.bar(categories, counts, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Number of Cytokines', fontsize=12)
    ax1.set_title('Cytokine Program Classification', fontsize=14)

    for bar, count in zip(bars, counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 str(count), ha='center', fontsize=11, fontweight='bold')

    # Right: List core inflammatory cytokines
    ax2 = axes[1]
    ax2.axis('off')

    if len(shared_4_plus) > 0:
        text = "Core Inflammatory Cytokines\n(elevated in ≥4 diseases):\n\n"
        text += "\n".join([f"• {p}" for p in sorted(shared_4_plus)[:15]])
        ax2.text(0.1, 0.9, text, transform=ax2.transAxes, fontsize=11,
                 verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fig6_cross_disease_comparison.pdf')
    plt.savefig(FIGURES_DIR / 'fig6_cross_disease_comparison.png')
    plt.close()
    log("  Saved: fig6_cross_disease_comparison.pdf/png")


# ==============================================================================
# Summary Statistics Table
# ==============================================================================

def generate_summary_table():
    """
    Generate summary statistics table for manuscript.
    """
    log("Generating Summary Statistics Table...")

    summary = {
        'Atlas': [],
        'Cells': [],
        'Samples': [],
        'Diseases': [],
        'CytoSig Significant': [],
        'SecAct Significant': [],
    }

    # CIMA
    summary['Atlas'].append('CIMA')
    summary['Cells'].append('6,484,974')
    summary['Samples'].append('428')
    summary['Diseases'].append('Healthy only')

    corr_path = CIMA_DIR / 'CIMA_correlation_biochemistry.csv'
    if corr_path.exists():
        corr_df = pd.read_csv(corr_path)
        cyto_sig = len(corr_df[(corr_df['signature'] == 'CytoSig') & (corr_df['qvalue'] < 0.05)])
        secact_sig = len(corr_df[(corr_df['signature'] == 'SecAct') & (corr_df['qvalue'] < 0.05)])
        summary['CytoSig Significant'].append(cyto_sig)
        summary['SecAct Significant'].append(secact_sig)
    else:
        summary['CytoSig Significant'].append('N/A')
        summary['SecAct Significant'].append('N/A')

    # Inflammation Atlas
    summary['Atlas'].append('Inflammation Atlas')
    summary['Cells'].append('6,340,934')
    summary['Samples'].append('1,047')
    summary['Diseases'].append('20')

    diff_path = INFLAM_DIR / 'disease_differential.csv'
    if diff_path.exists():
        diff_df = pd.read_csv(diff_path)
        cyto_sig = len(diff_df[(diff_df['signature'] == 'CytoSig') & (diff_df['qvalue'] < 0.05)])
        secact_sig = len(diff_df[(diff_df['signature'] == 'SecAct') & (diff_df['qvalue'] < 0.05)])
        summary['CytoSig Significant'].append(cyto_sig)
        summary['SecAct Significant'].append(secact_sig)
    else:
        summary['CytoSig Significant'].append('N/A')
        summary['SecAct Significant'].append('N/A')

    # scAtlas
    summary['Atlas'].append('scAtlas')
    summary['Cells'].append('6,440,926')
    summary['Samples'].append('Multiple')
    summary['Diseases'].append('35 organs + cancers')

    organ_path = SCATLAS_DIR / 'normal_organ_signatures.csv'
    if organ_path.exists():
        organ_df = pd.read_csv(organ_path)
        cyto_sig = len(organ_df[organ_df['signature_type'] == 'CytoSig'])
        secact_sig = len(organ_df[organ_df['signature_type'] == 'SecAct'])
        summary['CytoSig Significant'].append(cyto_sig)
        summary['SecAct Significant'].append(secact_sig)
    else:
        summary['CytoSig Significant'].append('N/A')
        summary['SecAct Significant'].append('N/A')

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(FIGURES_DIR / 'table_summary_statistics.csv', index=False)
    log(f"  Saved: table_summary_statistics.csv")

    return summary_df


# ==============================================================================
# Main Entry Point
# ==============================================================================

def main():
    """Generate all publication figures."""
    import argparse

    parser = argparse.ArgumentParser(description='Generate Publication Figures')
    parser.add_argument('--figure', type=int, choices=[1, 2, 3, 4, 5, 6],
                        help='Generate specific figure only')
    parser.add_argument('--all', action='store_true', default=True,
                        help='Generate all figures')
    args = parser.parse_args()

    log("=" * 60)
    log("PUBLICATION FIGURE GENERATION")
    log("=" * 60)
    log(f"Output directory: {FIGURES_DIR}")

    figure_functions = {
        1: figure1_overview_schema,
        2: figure2_activity_heatmap,
        3: figure3_treatment_response,
        4: figure4_volcano_plots,
        5: figure5_metabolome_correlations,
        6: figure6_cross_disease_comparison,
    }

    if args.figure:
        log(f"\nGenerating Figure {args.figure} only...")
        figure_functions[args.figure]()
    else:
        log("\nGenerating all figures...")
        for fig_num, func in figure_functions.items():
            try:
                func()
            except Exception as e:
                log(f"  Error in Figure {fig_num}: {e}")

    # Generate summary table
    generate_summary_table()

    log("\nFigure generation complete!")
    log(f"Output directory: {FIGURES_DIR}")


if __name__ == '__main__':
    main()
