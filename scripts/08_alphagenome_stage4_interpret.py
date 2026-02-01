#!/usr/bin/env python3
"""
AlphaGenome Stage 4: Interpret and Score Predictions
======================================================
Analyze AlphaGenome predictions to prioritize regulatory variants.

Analysis:
1. Compute regulatory impact scores from multimodal predictions
2. Classify mechanism (enhancer/promoter/splice/TF binding)
3. Check direction concordance: AlphaGenome prediction vs observed eQTL beta
4. Prioritize variants with >99th percentile quantile scores

Input:
- results/alphagenome/stage3_predictions.h5ad

Output:
- results/alphagenome/stage4_scored_variants.csv
- results/alphagenome/stage4_prioritized.csv (high-confidence set)
- results/alphagenome/stage4_summary.json
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import anndata as ad
from scipy import stats

# ==============================================================================
# Configuration
# ==============================================================================

INPUT_DIR = Path('/vf/users/parks34/projects/2secactpy/results/alphagenome')
OUTPUT_DIR = INPUT_DIR

# Quantile thresholds for prioritization
HIGH_CONFIDENCE_QUANTILE = 0.99
MODERATE_CONFIDENCE_QUANTILE = 0.95

# Track patterns for mechanism classification
MECHANISM_PATTERNS = {
    'enhancer': {
        'positive': ['H3K27ac', 'H3K4me1', 'ATAC', 'DNase', 'EP300'],
        'negative': ['H3K4me3'],  # Promoter mark
        'description': 'Active enhancer marks (H3K27ac, H3K4me1) with accessibility'
    },
    'promoter': {
        'positive': ['H3K4me3', 'H3K27ac', 'TSS', 'promoter'],
        'negative': [],
        'description': 'Promoter marks (H3K4me3) near TSS'
    },
    'tf_binding': {
        'positive': ['ChIP', 'TF', 'CTCF', 'RUNX', 'PAX', 'PU1', 'SPI1', 'GATA', 'NF'],
        'negative': [],
        'description': 'Transcription factor binding disruption'
    },
    'repressive': {
        'positive': ['H3K27me3', 'H3K9me3', 'EZH2', 'Polycomb'],
        'negative': [],
        'description': 'Repressive chromatin marks'
    },
}


def log(msg: str):
    """Print timestamped log message."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def compute_impact_score(
    track_diffs: np.ndarray,
    track_names: List[str]
) -> Tuple[float, Dict[str, float]]:
    """
    Compute overall regulatory impact score from track differences.

    Uses absolute differences with track-type weighting.

    Returns:
        (overall_score, breakdown_dict)
    """
    # Track type weights (epigenetic marks tend to be more reliable)
    type_weights = {
        'H3K27ac': 2.0,    # Active enhancer
        'H3K4me3': 2.0,    # Active promoter
        'H3K4me1': 1.5,    # Enhancer mark
        'ATAC': 1.5,       # Accessibility
        'DNase': 1.5,      # Accessibility
        'ChIP': 1.0,       # TF binding
        'CTCF': 1.5,       # Insulator
        'Expression': 1.0,  # Expression
    }

    weighted_sum = 0.0
    weight_total = 0.0
    breakdown = {}

    for i, track in enumerate(track_names):
        abs_diff = abs(track_diffs[i])

        # Get weight based on track type
        weight = 1.0
        for pattern, w in type_weights.items():
            if pattern.lower() in track.lower():
                weight = w
                break

        weighted_sum += abs_diff * weight
        weight_total += weight

        # Group by track type for breakdown
        track_type = track.split('_')[0] if '_' in track else track
        if track_type not in breakdown:
            breakdown[track_type] = []
        breakdown[track_type].append(abs_diff)

    # Average breakdown values
    for k, v in breakdown.items():
        breakdown[k] = np.mean(v)

    overall_score = weighted_sum / weight_total if weight_total > 0 else 0.0
    return overall_score, breakdown


def classify_mechanism(
    track_diffs: np.ndarray,
    track_names: List[str],
    threshold: float = 0.1
) -> Tuple[str, float]:
    """
    Classify regulatory mechanism based on track patterns.

    Returns:
        (mechanism, confidence_score)
    """
    mechanism_scores = {}

    for mechanism, patterns in MECHANISM_PATTERNS.items():
        positive_score = 0.0
        negative_penalty = 0.0

        for i, track in enumerate(track_names):
            track_lower = track.lower()
            abs_diff = abs(track_diffs[i])

            if abs_diff < threshold:
                continue

            # Check positive patterns
            for pattern in patterns['positive']:
                if pattern.lower() in track_lower:
                    positive_score += abs_diff
                    break

            # Check negative patterns
            for pattern in patterns['negative']:
                if pattern.lower() in track_lower:
                    negative_penalty += abs_diff
                    break

        mechanism_scores[mechanism] = positive_score - negative_penalty * 0.5

    if not mechanism_scores:
        return 'unknown', 0.0

    best_mechanism = max(mechanism_scores, key=mechanism_scores.get)
    best_score = mechanism_scores[best_mechanism]

    return best_mechanism, best_score


def check_direction_concordance(
    eqtl_beta: float,
    track_diffs: np.ndarray,
    track_names: List[str]
) -> Tuple[bool, float]:
    """
    Check if AlphaGenome predictions agree with eQTL direction.

    Positive eQTL beta = ALT increases expression
    Positive AlphaGenome diff (for activation marks) should also indicate increased activity

    Returns:
        (concordant, agreement_score)
    """
    # Focus on expression-relevant tracks
    expression_patterns = ['expression', 'h3k27ac', 'atac', 'dnase', 'h3k4me3']

    concordant_count = 0
    total_count = 0

    for i, track in enumerate(track_names):
        track_lower = track.lower()
        diff = track_diffs[i]

        if abs(diff) < 0.05:  # Skip small effects
            continue

        for pattern in expression_patterns:
            if pattern in track_lower:
                total_count += 1

                # Positive eQTL = increased expression
                # Positive diff in activation marks = increased activity
                if (eqtl_beta > 0 and diff > 0) or (eqtl_beta < 0 and diff < 0):
                    concordant_count += 1
                break

    if total_count == 0:
        return True, 0.5  # Neutral if no relevant tracks

    agreement = concordant_count / total_count
    concordant = agreement > 0.5

    return concordant, agreement


def main():
    log("=" * 60)
    log("ALPHAGENOME STAGE 4: INTERPRET AND SCORE PREDICTIONS")
    log("=" * 60)

    # Load Stage 3 predictions
    input_path = INPUT_DIR / 'stage3_predictions.h5ad'
    log(f"Loading: {input_path}")
    adata = ad.read_h5ad(input_path)
    log(f"  Variants: {adata.n_obs}")
    log(f"  Tracks: {adata.n_vars}")

    # Get matrices
    diff_matrix = adata.X
    track_names = list(adata.var_names)
    variants_df = adata.obs.copy()

    log("\nScoring variants...")

    # Initialize result columns
    results = []

    for i in range(adata.n_obs):
        row = variants_df.iloc[i]
        track_diffs = diff_matrix[i, :]

        # Compute impact score
        impact_score, breakdown = compute_impact_score(track_diffs, track_names)

        # Classify mechanism
        mechanism, mechanism_score = classify_mechanism(track_diffs, track_names)

        # Check direction concordance
        eqtl_beta = row.get('eqtl_beta', 0)
        concordant, agreement = check_direction_concordance(
            eqtl_beta, track_diffs, track_names
        )

        # Compute quantile score (max absolute diff across tracks)
        max_abs_diff = np.max(np.abs(track_diffs))

        # Get top affected tracks
        top_track_idx = np.argsort(np.abs(track_diffs))[-5:][::-1]
        top_tracks = [track_names[j] for j in top_track_idx]
        top_diffs = [track_diffs[j] for j in top_track_idx]

        results.append({
            'variant_id': row['variant_id'],
            'impact_score': impact_score,
            'max_track_diff': max_abs_diff,
            'mechanism': mechanism,
            'mechanism_score': mechanism_score,
            'eqtl_concordant': concordant,
            'concordance_score': agreement,
            'n_affected_tracks': np.sum(np.abs(track_diffs) > 0.1),
            'top_track_1': top_tracks[0] if len(top_tracks) > 0 else '',
            'top_diff_1': top_diffs[0] if len(top_diffs) > 0 else 0,
            'top_track_2': top_tracks[1] if len(top_tracks) > 1 else '',
            'top_diff_2': top_diffs[1] if len(top_diffs) > 1 else 0,
            'top_track_3': top_tracks[2] if len(top_tracks) > 2 else '',
            'top_diff_3': top_diffs[2] if len(top_diffs) > 2 else 0,
        })

        if (i + 1) % 500 == 0:
            log(f"  Processed {i + 1}/{adata.n_obs}")

    # Create results DataFrame
    scores_df = pd.DataFrame(results)

    # Merge with original variant info
    merge_cols = ['variant_id', 'CHROM', 'POS', 'REF', 'ALT',
                  'primary_gene', 'celltype', 'eqtl_beta', 'pval_nominal',
                  'variant_type', 'in_cytosig', 'in_secact',
                  'target_genes_str', 'cell_types_str']
    available_cols = [c for c in merge_cols if c in variants_df.columns]
    scores_df = scores_df.merge(variants_df[available_cols], on='variant_id', how='left')

    # Compute quantile scores
    log("\nComputing quantile scores...")
    scores_df['impact_quantile'] = stats.rankdata(scores_df['impact_score']) / len(scores_df)
    scores_df['max_diff_quantile'] = stats.rankdata(scores_df['max_track_diff']) / len(scores_df)

    # Combined priority score
    scores_df['priority_score'] = (
        scores_df['impact_quantile'] * 0.4 +
        scores_df['max_diff_quantile'] * 0.3 +
        scores_df['concordance_score'].fillna(0.5) * 0.2 +
        scores_df['eqtl_concordant'].astype(float) * 0.1
    )

    # Sort by priority score
    scores_df = scores_df.sort_values('priority_score', ascending=False)

    # Save all scored variants
    output_csv = OUTPUT_DIR / 'stage4_scored_variants.csv'
    scores_df.to_csv(output_csv, index=False)
    log(f"\nSaved: {output_csv}")
    log(f"  Total variants: {len(scores_df)}")

    # Filter to high-confidence prioritized variants
    high_confidence = scores_df[
        (scores_df['impact_quantile'] >= HIGH_CONFIDENCE_QUANTILE) |
        (scores_df['max_diff_quantile'] >= HIGH_CONFIDENCE_QUANTILE)
    ].copy()

    moderate_confidence = scores_df[
        ((scores_df['impact_quantile'] >= MODERATE_CONFIDENCE_QUANTILE) |
         (scores_df['max_diff_quantile'] >= MODERATE_CONFIDENCE_QUANTILE)) &
        (scores_df['eqtl_concordant'] == True)
    ].copy()

    # Combine (unique variants)
    prioritized = pd.concat([high_confidence, moderate_confidence]).drop_duplicates('variant_id')
    prioritized = prioritized.sort_values('priority_score', ascending=False)

    prioritized_csv = OUTPUT_DIR / 'stage4_prioritized.csv'
    prioritized.to_csv(prioritized_csv, index=False)
    log(f"\nSaved: {prioritized_csv}")
    log(f"  Prioritized variants: {len(prioritized)}")

    # Mechanism distribution
    log("\nMechanism distribution (prioritized):")
    mech_counts = prioritized['mechanism'].value_counts()
    for mech, count in mech_counts.items():
        log(f"  {mech}: {count}")

    # Gene distribution
    log("\nTop genes by prioritized variant count:")
    if 'primary_gene' in prioritized.columns:
        gene_counts = prioritized['primary_gene'].value_counts()
        for gene in gene_counts.head(10).index:
            log(f"  {gene}: {gene_counts[gene]}")

    # Summary statistics
    summary = {
        'stage': 4,
        'description': 'Interpret and score AlphaGenome predictions',
        'input': {
            'predictions_file': str(input_path),
            'n_variants': adata.n_obs,
            'n_tracks': adata.n_vars,
        },
        'output': {
            'scored_variants': len(scores_df),
            'prioritized_variants': len(prioritized),
            'high_confidence_threshold': HIGH_CONFIDENCE_QUANTILE,
            'moderate_confidence_threshold': MODERATE_CONFIDENCE_QUANTILE,
        },
        'mechanisms': mech_counts.to_dict(),
        'statistics': {
            'mean_impact_score': float(scores_df['impact_score'].mean()),
            'median_impact_score': float(scores_df['impact_score'].median()),
            'mean_max_diff': float(scores_df['max_track_diff'].mean()),
            'concordance_rate': float(scores_df['eqtl_concordant'].mean()),
        },
        'top_variants': prioritized.head(20)[
            ['variant_id', 'primary_gene', 'mechanism', 'impact_score', 'priority_score']
        ].to_dict('records') if 'primary_gene' in prioritized.columns else [],
    }

    summary_path = OUTPUT_DIR / 'stage4_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    log(f"\nSaved: {summary_path}")

    log("\nStage 4 complete!")
    log(f"  High-confidence variants: {len(prioritized)}")

    return scores_df, prioritized


if __name__ == '__main__':
    main()
