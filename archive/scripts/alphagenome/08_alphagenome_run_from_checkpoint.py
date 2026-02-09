#!/usr/bin/env python3
"""
AlphaGenome: Run Stage 4/5 from Checkpoint
============================================
Convert checkpoint predictions to h5ad and run downstream analysis.

This allows running Stage 4/5 on partial results while Stage 3 is still running.

Usage:
    python scripts/08_alphagenome_run_from_checkpoint.py
    python scripts/08_alphagenome_run_from_checkpoint.py --suffix _v2
    python scripts/08_alphagenome_run_from_checkpoint.py --suffix _v2 --stage4-only
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import anndata as ad


# Configuration
RESULTS_DIR = Path('/vf/users/parks34/projects/2secactpy/results/alphagenome')
SCRIPTS_DIR = Path('/vf/users/parks34/projects/2secactpy/scripts')


def log(msg: str):
    """Print timestamped log message."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def load_checkpoint_predictions(suffix: str = '') -> dict:
    """Load predictions from checkpoint file."""
    pred_path = RESULTS_DIR / f'stage3_predictions_checkpoint{suffix}.json'

    if not pred_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {pred_path}")

    with open(pred_path) as f:
        predictions = json.load(f)

    log(f"Loaded {len(predictions)} predictions from checkpoint")
    return predictions


def create_h5ad_from_checkpoint(predictions: dict, suffix: str = '') -> Path:
    """Convert checkpoint predictions to h5ad format."""

    # Load variants metadata
    variants_df = pd.read_csv(RESULTS_DIR / 'stage2_alphagenome_input.csv')

    # Filter to variants with predictions
    pred_variants = set(predictions.keys())
    variants_df = variants_df[variants_df['variant_id'].isin(pred_variants)].copy()
    log(f"Variants with predictions: {len(variants_df)}")

    if len(variants_df) == 0:
        raise ValueError("No variants with predictions found")

    # Collect all tracks
    all_tracks = set()
    for pred in predictions.values():
        if 'tracks' in pred:
            all_tracks.update(pred['tracks'].keys())
    all_tracks = sorted(all_tracks)
    log(f"Tracks: {len(all_tracks)}")

    if len(all_tracks) == 0:
        raise ValueError("No tracks found in predictions")

    # Build matrices
    n_variants = len(variants_df)
    n_tracks = len(all_tracks)

    diff_matrix = np.zeros((n_variants, n_tracks))
    ref_matrix = np.zeros((n_variants, n_tracks))
    alt_matrix = np.zeros((n_variants, n_tracks))

    track_to_idx = {t: i for i, t in enumerate(all_tracks)}

    for i, (_, row) in enumerate(variants_df.iterrows()):
        vid = row['variant_id']
        if vid in predictions and 'tracks' in predictions[vid]:
            for track_name, scores in predictions[vid]['tracks'].items():
                j = track_to_idx[track_name]
                diff_matrix[i, j] = scores.get('diff', 0)
                ref_matrix[i, j] = scores.get('ref_score', 0)
                alt_matrix[i, j] = scores.get('alt_score', 0)

    # Create AnnData
    adata = ad.AnnData(
        X=diff_matrix,
        obs=variants_df.reset_index(drop=True),
        var=pd.DataFrame({'track_name': all_tracks}, index=all_tracks)
    )
    adata.layers['ref_score'] = ref_matrix
    adata.layers['alt_score'] = alt_matrix
    adata.uns['created_from'] = 'checkpoint'
    adata.uns['n_predictions'] = len(predictions)

    # Save
    output_path = RESULTS_DIR / f'stage3_predictions{suffix}.h5ad'
    adata.write_h5ad(output_path, compression='gzip')
    log(f"Saved h5ad: {output_path} ({adata.shape[0]} variants x {adata.shape[1]} tracks)")

    return output_path


def run_stage4(h5ad_path: Path, suffix: str = '') -> Path:
    """Run Stage 4 interpretation."""
    log("Running Stage 4: Interpret predictions...")

    cmd = [
        'python', str(SCRIPTS_DIR / '08_alphagenome_stage4_interpret.py'),
        '--input', str(h5ad_path),
        '--output-suffix', suffix
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        log(f"Stage 4 error: {result.stderr}")
        raise RuntimeError("Stage 4 failed")

    # Print last few lines of output
    for line in result.stdout.strip().split('\n')[-10:]:
        print(f"  {line}")

    return RESULTS_DIR / f'stage4_prioritized{suffix}.csv'


def run_stage5(prioritized_path: Path, suffix: str = '') -> dict:
    """Run Stage 5 validation."""
    log("Running Stage 5: GTEx validation...")

    cmd = [
        'python', str(SCRIPTS_DIR / '08_alphagenome_stage5_validate.py'),
        '--input', str(prioritized_path),
        '--output-suffix', suffix
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        log(f"Stage 5 error: {result.stderr}")
        raise RuntimeError("Stage 5 failed")

    # Print last few lines of output
    for line in result.stdout.strip().split('\n')[-10:]:
        print(f"  {line}")

    # Load and return metrics
    metrics_path = RESULTS_DIR / f'stage5_validation_metrics{suffix}.json'
    if metrics_path.exists():
        with open(metrics_path) as f:
            return json.load(f)
    return {}


def main():
    parser = argparse.ArgumentParser(
        description='Run Stage 4/5 from checkpoint predictions'
    )
    parser.add_argument('--suffix', type=str, default='_v2',
                       help='Suffix for input/output files (default: _v2)')
    parser.add_argument('--stage4-only', action='store_true',
                       help='Only run Stage 4, skip Stage 5')
    parser.add_argument('--skip-h5ad', action='store_true',
                       help='Skip h5ad creation (use existing file)')
    args = parser.parse_args()

    suffix = args.suffix

    log("=" * 60)
    log("ALPHAGENOME: RUN FROM CHECKPOINT")
    log("=" * 60)
    log(f"Suffix: {suffix}")

    try:
        # Step 1: Create h5ad from checkpoint
        if not args.skip_h5ad:
            predictions = load_checkpoint_predictions(suffix)
            h5ad_path = create_h5ad_from_checkpoint(predictions, suffix)
        else:
            h5ad_path = RESULTS_DIR / f'stage3_predictions{suffix}.h5ad'
            if not h5ad_path.exists():
                raise FileNotFoundError(f"H5AD not found: {h5ad_path}")
            log(f"Using existing h5ad: {h5ad_path}")

        # Step 2: Run Stage 4
        prioritized_path = run_stage4(h5ad_path, suffix)

        # Step 3: Run Stage 5
        if not args.stage4_only:
            metrics = run_stage5(prioritized_path, suffix)

            log("\n" + "=" * 60)
            log("SUMMARY")
            log("=" * 60)

            if metrics:
                n_matched = metrics.get('output', {}).get('matched_variants', 0)
                concordance = metrics.get('concordance', {}).get('concordance', 0)
                log(f"  Matched variants: {n_matched}")
                log(f"  Direction concordance: {concordance*100:.1f}%")

        log("\nComplete!")
        log(f"Output files in: {RESULTS_DIR}")

    except Exception as e:
        log(f"ERROR: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
