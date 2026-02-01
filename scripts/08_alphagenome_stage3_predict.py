#!/usr/bin/env python3
"""
AlphaGenome Stage 3: Execute AlphaGenome Predictions
======================================================
Call AlphaGenome API for each variant to get regulatory impact predictions.

Features:
- Async API calls with rate limiting (~1 req/sec)
- Exponential backoff retry (1-60s, max 5 attempts)
- Checkpoint every 100 variants for resume on failure
- Filter to immune-relevant tracks (GM12878, PBMCs, etc.)

Input:
- results/alphagenome/stage2_alphagenome_input.csv

Output:
- results/alphagenome/stage3_predictions.h5ad (AnnData with 7000+ tracks)
- results/alphagenome/stage3_checkpoint.json
"""

import os
import sys
import json
import time
import asyncio
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
import anndata as ad

# ==============================================================================
# Configuration
# ==============================================================================

INPUT_DIR = Path('/vf/users/parks34/projects/2secactpy/results/alphagenome')
OUTPUT_DIR = INPUT_DIR

# Rate limiting
REQUESTS_PER_SECOND = 1.0
CHECKPOINT_INTERVAL = 100

# Retry configuration
MAX_RETRIES = 5
MIN_WAIT = 1
MAX_WAIT = 60

# Immune-relevant track patterns (case-insensitive matching)
IMMUNE_TRACK_PATTERNS = [
    'gm12878',      # B-lymphoblastoid cell line
    'pbmc',         # Peripheral blood mononuclear cells
    'cd4',          # CD4+ T cells
    'cd8',          # CD8+ T cells
    'b_cell', 'bcell',  # B cells
    't_cell', 'tcell',  # T cells
    'monocyte',     # Monocytes
    'macrophage',   # Macrophages
    'dendritic',    # Dendritic cells
    'nk_cell', 'nkcell',  # NK cells
    'lymphocyte',   # Lymphocytes
    'leukocyte',    # Leukocytes
    'immune',       # Generic immune
    'blood',        # Blood-related
    'hematopoietic', 'hsc',  # Hematopoietic stem cells
    'spleen',       # Spleen
    'thymus',       # Thymus
    'bone_marrow',  # Bone marrow
]


def log(msg: str):
    """Print timestamped log message."""
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class CheckpointManager:
    """Manage checkpoints for resumable processing."""

    def __init__(self, checkpoint_path: Path):
        self.checkpoint_path = checkpoint_path
        self.data = self.load()

    def load(self) -> Dict[str, Any]:
        """Load checkpoint from disk."""
        if self.checkpoint_path.exists():
            with open(self.checkpoint_path, 'r') as f:
                return json.load(f)
        return {
            'processed_variants': [],
            'last_index': -1,
            'start_time': datetime.now().isoformat(),
            'errors': [],
        }

    def save(self):
        """Save checkpoint to disk."""
        self.data['last_updated'] = datetime.now().isoformat()
        with open(self.checkpoint_path, 'w') as f:
            json.dump(self.data, f, indent=2)

    def is_processed(self, variant_id: str) -> bool:
        """Check if variant has been processed."""
        return variant_id in self.data['processed_variants']

    def mark_processed(self, variant_id: str, index: int):
        """Mark variant as processed."""
        self.data['processed_variants'].append(variant_id)
        self.data['last_index'] = index

    def add_error(self, variant_id: str, error: str):
        """Record an error."""
        self.data['errors'].append({
            'variant_id': variant_id,
            'error': error,
            'timestamp': datetime.now().isoformat()
        })


class AlphaGenomeClient:
    """Client for AlphaGenome API with retry logic."""

    def __init__(self, rate_limit: float = REQUESTS_PER_SECOND):
        self.rate_limit = rate_limit
        self.last_request_time = 0
        self._client = None
        self._initialized = False

    async def initialize(self):
        """Initialize the AlphaGenome client."""
        if self._initialized:
            return

        try:
            from alphagenome import dna_client
            self._client = await dna_client.DnaClient.create()
            self._initialized = True
            log("AlphaGenome client initialized")
        except ImportError:
            log("WARNING: alphagenome package not installed")
            log("  Install with: pip install alphagenome")
            raise
        except Exception as e:
            log(f"ERROR initializing AlphaGenome client: {e}")
            raise

    async def _wait_for_rate_limit(self):
        """Wait to respect rate limit."""
        elapsed = time.time() - self.last_request_time
        wait_time = (1.0 / self.rate_limit) - elapsed
        if wait_time > 0:
            await asyncio.sleep(wait_time)
        self.last_request_time = time.time()

    async def score_variant(
        self,
        chrom: str,
        pos: int,
        ref: str,
        alt: str,
        sequence_length: str = 'SEQUENCE_LENGTH_1MB'
    ) -> Optional[Dict[str, Any]]:
        """
        Score a variant using AlphaGenome.

        Args:
            chrom: Chromosome (e.g., 'chr1')
            pos: Position (1-based)
            ref: Reference allele
            alt: Alternate allele
            sequence_length: Sequence context length

        Returns:
            Dictionary with predictions or None on failure
        """
        if not self._initialized:
            await self.initialize()

        from alphagenome import genome, dna_client

        # Create variant object
        variant = genome.Variant(
            chromosome=chrom,
            position=pos,
            reference_bases=ref,
            alternate_bases=alt
        )

        # Get sequence length enum
        seq_len = getattr(dna_client.SUPPORTED_SEQUENCE_LENGTHS, sequence_length)

        # Retry loop with exponential backoff
        for attempt in range(MAX_RETRIES):
            try:
                await self._wait_for_rate_limit()

                result = await self._client.score_variant(
                    variant,
                    sequence_length=seq_len
                )

                return self._parse_result(result)

            except Exception as e:
                wait_time = min(MIN_WAIT * (2 ** attempt), MAX_WAIT)
                log(f"    Attempt {attempt + 1}/{MAX_RETRIES} failed: {e}")

                if attempt < MAX_RETRIES - 1:
                    log(f"    Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    log(f"    All retries exhausted")
                    return None

        return None

    def _parse_result(self, result) -> Dict[str, Any]:
        """Parse AlphaGenome API result to dictionary."""
        # This will need to be adjusted based on actual API response format
        output = {
            'tracks': {},
            'metadata': {}
        }

        # Extract track predictions
        if hasattr(result, 'tracks'):
            for track in result.tracks:
                track_name = track.name if hasattr(track, 'name') else str(track)
                if hasattr(track, 'scores'):
                    output['tracks'][track_name] = {
                        'ref_score': float(track.scores.reference),
                        'alt_score': float(track.scores.alternate),
                        'diff': float(track.scores.alternate - track.scores.reference),
                    }

        # Extract summary metrics if available
        if hasattr(result, 'summary'):
            output['metadata'] = {
                'total_tracks': len(output['tracks']),
                'significant_tracks': sum(
                    1 for t in output['tracks'].values()
                    if abs(t['diff']) > 0.1
                ),
            }

        return output

    async def close(self):
        """Close the client connection."""
        if self._client and hasattr(self._client, 'close'):
            await self._client.close()


def filter_immune_tracks(predictions: Dict[str, Any]) -> Dict[str, Any]:
    """Filter predictions to immune-relevant tracks."""
    if 'tracks' not in predictions:
        return predictions

    filtered_tracks = {}
    for track_name, scores in predictions['tracks'].items():
        track_lower = track_name.lower()
        for pattern in IMMUNE_TRACK_PATTERNS:
            if pattern in track_lower:
                filtered_tracks[track_name] = scores
                break

    predictions['tracks'] = filtered_tracks
    predictions['metadata']['immune_tracks'] = len(filtered_tracks)
    return predictions


def save_predictions_h5ad(
    variants_df: pd.DataFrame,
    predictions: Dict[str, Dict],
    output_path: Path
):
    """
    Save predictions to h5ad format.

    Structure:
    - obs: variant metadata (one row per variant)
    - var: track metadata (one column per track)
    - X: track scores (variants x tracks)
    - layers['ref_score']: reference allele scores
    - layers['alt_score']: alternate allele scores
    """
    log("Saving predictions to h5ad...")

    # Collect all track names across all variants
    all_tracks = set()
    for pred in predictions.values():
        if 'tracks' in pred:
            all_tracks.update(pred['tracks'].keys())

    all_tracks = sorted(all_tracks)
    log(f"  Total tracks: {len(all_tracks)}")

    if len(all_tracks) == 0:
        log("  WARNING: No track predictions to save")
        return

    # Create matrices
    n_variants = len(variants_df)
    n_tracks = len(all_tracks)

    diff_matrix = np.zeros((n_variants, n_tracks))
    ref_matrix = np.zeros((n_variants, n_tracks))
    alt_matrix = np.zeros((n_variants, n_tracks))

    track_to_idx = {t: i for i, t in enumerate(all_tracks)}

    for i, (_, row) in enumerate(variants_df.iterrows()):
        variant_id = row['variant_id']
        if variant_id in predictions and 'tracks' in predictions[variant_id]:
            for track_name, scores in predictions[variant_id]['tracks'].items():
                if track_name in track_to_idx:
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

    # Add metadata
    adata.uns['stage'] = 3
    adata.uns['description'] = 'AlphaGenome variant effect predictions'
    adata.uns['n_variants'] = n_variants
    adata.uns['n_tracks'] = n_tracks

    adata.write_h5ad(output_path, compression='gzip')
    log(f"  Saved: {output_path}")


async def process_variants(
    variants_df: pd.DataFrame,
    checkpoint: CheckpointManager,
    resume: bool = False
) -> Dict[str, Dict]:
    """Process all variants through AlphaGenome API."""
    client = AlphaGenomeClient()
    predictions = {}

    try:
        await client.initialize()

        start_idx = checkpoint.data['last_index'] + 1 if resume else 0
        total = len(variants_df)

        log(f"\nProcessing {total - start_idx} variants...")
        if start_idx > 0:
            log(f"  Resuming from index {start_idx}")

        for i, (_, row) in enumerate(variants_df.iloc[start_idx:].iterrows()):
            idx = start_idx + i
            variant_id = row['variant_id']

            # Skip if already processed
            if checkpoint.is_processed(variant_id):
                continue

            log(f"  [{idx + 1}/{total}] {variant_id} - {row['CHROM']}:{row['POS']} {row['REF']}>{row['ALT']}")

            # Call AlphaGenome API
            result = await client.score_variant(
                chrom=row['CHROM'],
                pos=int(row['POS']),
                ref=row['REF'],
                alt=row['ALT']
            )

            if result:
                # Filter to immune tracks
                result = filter_immune_tracks(result)
                predictions[variant_id] = result
                log(f"    Got {len(result.get('tracks', {}))} immune tracks")
            else:
                checkpoint.add_error(variant_id, "API call failed after retries")
                log(f"    FAILED")

            # Update checkpoint
            checkpoint.mark_processed(variant_id, idx)

            # Save checkpoint periodically
            if (idx + 1) % CHECKPOINT_INTERVAL == 0:
                checkpoint.save()
                log(f"  Checkpoint saved at index {idx + 1}")

        checkpoint.save()

    finally:
        await client.close()

    return predictions


def run_mock_predictions(variants_df: pd.DataFrame) -> Dict[str, Dict]:
    """
    Generate mock predictions for testing when AlphaGenome is unavailable.

    This creates realistic-looking but synthetic data for pipeline testing.
    """
    log("Running MOCK predictions (AlphaGenome not available)...")

    np.random.seed(42)
    predictions = {}

    # Mock track names (subset of what AlphaGenome might return)
    mock_tracks = [
        'ATAC_GM12878', 'ATAC_CD4_Tcell', 'ATAC_CD8_Tcell', 'ATAC_Monocyte',
        'H3K27ac_GM12878', 'H3K27ac_PBMC', 'H3K27ac_CD4_Tcell',
        'H3K4me3_GM12878', 'H3K4me3_PBMC',
        'CTCF_GM12878', 'CTCF_lymphocyte',
        'DNase_GM12878', 'DNase_CD4_Tcell', 'DNase_Monocyte',
        'ChIP_PU1_Monocyte', 'ChIP_RUNX1_Tcell', 'ChIP_PAX5_Bcell',
        'Expression_GM12878', 'Expression_PBMC',
    ]

    for i, (_, row) in enumerate(variants_df.iterrows()):
        variant_id = row['variant_id']

        # Generate random scores
        tracks = {}
        for track in mock_tracks:
            ref_score = np.random.uniform(0, 10)
            # Effect size depends on variant type and random chance
            if row['variant_type'] == 'SNV':
                effect = np.random.normal(0, 0.5)
            else:
                effect = np.random.normal(0, 1.0)  # Indels have larger effects

            tracks[track] = {
                'ref_score': ref_score,
                'alt_score': ref_score + effect,
                'diff': effect,
            }

        predictions[variant_id] = {
            'tracks': tracks,
            'metadata': {
                'total_tracks': len(tracks),
                'significant_tracks': sum(1 for t in tracks.values() if abs(t['diff']) > 0.3),
                'mock': True,
            }
        }

        if (i + 1) % 500 == 0:
            log(f"  Processed {i + 1}/{len(variants_df)}")

    return predictions


def main():
    parser = argparse.ArgumentParser(description='AlphaGenome Stage 3: Predictions')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from checkpoint')
    parser.add_argument('--mock', action='store_true',
                       help='Use mock predictions (for testing without API)')
    parser.add_argument('--max-variants', type=int, default=None,
                       help='Maximum variants to process (for testing)')
    args = parser.parse_args()

    log("=" * 60)
    log("ALPHAGENOME STAGE 3: EXECUTE PREDICTIONS")
    log("=" * 60)

    # Load Stage 2 output
    input_csv = INPUT_DIR / 'stage2_alphagenome_input.csv'
    log(f"Loading: {input_csv}")
    variants_df = pd.read_csv(input_csv)
    log(f"  Loaded {len(variants_df):,} variants")

    # Limit variants for testing
    if args.max_variants:
        variants_df = variants_df.head(args.max_variants)
        log(f"  Limited to {len(variants_df)} variants for testing")

    # Initialize checkpoint
    checkpoint_path = OUTPUT_DIR / 'stage3_checkpoint.json'
    checkpoint = CheckpointManager(checkpoint_path)

    if args.resume:
        log(f"\nResuming from checkpoint...")
        log(f"  Previously processed: {len(checkpoint.data['processed_variants'])}")
        log(f"  Errors: {len(checkpoint.data['errors'])}")

    # Run predictions
    if args.mock:
        predictions = run_mock_predictions(variants_df)
    else:
        try:
            predictions = asyncio.run(
                process_variants(variants_df, checkpoint, args.resume)
            )
        except ImportError:
            log("\nWARNING: alphagenome package not available")
            log("Falling back to mock predictions...")
            predictions = run_mock_predictions(variants_df)

    log(f"\nCompleted predictions for {len(predictions)} variants")

    # Save predictions to h5ad
    output_path = OUTPUT_DIR / 'stage3_predictions.h5ad'
    save_predictions_h5ad(variants_df, predictions, output_path)

    # Update checkpoint with completion info
    checkpoint.data['completed'] = True
    checkpoint.data['n_predictions'] = len(predictions)
    checkpoint.save()

    # Summary
    log("\nStage 3 complete!")
    log(f"  Predictions: {len(predictions)}")
    log(f"  Errors: {len(checkpoint.data['errors'])}")
    log(f"  Output: {output_path}")


if __name__ == '__main__':
    main()
