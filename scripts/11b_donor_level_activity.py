#!/usr/bin/env python3
"""
Run activity inference on existing donor-level pseudobulk files.

This standalone script computes CytoSig/LinCytoSig/SecAct activity
on pre-generated pseudobulk files. It handles gene symbol mapping
for Ensembl ID datasets (e.g., Inflammation Atlas).

Usage:
    python scripts/11b_donor_level_activity.py

Author: Claude Code (2026-02-04)
"""

import argparse
import gzip
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import anndata as ad
import numpy as np
import pandas as pd

from secactpy import load_cytosig, load_secact, ridge


def log(msg: str) -> None:
    """Print timestamped log message."""
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {msg}", flush=True)


def load_signatures() -> Dict[str, pd.DataFrame]:
    """Load all signature matrices."""
    cytosig = load_cytosig()
    secact = load_secact()

    # Load LinCytoSig
    lincytosig_path = Path("/data/parks34/projects/1ridgesig/SecActpy/secactpy/data/LinCytoSig.tsv.gz")
    with gzip.open(lincytosig_path, 'rt') as f:
        lincytosig = pd.read_csv(f, sep='\t', index_col=0)

    return {
        'cytosig': cytosig,
        'lincytosig': lincytosig,
        'secact': secact,
    }


def compute_activity(
    expr_matrix: np.ndarray,
    gene_names: List[str],
    signature: pd.DataFrame,
    lambda_: float = 5e5,
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Compute activity scores using ridge regression.

    Returns:
        Tuple of (activity_matrix, common_genes, target_names)
    """
    # Find common genes
    expr_genes = set(gene_names)
    sig_genes = set(signature.index)
    common = sorted(expr_genes & sig_genes)

    if len(common) < 100:
        raise ValueError(f"Too few common genes: {len(common)}")

    # Subset and align
    gene_idx = [gene_names.index(g) for g in common]
    Y = expr_matrix[:, gene_idx].T  # (genes × samples)
    X = signature.loc[common].values  # (genes × targets)
    X = np.nan_to_num(X, nan=0.0)

    # Ridge: X (genes × targets), Y (genes × samples) -> zscore (targets × samples)
    result = ridge(X, Y, lambda_=lambda_, n_rand=1000, verbose=False)
    activity = result['zscore'].T  # (samples × targets)

    return activity, common, list(signature.columns)


def run_activity_for_pseudobulk(
    pb_path: Path,
    output_dir: Path,
    force: bool = False,
) -> Dict[str, Path]:
    """
    Run activity inference on a single pseudobulk file.

    Args:
        pb_path: Path to pseudobulk H5AD file
        output_dir: Output directory for activity files
        force: If True, overwrite existing activity files

    Returns:
        Dict mapping signature name to output path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse file name to get atlas_level prefix
    # e.g., inflammation_external_L1_donor_pseudobulk.h5ad -> inflammation_external_L1
    file_stem = pb_path.stem.replace('_donor_pseudobulk', '')

    log("=" * 70)
    log(f"Activity Inference: {file_stem}")
    log("=" * 70)

    # Load pseudobulk
    log(f"Loading: {pb_path}")
    adata_pb = ad.read_h5ad(pb_path)
    expr_matrix = adata_pb.X
    if hasattr(expr_matrix, 'toarray'):
        expr_matrix = expr_matrix.toarray()

    log(f"Shape: {adata_pb.shape[0]:,} groups × {adata_pb.shape[1]:,} genes")

    # Get gene names - use symbols if available (Ensembl ID mapping)
    if 'symbol' in adata_pb.var.columns:
        gene_names = adata_pb.var['symbol'].tolist()
        log(f"Using gene symbols from var['symbol'] for signature matching")
    else:
        gene_names = list(adata_pb.var_names)
        log(f"Using var_names for signature matching")

    # Load signatures
    signatures = load_signatures()
    output_paths = {}

    for sig_name, sig_matrix in signatures.items():
        out_path = output_dir / f"{file_stem}_donor_{sig_name}.h5ad"

        if out_path.exists() and not force:
            log(f"Skipping {sig_name} (exists): {out_path.name}")
            output_paths[sig_name] = out_path
            continue

        log(f"Computing {sig_name} activity...")
        try:
            activity, common_genes, targets = compute_activity(
                expr_matrix, gene_names, sig_matrix
            )

            # Create activity AnnData
            adata_act = ad.AnnData(
                X=activity.astype(np.float32),
                obs=adata_pb.obs.copy(),
                var=pd.DataFrame(index=targets),
            )
            adata_act.uns['common_genes'] = len(common_genes)
            adata_act.uns['signature'] = sig_name
            adata_act.uns['source_pseudobulk'] = str(pb_path)

            # Save
            adata_act.write_h5ad(out_path, compression='gzip')
            output_paths[sig_name] = out_path
            log(f"  Saved: {out_path.name} ({len(common_genes)} common genes)")

        except Exception as e:
            log(f"  Failed: {e}")

    return output_paths


def main():
    parser = argparse.ArgumentParser(description="Run activity on donor-level pseudobulk")
    parser.add_argument('--input-dir', type=Path,
                        default=Path('/vf/users/parks34/projects/2secactpy/results/donor_level'),
                        help='Input directory containing pseudobulk files')
    parser.add_argument('--force', action='store_true',
                        help='Overwrite existing activity files')
    parser.add_argument('--filter', type=str, default=None,
                        help='Only process files matching this pattern')

    args = parser.parse_args()

    # Find all pseudobulk files
    pb_files = sorted(args.input_dir.rglob('*_donor_pseudobulk.h5ad'))

    if args.filter:
        pb_files = [f for f in pb_files if args.filter in str(f)]

    log(f"Found {len(pb_files)} pseudobulk files")

    all_outputs = {}

    for pb_path in pb_files:
        # Output to donor_activity subdirectory parallel to donor_pseudobulk
        output_dir = pb_path.parent.parent / 'donor_activity'

        outputs = run_activity_for_pseudobulk(
            pb_path=pb_path,
            output_dir=output_dir,
            force=args.force,
        )
        all_outputs[pb_path.stem] = outputs

    log("\n" + "=" * 70)
    log("ALL ACTIVITY INFERENCE COMPLETE")
    log("=" * 70)

    # Summary
    success_count = sum(len(v) for v in all_outputs.values())
    log(f"Total activity files: {success_count}")


if __name__ == '__main__':
    main()
