#!/usr/bin/env python3
"""
Single-Cell Activity Inference using secactpy batch processing.

Uses secactpy's ridge_batch with H5AD streaming for memory-efficient
processing of large single-cell datasets.

Usage:
    # Test mode
    python run_singlecell_secactpy.py --atlas cima --signature cytosig --test

    # Full run
    python run_singlecell_secactpy.py --atlas cima --signature cytosig

    # All signatures for an atlas
    python run_singlecell_secactpy.py --atlas cima
"""

import argparse
import sys
import time
from pathlib import Path
import numpy as np
import anndata as ad
from scipy import sparse

# Ensure line buffering for SLURM logs
sys.stdout.reconfigure(line_buffering=True)

# Add secactpy path
sys.path.insert(0, "/vf/users/parks34/projects/1ridgesig/SecActpy")

from secactpy import load_cytosig, load_secact, load_lincytosig
from secactpy.batch import ridge_batch

# =============================================================================
# Configuration
# =============================================================================

ATLAS_CONFIG = {
    'cima': {
        'input': '/data/Jiang_Lab/Data/Seongyong/CIMA/Cell_Atlas/CIMA_RNA_6484974cells_36326genes_compressed.h5ad',
    },
    'inflammation_main': {
        'input': '/data/Jiang_Lab/Data/Seongyong/Inflammation_Atlas/INFLAMMATION_ATLAS_main_afterQC.h5ad',
    },
    'inflammation_val': {
        'input': '/data/Jiang_Lab/Data/Seongyong/Inflammation_Atlas/INFLAMMATION_ATLAS_validation_afterQC.h5ad',
    },
    'inflammation_ext': {
        'input': '/data/Jiang_Lab/Data/Seongyong/Inflammation_Atlas/INFLAMMATION_ATLAS_external_afterQC.h5ad',
    },
    'scatlas_normal': {
        'input': '/data/Jiang_Lab/Data/Seongyong/scAtlas_2025/igt_s9_fine_counts.h5ad',
    },
    'scatlas_cancer': {
        'input': '/data/Jiang_Lab/Data/Seongyong/scAtlas_2025/PanCancer_igt_s9_fine_counts.h5ad',
    },
}

# Batch sizes per signature (smaller for larger signatures)
BATCH_SIZES = {
    'cytosig': 50000,     # 43 signatures
    'lincytosig': 20000,  # 178 signatures
    'secact': 10000,      # 1170 signatures
}

OUTPUT_DIR = Path('/vf/users/parks34/projects/2cytoatlas/results/atlas_validation')


def log(msg):
    from datetime import datetime
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def load_signature(sig_type):
    """Load signature matrix."""
    if sig_type == 'cytosig':
        return load_cytosig()
    elif sig_type == 'lincytosig':
        return load_lincytosig()
    elif sig_type == 'secact':
        return load_secact()
    else:
        raise ValueError(f"Unknown signature type: {sig_type}")


def run_singlecell_activity(
    atlas: str,
    signature: str,
    max_cells: int = None,
    batch_size: int = None,
    skip_existing: bool = True,
):
    """Run single-cell activity inference using secactpy batch processing."""

    config = ATLAS_CONFIG[atlas]
    input_path = config['input']

    output_dir = OUTPUT_DIR / atlas / 'singlecell'
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{atlas}_singlecell_{signature}.h5ad"

    # Skip if exists
    if skip_existing and output_path.exists():
        log(f"SKIP: {output_path.name} already exists")
        return str(output_path)

    log("=" * 70)
    log(f"Single-Cell Activity: {atlas} / {signature.upper()}")
    log("=" * 70)
    log(f"Input: {input_path}")
    log(f"Output: {output_path}")

    # Load signature
    log(f"Loading {signature} signature...")
    sig = load_signature(signature)
    sig_genes = list(sig.index)
    log(f"  Signature: {len(sig_genes)} genes × {sig.shape[1]} features")

    # Open expression data in backed mode
    log("Opening H5AD in backed mode...")
    adata = ad.read_h5ad(input_path, backed='r')
    log(f"  Shape: {adata.n_obs:,} cells × {adata.n_vars:,} genes")

    # Match genes
    expr_genes = list(adata.var_names)
    common_genes = [g for g in sig_genes if g in expr_genes]
    log(f"  Matched genes: {len(common_genes)}")

    if len(common_genes) < 100:
        raise ValueError(f"Too few matched genes: {len(common_genes)}")

    # Get gene indices
    sig_gene_idx = [sig_genes.index(g) for g in common_genes]
    expr_gene_idx = [expr_genes.index(g) for g in common_genes]

    # Subset signature to matched genes
    X = sig.iloc[sig_gene_idx].values  # (n_genes, n_features)
    feature_names = list(sig.columns)

    # Determine cells to process
    n_cells = adata.n_obs
    if max_cells and max_cells < n_cells:
        log(f"  Processing: {max_cells:,} cells (test mode)")
        n_cells = max_cells
    else:
        log(f"  Processing: {n_cells:,} cells")

    # Set batch size
    if batch_size is None:
        batch_size = BATCH_SIZES.get(signature, 10000)
    log(f"  Batch size: {batch_size:,}")

    # Extract expression matrix for matched genes
    log("Extracting expression data...")
    start_time = time.time()

    Y_raw = adata.X[:n_cells, expr_gene_idx]
    if sparse.issparse(Y_raw):
        # Keep as sparse for secactpy batch processing
        Y = Y_raw.T.tocsc()  # (n_genes, n_samples), CSC for column access
    else:
        Y = Y_raw.T  # (n_genes, n_samples)

    log(f"  Y shape: {Y.shape}, sparse: {sparse.issparse(Y)}")

    # Get cell names
    cell_names = list(adata.obs_names[:n_cells])

    # Close backed file
    adata.file.close()

    extract_time = time.time() - start_time
    log(f"  Extraction time: {extract_time:.1f}s")

    # Run ridge_batch with streaming output
    log("Running ridge_batch...")
    log(f"  Backend: auto (GPU if available)")

    result = ridge_batch(
        X=X,
        Y=Y,
        batch_size=batch_size,
        backend='auto',
        feature_names=feature_names,
        sample_names=cell_names,
        verbose=True,
    )

    # Save as AnnData
    log("Saving results as AnnData...")

    adata_out = ad.AnnData(
        X=result['zscore'].T,  # (cells, features)
        obs={'cell_id': cell_names},
        var={'feature_name': feature_names},
        layers={
            'beta': result['beta'].T,
            'se': result['se'].T,
            'pvalue': result['pvalue'].T,
        }
    )
    adata_out.obs_names = cell_names
    adata_out.var_names = feature_names

    adata_out.write_h5ad(output_path)

    total_time = time.time() - start_time
    file_size = output_path.stat().st_size / 1e6
    log(f"Complete: {output_path.name} ({file_size:.1f} MB) in {total_time/60:.1f} min")

    return str(output_path)


def main():
    parser = argparse.ArgumentParser(description='Single-cell activity inference using secactpy')
    parser.add_argument('--atlas', required=True, choices=list(ATLAS_CONFIG.keys()),
                       help='Atlas to process')
    parser.add_argument('--signature', choices=['cytosig', 'lincytosig', 'secact'],
                       help='Signature to use (default: all)')
    parser.add_argument('--test', action='store_true',
                       help='Test mode: process max 100K cells')
    parser.add_argument('--batch-size', type=int,
                       help='Override batch size')
    parser.add_argument('--skip-existing', action='store_true', default=True,
                       help='Skip if output exists')
    parser.add_argument('--no-skip', dest='skip_existing', action='store_false',
                       help='Overwrite existing files')

    args = parser.parse_args()

    max_cells = 100000 if args.test else None
    signatures = [args.signature] if args.signature else ['cytosig', 'lincytosig', 'secact']

    log(f"Atlas: {args.atlas}")
    log(f"Signatures: {', '.join(signatures)}")
    if args.test:
        log(f"Test mode: max {max_cells:,} cells")
    log("")

    results = {}
    for sig in signatures:
        try:
            path = run_singlecell_activity(
                atlas=args.atlas,
                signature=sig,
                max_cells=max_cells,
                batch_size=args.batch_size,
                skip_existing=args.skip_existing,
            )
            results[sig] = ('success', path)
        except Exception as e:
            log(f"ERROR: {sig} failed - {e}")
            import traceback
            traceback.print_exc()
            results[sig] = ('failed', str(e))

    # Summary
    log("")
    log("=" * 70)
    log("SUMMARY")
    log("=" * 70)
    for sig, (status, info) in results.items():
        if status == 'success':
            log(f"  ✓ {sig}: {Path(info).name}")
        else:
            log(f"  ✗ {sig}: {info}")


if __name__ == '__main__':
    main()
