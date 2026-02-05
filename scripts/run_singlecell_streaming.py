#!/usr/bin/env python3
"""
Single-Cell Activity Inference with True H5AD Streaming.

Uses backed mode to read H5AD and processes in batches using secactpy.
Supports cytosig, lincytosig, and secact signatures.

Usage:
    # Test mode
    python run_singlecell_streaming.py --atlas cima --signature cytosig --test

    # Full run
    python run_singlecell_streaming.py --atlas cima --signature cytosig
"""

import argparse
import sys
import time
import gc
from pathlib import Path
from datetime import datetime
import numpy as np
import anndata as ad
from scipy import sparse

# Ensure line buffering
sys.stdout.reconfigure(line_buffering=True)

# Add secactpy path
sys.path.insert(0, "/vf/users/parks34/projects/1ridgesig/SecActpy")

from secactpy import load_cytosig, load_secact, load_lincytosig
from secactpy.ridge import CUPY_AVAILABLE
from secactpy.batch import ridge_batch, _compute_T_numpy
from secactpy.rng import get_cached_inverse_perm_table

if CUPY_AVAILABLE:
    import cupy as cp
    from secactpy.batch import _compute_T_cupy, _process_batch_cupy

# =============================================================================
# Configuration
# =============================================================================

ATLAS_CONFIG = {
    'cima': '/data/Jiang_Lab/Data/Seongyong/CIMA/Cell_Atlas/CIMA_RNA_6484974cells_36326genes_compressed.h5ad',
    'inflammation_main': '/data/Jiang_Lab/Data/Seongyong/Inflammation_Atlas/INFLAMMATION_ATLAS_main_afterQC.h5ad',
    'inflammation_val': '/data/Jiang_Lab/Data/Seongyong/Inflammation_Atlas/INFLAMMATION_ATLAS_validation_afterQC.h5ad',
    'inflammation_ext': '/data/Jiang_Lab/Data/Seongyong/Inflammation_Atlas/INFLAMMATION_ATLAS_external_afterQC.h5ad',
    'scatlas_normal': '/data/Jiang_Lab/Data/Seongyong/scAtlas_2025/igt_s9_fine_counts.h5ad',
    'scatlas_cancer': '/data/Jiang_Lab/Data/Seongyong/scAtlas_2025/PanCancer_igt_s9_fine_counts.h5ad',
}

BATCH_SIZES = {
    'cytosig': 50000,
    'lincytosig': 20000,
    'secact': 10000,
}

OUTPUT_DIR = Path('/vf/users/parks34/projects/2secactpy/results/atlas_validation')


def log(msg):
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
        raise ValueError(f"Unknown signature: {sig_type}")


def run_singlecell_streaming(
    atlas: str,
    signature: str,
    max_cells: int = None,
    batch_size: int = None,
    n_rand: int = 1000,
    seed: int = 0,
    skip_existing: bool = True,
):
    """Run single-cell activity with true H5AD streaming."""

    input_path = ATLAS_CONFIG[atlas]
    output_dir = OUTPUT_DIR / atlas / 'singlecell'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{atlas}_singlecell_{signature}.h5ad"

    if skip_existing and output_path.exists():
        log(f"SKIP: {output_path.name} exists")
        return str(output_path)

    log("=" * 70)
    log(f"Single-Cell Activity: {atlas} / {signature.upper()}")
    log("=" * 70)
    log(f"Input: {input_path}")
    log(f"Output: {output_path}")

    # Check GPU
    use_gpu = CUPY_AVAILABLE
    if use_gpu:
        log(f"Backend: CuPy (GPU)")
    else:
        log(f"Backend: NumPy (CPU)")

    # Load signature
    log(f"Loading {signature} signature...")
    sig = load_signature(signature)
    sig_genes = [g.upper() for g in sig.index]
    feature_names = list(sig.columns)
    log(f"  Signature: {len(sig_genes)} genes × {len(feature_names)} features")

    # Open H5AD in backed mode
    log("Opening H5AD in backed mode...")
    adata = ad.read_h5ad(input_path, backed='r')
    log(f"  Shape: {adata.n_obs:,} cells × {adata.n_vars:,} genes")

    # Match genes
    expr_genes = [g.upper() for g in adata.var_names]
    common_genes = [g for g in sig_genes if g in expr_genes]
    log(f"  Matched genes: {len(common_genes)}")

    if len(common_genes) < 100:
        raise ValueError(f"Too few matched genes: {len(common_genes)}")

    # Get indices
    sig_idx = [sig_genes.index(g) for g in common_genes]
    expr_idx = [expr_genes.index(g) for g in common_genes]

    # Subset signature matrix
    X = sig.iloc[sig_idx].values.astype(np.float64)  # (n_genes, n_features)

    # Determine cells to process
    n_cells = adata.n_obs
    if max_cells and max_cells < n_cells:
        n_cells = max_cells
        log(f"  Processing: {n_cells:,} cells (test mode)")
    else:
        log(f"  Processing: {n_cells:,} cells")

    # Set batch size
    if batch_size is None:
        batch_size = BATCH_SIZES.get(signature, 10000)
    log(f"  Batch size: {batch_size:,}")

    # Compute projection matrix T
    log("Computing projection matrix...")
    lambda_ = 5e5
    if use_gpu:
        X_gpu = cp.asarray(X, dtype=cp.float64)
        T_gpu = _compute_T_cupy(X_gpu, lambda_)
        del X_gpu
        cp.get_default_memory_pool().free_all_blocks()
    else:
        T = _compute_T_numpy(X, lambda_)

    # Generate inverse permutation table
    log("Generating permutation table...")
    n_matched = len(common_genes)
    inv_perm_table = get_cached_inverse_perm_table(n_matched, n_rand, seed)

    # Get cell names
    cell_names = list(adata.obs_names[:n_cells])

    # Process in batches
    n_batches = (n_cells + batch_size - 1) // batch_size
    log(f"Processing {n_batches} batches...")

    all_zscore = []
    all_pvalue = []
    batch_times = []
    start_time = time.time()

    for batch_idx in range(n_batches):
        batch_start = time.time()

        start = batch_idx * batch_size
        end = min(start + batch_size, n_cells)

        # Read batch from backed H5AD
        X_batch = adata.X[start:end][:, expr_idx]

        if sparse.issparse(X_batch):
            X_dense = X_batch.toarray()
        else:
            X_dense = np.asarray(X_batch)

        # Transpose: (n_genes, batch_size)
        Y_batch = X_dense.T.astype(np.float64)

        # Process batch
        if use_gpu:
            result = _process_batch_cupy(T_gpu, Y_batch, inv_perm_table, n_rand)
        else:
            from secactpy.batch import _process_batch_numpy
            T_np = _compute_T_numpy(X, lambda_) if batch_idx == 0 else T
            result = _process_batch_numpy(T_np, Y_batch, inv_perm_table, n_rand)

        # Store results (transpose to cells × features)
        all_zscore.append(result['zscore'].T)
        all_pvalue.append(result['pvalue'].T)

        batch_time = time.time() - batch_start
        batch_times.append(batch_time)

        if batch_idx % 10 == 0 or batch_idx == n_batches - 1:
            avg_time = np.mean(batch_times[-10:])
            eta = avg_time * (n_batches - batch_idx - 1)
            log(f"  Batch {batch_idx+1}/{n_batches} ({batch_time:.1f}s, ETA: {eta/60:.1f}min)")

        del X_batch, X_dense, Y_batch, result
        gc.collect()

    # Cleanup GPU
    if use_gpu:
        del T_gpu
        cp.get_default_memory_pool().free_all_blocks()

    # Close backed file
    adata.file.close()

    # Concatenate results
    log("Concatenating results...")
    zscore_matrix = np.vstack(all_zscore)
    pvalue_matrix = np.vstack(all_pvalue)

    # Create output AnnData
    log("Saving results...")
    adata_out = ad.AnnData(
        X=zscore_matrix,
        obs={'cell_id': cell_names},
        var={'feature_name': feature_names},
        layers={'pvalue': pvalue_matrix},
    )
    adata_out.obs_names = cell_names
    adata_out.var_names = feature_names

    adata_out.write_h5ad(output_path)

    total_time = time.time() - start_time
    file_size = output_path.stat().st_size / 1e6
    log(f"Complete: {output_path.name} ({file_size:.1f} MB) in {total_time/60:.1f} min")

    return str(output_path)


def main():
    parser = argparse.ArgumentParser(description='Single-cell activity with H5AD streaming')
    parser.add_argument('--atlas', required=True, choices=list(ATLAS_CONFIG.keys()))
    parser.add_argument('--signature', choices=['cytosig', 'lincytosig', 'secact'])
    parser.add_argument('--test', action='store_true', help='Test mode: 100K cells')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--skip-existing', action='store_true', default=True)
    parser.add_argument('--no-skip', dest='skip_existing', action='store_false')

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
            path = run_singlecell_streaming(
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
