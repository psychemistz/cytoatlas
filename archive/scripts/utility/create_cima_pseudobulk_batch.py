#!/usr/bin/env python
"""
Create celltype-specific pseudobulk from CIMA using BATCH PROCESSING.

Uses backed mode + chunked reading to avoid loading entire file into memory.
GPU accelerated aggregation where possible.
"""

import sys
import time
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse
from collections import defaultdict

# Paths
CIMA_H5AD = "/data/Jiang_Lab/Data/Seongyong/CIMA/Cell_Atlas/CIMA_RNA_6484974cells_36326genes_compressed.h5ad"
OUTPUT_DIR = Path("/vf/users/parks34/projects/2cytoatlas/results/cima/pseudobulk")

# Parameters
BATCH_SIZE = 50000  # Process 50K cells at a time
CELLTYPE_LEVELS = ["cell_type_l1", "cell_type_l2", "cell_type_l3", "cell_type_l4"]


def check_gpu():
    """Check GPU availability."""
    try:
        import cupy as cp
        n = cp.cuda.runtime.getDeviceCount()
        mem = cp.cuda.runtime.memGetInfo()
        print(f"GPU: {n} device(s), {mem[0]/1024**3:.1f}/{mem[1]/1024**3:.1f} GB free")
        return True
    except Exception as e:
        print(f"GPU: Not available ({e})")
        return False


def aggregate_batch_gpu(X_batch, group_labels, unique_groups, group_to_idx, accumulators):
    """Aggregate a batch using GPU."""
    import cupy as cp

    # Transfer to GPU
    if sparse.issparse(X_batch):
        X_dense = X_batch.toarray()
    else:
        X_dense = np.asarray(X_batch)

    X_gpu = cp.asarray(X_dense, dtype=cp.float32)

    # Aggregate per group
    for g in np.unique(group_labels):
        if g not in group_to_idx:
            continue
        mask = group_labels == g
        group_sum = cp.asnumpy(X_gpu[mask].sum(axis=0))
        accumulators[g] += group_sum

    # Free GPU memory
    del X_gpu
    cp.get_default_memory_pool().free_all_blocks()


def aggregate_batch_cpu(X_batch, group_labels, unique_groups, group_to_idx, accumulators):
    """Aggregate a batch using CPU."""
    for g in np.unique(group_labels):
        if g not in group_to_idx:
            continue
        mask = group_labels == g
        if sparse.issparse(X_batch):
            group_sum = np.asarray(X_batch[mask].sum(axis=0)).flatten()
        else:
            group_sum = X_batch[mask].sum(axis=0)
        accumulators[g] += group_sum


def main():
    total_start = time.time()

    print("=" * 70)
    print("CIMA Pseudobulk Generation (BATCH MODE)")
    print("=" * 70)

    use_gpu = check_gpu()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output: {OUTPUT_DIR}")
    print(f"Batch size: {BATCH_SIZE:,} cells")

    # Open file in backed mode (memory efficient)
    print(f"\n[1] Opening CIMA H5AD in backed mode...")
    open_start = time.time()
    adata = ad.read_h5ad(CIMA_H5AD, backed='r')
    n_cells, n_genes = adata.shape
    print(f"    Shape: {n_cells:,} cells x {n_genes:,} genes")
    print(f"    Opened in {time.time() - open_start:.1f}s")

    # Get gene names
    var_names = adata.var_names.tolist()

    # Read all cell metadata at once (this is small compared to expression)
    print(f"\n[2] Loading cell metadata...")
    meta_start = time.time()
    obs_df = adata.obs[CELLTYPE_LEVELS].copy()
    print(f"    Loaded in {time.time() - meta_start:.1f}s")

    # Process each level
    results = {}

    for level in CELLTYPE_LEVELS:
        print(f"\n[{level}] Processing...")
        level_start = time.time()

        # Get unique groups
        unique_groups = obs_df[level].unique().tolist()
        group_to_idx = {g: i for i, g in enumerate(unique_groups)}
        n_groups = len(unique_groups)
        print(f"    Groups: {n_groups}")

        # Initialize accumulators (one array per group)
        accumulators = {g: np.zeros(n_genes, dtype=np.float64) for g in unique_groups}
        cell_counts = defaultdict(int)

        # Process in batches
        n_batches = (n_cells + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"    Batches: {n_batches}")

        batch_times = []
        for batch_idx in range(n_batches):
            batch_start = time.time()
            start_idx = batch_idx * BATCH_SIZE
            end_idx = min(start_idx + BATCH_SIZE, n_cells)

            # Load batch
            X_batch = adata.X[start_idx:end_idx]
            group_labels = obs_df[level].iloc[start_idx:end_idx].values

            # Count cells per group
            for g in group_labels:
                cell_counts[g] += 1

            # Aggregate
            if use_gpu:
                try:
                    aggregate_batch_gpu(X_batch, group_labels, unique_groups, group_to_idx, accumulators)
                except Exception as e:
                    if batch_idx == 0:
                        print(f"    GPU failed, using CPU: {e}")
                    aggregate_batch_cpu(X_batch, group_labels, unique_groups, group_to_idx, accumulators)
            else:
                aggregate_batch_cpu(X_batch, group_labels, unique_groups, group_to_idx, accumulators)

            batch_time = time.time() - batch_start
            batch_times.append(batch_time)

            if batch_idx % 10 == 0 or batch_idx == n_batches - 1:
                avg_time = np.mean(batch_times[-10:])
                eta = avg_time * (n_batches - batch_idx - 1)
                print(f"    Batch {batch_idx+1}/{n_batches} ({batch_time:.1f}s, ETA: {eta/60:.1f}min)")

        # Build pseudobulk matrix
        pseudobulk = np.vstack([accumulators[g] for g in unique_groups])

        # Normalize to CPM + log1p
        total = pseudobulk.sum(axis=1, keepdims=True)
        total = np.maximum(total, 1)
        cpm = pseudobulk / total * 1e6
        log_cpm = np.log1p(cpm)

        # Create AnnData
        pb_adata = ad.AnnData(
            X=log_cpm.astype(np.float32),
            obs=pd.DataFrame(index=unique_groups),
            var=pd.DataFrame(index=var_names)
        )
        pb_adata.layers["counts"] = pseudobulk.astype(np.float32)
        pb_adata.obs["n_cells"] = [cell_counts[g] for g in unique_groups]

        level_time = time.time() - level_start
        print(f"    Shape: {pb_adata.shape}")
        print(f"    Time: {level_time:.1f}s ({level_time/60:.1f}min)")

        # Save
        output_path = OUTPUT_DIR / f"CIMA_pseudobulk_{level}.h5ad"
        pb_adata.write_h5ad(output_path)
        print(f"    Saved: {output_path}")

        results[level] = {
            "n_groups": n_groups,
            "time": level_time,
            "path": str(output_path)
        }

    # Summary
    total_time = time.time() - total_start

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print("=" * 70)

    print(f"\n--- Timing ---")
    for level, info in results.items():
        print(f"  {level}: {info['time']/60:5.1f} min ({info['n_groups']} groups)")
    print(f"  TOTAL: {total_time/60:.1f} min ({total_time:.0f}s)")

    print(f"\n--- Output Files ---")
    for level, info in results.items():
        print(f"  {info['path']}")

    import psutil
    mem_gb = psutil.Process().memory_info().rss / 1024**3
    print(f"\n--- Memory ---")
    print(f"  Peak: {mem_gb:.1f} GB")

    return results


if __name__ == "__main__":
    results = main()
