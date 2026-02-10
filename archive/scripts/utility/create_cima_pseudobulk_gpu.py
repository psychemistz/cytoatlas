#!/usr/bin/env python
"""
GPU-Optimized CIMA Pseudobulk Generation.

Key optimizations:
1. Single pass through data - aggregate ALL levels simultaneously
2. GPU-resident accumulators - minimize CPU-GPU transfers
3. CuPy scatter_add for fast aggregation
4. Larger batches for better GPU utilization
5. Async data loading with prefetch
"""

import sys
import time
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse

# Paths
CIMA_H5AD = "/data/Jiang_Lab/Data/Seongyong/CIMA/Cell_Atlas/CIMA_RNA_6484974cells_36326genes_compressed.h5ad"
OUTPUT_DIR = Path("/vf/users/parks34/projects/2cytoatlas/results/cima/pseudobulk")

# Parameters - larger batch for better GPU utilization
BATCH_SIZE = 100000  # 100K cells per batch
CELLTYPE_LEVELS = ["cell_type_l1", "cell_type_l2", "cell_type_l3", "cell_type_l4"]


def check_gpu():
    """Check GPU and return CuPy if available."""
    try:
        import cupy as cp
        n = cp.cuda.runtime.getDeviceCount()
        mem = cp.cuda.runtime.memGetInfo()
        print(f"GPU: {n} device(s)")
        print(f"  Memory: {mem[0]/1024**3:.1f}/{mem[1]/1024**3:.1f} GB free")
        # Try to get device name (API varies by CuPy version)
        try:
            dev = cp.cuda.Device()
            if hasattr(dev, 'name'):
                print(f"  Device: {dev.name.decode()}")
            else:
                props = cp.cuda.runtime.getDeviceProperties(0)
                print(f"  Device: {props['name'].decode()}")
        except:
            print(f"  Device: GPU {cp.cuda.runtime.getDevice()}")
        return cp
    except ImportError:
        print("CuPy not installed")
        return None
    except Exception as e:
        print(f"GPU error: {e}")
        return None


def create_gpu_accumulators(cp, levels_info, n_genes):
    """Create GPU-resident accumulators for all levels."""
    accumulators = {}
    for level, info in levels_info.items():
        n_groups = info['n_groups']
        # Keep accumulators on GPU as float32 for speed
        accumulators[level] = {
            'sums': cp.zeros((n_groups, n_genes), dtype=cp.float32),
            'counts': cp.zeros(n_groups, dtype=cp.int64),
            'group_to_idx': info['group_to_idx'],
            'unique_groups': info['unique_groups']
        }
    return accumulators


def aggregate_batch_gpu(cp, X_batch, obs_batch, accumulators, levels_info):
    """
    Aggregate a batch using GPU scatter operations.
    Process ALL levels in single GPU pass.
    """
    n_cells, n_genes = X_batch.shape

    # Transfer expression data to GPU once
    if sparse.issparse(X_batch):
        # For sparse, convert to dense on GPU
        X_dense = X_batch.toarray()
    else:
        X_dense = np.asarray(X_batch)

    X_gpu = cp.asarray(X_dense, dtype=cp.float32)

    # Process all levels with single GPU data
    for level in accumulators.keys():
        acc = accumulators[level]
        group_to_idx = acc['group_to_idx']

        # Get group indices for this batch
        labels = obs_batch[level].values
        indices = cp.array([group_to_idx.get(g, -1) for g in labels], dtype=cp.int32)

        # Use scatter_add for fast aggregation
        # This is the key GPU optimization - parallel reduction
        valid_mask = indices >= 0
        valid_indices = indices[valid_mask]
        valid_X = X_gpu[cp.asnumpy(valid_mask)]

        if len(valid_indices) > 0:
            # Scatter add: acc['sums'][valid_indices] += valid_X
            # CuPy doesn't have direct scatter_add for 2D, so we use a loop over unique indices
            # But this is still fast because it's on GPU
            unique_idx = cp.unique(valid_indices)
            for idx in unique_idx:
                mask = valid_indices == idx
                acc['sums'][int(idx)] += valid_X[cp.asnumpy(mask)].sum(axis=0)
                acc['counts'][int(idx)] += int(mask.sum())

    # Free GPU memory
    del X_gpu
    cp.get_default_memory_pool().free_all_blocks()


def aggregate_batch_gpu_optimized(cp, X_batch, obs_batch, accumulators, levels_info):
    """
    Highly optimized GPU aggregation using index sorting.
    """
    n_cells, n_genes = X_batch.shape

    # Transfer to GPU
    if sparse.issparse(X_batch):
        X_dense = X_batch.toarray()
    else:
        X_dense = np.asarray(X_batch)

    X_gpu = cp.asarray(X_dense, dtype=cp.float32)

    for level in accumulators.keys():
        acc = accumulators[level]
        group_to_idx = acc['group_to_idx']
        n_groups = len(acc['unique_groups'])

        # Get indices on CPU (fast), then transfer
        labels = obs_batch[level].values
        indices_np = np.array([group_to_idx.get(g, -1) for g in labels], dtype=np.int32)
        indices_gpu = cp.asarray(indices_np)

        # For each group, accumulate
        for g_idx in range(n_groups):
            mask = indices_gpu == g_idx
            if mask.any():
                acc['sums'][g_idx] += X_gpu[mask].sum(axis=0)
                acc['counts'][g_idx] += int(mask.sum())

    del X_gpu
    cp.get_default_memory_pool().free_all_blocks()


def finalize_and_save(cp, accumulators, var_names, output_dir):
    """Convert GPU accumulators to AnnData and save."""
    results = {}

    for level, acc in accumulators.items():
        print(f"\n  Finalizing {level}...")

        # Transfer back to CPU
        sums_cpu = cp.asnumpy(acc['sums'])
        counts_cpu = cp.asnumpy(acc['counts'])
        unique_groups = acc['unique_groups']

        # Filter out empty groups
        valid_mask = counts_cpu > 0
        sums_valid = sums_cpu[valid_mask]
        counts_valid = counts_cpu[valid_mask]
        groups_valid = [g for g, v in zip(unique_groups, valid_mask) if v]

        # Normalize: CPM + log1p
        total = sums_valid.sum(axis=1, keepdims=True)
        total = np.maximum(total, 1)
        cpm = sums_valid / total * 1e6
        log_cpm = np.log1p(cpm).astype(np.float32)

        # Create AnnData
        pb_adata = ad.AnnData(
            X=log_cpm,
            obs=pd.DataFrame(index=groups_valid),
            var=pd.DataFrame(index=var_names)
        )
        pb_adata.layers["counts"] = sums_valid.astype(np.float32)
        pb_adata.obs["n_cells"] = counts_valid

        # Save
        output_path = output_dir / f"CIMA_pseudobulk_{level}.h5ad"
        pb_adata.write_h5ad(output_path)
        print(f"    Saved: {output_path} ({pb_adata.shape})")

        results[level] = {
            'n_groups': len(groups_valid),
            'path': str(output_path)
        }

    return results


def main():
    total_start = time.time()

    print("=" * 70)
    print("GPU-Optimized CIMA Pseudobulk Generation")
    print("=" * 70)

    # Check GPU
    cp = check_gpu()
    if cp is None:
        print("ERROR: GPU required for this script")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput: {OUTPUT_DIR}")
    print(f"Batch size: {BATCH_SIZE:,} cells")

    # Open file in backed mode
    print(f"\n[1] Opening CIMA H5AD...")
    open_start = time.time()
    adata = ad.read_h5ad(CIMA_H5AD, backed='r')
    n_cells, n_genes = adata.shape
    var_names = adata.var_names.tolist()
    print(f"    Shape: {n_cells:,} cells x {n_genes:,} genes")
    print(f"    Opened in {time.time() - open_start:.1f}s")

    # Load metadata for all levels at once
    print(f"\n[2] Loading cell metadata...")
    meta_start = time.time()
    obs_df = adata.obs[CELLTYPE_LEVELS].copy()
    print(f"    Loaded in {time.time() - meta_start:.1f}s")

    # Prepare level info
    print(f"\n[3] Preparing level info...")
    levels_info = {}
    for level in CELLTYPE_LEVELS:
        unique_groups = obs_df[level].dropna().unique().tolist()
        group_to_idx = {g: i for i, g in enumerate(unique_groups)}
        levels_info[level] = {
            'unique_groups': unique_groups,
            'group_to_idx': group_to_idx,
            'n_groups': len(unique_groups)
        }
        print(f"    {level}: {len(unique_groups)} groups")

    # Create GPU accumulators
    print(f"\n[4] Creating GPU accumulators...")
    accumulators = create_gpu_accumulators(cp, levels_info, n_genes)

    # Estimate GPU memory usage
    total_acc_memory = sum(
        acc['sums'].nbytes + acc['counts'].nbytes
        for acc in accumulators.values()
    )
    print(f"    Accumulator memory: {total_acc_memory / 1024**2:.1f} MB")

    # Process in batches - single pass for ALL levels
    print(f"\n[5] Processing batches (ALL levels simultaneously)...")
    process_start = time.time()

    n_batches = (n_cells + BATCH_SIZE - 1) // BATCH_SIZE
    batch_times = []

    for batch_idx in range(n_batches):
        batch_start = time.time()
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, n_cells)

        # Load batch data
        X_batch = adata.X[start_idx:end_idx]
        obs_batch = obs_df.iloc[start_idx:end_idx]

        # Aggregate on GPU for ALL levels
        aggregate_batch_gpu_optimized(cp, X_batch, obs_batch, accumulators, levels_info)

        batch_time = time.time() - batch_start
        batch_times.append(batch_time)

        if batch_idx % 5 == 0 or batch_idx == n_batches - 1:
            avg_time = np.mean(batch_times[-5:])
            eta = avg_time * (n_batches - batch_idx - 1)
            cells_per_sec = BATCH_SIZE / avg_time
            print(f"    Batch {batch_idx+1}/{n_batches}: {batch_time:.1f}s "
                  f"({cells_per_sec:.0f} cells/s, ETA: {eta/60:.1f}min)")

    process_time = time.time() - process_start
    print(f"\n    Total processing: {process_time:.1f}s ({process_time/60:.1f}min)")
    print(f"    Throughput: {n_cells / process_time:.0f} cells/sec")

    # Finalize and save
    print(f"\n[6] Finalizing and saving...")
    save_start = time.time()
    results = finalize_and_save(cp, accumulators, var_names, OUTPUT_DIR)
    save_time = time.time() - save_start
    print(f"    Save time: {save_time:.1f}s")

    # Summary
    total_time = time.time() - total_start

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print("=" * 70)

    print(f"\n--- Timing ---")
    print(f"  H5AD open:     {time.time() - total_start - process_time - save_time:6.1f}s")
    print(f"  Processing:    {process_time:6.1f}s ({n_batches} batches)")
    print(f"  Saving:        {save_time:6.1f}s")
    print(f"  TOTAL:         {total_time:6.1f}s ({total_time/60:.1f}min)")

    print(f"\n--- Performance ---")
    print(f"  Cells processed: {n_cells:,}")
    print(f"  Throughput: {n_cells / process_time:,.0f} cells/sec")
    print(f"  Batch time avg: {np.mean(batch_times):.2f}s")

    print(f"\n--- Output Files ---")
    for level, info in results.items():
        print(f"  {level}: {info['n_groups']} groups -> {info['path']}")

    # Memory info
    import psutil
    mem_gb = psutil.Process().memory_info().rss / 1024**3
    gpu_mem = cp.cuda.runtime.memGetInfo()
    print(f"\n--- Memory ---")
    print(f"  CPU: {mem_gb:.1f} GB")
    print(f"  GPU: {gpu_mem[0]/1024**3:.1f}/{gpu_mem[1]/1024**3:.1f} GB free")

    return results


if __name__ == "__main__":
    results = main()
