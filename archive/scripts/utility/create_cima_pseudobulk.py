#!/usr/bin/env python
"""
Create celltype-specific pseudobulk from CIMA raw expression data.

Generates pseudobulk aggregations at L1, L2, L3, L4 cell type levels.
Uses GPU acceleration where possible.

Output: results/cima/pseudobulk/CIMA_pseudobulk_{level}.h5ad
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
OUTPUT_DIR = Path("/vf/users/parks34/projects/2secactpy/results/cima/pseudobulk")

# Cell type levels
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


def create_pseudobulk_gpu(X, group_labels, unique_groups):
    """Create pseudobulk using GPU (CuPy)."""
    import cupy as cp
    from cupyx.scipy import sparse as cp_sparse

    n_cells, n_genes = X.shape
    n_groups = len(unique_groups)

    # Transfer data to GPU
    if sparse.issparse(X):
        X_gpu = cp_sparse.csr_matrix(X)
    else:
        X_gpu = cp.asarray(X)

    # Create group index mapping
    group_to_idx = {g: i for i, g in enumerate(unique_groups)}
    group_indices = cp.array([group_to_idx[g] for g in group_labels])

    # Sum per group using scatter_add
    result = cp.zeros((n_groups, n_genes), dtype=cp.float32)

    for i, g in enumerate(unique_groups):
        mask = group_indices == i
        if sparse.issparse(X):
            group_sum = X_gpu[cp.asnumpy(mask)].sum(axis=0)
            result[i] = cp.asarray(group_sum).flatten()
        else:
            result[i] = X_gpu[mask].sum(axis=0)

    return cp.asnumpy(result)


def create_pseudobulk_cpu(X, group_labels, unique_groups):
    """Create pseudobulk using CPU (NumPy/SciPy)."""
    n_groups = len(unique_groups)
    n_genes = X.shape[1]

    result = np.zeros((n_groups, n_genes), dtype=np.float32)

    for i, g in enumerate(unique_groups):
        mask = group_labels == g
        if sparse.issparse(X):
            group_sum = np.asarray(X[mask].sum(axis=0)).flatten()
        else:
            group_sum = X[mask].sum(axis=0)
        result[i] = group_sum

    return result


def normalize_cpm(counts, log1p=True):
    """Normalize to CPM and optionally log-transform."""
    # Sum per sample (column)
    total = counts.sum(axis=1, keepdims=True)
    total = np.maximum(total, 1)  # Avoid division by zero
    cpm = counts / total * 1e6

    if log1p:
        cpm = np.log1p(cpm)

    return cpm


def main():
    total_start = time.time()

    print("=" * 70)
    print("CIMA Celltype Pseudobulk Generation")
    print("=" * 70)

    # Check GPU
    use_gpu = check_gpu()

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {OUTPUT_DIR}")

    # Load data
    print(f"\n[1] Loading CIMA H5AD...")
    load_start = time.time()

    adata = ad.read_h5ad(CIMA_H5AD)  # Load fully into memory

    load_time = time.time() - load_start
    print(f"    Loaded in {load_time:.1f}s")
    print(f"    Shape: {adata.n_obs:,} cells x {adata.n_vars:,} genes")
    print(f"    X type: {type(adata.X)}, dtype: {adata.X.dtype}")

    # Memory info
    import psutil
    proc = psutil.Process()
    mem_gb = proc.memory_info().rss / 1024**3
    print(f"    Memory: {mem_gb:.1f} GB")

    # Process each level
    results = {}

    for level in CELLTYPE_LEVELS:
        print(f"\n[{level}] Creating pseudobulk...")
        level_start = time.time()

        # Get group labels
        group_labels = adata.obs[level].values
        unique_groups = adata.obs[level].unique()
        n_groups = len(unique_groups)

        print(f"    Groups: {n_groups}")

        # Create pseudobulk
        if use_gpu:
            try:
                pseudobulk = create_pseudobulk_gpu(adata.X, group_labels, unique_groups)
                method = "GPU"
            except Exception as e:
                print(f"    GPU failed ({e}), falling back to CPU")
                pseudobulk = create_pseudobulk_cpu(adata.X, group_labels, unique_groups)
                method = "CPU"
        else:
            pseudobulk = create_pseudobulk_cpu(adata.X, group_labels, unique_groups)
            method = "CPU"

        # Normalize
        pseudobulk_cpm = normalize_cpm(pseudobulk, log1p=True)

        # Create AnnData
        pb_adata = ad.AnnData(
            X=pseudobulk_cpm,
            obs=pd.DataFrame(index=unique_groups),
            var=adata.var.copy()
        )

        # Add raw counts as layer
        pb_adata.layers["counts"] = pseudobulk

        # Add cell counts per group
        cell_counts = pd.Series(group_labels).value_counts()
        pb_adata.obs["n_cells"] = cell_counts[pb_adata.obs_names].values

        level_time = time.time() - level_start
        print(f"    Shape: {pb_adata.shape}")
        print(f"    Time: {level_time:.1f}s ({method})")

        # Save
        output_path = OUTPUT_DIR / f"CIMA_pseudobulk_{level}.h5ad"
        pb_adata.write_h5ad(output_path)
        print(f"    Saved: {output_path}")

        results[level] = {
            "n_groups": n_groups,
            "time": level_time,
            "method": method,
            "path": str(output_path)
        }

    # Summary
    total_time = time.time() - total_start

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print("=" * 70)

    print(f"\n--- Timing ---")
    print(f"  Data loading:  {load_time:6.1f}s")
    for level, info in results.items():
        print(f"  {level}: {info['time']:6.1f}s ({info['n_groups']} groups, {info['method']})")
    print(f"  TOTAL:         {total_time:6.1f}s")

    print(f"\n--- Output Files ---")
    for level, info in results.items():
        print(f"  {level}: {info['path']}")

    print(f"\n--- Memory ---")
    mem_gb = proc.memory_info().rss / 1024**3
    print(f"  Peak process memory: {mem_gb:.1f} GB")

    return results


if __name__ == "__main__":
    results = main()
