"""
Streaming pseudobulk aggregator with GPU acceleration.

Provides memory-efficient batch processing for large single-cell atlases
using backed H5AD reading and GPU-accelerated aggregation.
"""

import gc
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

from cytoatlas_pipeline.batch.atlas_config import (
    AtlasConfig,
    PseudobulkConfig,
    get_atlas_config,
)

# Try importing CuPy for GPU acceleration
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None


def log(msg: str):
    """Print timestamped log message."""
    timestamp = time.strftime('%H:%M:%S')
    print(f"[{timestamp}] {msg}", flush=True)


class StreamingPseudobulkAggregator:
    """
    GPU-accelerated streaming pseudobulk aggregator.

    Processes large single-cell atlases in batches using backed H5AD mode,
    aggregating expression by cell type groups with GPU acceleration.

    Key Features:
    - Memory-efficient backed mode reading
    - GPU-accelerated aggregation with CuPy
    - CPM normalization + log1p transformation
    - Atlas-level z-score normalization
    - Checkpoint support for resumption

    Output H5AD Structure:
    - X: log1p(CPM) normalized expression
    - layers['counts']: Raw sum counts
    - layers['zscore']: Atlas-level z-scored expression
    - obs: Cell type metadata with n_cells counts
    - var: Gene metadata
    - uns['atlas_stats']: Atlas-level gene mean/std for z-score calculation
    """

    def __init__(
        self,
        atlas_config: Union[str, AtlasConfig],
        pseudobulk_config: Optional[PseudobulkConfig] = None,
        use_gpu: bool = True,
    ):
        """
        Initialize aggregator.

        Args:
            atlas_config: Atlas name (str) or AtlasConfig instance
            pseudobulk_config: Pseudobulk generation parameters
            use_gpu: Whether to use GPU acceleration if available
        """
        if isinstance(atlas_config, str):
            self.atlas_config = get_atlas_config(atlas_config)
        else:
            self.atlas_config = atlas_config

        self.pb_config = pseudobulk_config or PseudobulkConfig()
        self.use_gpu = use_gpu and CUPY_AVAILABLE

        if self.use_gpu:
            self._check_gpu()

    def _check_gpu(self) -> bool:
        """Check GPU availability and memory."""
        try:
            n = cp.cuda.runtime.getDeviceCount()
            mem = cp.cuda.runtime.memGetInfo()
            log(f"GPU: {n} device(s), {mem[0]/1024**3:.1f}/{mem[1]/1024**3:.1f} GB free")
            return True
        except Exception as e:
            log(f"GPU: Not available ({e})")
            self.use_gpu = False
            return False

    def aggregate(
        self,
        level: str,
        output_path: Path,
        checkpoint_dir: Optional[Path] = None,
    ) -> ad.AnnData:
        """
        Aggregate expression by cell type at specified annotation level.

        Args:
            level: Annotation level (e.g., "L1", "L2")
            output_path: Path to save output H5AD
            checkpoint_dir: Directory for checkpoints (optional)

        Returns:
            AnnData with pseudobulk expression
        """
        h5ad_path = self.atlas_config.h5ad_path
        celltype_col = self.atlas_config.get_level_column(level)
        batch_size = self.pb_config.batch_size

        log("=" * 70)
        log(f"Pseudobulk Generation: {self.atlas_config.name} / {level}")
        log("=" * 70)
        log(f"H5AD: {h5ad_path}")
        log(f"Cell type column: {celltype_col}")
        log(f"Batch size: {batch_size:,}")
        log(f"Backend: {'CuPy (GPU)' if self.use_gpu else 'NumPy (CPU)'}")

        # Open file in backed mode
        log("Opening H5AD in backed mode...")
        start_time = time.time()
        adata = ad.read_h5ad(h5ad_path, backed='r')
        n_cells, n_genes = adata.shape
        log(f"  Shape: {n_cells:,} cells x {n_genes:,} genes")
        log(f"  Opened in {time.time() - start_time:.1f}s")

        # Get gene names
        var_names = adata.var_names.tolist()

        # Load cell metadata
        log("Loading cell metadata...")
        meta_start = time.time()

        # Handle different column scenarios
        if celltype_col in adata.obs.columns:
            obs_df = pd.DataFrame({celltype_col: adata.obs[celltype_col]})
        elif celltype_col == "organ_cellType1":
            # Combined annotation for scAtlas
            obs_df = pd.DataFrame({
                'organ': adata.obs['organ'],
                'cellType1': adata.obs['cellType1'],
            })
            obs_df[celltype_col] = obs_df['organ'] + '_' + obs_df['cellType1']
        else:
            raise ValueError(f"Column '{celltype_col}' not found in obs")

        log(f"  Loaded in {time.time() - meta_start:.1f}s")

        # Get unique groups
        unique_groups = sorted(obs_df[celltype_col].dropna().unique().tolist())
        group_to_idx = {g: i for i, g in enumerate(unique_groups)}
        n_groups = len(unique_groups)
        log(f"  Groups: {n_groups}")

        # Initialize accumulators
        accumulators = {g: np.zeros(n_genes, dtype=np.float64) for g in unique_groups}
        cell_counts = defaultdict(int)

        # For atlas-level statistics (mean/std per gene)
        # We use Welford's online algorithm for numerical stability
        gene_count = np.zeros(n_genes, dtype=np.int64)
        gene_mean = np.zeros(n_genes, dtype=np.float64)
        gene_m2 = np.zeros(n_genes, dtype=np.float64)

        # Process in batches
        n_batches = (n_cells + batch_size - 1) // batch_size
        log(f"Processing {n_batches} batches...")

        batch_times = []
        group_labels_all = obs_df[celltype_col].values

        for batch_idx in range(n_batches):
            batch_start = time.time()
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_cells)

            # Load batch
            X_batch = adata.X[start_idx:end_idx]
            group_labels = group_labels_all[start_idx:end_idx]

            # Count cells per group
            for g in group_labels:
                if g is not None and not pd.isna(g):
                    cell_counts[g] += 1

            # Aggregate by group
            if self.use_gpu:
                try:
                    self._aggregate_batch_gpu(
                        X_batch, group_labels, group_to_idx, accumulators
                    )
                except Exception as e:
                    if batch_idx == 0:
                        log(f"  GPU failed, falling back to CPU: {e}")
                        self.use_gpu = False
                    self._aggregate_batch_cpu(
                        X_batch, group_labels, group_to_idx, accumulators
                    )
            else:
                self._aggregate_batch_cpu(
                    X_batch, group_labels, group_to_idx, accumulators
                )

            # Update atlas-level statistics (for z-score normalization)
            self._update_atlas_stats(
                X_batch, gene_count, gene_mean, gene_m2
            )

            batch_time = time.time() - batch_start
            batch_times.append(batch_time)

            if batch_idx % 10 == 0 or batch_idx == n_batches - 1:
                avg_time = np.mean(batch_times[-10:])
                eta = avg_time * (n_batches - batch_idx - 1)
                log(f"  Batch {batch_idx+1}/{n_batches} ({batch_time:.1f}s, ETA: {eta/60:.1f}min)")

        # Compute atlas-level standard deviation
        gene_var = np.divide(
            gene_m2, gene_count,
            out=np.zeros_like(gene_m2),
            where=gene_count > 1
        )
        gene_std = np.sqrt(gene_var)
        gene_std[gene_std == 0] = 1.0  # Avoid division by zero

        # Build pseudobulk matrix
        log("Building pseudobulk matrix...")
        raw_counts = np.vstack([accumulators[g] for g in unique_groups])

        # Normalize to CPM + log1p
        if self.pb_config.normalize:
            total = raw_counts.sum(axis=1, keepdims=True)
            total = np.maximum(total, 1)
            cpm = raw_counts / total * 1e6
        else:
            cpm = raw_counts.copy()

        if self.pb_config.log_transform:
            log_cpm = np.log1p(cpm)
        else:
            log_cpm = cpm.copy()

        # Compute z-scored expression (using atlas-level statistics)
        if self.pb_config.zscore:
            zscore_expr = (log_cpm - gene_mean) / gene_std
        else:
            zscore_expr = log_cpm.copy()

        # Create AnnData
        log("Creating AnnData...")
        pb_adata = ad.AnnData(
            X=log_cpm.astype(np.float32),
            obs=pd.DataFrame(
                {
                    'cell_type': unique_groups,
                    'n_cells': [cell_counts[g] for g in unique_groups],
                },
                index=unique_groups,
            ),
            var=pd.DataFrame(index=var_names),
        )

        # Add layers
        pb_adata.layers['counts'] = raw_counts.astype(np.float32)
        pb_adata.layers['zscore'] = zscore_expr.astype(np.float32)

        # Store atlas statistics in uns for reference
        pb_adata.uns['atlas_stats'] = {
            'gene_mean': gene_mean.astype(np.float32),
            'gene_std': gene_std.astype(np.float32),
            'n_cells_total': int(gene_count[0]) if len(gene_count) > 0 else 0,
        }

        # Metadata
        pb_adata.uns['atlas'] = self.atlas_config.name
        pb_adata.uns['level'] = level
        pb_adata.uns['celltype_col'] = celltype_col

        # Save
        log(f"Saving to {output_path}...")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pb_adata.write_h5ad(output_path, compression=self.pb_config.output_compression)

        total_time = time.time() - start_time
        log(f"Completed in {total_time/60:.1f} min")
        log(f"  Shape: {pb_adata.shape}")
        log(f"  Output: {output_path}")

        return pb_adata

    def _aggregate_batch_gpu(
        self,
        X_batch: Union[np.ndarray, sparse.spmatrix],
        group_labels: np.ndarray,
        group_to_idx: Dict[str, int],
        accumulators: Dict[str, np.ndarray],
    ):
        """Aggregate batch using GPU."""
        # Transfer to GPU
        if sparse.issparse(X_batch):
            X_dense = X_batch.toarray()
        else:
            X_dense = np.asarray(X_batch)

        X_gpu = cp.asarray(X_dense, dtype=cp.float32)

        # Aggregate per group
        for g in np.unique(group_labels):
            if g is None or pd.isna(g) or g not in group_to_idx:
                continue
            mask = group_labels == g
            group_sum = cp.asnumpy(X_gpu[mask].sum(axis=0))
            accumulators[g] += group_sum.flatten()

        # Free GPU memory
        del X_gpu
        cp.get_default_memory_pool().free_all_blocks()

    def _aggregate_batch_cpu(
        self,
        X_batch: Union[np.ndarray, sparse.spmatrix],
        group_labels: np.ndarray,
        group_to_idx: Dict[str, int],
        accumulators: Dict[str, np.ndarray],
    ):
        """Aggregate batch using CPU."""
        for g in np.unique(group_labels):
            if g is None or pd.isna(g) or g not in group_to_idx:
                continue
            mask = group_labels == g
            if sparse.issparse(X_batch):
                group_sum = np.asarray(X_batch[mask].sum(axis=0)).flatten()
            else:
                group_sum = X_batch[mask].sum(axis=0)
            accumulators[g] += group_sum

    def _update_atlas_stats(
        self,
        X_batch: Union[np.ndarray, sparse.spmatrix],
        gene_count: np.ndarray,
        gene_mean: np.ndarray,
        gene_m2: np.ndarray,
    ):
        """
        Update atlas-level gene statistics using Chan's parallel algorithm.

        This computes running mean and variance across all cells in the atlas,
        which is used for z-score normalization of pseudobulk expression.

        Uses vectorized batch statistics + parallel merge (much faster than
        cell-by-cell Welford's algorithm).

        Note: We compute statistics on log1p(CPM) normalized values to match
        what will be z-scored in the final output.
        """
        if sparse.issparse(X_batch):
            X_dense = X_batch.toarray()
        else:
            X_dense = np.asarray(X_batch)

        # Normalize batch to CPM + log1p for consistent statistics
        row_sums = X_dense.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1)
        cpm = X_dense / row_sums * 1e6
        log_cpm = np.log1p(cpm)

        # Vectorized batch statistics
        n_batch = log_cpm.shape[0]
        batch_mean = log_cpm.mean(axis=0)
        batch_var = log_cpm.var(axis=0, ddof=0)  # Population variance
        batch_m2 = batch_var * n_batch

        # Chan's parallel algorithm to merge batch stats with running stats
        n_a = gene_count[0] if gene_count[0] > 0 else 0
        n_b = n_batch
        n_total = n_a + n_b

        if n_a == 0:
            # First batch - just copy
            gene_count[:] = n_batch
            gene_mean[:] = batch_mean
            gene_m2[:] = batch_m2
        else:
            # Merge using Chan's algorithm
            delta = batch_mean - gene_mean
            gene_mean[:] = (n_a * gene_mean + n_b * batch_mean) / n_total
            gene_m2[:] = gene_m2 + batch_m2 + delta ** 2 * n_a * n_b / n_total
            gene_count[:] = n_total


def aggregate_atlas_pseudobulk(
    atlas_name: str,
    level: str,
    output_dir: Path,
    use_gpu: bool = True,
    **kwargs,
) -> ad.AnnData:
    """
    Convenience function to aggregate pseudobulk for an atlas/level.

    Args:
        atlas_name: Name of the atlas (e.g., "cima", "inflammation_main")
        level: Annotation level (e.g., "L1", "L2")
        output_dir: Output directory
        use_gpu: Whether to use GPU acceleration
        **kwargs: Additional arguments for PseudobulkConfig

    Returns:
        AnnData with pseudobulk expression
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pb_config = PseudobulkConfig(**kwargs)
    aggregator = StreamingPseudobulkAggregator(
        atlas_config=atlas_name,
        pseudobulk_config=pb_config,
        use_gpu=use_gpu,
    )

    output_path = output_dir / f"{atlas_name}_{level}_pseudobulk.h5ad"
    return aggregator.aggregate(level=level, output_path=output_path)
