"""
Multi-level streaming pseudobulk aggregator.

Generates pseudobulk for ALL annotation levels in a single pass through the H5AD file,
plus bootstrap resampled pseudobulk for confidence interval estimation.

Output files (8 total for CIMA):
- {atlas}_pseudobulk_l1.h5ad
- {atlas}_pseudobulk_l2.h5ad
- {atlas}_pseudobulk_l3.h5ad
- {atlas}_pseudobulk_l4.h5ad
- {atlas}_pseudobulk_l1_resampled.h5ad (K bootstrap samples)
- {atlas}_pseudobulk_l2_resampled.h5ad
- {atlas}_pseudobulk_l3_resampled.h5ad
- {atlas}_pseudobulk_l4_resampled.h5ad
"""

import gc
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


class MultiLevelAggregator:
    """
    Multi-level streaming pseudobulk aggregator.

    Processes all annotation levels in a single pass through the H5AD file,
    generating both standard pseudobulk and bootstrap resampled versions.

    For CIMA with L1-L4 levels, this produces 8 output files in one ~60 min pass
    instead of 4 separate passes taking ~4 hours.

    Bootstrap Resampling Strategy:
    - Aggregate to (sample × cell_type) level during streaming
    - After processing, bootstrap resample samples within each cell type
    - This provides confidence intervals for activity inference
    """

    def __init__(
        self,
        atlas_config: Union[str, AtlasConfig],
        levels: Optional[List[str]] = None,
        batch_size: int = 50000,
        n_bootstrap: int = 100,
        min_cells_per_group: int = 10,
        min_samples_for_bootstrap: int = 5,
        use_gpu: bool = True,
        seed: int = 42,
    ):
        """
        Initialize multi-level aggregator.

        Args:
            atlas_config: Atlas name or AtlasConfig instance
            levels: Annotation levels to process (default: all available)
            batch_size: Cells per batch
            n_bootstrap: Number of bootstrap resamples
            min_cells_per_group: Minimum cells for a group to be included
            min_samples_for_bootstrap: Minimum samples for bootstrap resampling
            use_gpu: Use GPU acceleration if available
            seed: Random seed for bootstrap
        """
        if isinstance(atlas_config, str):
            self.atlas_config = get_atlas_config(atlas_config)
        else:
            self.atlas_config = atlas_config

        self.levels = levels or list(self.atlas_config.annotation_levels.keys())
        self.batch_size = batch_size
        self.n_bootstrap = n_bootstrap
        self.min_cells = min_cells_per_group
        self.min_samples_bootstrap = min_samples_for_bootstrap
        self.use_gpu = use_gpu and CUPY_AVAILABLE
        self.seed = seed

        if self.use_gpu:
            self._check_gpu()

    def _check_gpu(self) -> bool:
        """Check GPU availability."""
        try:
            n = cp.cuda.runtime.getDeviceCount()
            mem = cp.cuda.runtime.memGetInfo()
            log(f"GPU: {n} device(s), {mem[0]/1024**3:.1f}/{mem[1]/1024**3:.1f} GB free")
            return True
        except Exception as e:
            log(f"GPU not available: {e}")
            self.use_gpu = False
            return False

    def aggregate_all_levels(
        self,
        output_dir: Path,
        sample_col: Optional[str] = None,
        skip_existing: bool = True,
    ) -> Dict[str, Path]:
        """
        Aggregate pseudobulk for all levels in a single pass.

        Args:
            output_dir: Output directory for H5AD files
            sample_col: Sample/donor column for sample-level aggregation
            skip_existing: Skip levels that already have output files

        Returns:
            Dict mapping level names to output file paths
        """
        h5ad_path = self.atlas_config.h5ad_path
        sample_col = sample_col or self.atlas_config.sample_col

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Check which levels need processing
        levels_to_process = []
        output_paths = {}

        for level in self.levels:
            out_path = output_dir / f"{self.atlas_config.name.lower()}_pseudobulk_{level.lower()}.h5ad"
            output_paths[level] = out_path

            if skip_existing and out_path.exists():
                log(f"Skipping {level} - already exists: {out_path}")
            else:
                levels_to_process.append(level)

        if not levels_to_process:
            log("All levels already processed, nothing to do")
            return output_paths

        log("=" * 70)
        log(f"Multi-Level Pseudobulk Generation: {self.atlas_config.name}")
        log("=" * 70)
        log(f"H5AD: {h5ad_path}")
        log(f"Levels to process: {levels_to_process}")
        log(f"Batch size: {self.batch_size:,}")
        log(f"Bootstrap samples: {self.n_bootstrap}")
        log(f"Backend: {'CuPy (GPU)' if self.use_gpu else 'NumPy (CPU)'}")

        # Open file in backed mode
        log("Opening H5AD in backed mode...")
        start_time = time.time()
        adata = ad.read_h5ad(h5ad_path, backed='r')
        n_cells, n_genes = adata.shape
        open_time = time.time() - start_time
        log(f"  Shape: {n_cells:,} cells × {n_genes:,} genes")
        log(f"  Opened in {open_time:.1f}s ({open_time/60:.1f} min)")

        # Get gene names
        var_names = list(adata.var_names)

        # Load metadata for all levels
        log("Loading cell metadata...")
        level_columns = {level: self.atlas_config.get_level_column(level)
                        for level in levels_to_process}

        # Build obs DataFrame with all needed columns
        # Handle composite columns (e.g., "tissue+cellType1" -> creates "tissue_cellType1")
        cols_to_load = set()
        composite_columns = {}  # {composite_col_name: [col1, col2]}

        for level, col_spec in level_columns.items():
            if '+' in col_spec:
                # Composite column: "col1+col2" -> combine into "col1_col2"
                parts = col_spec.split('+')
                cols_to_load.update(parts)
                composite_col_name = '_'.join(parts)
                composite_columns[col_spec] = (parts, composite_col_name)
                level_columns[level] = composite_col_name  # Update to use composite name
            else:
                cols_to_load.add(col_spec)

        if sample_col and sample_col in adata.obs.columns:
            cols_to_load.add(sample_col)

        obs_df = adata.obs[list(cols_to_load)].copy()
        log(f"  Loaded {len(cols_to_load)} columns")

        # Create composite columns (handle NaN values)
        for col_spec, (parts, composite_name) in composite_columns.items():
            # Check for NaN in component columns
            mask_na0 = obs_df[parts[0]].isna()
            mask_na1 = obs_df[parts[1]].isna()
            mask_any_na = mask_na0 | mask_na1

            # Convert to string for concatenation
            part0 = obs_df[parts[0]].astype(str)
            part1 = obs_df[parts[1]].astype(str)

            # Create composite as regular string column (not categorical)
            composite_values = part0 + '_' + part1

            # Set NaN for rows with missing values in either component
            # Use pd.Series to ensure it's not categorical
            composite_series = pd.Series(composite_values.values, index=obs_df.index, dtype=object)
            composite_series.loc[mask_any_na] = np.nan
            obs_df[composite_name] = composite_series

            n_missing = mask_any_na.sum()
            log(f"  Created composite column: {composite_name}")
            if n_missing > 0:
                log(f"    ({n_missing:,} cells with missing annotations excluded)")

        # Initialize accumulators for each level
        # For cell-type-only pseudobulk
        celltype_accum = {}  # {level: {celltype: sum_array}}
        celltype_counts = {}  # {level: {celltype: n_cells}}

        # For sample × celltype pseudobulk (for bootstrap)
        sample_celltype_accum = {}  # {level: {(sample, celltype): sum_array}}
        sample_celltype_counts = {}  # {level: {(sample, celltype): n_cells}}

        # For atlas-level statistics (z-score normalization)
        # Use array for gene_count to make it mutable in _update_atlas_stats
        gene_count = np.array([0], dtype=np.int64)
        gene_mean = np.zeros(n_genes, dtype=np.float64)
        gene_m2 = np.zeros(n_genes, dtype=np.float64)

        for level in levels_to_process:
            col = level_columns[level]
            unique_groups = sorted(obs_df[col].dropna().unique().tolist())

            celltype_accum[level] = {g: np.zeros(n_genes, dtype=np.float64) for g in unique_groups}
            celltype_counts[level] = defaultdict(int)
            sample_celltype_accum[level] = defaultdict(lambda: np.zeros(n_genes, dtype=np.float64))
            sample_celltype_counts[level] = defaultdict(int)

            log(f"  {level}: {len(unique_groups)} groups")

        # Process in batches
        n_batches = (n_cells + self.batch_size - 1) // self.batch_size
        log(f"Processing {n_batches} batches...")

        batch_times = []

        for batch_idx in range(n_batches):
            batch_start = time.time()
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, n_cells)

            # Load batch expression
            X_batch = adata.X[start_idx:end_idx]
            if sparse.issparse(X_batch):
                X_dense = X_batch.toarray()
            else:
                X_dense = np.asarray(X_batch)

            # Get batch metadata
            batch_obs = obs_df.iloc[start_idx:end_idx]
            batch_samples = batch_obs[sample_col].values if sample_col else None

            # Process each level
            for level in levels_to_process:
                col = level_columns[level]
                batch_celltypes = batch_obs[col].values

                # Filter out NaN values before np.unique to avoid type comparison error
                # (NaN is float, celltypes are strings)
                valid_mask = pd.notna(batch_celltypes)
                valid_celltypes = batch_celltypes[valid_mask]

                # Aggregate by cell type
                for ct in np.unique(valid_celltypes) if len(valid_celltypes) > 0 else []:
                    if ct is None or pd.isna(ct):
                        continue

                    mask = batch_celltypes == ct
                    ct_sum = X_dense[mask].sum(axis=0)
                    ct_count = mask.sum()

                    celltype_accum[level][ct] += ct_sum
                    celltype_counts[level][ct] += ct_count

                    # Also aggregate by sample × celltype (for bootstrap)
                    if batch_samples is not None:
                        # Filter out NaN samples to avoid type comparison error
                        ct_samples = batch_samples[mask]
                        valid_sample_mask = pd.notna(ct_samples)
                        valid_samples = ct_samples[valid_sample_mask]
                        for sample in np.unique(valid_samples) if len(valid_samples) > 0 else []:
                            if sample is None or pd.isna(sample):
                                continue
                            sample_mask = mask & (batch_samples == sample)
                            if sample_mask.any():
                                key = (sample, ct)
                                sample_celltype_accum[level][key] += X_dense[sample_mask].sum(axis=0)
                                sample_celltype_counts[level][key] += sample_mask.sum()

            # Update atlas-level statistics for z-score
            self._update_atlas_stats(X_dense, gene_count, gene_mean, gene_m2)

            batch_time = time.time() - batch_start
            batch_times.append(batch_time)

            if batch_idx % 10 == 0 or batch_idx == n_batches - 1:
                avg_time = np.mean(batch_times[-10:])
                eta = avg_time * (n_batches - batch_idx - 1)
                log(f"  Batch {batch_idx+1}/{n_batches} ({batch_time:.1f}s, ETA: {eta/60:.1f}min)")

        # Compute atlas-level std
        gene_var = gene_m2 / max(gene_count[0] - 1, 1)
        gene_std = np.sqrt(gene_var)
        gene_std[gene_std == 0] = 1.0

        # Write output files for each level
        log("Writing output files...")

        for level in levels_to_process:
            self._write_level_output(
                level=level,
                output_path=output_paths[level],
                celltype_accum=celltype_accum[level],
                celltype_counts=celltype_counts[level],
                sample_celltype_accum=sample_celltype_accum[level],
                sample_celltype_counts=sample_celltype_counts[level],
                var_names=var_names,
                gene_mean=gene_mean,
                gene_std=gene_std,
                output_dir=output_dir,
            )

        total_time = time.time() - start_time
        log(f"\nCompleted in {total_time/60:.1f} min")
        log(f"Output files: {len(levels_to_process) * 2} (pseudobulk + resampled)")

        return output_paths

    def _update_atlas_stats(
        self,
        X_dense: np.ndarray,
        gene_count: np.ndarray,
        gene_mean: np.ndarray,
        gene_m2: np.ndarray,
    ):
        """Update atlas-level statistics using Chan's parallel algorithm."""
        # Normalize to CPM + log1p
        row_sums = X_dense.sum(axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1)
        cpm = X_dense / row_sums * 1e6
        log_cpm = np.log1p(cpm)

        # Batch statistics
        n_batch = log_cpm.shape[0]
        batch_mean = log_cpm.mean(axis=0)
        batch_var = log_cpm.var(axis=0, ddof=0)
        batch_m2 = batch_var * n_batch

        # Merge with running statistics (gene_count[0] is mutable)
        n_a = gene_count[0]
        n_total = n_a + n_batch

        if n_a == 0:
            gene_mean[:] = batch_mean
            gene_m2[:] = batch_m2
        else:
            delta = batch_mean - gene_mean
            gene_mean[:] = (n_a * gene_mean + n_batch * batch_mean) / n_total
            gene_m2[:] = gene_m2 + batch_m2 + delta ** 2 * n_a * n_batch / n_total

        gene_count[0] = n_total

    def _write_level_output(
        self,
        level: str,
        output_path: Path,
        celltype_accum: Dict[str, np.ndarray],
        celltype_counts: Dict[str, int],
        sample_celltype_accum: Dict[Tuple[str, str], np.ndarray],
        sample_celltype_counts: Dict[Tuple[str, str], int],
        var_names: List[str],
        gene_mean: np.ndarray,
        gene_std: np.ndarray,
        output_dir: Path,
    ):
        """Write pseudobulk and resampled pseudobulk for a level."""
        log(f"  Writing {level}...")

        # Filter cell types with minimum cells
        valid_celltypes = [ct for ct, count in celltype_counts.items()
                          if count >= self.min_cells]

        if not valid_celltypes:
            log(f"    WARNING: No valid cell types for {level}")
            return

        # Build pseudobulk matrix (celltypes × genes)
        raw_counts = np.vstack([celltype_accum[ct] for ct in valid_celltypes])
        n_cells_per_ct = np.array([celltype_counts[ct] for ct in valid_celltypes])

        # Normalize to CPM + log1p
        total = raw_counts.sum(axis=1, keepdims=True)
        total = np.maximum(total, 1)
        cpm = raw_counts / total * 1e6
        log_cpm = np.log1p(cpm)

        # Z-score using atlas-level statistics
        zscore_expr = (log_cpm - gene_mean) / gene_std

        # Create AnnData
        pb_adata = ad.AnnData(
            X=log_cpm.astype(np.float32),
            obs=pd.DataFrame({
                'cell_type': valid_celltypes,
                'n_cells': n_cells_per_ct,
            }, index=valid_celltypes),
            var=pd.DataFrame(index=var_names),
        )
        pb_adata.layers['counts'] = raw_counts.astype(np.float32)
        pb_adata.layers['zscore'] = zscore_expr.astype(np.float32)
        pb_adata.uns['atlas'] = self.atlas_config.name
        pb_adata.uns['level'] = level
        pb_adata.uns['atlas_stats'] = {
            'gene_mean': gene_mean.astype(np.float32),
            'gene_std': gene_std.astype(np.float32),
        }

        pb_adata.write_h5ad(output_path, compression='gzip')
        log(f"    Saved: {output_path.name} ({len(valid_celltypes)} cell types)")

        # Generate bootstrap resampled pseudobulk
        self._write_bootstrap_pseudobulk(
            level=level,
            valid_celltypes=valid_celltypes,
            sample_celltype_accum=sample_celltype_accum,
            sample_celltype_counts=sample_celltype_counts,
            var_names=var_names,
            gene_mean=gene_mean,
            gene_std=gene_std,
            output_dir=output_dir,
        )

    def _write_bootstrap_pseudobulk(
        self,
        level: str,
        valid_celltypes: List[str],
        sample_celltype_accum: Dict[Tuple[str, str], np.ndarray],
        sample_celltype_counts: Dict[Tuple[str, str], int],
        var_names: List[str],
        gene_mean: np.ndarray,
        gene_std: np.ndarray,
        output_dir: Path,
    ):
        """Generate bootstrap resampled pseudobulk for confidence intervals."""
        n_genes = len(var_names)
        rng = np.random.default_rng(self.seed)

        # For each cell type, get sample-level data
        bootstrap_data = []  # List of (celltype, bootstrap_idx, expression)

        for ct in valid_celltypes:
            # Get all samples for this cell type
            samples = [(s, ct) for (s, c) in sample_celltype_accum.keys() if c == ct]

            if len(samples) < self.min_samples_bootstrap:
                continue

            # Get sample-level expression (already summed)
            sample_expr = np.vstack([sample_celltype_accum[key] for key in samples])
            sample_counts = np.array([sample_celltype_counts[key] for key in samples])

            # Normalize each sample to CPM
            sample_totals = sample_expr.sum(axis=1, keepdims=True)
            sample_totals = np.maximum(sample_totals, 1)
            sample_cpm = sample_expr / sample_totals * 1e6

            # Bootstrap: resample samples with replacement
            n_samples = len(samples)
            for boot_idx in range(self.n_bootstrap):
                # Resample sample indices
                boot_indices = rng.choice(n_samples, size=n_samples, replace=True)

                # Sum resampled CPM and normalize
                boot_sum = sample_cpm[boot_indices].sum(axis=0)
                boot_total = boot_sum.sum()
                if boot_total > 0:
                    boot_cpm = boot_sum / boot_total * 1e6
                else:
                    boot_cpm = boot_sum

                boot_log_cpm = np.log1p(boot_cpm)
                bootstrap_data.append((ct, boot_idx, boot_log_cpm))

        if not bootstrap_data:
            log(f"    No cell types have enough samples for bootstrap")
            return

        # Create resampled AnnData
        # Shape: (n_celltypes × n_bootstrap) × n_genes
        celltypes = [d[0] for d in bootstrap_data]
        boot_indices = [d[1] for d in bootstrap_data]
        expressions = np.vstack([d[2] for d in bootstrap_data])

        # Z-score
        zscore_expr = (expressions - gene_mean) / gene_std

        resampled_adata = ad.AnnData(
            X=expressions.astype(np.float32),
            obs=pd.DataFrame({
                'cell_type': celltypes,
                'bootstrap_idx': boot_indices,
            }),
            var=pd.DataFrame(index=var_names),
        )
        resampled_adata.layers['zscore'] = zscore_expr.astype(np.float32)
        resampled_adata.uns['atlas'] = self.atlas_config.name
        resampled_adata.uns['level'] = level
        resampled_adata.uns['n_bootstrap'] = self.n_bootstrap

        output_path = output_dir / f"{self.atlas_config.name.lower()}_pseudobulk_{level.lower()}_resampled.h5ad"
        resampled_adata.write_h5ad(output_path, compression='gzip')

        n_ct_with_bootstrap = len(set(celltypes))
        log(f"    Saved: {output_path.name} ({n_ct_with_bootstrap} cell types × {self.n_bootstrap} bootstraps)")


def aggregate_all_levels(
    atlas_name: str,
    output_dir: Path,
    levels: Optional[List[str]] = None,
    n_bootstrap: int = 100,
    batch_size: int = 50000,
    use_gpu: bool = True,
    skip_existing: bool = True,
) -> Dict[str, Path]:
    """
    Convenience function for multi-level aggregation.

    Args:
        atlas_name: Name of the atlas
        output_dir: Output directory
        levels: Levels to process (default: all)
        n_bootstrap: Number of bootstrap samples
        batch_size: Batch size
        use_gpu: Use GPU acceleration
        skip_existing: Skip existing files

    Returns:
        Dict mapping levels to output paths
    """
    aggregator = MultiLevelAggregator(
        atlas_config=atlas_name,
        levels=levels,
        batch_size=batch_size,
        n_bootstrap=n_bootstrap,
        use_gpu=use_gpu,
    )

    return aggregator.aggregate_all_levels(
        output_dir=Path(output_dir),
        skip_existing=skip_existing,
    )
