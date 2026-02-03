"""
Single-cell streaming aggregation.

Provides streaming processing for per-cell analysis without
full aggregation, enabling memory-efficient processing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generator, Iterator, Optional, Union

import numpy as np
import pandas as pd
from scipy import sparse as sp

from cytoatlas_pipeline.aggregation.base import AggregationConfig


@dataclass
class StreamingConfig(AggregationConfig):
    """Configuration for streaming single-cell processing."""

    batch_size: int = 10000
    """Number of cells per batch."""

    normalize_per_cell: bool = True
    """Normalize each cell to sum to 1."""

    scale_factor: float = 10000.0
    """Scale factor after normalization."""


@dataclass
class CellBatch:
    """A batch of single cells for streaming processing."""

    X: Union[np.ndarray, sp.spmatrix]
    """Expression matrix (cells x genes)."""

    obs: pd.DataFrame
    """Cell metadata."""

    cell_indices: np.ndarray
    """Original cell indices."""

    batch_index: int
    """Index of this batch."""

    total_batches: int
    """Total number of batches."""


class SingleCellStreamer:
    """
    Streams single-cell data for per-cell processing.

    Enables memory-efficient processing of large datasets by
    yielding batches of cells with their metadata.

    Example:
        >>> streamer = SingleCellStreamer(config)
        >>> for batch in streamer.stream(X, obs):
        ...     # Process each cell batch
        ...     activities = compute_activity(batch.X, signature)
        ...     # Stream results to disk
        ...     writer.write(activities, batch.cell_indices)
    """

    def __init__(self, config: Optional[StreamingConfig] = None):
        """
        Initialize single-cell streamer.

        Args:
            config: Streaming configuration.
        """
        self.config = config or StreamingConfig()

    def stream(
        self,
        X: Union[np.ndarray, sp.spmatrix],
        obs: pd.DataFrame,
        gene_filter: Optional[np.ndarray] = None,
    ) -> Generator[CellBatch, None, None]:
        """
        Stream cells in batches.

        Args:
            X: Expression matrix (cells x genes).
            obs: Cell metadata.
            gene_filter: Boolean mask for genes to include.

        Yields:
            CellBatch for each batch of cells.
        """
        n_cells = X.shape[0]
        batch_size = self.config.batch_size
        n_batches = (n_cells + batch_size - 1) // batch_size

        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_cells)

            cell_indices = np.arange(start_idx, end_idx)

            # Get batch data
            X_batch = X[start_idx:end_idx]
            if gene_filter is not None:
                if sp.issparse(X_batch):
                    X_batch = X_batch[:, gene_filter]
                else:
                    X_batch = X_batch[:, gene_filter]

            # Normalize if configured
            if self.config.normalize_per_cell:
                X_batch = self._normalize_batch(X_batch)

            obs_batch = obs.iloc[start_idx:end_idx].copy()

            yield CellBatch(
                X=X_batch,
                obs=obs_batch,
                cell_indices=cell_indices,
                batch_index=batch_idx,
                total_batches=n_batches,
            )

    def _normalize_batch(
        self,
        X: Union[np.ndarray, sp.spmatrix],
    ) -> Union[np.ndarray, sp.spmatrix]:
        """Normalize expression per cell."""
        if sp.issparse(X):
            # Per-cell normalization for sparse matrix
            row_sums = np.asarray(X.sum(axis=1)).ravel()
            row_sums = np.where(row_sums == 0, 1, row_sums)

            # Create diagonal scaling matrix
            scaling = sp.diags(self.config.scale_factor / row_sums)
            return scaling @ X
        else:
            row_sums = X.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums == 0, 1, row_sums)
            return X / row_sums * self.config.scale_factor

    def process(
        self,
        X: Union[np.ndarray, sp.spmatrix],
        obs: pd.DataFrame,
        process_fn: Callable[[CellBatch], np.ndarray],
        gene_filter: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Process all cells and collect results.

        Args:
            X: Expression matrix.
            obs: Cell metadata.
            process_fn: Function to apply to each batch.
            gene_filter: Boolean mask for genes.

        Returns:
            Concatenated results from all batches.
        """
        results = []
        for batch in self.stream(X, obs, gene_filter):
            batch_result = process_fn(batch)
            results.append(batch_result)

        return np.vstack(results)

    def process_to_file(
        self,
        X: Union[np.ndarray, sp.spmatrix],
        obs: pd.DataFrame,
        process_fn: Callable[[CellBatch], np.ndarray],
        output_path: str,
        gene_filter: Optional[np.ndarray] = None,
        feature_names: Optional[list[str]] = None,
    ) -> None:
        """
        Process all cells and stream results to file.

        Args:
            X: Expression matrix.
            obs: Cell metadata.
            process_fn: Function to apply to each batch.
            output_path: Path to output HDF5 file.
            gene_filter: Boolean mask for genes.
            feature_names: Names for result features.
        """
        import h5py

        first_batch = True
        h5_file = None

        try:
            for batch in self.stream(X, obs, gene_filter):
                batch_result = process_fn(batch)

                if first_batch:
                    # Initialize HDF5 file
                    n_features = batch_result.shape[1]
                    n_cells = X.shape[0]

                    h5_file = h5py.File(output_path, "w")
                    h5_file.create_dataset(
                        "data",
                        shape=(n_cells, n_features),
                        dtype="float32",
                        chunks=(min(1000, n_cells), n_features),
                        compression="gzip",
                    )

                    if feature_names is not None:
                        h5_file.create_dataset(
                            "feature_names",
                            data=np.array(feature_names, dtype="S"),
                        )

                    first_batch = False

                # Write batch
                start_idx = batch.cell_indices[0]
                end_idx = batch.cell_indices[-1] + 1
                h5_file["data"][start_idx:end_idx] = batch_result

        finally:
            if h5_file is not None:
                h5_file.close()
