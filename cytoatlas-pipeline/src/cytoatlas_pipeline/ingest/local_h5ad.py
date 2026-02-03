"""
Local H5AD file data source.

Provides efficient chunked loading from local H5AD files using AnnData's
backed mode for memory efficiency.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterator, Optional, Union

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse as sp

from cytoatlas_pipeline.ingest.base import DataChunk, DataSource, DataSourceConfig


class LocalH5ADSource(DataSource):
    """
    Data source for local H5AD files.

    Uses AnnData's backed mode for memory-efficient reading of large files.

    Example:
        >>> source = LocalH5ADSource("/path/to/data.h5ad", backed=True)
        >>> print(f"Dataset: {source.n_cells} cells, {source.n_genes} genes")
        >>>
        >>> for chunk in source.iter_chunks(batch_size=10000):
        ...     # Process chunk
        ...     pass
    """

    def __init__(
        self,
        path: Union[str, Path],
        config: Optional[DataSourceConfig] = None,
        backed: Optional[bool] = None,
        layer: Optional[str] = None,
    ):
        """
        Initialize H5AD data source.

        Args:
            path: Path to H5AD file.
            config: Source configuration.
            backed: Override config backed mode.
            layer: Override config layer.
        """
        super().__init__(config)

        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"H5AD file not found: {self.path}")

        # Override config if specified
        if backed is not None:
            self.config.backed = backed
        if layer is not None:
            self.config.layer = layer

        # Load AnnData
        if self.config.backed:
            self._adata = ad.read_h5ad(self.path, backed="r")
        else:
            self._adata = ad.read_h5ad(self.path)

    @property
    def n_cells(self) -> int:
        """Total number of cells."""
        return self._adata.n_obs

    @property
    def n_genes(self) -> int:
        """Total number of genes."""
        return self._adata.n_vars

    @property
    def gene_names(self) -> list[str]:
        """List of gene names."""
        return list(self._adata.var_names)

    @property
    def obs_columns(self) -> list[str]:
        """Available cell metadata columns."""
        return list(self._adata.obs.columns)

    @property
    def var_columns(self) -> list[str]:
        """Available gene metadata columns."""
        return list(self._adata.var.columns)

    @property
    def layers(self) -> list[str]:
        """Available layers."""
        return list(self._adata.layers.keys())

    def iter_chunks(
        self,
        batch_size: Optional[int] = None,
        gene_filter: Optional[list[str]] = None,
    ) -> Iterator[DataChunk]:
        """
        Iterate over data in chunks.

        Args:
            batch_size: Number of cells per chunk.
            gene_filter: Only include these genes.

        Yields:
            DataChunk for each batch.
        """
        if batch_size is None:
            batch_size = self.config.batch_size

        n_cells = self.n_cells
        n_chunks = math.ceil(n_cells / batch_size)

        # Prepare gene filter mask
        if gene_filter is not None:
            gene_filter_upper = [g.upper() for g in gene_filter]
            gene_mask = np.array([
                g.upper() in gene_filter_upper for g in self._adata.var_names
            ])
            var_filtered = self._adata.var.iloc[gene_mask].copy()
            n_genes_filtered = gene_mask.sum()
        else:
            gene_mask = None
            var_filtered = self._adata.var.copy()
            n_genes_filtered = self.n_genes

        for chunk_idx in range(n_chunks):
            start_idx = chunk_idx * batch_size
            end_idx = min(start_idx + batch_size, n_cells)

            # Get expression data
            if self.config.layer is not None and self.config.layer in self._adata.layers:
                X_chunk = self._adata.layers[self.config.layer][start_idx:end_idx]
            else:
                X_chunk = self._adata.X[start_idx:end_idx]

            # Convert to dense if backed AnnData returns a backed array
            if hasattr(X_chunk, "toarray"):
                # Keep sparse for memory efficiency
                pass
            elif hasattr(X_chunk, "__array__"):
                X_chunk = np.asarray(X_chunk)

            # Apply gene filter
            if gene_mask is not None:
                if sp.issparse(X_chunk):
                    X_chunk = X_chunk[:, gene_mask]
                else:
                    X_chunk = X_chunk[:, gene_mask]

            # Get metadata
            obs_chunk = self._adata.obs.iloc[start_idx:end_idx].copy()

            yield DataChunk(
                X=X_chunk,
                obs=obs_chunk,
                var=var_filtered,
                chunk_index=chunk_idx,
                total_chunks=n_chunks,
                n_cells=end_idx - start_idx,
                n_genes=n_genes_filtered,
                source_info={
                    "path": str(self.path),
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "backed": self.config.backed,
                },
            )

    def get_obs(self, columns: Optional[list[str]] = None) -> pd.DataFrame:
        """Get cell metadata."""
        if columns is not None:
            return self._adata.obs[columns].copy()
        return self._adata.obs.copy()

    def get_var(self, columns: Optional[list[str]] = None) -> pd.DataFrame:
        """Get gene metadata."""
        if columns is not None:
            return self._adata.var[columns].copy()
        return self._adata.var.copy()

    def get_expression(
        self,
        cell_indices: Optional[np.ndarray] = None,
        gene_names: Optional[list[str]] = None,
        dense: bool = False,
    ) -> Union[np.ndarray, sp.spmatrix]:
        """
        Get expression matrix for specific cells/genes.

        Args:
            cell_indices: Cell indices to retrieve (None for all).
            gene_names: Gene names to retrieve (None for all).
            dense: Return dense array.

        Returns:
            Expression matrix.
        """
        # Get layer
        if self.config.layer is not None and self.config.layer in self._adata.layers:
            X = self._adata.layers[self.config.layer]
        else:
            X = self._adata.X

        # Slice cells
        if cell_indices is not None:
            X = X[cell_indices]

        # Slice genes
        if gene_names is not None:
            gene_indices = np.array([
                i for i, g in enumerate(self._adata.var_names)
                if g.upper() in [gn.upper() for gn in gene_names]
            ])
            X = X[:, gene_indices]

        if dense and sp.issparse(X):
            X = X.toarray()

        return X

    def filter(
        self,
        cell_types: Optional[list[str]] = None,
        samples: Optional[list[str]] = None,
        min_genes: Optional[int] = None,
        min_cells: Optional[int] = None,
    ) -> "LocalH5ADSource":
        """
        Create filtered view of data.

        Args:
            cell_types: Keep only these cell types.
            samples: Keep only these samples.
            min_genes: Minimum genes per cell.
            min_cells: Minimum cells per gene.

        Returns:
            New LocalH5ADSource with filtered data.
        """
        adata = self._adata

        # Apply filters
        cell_mask = np.ones(self.n_cells, dtype=bool)

        if cell_types is not None:
            cell_mask &= adata.obs[self.config.cell_type_col].isin(cell_types).values

        if samples is not None:
            cell_mask &= adata.obs[self.config.sample_col].isin(samples).values

        if min_genes is not None:
            if sp.issparse(adata.X):
                genes_per_cell = np.asarray((adata.X > 0).sum(axis=1)).ravel()
            else:
                genes_per_cell = (adata.X > 0).sum(axis=1)
            cell_mask &= genes_per_cell >= min_genes

        # Filter genes
        gene_mask = np.ones(self.n_genes, dtype=bool)

        if min_cells is not None:
            if sp.issparse(adata.X):
                cells_per_gene = np.asarray((adata.X > 0).sum(axis=0)).ravel()
            else:
                cells_per_gene = (adata.X > 0).sum(axis=0)
            gene_mask &= cells_per_gene >= min_cells

        # Create filtered AnnData
        adata_filtered = adata[cell_mask, gene_mask].copy()

        # Create new source (not backed since we're copying)
        new_source = LocalH5ADSource.__new__(LocalH5ADSource)
        new_source.config = self.config
        new_source.path = self.path
        new_source._adata = adata_filtered

        return new_source

    def close(self) -> None:
        """Close the H5AD file (for backed mode)."""
        if hasattr(self._adata, "file") and self._adata.file is not None:
            self._adata.file.close()

    def __enter__(self) -> "LocalH5ADSource":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
