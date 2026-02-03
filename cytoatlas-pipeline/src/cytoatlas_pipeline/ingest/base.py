"""
Base interfaces for data sources.

Defines the abstract DataSource interface that all data loaders implement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generator, Iterator, Optional, Union

import numpy as np
import pandas as pd
from scipy import sparse as sp


@dataclass
class DataSourceConfig:
    """Configuration for data sources."""

    batch_size: int = 10000
    """Number of cells per chunk."""

    backed: bool = True
    """Use backed/lazy loading when available."""

    cell_type_col: str = "cell_type"
    """Column name for cell type annotations."""

    sample_col: str = "sample"
    """Column name for sample identifiers."""

    gene_col: str = "gene_symbol"
    """Column name for gene symbols (in var)."""

    layer: Optional[str] = None
    """Layer to use for expression data (None for .X)."""

    filters: dict[str, Any] = field(default_factory=dict)
    """Filters to apply when loading data."""


@dataclass
class DataChunk:
    """
    A chunk of data from a data source.

    Contains expression matrix, cell metadata, and gene information.
    """

    X: Union[np.ndarray, sp.spmatrix]
    """Expression matrix (cells x genes)."""

    obs: pd.DataFrame
    """Cell metadata (cell annotations)."""

    var: pd.DataFrame
    """Gene metadata (gene annotations)."""

    chunk_index: int
    """Index of this chunk."""

    total_chunks: int
    """Total number of chunks."""

    n_cells: int
    """Number of cells in this chunk."""

    n_genes: int
    """Number of genes."""

    source_info: dict[str, Any] = field(default_factory=dict)
    """Additional source-specific information."""

    @property
    def gene_names(self) -> list[str]:
        """Get gene names as list."""
        return list(self.var.index)

    @property
    def cell_ids(self) -> list[str]:
        """Get cell IDs as list."""
        return list(self.obs.index)

    @property
    def is_sparse(self) -> bool:
        """Check if expression matrix is sparse."""
        return sp.issparse(self.X)

    def to_dense(self) -> "DataChunk":
        """Convert to dense expression matrix."""
        if self.is_sparse:
            return DataChunk(
                X=self.X.toarray(),
                obs=self.obs,
                var=self.var,
                chunk_index=self.chunk_index,
                total_chunks=self.total_chunks,
                n_cells=self.n_cells,
                n_genes=self.n_genes,
                source_info=self.source_info,
            )
        return self

    def to_sparse(self, format: str = "csr") -> "DataChunk":
        """Convert to sparse expression matrix."""
        if not self.is_sparse:
            if format == "csr":
                X_sparse = sp.csr_matrix(self.X)
            elif format == "csc":
                X_sparse = sp.csc_matrix(self.X)
            else:
                raise ValueError(f"Unknown sparse format: {format}")

            return DataChunk(
                X=X_sparse,
                obs=self.obs,
                var=self.var,
                chunk_index=self.chunk_index,
                total_chunks=self.total_chunks,
                n_cells=self.n_cells,
                n_genes=self.n_genes,
                source_info=self.source_info,
            )
        return self

    def filter_genes(self, gene_names: list[str]) -> "DataChunk":
        """Filter to specific genes."""
        gene_mask = self.var.index.isin(gene_names)
        var_filtered = self.var.loc[gene_mask]

        if self.is_sparse:
            X_filtered = self.X[:, gene_mask]
        else:
            X_filtered = self.X[:, gene_mask]

        return DataChunk(
            X=X_filtered,
            obs=self.obs,
            var=var_filtered,
            chunk_index=self.chunk_index,
            total_chunks=self.total_chunks,
            n_cells=self.n_cells,
            n_genes=len(var_filtered),
            source_info=self.source_info,
        )


class DataSource(ABC):
    """
    Abstract base class for data sources.

    Provides a unified interface for loading single-cell data from
    various sources (local files, remote, cloud databases).

    Example:
        >>> source = LocalH5ADSource("/path/to/data.h5ad")
        >>> for chunk in source.iter_chunks(batch_size=10000):
        ...     # Process each chunk
        ...     activities = compute_activity(chunk.X, signature)
    """

    def __init__(self, config: Optional[DataSourceConfig] = None):
        """
        Initialize data source.

        Args:
            config: Source configuration.
        """
        self.config = config or DataSourceConfig()

    @property
    @abstractmethod
    def n_cells(self) -> int:
        """Total number of cells in the data source."""
        ...

    @property
    @abstractmethod
    def n_genes(self) -> int:
        """Total number of genes in the data source."""
        ...

    @property
    @abstractmethod
    def gene_names(self) -> list[str]:
        """List of gene names."""
        ...

    @property
    @abstractmethod
    def obs_columns(self) -> list[str]:
        """Available observation (cell) metadata columns."""
        ...

    @property
    @abstractmethod
    def var_columns(self) -> list[str]:
        """Available variable (gene) metadata columns."""
        ...

    @abstractmethod
    def iter_chunks(
        self,
        batch_size: Optional[int] = None,
        gene_filter: Optional[list[str]] = None,
    ) -> Iterator[DataChunk]:
        """
        Iterate over data in chunks.

        Args:
            batch_size: Number of cells per chunk (None uses config default).
            gene_filter: Only include these genes.

        Yields:
            DataChunk for each batch of cells.
        """
        ...

    @abstractmethod
    def get_obs(self, columns: Optional[list[str]] = None) -> pd.DataFrame:
        """
        Get cell metadata.

        Args:
            columns: Specific columns to retrieve (None for all).

        Returns:
            DataFrame with cell metadata.
        """
        ...

    @abstractmethod
    def get_var(self, columns: Optional[list[str]] = None) -> pd.DataFrame:
        """
        Get gene metadata.

        Args:
            columns: Specific columns to retrieve (None for all).

        Returns:
            DataFrame with gene metadata.
        """
        ...

    def get_cell_types(self) -> list[str]:
        """Get unique cell types."""
        obs = self.get_obs([self.config.cell_type_col])
        return list(obs[self.config.cell_type_col].unique())

    def get_samples(self) -> list[str]:
        """Get unique samples."""
        obs = self.get_obs([self.config.sample_col])
        return list(obs[self.config.sample_col].unique())

    def summary(self) -> dict[str, Any]:
        """Get summary statistics about the data source."""
        return {
            "n_cells": self.n_cells,
            "n_genes": self.n_genes,
            "n_cell_types": len(self.get_cell_types()),
            "n_samples": len(self.get_samples()),
            "obs_columns": self.obs_columns,
            "var_columns": self.var_columns,
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(n_cells={self.n_cells}, n_genes={self.n_genes})"
