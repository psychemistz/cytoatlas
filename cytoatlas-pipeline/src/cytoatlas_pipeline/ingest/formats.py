"""
Support for various single-cell data formats.

Provides converters and data sources for:
- Loom files
- 10X Genomics formats (MTX, H5)
- CSV/TSV expression matrices
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


class LoomSource(DataSource):
    """
    Data source for Loom files.

    Example:
        >>> source = LoomSource("/path/to/data.loom")
        >>> for chunk in source.iter_chunks():
        ...     process(chunk)
    """

    def __init__(
        self,
        path: Union[str, Path],
        config: Optional[DataSourceConfig] = None,
    ):
        """
        Initialize Loom data source.

        Args:
            path: Path to Loom file.
            config: Source configuration.
        """
        super().__init__(config)
        self.path = Path(path)

        if not self.path.exists():
            raise FileNotFoundError(f"Loom file not found: {self.path}")

        # Convert to AnnData for consistent interface
        self._adata = ad.read_loom(self.path)

    @property
    def n_cells(self) -> int:
        return self._adata.n_obs

    @property
    def n_genes(self) -> int:
        return self._adata.n_vars

    @property
    def gene_names(self) -> list[str]:
        return list(self._adata.var_names)

    @property
    def obs_columns(self) -> list[str]:
        return list(self._adata.obs.columns)

    @property
    def var_columns(self) -> list[str]:
        return list(self._adata.var.columns)

    def iter_chunks(
        self,
        batch_size: Optional[int] = None,
        gene_filter: Optional[list[str]] = None,
    ) -> Iterator[DataChunk]:
        """Iterate over data in chunks."""
        if batch_size is None:
            batch_size = self.config.batch_size

        n_cells = self.n_cells
        n_chunks = math.ceil(n_cells / batch_size)

        # Gene filter
        if gene_filter is not None:
            gene_filter_upper = set(g.upper() for g in gene_filter)
            gene_mask = np.array([
                g.upper() in gene_filter_upper for g in self._adata.var_names
            ])
            var_filtered = self._adata.var.iloc[gene_mask].copy()
        else:
            gene_mask = None
            var_filtered = self._adata.var.copy()

        for chunk_idx in range(n_chunks):
            start_idx = chunk_idx * batch_size
            end_idx = min(start_idx + batch_size, n_cells)

            X_chunk = self._adata.X[start_idx:end_idx]
            if gene_mask is not None:
                X_chunk = X_chunk[:, gene_mask]

            obs_chunk = self._adata.obs.iloc[start_idx:end_idx].copy()

            yield DataChunk(
                X=X_chunk,
                obs=obs_chunk,
                var=var_filtered,
                chunk_index=chunk_idx,
                total_chunks=n_chunks,
                n_cells=end_idx - start_idx,
                n_genes=len(var_filtered),
                source_info={"path": str(self.path), "format": "loom"},
            )

    def get_obs(self, columns: Optional[list[str]] = None) -> pd.DataFrame:
        if columns is not None:
            return self._adata.obs[columns].copy()
        return self._adata.obs.copy()

    def get_var(self, columns: Optional[list[str]] = None) -> pd.DataFrame:
        if columns is not None:
            return self._adata.var[columns].copy()
        return self._adata.var.copy()


class TenXSource(DataSource):
    """
    Data source for 10X Genomics formats.

    Supports:
    - 10X HDF5 files (*.h5)
    - 10X MTX directories (containing matrix.mtx, barcodes.tsv, genes.tsv)

    Example:
        >>> # From H5 file
        >>> source = TenXSource("/path/to/filtered_feature_bc_matrix.h5")
        >>>
        >>> # From MTX directory
        >>> source = TenXSource("/path/to/filtered_feature_bc_matrix/")
    """

    def __init__(
        self,
        path: Union[str, Path],
        config: Optional[DataSourceConfig] = None,
    ):
        """
        Initialize 10X data source.

        Args:
            path: Path to 10X H5 file or MTX directory.
            config: Source configuration.
        """
        super().__init__(config)
        self.path = Path(path)

        if not self.path.exists():
            raise FileNotFoundError(f"10X data not found: {self.path}")

        # Detect format and load
        if self.path.is_dir():
            # MTX directory
            self._adata = ad.read_10x_mtx(self.path)
        elif self.path.suffix == ".h5":
            # 10X H5 file
            self._adata = ad.read_10x_h5(self.path)
        else:
            raise ValueError(f"Unknown 10X format: {self.path}")

    @property
    def n_cells(self) -> int:
        return self._adata.n_obs

    @property
    def n_genes(self) -> int:
        return self._adata.n_vars

    @property
    def gene_names(self) -> list[str]:
        return list(self._adata.var_names)

    @property
    def obs_columns(self) -> list[str]:
        return list(self._adata.obs.columns)

    @property
    def var_columns(self) -> list[str]:
        return list(self._adata.var.columns)

    def iter_chunks(
        self,
        batch_size: Optional[int] = None,
        gene_filter: Optional[list[str]] = None,
    ) -> Iterator[DataChunk]:
        """Iterate over data in chunks."""
        if batch_size is None:
            batch_size = self.config.batch_size

        n_cells = self.n_cells
        n_chunks = math.ceil(n_cells / batch_size)

        if gene_filter is not None:
            gene_filter_upper = set(g.upper() for g in gene_filter)
            gene_mask = np.array([
                g.upper() in gene_filter_upper for g in self._adata.var_names
            ])
            var_filtered = self._adata.var.iloc[gene_mask].copy()
        else:
            gene_mask = None
            var_filtered = self._adata.var.copy()

        for chunk_idx in range(n_chunks):
            start_idx = chunk_idx * batch_size
            end_idx = min(start_idx + batch_size, n_cells)

            X_chunk = self._adata.X[start_idx:end_idx]
            if gene_mask is not None:
                X_chunk = X_chunk[:, gene_mask]

            obs_chunk = self._adata.obs.iloc[start_idx:end_idx].copy()

            yield DataChunk(
                X=X_chunk,
                obs=obs_chunk,
                var=var_filtered,
                chunk_index=chunk_idx,
                total_chunks=n_chunks,
                n_cells=end_idx - start_idx,
                n_genes=len(var_filtered),
                source_info={"path": str(self.path), "format": "10x"},
            )

    def get_obs(self, columns: Optional[list[str]] = None) -> pd.DataFrame:
        if columns is not None:
            return self._adata.obs[columns].copy()
        return self._adata.obs.copy()

    def get_var(self, columns: Optional[list[str]] = None) -> pd.DataFrame:
        if columns is not None:
            return self._adata.var[columns].copy()
        return self._adata.var.copy()


def convert_to_anndata(
    path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    format: Optional[str] = None,
) -> ad.AnnData:
    """
    Convert various formats to AnnData.

    Supported formats:
    - .loom: Loom files
    - .h5: 10X HDF5 files
    - .mtx / directory: 10X MTX format
    - .csv/.tsv: Expression matrices (genes x cells)

    Args:
        path: Input file path.
        output_path: Optional path to save converted H5AD.
        format: Force format detection (auto-detect if None).

    Returns:
        AnnData object.
    """
    path = Path(path)

    # Detect format
    if format is None:
        if path.suffix == ".loom":
            format = "loom"
        elif path.suffix == ".h5":
            format = "10x_h5"
        elif path.is_dir() or path.suffix == ".mtx":
            format = "10x_mtx"
        elif path.suffix in (".csv", ".tsv"):
            format = "csv"
        elif path.suffix in (".h5ad",):
            format = "h5ad"
        else:
            raise ValueError(f"Unknown format for: {path}")

    # Load based on format
    if format == "loom":
        adata = ad.read_loom(path)
    elif format == "10x_h5":
        adata = ad.read_10x_h5(path)
    elif format == "10x_mtx":
        if path.is_dir():
            adata = ad.read_10x_mtx(path)
        else:
            # Path to matrix.mtx file
            adata = ad.read_10x_mtx(path.parent)
    elif format == "csv":
        df = pd.read_csv(path, index_col=0)
        adata = ad.AnnData(df.T)  # Transpose: genes x cells -> cells x genes
    elif format == "h5ad":
        adata = ad.read_h5ad(path)
    else:
        raise ValueError(f"Unsupported format: {format}")

    # Save if output path provided
    if output_path is not None:
        output_path = Path(output_path)
        adata.write_h5ad(output_path)

    return adata


def read_expression_csv(
    path: Union[str, Path],
    genes_as_rows: bool = True,
    gene_col: Optional[str] = None,
    sample_cols: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    Read expression matrix from CSV/TSV.

    Args:
        path: Path to CSV/TSV file.
        genes_as_rows: If True, genes are rows and samples are columns.
        gene_col: Column containing gene names (if not index).
        sample_cols: Columns containing expression data.

    Returns:
        Tuple of (expression_df, gene_names, sample_names).
    """
    path = Path(path)

    # Read file
    sep = "\t" if path.suffix == ".tsv" else ","
    df = pd.read_csv(path, sep=sep, index_col=0 if gene_col is None else None)

    # Handle gene column
    if gene_col is not None:
        df = df.set_index(gene_col)

    # Select sample columns
    if sample_cols is not None:
        df = df[sample_cols]

    # Get names
    gene_names = list(df.index)
    sample_names = list(df.columns)

    # Transpose if needed
    if not genes_as_rows:
        df = df.T
        gene_names, sample_names = sample_names, gene_names

    return df, gene_names, sample_names
