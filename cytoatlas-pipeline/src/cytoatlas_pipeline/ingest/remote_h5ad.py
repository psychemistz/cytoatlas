"""
Remote H5AD file data source.

Provides streaming access to H5AD files from URLs (HTTP/HTTPS/S3).
"""

from __future__ import annotations

import math
import tempfile
from pathlib import Path
from typing import Any, Iterator, Optional, Union
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from scipy import sparse as sp

from cytoatlas_pipeline.ingest.base import DataChunk, DataSource, DataSourceConfig

# Lazy imports
_FSSPEC_AVAILABLE = None


def _check_fsspec_available() -> bool:
    """Check if fsspec is available."""
    global _FSSPEC_AVAILABLE
    if _FSSPEC_AVAILABLE is None:
        try:
            import fsspec

            _FSSPEC_AVAILABLE = True
        except ImportError:
            _FSSPEC_AVAILABLE = False
    return _FSSPEC_AVAILABLE


class RemoteH5ADSource(DataSource):
    """
    Data source for remote H5AD files.

    Supports HTTP, HTTPS, and S3 URLs. Downloads file chunks as needed
    or can cache the entire file locally.

    Example:
        >>> source = RemoteH5ADSource(
        ...     "https://example.com/data.h5ad",
        ...     cache_dir="/tmp/cache"
        ... )
        >>> for chunk in source.iter_chunks():
        ...     process(chunk)

    Note:
        For S3 access, requires s3fs: pip install s3fs
        For HTTP access, may require aiohttp: pip install aiohttp
    """

    def __init__(
        self,
        url: str,
        config: Optional[DataSourceConfig] = None,
        cache_dir: Optional[Union[str, Path]] = None,
        download_full: bool = True,
    ):
        """
        Initialize remote H5AD source.

        Args:
            url: URL to H5AD file (http://, https://, s3://).
            config: Source configuration.
            cache_dir: Directory for caching downloaded data.
            download_full: Download entire file vs streaming.
        """
        super().__init__(config)

        self.url = url
        self.download_full = download_full

        # Parse URL
        parsed = urlparse(url)
        self.scheme = parsed.scheme
        self.filename = Path(parsed.path).name

        # Setup cache directory
        if cache_dir is not None:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.cache_dir = Path(tempfile.gettempdir()) / "cytoatlas_cache"
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._local_path: Optional[Path] = None
        self._adata = None

    def _ensure_downloaded(self) -> Path:
        """Ensure file is downloaded and return local path."""
        if self._local_path is not None and self._local_path.exists():
            return self._local_path

        # Check cache
        cache_path = self.cache_dir / self.filename
        if cache_path.exists():
            self._local_path = cache_path
            return self._local_path

        # Download
        if _check_fsspec_available():
            import fsspec

            fs, path = fsspec.url_to_fs(self.url)
            fs.get(path, str(cache_path))
        else:
            # Fallback to urllib for simple HTTP
            if self.scheme in ("http", "https"):
                import urllib.request

                urllib.request.urlretrieve(self.url, cache_path)
            else:
                raise ImportError(
                    f"fsspec required for {self.scheme}:// URLs. "
                    "Install with: pip install fsspec"
                )

        self._local_path = cache_path
        return self._local_path

    def _ensure_loaded(self) -> None:
        """Ensure AnnData is loaded."""
        if self._adata is not None:
            return

        import anndata as ad

        local_path = self._ensure_downloaded()
        self._adata = ad.read_h5ad(local_path, backed="r" if self.config.backed else None)

    @property
    def n_cells(self) -> int:
        """Total number of cells."""
        self._ensure_loaded()
        return self._adata.n_obs

    @property
    def n_genes(self) -> int:
        """Total number of genes."""
        self._ensure_loaded()
        return self._adata.n_vars

    @property
    def gene_names(self) -> list[str]:
        """List of gene names."""
        self._ensure_loaded()
        return list(self._adata.var_names)

    @property
    def obs_columns(self) -> list[str]:
        """Available cell metadata columns."""
        self._ensure_loaded()
        return list(self._adata.obs.columns)

    @property
    def var_columns(self) -> list[str]:
        """Available gene metadata columns."""
        self._ensure_loaded()
        return list(self._adata.var.columns)

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

        self._ensure_loaded()

        n_cells = self.n_cells
        n_chunks = math.ceil(n_cells / batch_size)

        # Prepare gene filter
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

            # Get expression
            if self.config.layer and self.config.layer in self._adata.layers:
                X_chunk = self._adata.layers[self.config.layer][start_idx:end_idx]
            else:
                X_chunk = self._adata.X[start_idx:end_idx]

            # Apply gene filter
            if gene_mask is not None:
                if sp.issparse(X_chunk):
                    X_chunk = X_chunk[:, gene_mask]
                else:
                    X_chunk = X_chunk[:, gene_mask]

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
                    "url": self.url,
                    "local_path": str(self._local_path),
                },
            )

    def get_obs(self, columns: Optional[list[str]] = None) -> pd.DataFrame:
        """Get cell metadata."""
        self._ensure_loaded()
        if columns is not None:
            return self._adata.obs[columns].copy()
        return self._adata.obs.copy()

    def get_var(self, columns: Optional[list[str]] = None) -> pd.DataFrame:
        """Get gene metadata."""
        self._ensure_loaded()
        if columns is not None:
            return self._adata.var[columns].copy()
        return self._adata.var.copy()

    def clear_cache(self) -> None:
        """Clear cached files."""
        if self._local_path and self._local_path.exists():
            self._local_path.unlink()
            self._local_path = None
            self._adata = None

    def close(self) -> None:
        """Close connections and optionally clear cache."""
        if self._adata is not None:
            if hasattr(self._adata, "file") and self._adata.file is not None:
                self._adata.file.close()
            self._adata = None

    def __enter__(self) -> "RemoteH5ADSource":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
