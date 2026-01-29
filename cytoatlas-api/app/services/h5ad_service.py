"""H5AD file access service with backed mode support."""

from functools import lru_cache
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd

from app.config import get_settings

settings = get_settings()


class H5ADService:
    """Service for accessing H5AD files in backed mode."""

    _instances: dict[str, "H5ADService"] = {}

    def __init__(self, h5ad_path: Path | str):
        """
        Initialize H5AD service.

        Args:
            h5ad_path: Path to H5AD file
        """
        self.path = Path(h5ad_path)
        self._adata: ad.AnnData | None = None
        self._metadata_cache: dict[str, Any] = {}

    @classmethod
    def get_instance(cls, h5ad_path: Path | str) -> "H5ADService":
        """Get or create singleton instance for a file."""
        path_str = str(h5ad_path)
        if path_str not in cls._instances:
            cls._instances[path_str] = cls(h5ad_path)
        return cls._instances[path_str]

    def open(self) -> ad.AnnData:
        """Open H5AD file in backed mode."""
        if self._adata is None:
            if not self.path.exists():
                raise FileNotFoundError(f"H5AD file not found: {self.path}")
            self._adata = ad.read_h5ad(self.path, backed="r")
        return self._adata

    def close(self) -> None:
        """Close H5AD file."""
        if self._adata is not None:
            if hasattr(self._adata, "file"):
                self._adata.file.close()
            self._adata = None

    @property
    def is_open(self) -> bool:
        """Check if file is open."""
        return self._adata is not None

    @property
    def shape(self) -> tuple[int, int]:
        """Get data shape (n_obs, n_vars)."""
        adata = self.open()
        return adata.shape

    @property
    def n_cells(self) -> int:
        """Get number of cells/observations."""
        return self.shape[0]

    @property
    def n_genes(self) -> int:
        """Get number of genes/variables."""
        return self.shape[1]

    @lru_cache(maxsize=1)
    def get_obs_columns(self) -> list[str]:
        """Get observation (cell) metadata column names."""
        adata = self.open()
        return adata.obs.columns.tolist()

    @lru_cache(maxsize=1)
    def get_var_columns(self) -> list[str]:
        """Get variable (gene) metadata column names."""
        adata = self.open()
        return adata.var.columns.tolist()

    def get_obs(
        self,
        columns: list[str] | None = None,
        indices: list[int] | None = None,
    ) -> pd.DataFrame:
        """
        Get observation metadata.

        Args:
            columns: Specific columns to return
            indices: Specific row indices to return

        Returns:
            DataFrame with observation metadata
        """
        adata = self.open()

        if indices is not None:
            obs = adata.obs.iloc[indices]
        else:
            obs = adata.obs

        if columns is not None:
            obs = obs[columns]

        return obs.copy()

    def get_var(
        self,
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Get variable metadata.

        Args:
            columns: Specific columns to return

        Returns:
            DataFrame with variable metadata
        """
        adata = self.open()
        var = adata.var

        if columns is not None:
            var = var[columns]

        return var.copy()

    def get_obs_names(self) -> list[str]:
        """Get observation names."""
        adata = self.open()
        return adata.obs_names.tolist()

    def get_var_names(self) -> list[str]:
        """Get variable names."""
        adata = self.open()
        return adata.var_names.tolist()

    def get_unique_values(self, column: str) -> list:
        """Get unique values for an observation column."""
        cache_key = f"unique_{column}"
        if cache_key not in self._metadata_cache:
            adata = self.open()
            self._metadata_cache[cache_key] = adata.obs[column].unique().tolist()
        return self._metadata_cache[cache_key]

    def get_expression_batch(
        self,
        cell_indices: list[int] | np.ndarray,
        gene_indices: list[int] | np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Get expression matrix for a batch of cells.

        Args:
            cell_indices: Indices of cells to retrieve
            gene_indices: Indices of genes to retrieve (optional)

        Returns:
            Expression matrix (cells x genes)
        """
        adata = self.open()

        if gene_indices is not None:
            return adata[cell_indices, gene_indices].X
        else:
            return adata[cell_indices, :].X

    def compute_pseudobulk(
        self,
        groupby: list[str],
        layer: str | None = None,
    ) -> pd.DataFrame:
        """
        Compute pseudobulk expression by grouping.

        Args:
            groupby: Columns to group by (e.g., ['sample', 'cell_type'])
            layer: Layer to use (None = X)

        Returns:
            Pseudobulk expression DataFrame (groups x genes)
        """
        adata = self.open()

        # Get grouping
        obs = adata.obs[groupby].copy()
        obs["_group"] = obs.apply(lambda x: "_".join(str(v) for v in x), axis=1)

        groups = obs["_group"].unique()
        gene_names = adata.var_names.tolist()

        # Initialize result matrix
        result = np.zeros((len(groups), len(gene_names)))

        for i, group in enumerate(groups):
            mask = obs["_group"] == group
            indices = np.where(mask)[0]

            # Get expression for this group
            if layer:
                expr = adata[indices, :].layers[layer]
            else:
                expr = adata[indices, :].X

            # Sum expression
            if hasattr(expr, "toarray"):
                result[i, :] = np.asarray(expr.sum(axis=0)).flatten()
            else:
                result[i, :] = expr.sum(axis=0)

        return pd.DataFrame(result, index=groups, columns=gene_names)

    def get_cell_type_counts(self, column: str = "cell_type") -> dict[str, int]:
        """Get cell counts per cell type."""
        cache_key = f"counts_{column}"
        if cache_key not in self._metadata_cache:
            adata = self.open()
            self._metadata_cache[cache_key] = adata.obs[column].value_counts().to_dict()
        return self._metadata_cache[cache_key]

    def __del__(self):
        """Cleanup on deletion."""
        self.close()


# Factory functions for common H5AD files
def get_cima_h5ad() -> H5ADService:
    """Get CIMA H5AD service."""
    return H5ADService.get_instance(settings.cima_h5ad)


def get_inflammation_main_h5ad() -> H5ADService:
    """Get Inflammation main cohort H5AD service."""
    return H5ADService.get_instance(settings.inflammation_main_h5ad)


def get_inflammation_validation_h5ad() -> H5ADService:
    """Get Inflammation validation cohort H5AD service."""
    return H5ADService.get_instance(settings.inflammation_validation_h5ad)


def get_inflammation_external_h5ad() -> H5ADService:
    """Get Inflammation external cohort H5AD service."""
    return H5ADService.get_instance(settings.inflammation_external_h5ad)


def get_scatlas_normal_h5ad() -> H5ADService:
    """Get scAtlas normal tissue H5AD service."""
    return H5ADService.get_instance(settings.scatlas_normal_h5ad)


def get_scatlas_cancer_h5ad() -> H5ADService:
    """Get scAtlas cancer H5AD service."""
    return H5ADService.get_instance(settings.scatlas_cancer_h5ad)
