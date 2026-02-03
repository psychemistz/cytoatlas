"""
cellxgene Census data source.

Provides access to the cellxgene Census, a cloud-based repository of
single-cell data with >50M cells across multiple datasets.
"""

from __future__ import annotations

import math
from typing import Any, Iterator, Optional, Union

import numpy as np
import pandas as pd
from scipy import sparse as sp

from cytoatlas_pipeline.ingest.base import DataChunk, DataSource, DataSourceConfig

# Lazy imports for optional dependencies
_CENSUS_AVAILABLE = None


def _check_census_available() -> bool:
    """Check if cellxgene-census is available."""
    global _CENSUS_AVAILABLE
    if _CENSUS_AVAILABLE is None:
        try:
            import cellxgene_census
            import tiledbsoma

            _CENSUS_AVAILABLE = True
        except ImportError:
            _CENSUS_AVAILABLE = False
    return _CENSUS_AVAILABLE


class CellxgeneCensusSource(DataSource):
    """
    Data source for cellxgene Census.

    Queries the cellxgene Census database for single-cell data matching
    specified filters (tissue, cell type, disease, etc.).

    Example:
        >>> source = CellxgeneCensusSource()
        >>> source = source.query(
        ...     tissue=["blood", "bone marrow"],
        ...     cell_type=["T cell", "B cell"],
        ...     disease="COVID-19",
        ...     max_cells=100000
        ... )
        >>>
        >>> for chunk in source.iter_chunks(batch_size=10000):
        ...     # Process chunk
        ...     pass

    Note:
        Requires cellxgene-census and tiledbsoma packages:
        pip install cellxgene-census tiledbsoma
    """

    def __init__(
        self,
        config: Optional[DataSourceConfig] = None,
        census_version: str = "stable",
        organism: str = "Homo sapiens",
    ):
        """
        Initialize Census data source.

        Args:
            config: Source configuration.
            census_version: Census version ("stable" or specific version).
            organism: Organism to query.
        """
        if not _check_census_available():
            raise ImportError(
                "cellxgene-census required. Install with: "
                "pip install cellxgene-census tiledbsoma"
            )

        super().__init__(config)

        self.census_version = census_version
        self.organism = organism

        # Query state
        self._query_params: dict[str, Any] = {}
        self._max_cells: Optional[int] = None

        # Cached metadata
        self._obs_df: Optional[pd.DataFrame] = None
        self._var_df: Optional[pd.DataFrame] = None
        self._n_cells: Optional[int] = None

    def query(
        self,
        tissue: Optional[Union[str, list[str]]] = None,
        cell_type: Optional[Union[str, list[str]]] = None,
        disease: Optional[Union[str, list[str]]] = None,
        sex: Optional[Union[str, list[str]]] = None,
        development_stage: Optional[Union[str, list[str]]] = None,
        dataset_id: Optional[Union[str, list[str]]] = None,
        assay: Optional[Union[str, list[str]]] = None,
        max_cells: Optional[int] = None,
        obs_value_filter: Optional[str] = None,
    ) -> "CellxgeneCensusSource":
        """
        Set query filters.

        Args:
            tissue: Tissue type(s) to include.
            cell_type: Cell type(s) to include.
            disease: Disease(s) to include.
            sex: Sex to include.
            development_stage: Development stage(s).
            dataset_id: Specific dataset ID(s).
            assay: Assay type(s).
            max_cells: Maximum cells to return.
            obs_value_filter: TileDB-SOMA value filter expression.

        Returns:
            Self for chaining.
        """
        # Build value filter
        filter_parts = []

        def add_filter(col: str, values: Union[str, list[str]]):
            if isinstance(values, str):
                values = [values]
            conditions = [f'{col} == "{v}"' for v in values]
            filter_parts.append(f"({' or '.join(conditions)})")

        if tissue is not None:
            add_filter("tissue", tissue)
        if cell_type is not None:
            add_filter("cell_type", cell_type)
        if disease is not None:
            add_filter("disease", disease)
        if sex is not None:
            add_filter("sex", sex)
        if development_stage is not None:
            add_filter("development_stage", development_stage)
        if dataset_id is not None:
            add_filter("dataset_id", dataset_id)
        if assay is not None:
            add_filter("assay", assay)

        if obs_value_filter is not None:
            filter_parts.append(f"({obs_value_filter})")

        self._query_params = {
            "obs_value_filter": " and ".join(filter_parts) if filter_parts else None,
        }
        self._max_cells = max_cells

        # Reset cached metadata
        self._obs_df = None
        self._var_df = None
        self._n_cells = None

        return self

    def _ensure_metadata(self) -> None:
        """Ensure metadata is loaded."""
        if self._obs_df is not None:
            return

        import cellxgene_census

        with cellxgene_census.open_soma(census_version=self.census_version) as census:
            # Get observation metadata
            obs_query = census["census_data"][self.organism].obs.read(
                value_filter=self._query_params.get("obs_value_filter"),
                column_names=[
                    "soma_joinid",
                    "cell_type",
                    "tissue",
                    "disease",
                    "sex",
                    "development_stage",
                    "assay",
                    "dataset_id",
                ],
            )
            self._obs_df = obs_query.concat().to_pandas()

            # Apply max_cells limit
            if self._max_cells is not None and len(self._obs_df) > self._max_cells:
                self._obs_df = self._obs_df.sample(n=self._max_cells, random_state=42)

            self._n_cells = len(self._obs_df)

            # Get variable metadata
            var_query = census["census_data"][self.organism].ms["RNA"].var.read(
                column_names=["soma_joinid", "feature_id", "feature_name"]
            )
            self._var_df = var_query.concat().to_pandas()
            self._var_df = self._var_df.set_index("feature_name")

    @property
    def n_cells(self) -> int:
        """Total number of cells matching query."""
        self._ensure_metadata()
        return self._n_cells

    @property
    def n_genes(self) -> int:
        """Total number of genes."""
        self._ensure_metadata()
        return len(self._var_df)

    @property
    def gene_names(self) -> list[str]:
        """List of gene names."""
        self._ensure_metadata()
        return list(self._var_df.index)

    @property
    def obs_columns(self) -> list[str]:
        """Available cell metadata columns."""
        self._ensure_metadata()
        return list(self._obs_df.columns)

    @property
    def var_columns(self) -> list[str]:
        """Available gene metadata columns."""
        self._ensure_metadata()
        return list(self._var_df.columns)

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
        import cellxgene_census

        if batch_size is None:
            batch_size = self.config.batch_size

        self._ensure_metadata()

        # Prepare gene filter
        if gene_filter is not None:
            gene_filter_upper = set(g.upper() for g in gene_filter)
            var_filtered = self._var_df[
                self._var_df.index.str.upper().isin(gene_filter_upper)
            ].copy()
            var_joinids = var_filtered["soma_joinid"].tolist()
        else:
            var_filtered = self._var_df.copy()
            var_joinids = None

        n_chunks = math.ceil(self._n_cells / batch_size)
        obs_joinids = self._obs_df["soma_joinid"].tolist()

        with cellxgene_census.open_soma(census_version=self.census_version) as census:
            experiment = census["census_data"][self.organism]

            for chunk_idx in range(n_chunks):
                start_idx = chunk_idx * batch_size
                end_idx = min(start_idx + batch_size, self._n_cells)

                chunk_joinids = obs_joinids[start_idx:end_idx]

                # Query expression data
                with experiment.axis_query(
                    measurement_name="RNA",
                    obs_query=cellxgene_census.tiledbsoma.AxisQuery(
                        coords=(chunk_joinids,)
                    ),
                    var_query=cellxgene_census.tiledbsoma.AxisQuery(
                        coords=(var_joinids,) if var_joinids else None
                    ),
                ) as query:
                    # Get sparse matrix
                    X_chunk = query.X("raw").tables().concat().to_scipy()

                    # Get metadata for this chunk
                    obs_chunk = self._obs_df.iloc[start_idx:end_idx].copy()

                yield DataChunk(
                    X=X_chunk,
                    obs=obs_chunk,
                    var=var_filtered,
                    chunk_index=chunk_idx,
                    total_chunks=n_chunks,
                    n_cells=end_idx - start_idx,
                    n_genes=len(var_filtered),
                    source_info={
                        "census_version": self.census_version,
                        "organism": self.organism,
                        "query_params": self._query_params,
                    },
                )

    def get_obs(self, columns: Optional[list[str]] = None) -> pd.DataFrame:
        """Get cell metadata."""
        self._ensure_metadata()
        if columns is not None:
            return self._obs_df[columns].copy()
        return self._obs_df.copy()

    def get_var(self, columns: Optional[list[str]] = None) -> pd.DataFrame:
        """Get gene metadata."""
        self._ensure_metadata()
        if columns is not None:
            return self._var_df[columns].copy()
        return self._var_df.copy()

    def available_tissues(self) -> list[str]:
        """Get list of available tissues in Census."""
        import cellxgene_census

        with cellxgene_census.open_soma(census_version=self.census_version) as census:
            obs = census["census_data"][self.organism].obs
            tissues = obs.read(column_names=["tissue"]).concat().to_pandas()
            return sorted(tissues["tissue"].unique().tolist())

    def available_cell_types(self, tissue: Optional[str] = None) -> list[str]:
        """Get list of available cell types."""
        import cellxgene_census

        with cellxgene_census.open_soma(census_version=self.census_version) as census:
            obs = census["census_data"][self.organism].obs

            if tissue:
                value_filter = f'tissue == "{tissue}"'
            else:
                value_filter = None

            cell_types = obs.read(
                column_names=["cell_type"], value_filter=value_filter
            ).concat().to_pandas()
            return sorted(cell_types["cell_type"].unique().tolist())

    def available_diseases(self) -> list[str]:
        """Get list of available diseases."""
        import cellxgene_census

        with cellxgene_census.open_soma(census_version=self.census_version) as census:
            obs = census["census_data"][self.organism].obs
            diseases = obs.read(column_names=["disease"]).concat().to_pandas()
            return sorted(diseases["disease"].unique().tolist())
