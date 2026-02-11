"""Base service class with common functionality."""

import logging
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

import orjson
import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.core.cache import CacheService, cached
from app.core.streaming import StreamingJSONResponse, create_streaming_response
from app.models.atlas import Atlas
from app.repositories.json_repository import JSONRepository

logger = logging.getLogger(__name__)
settings = get_settings()

# ---------------------------------------------------------------------------
#  JSON filename → DuckDB table name mapping
#
#  Used by load_json() to transparently query DuckDB instead of reading
#  JSON files from disk.  Keys are the bare JSON filenames (no path),
#  values are the DuckDB table names from convert_data_to_duckdb.py.
# ---------------------------------------------------------------------------
_JSON_TO_TABLE: dict[str, str] = {
    # Large visualization files (highest-impact for DuckDB migration)
    "activity_boxplot.json": "activity",
    "inflammation_disease.json": "inflammation_disease",
    "inflammation_disease_filtered.json": "inflammation_disease_filtered",
    "singlecell_activity.json": "singlecell_activity",
    "scatlas_celltypes.json": "scatlas_celltypes",
    "age_bmi_boxplots.json": "age_bmi_boxplots",
    "age_bmi_boxplots_filtered.json": "age_bmi_boxplots_filtered",
    "bulk_rnaseq_validation.json": "bulk_validation",
    # Medium-size flat JSON files
    "inflammation_severity.json": "inflammation_severity",
    "inflammation_severity_filtered.json": "inflammation_severity_filtered",
    "scatlas_organs.json": "scatlas_organs",
    "gene_expression.json": "gene_expression",
    "expression_boxplot.json": "expression_boxplot",
    "cima_celltype.json": "cima_celltype",
    "inflammation_celltype.json": "inflammation_celltype",
    "cima_differential.json": "cima_differential",
    "inflammation_differential.json": "inflammation_differential",
    "cima_metabolites_top.json": "cima_metabolites_top",
    "scatlas_organs_top.json": "scatlas_organs_top",
    # Single-cell files
    "cima_singlecell_cytosig.json": "cima_singlecell_cytosig",
    "scatlas_normal_singlecell_cytosig.json": "scatlas_normal_singlecell_cytosig",
    "scatlas_cancer_singlecell_cytosig.json": "scatlas_cancer_singlecell_cytosig",
    "scatlas_normal_singlecell_secact.json": "scatlas_normal_singlecell_secact",
    "scatlas_cancer_singlecell_secact.json": "scatlas_cancer_singlecell_secact",
    # Nested JSON files (flattened in DuckDB)
    "cima_correlations.json": "cima_correlations",
    "inflammation_correlations.json": "inflammation_correlations",
    "inflammation_celltype_correlations.json": "inflammation_celltype_correlations",
    "cima_celltype_correlations.json": "cima_celltype_correlations",
    "exhaustion.json": "exhaustion",
    "immune_infiltration.json": "immune_infiltration",
    "caf_signatures.json": "caf_signatures",
    "adjacent_tissue.json": "adjacent_tissue",
    "cancer_comparison.json": "cancer_comparison",
    "cancer_types.json": "cancer_types",
    "organ_cancer_matrix.json": "organ_cancer_matrix",
    "cross_atlas.json": "cross_atlas",
    "cohort_validation.json": "cohort_validation",
    # Additional files loaded by services (auto-discovered in DuckDB conversion)
    "cima_biochem_scatter.json": "cima_biochem_scatter",
    "cima_eqtl.json": "cima_eqtl",
    "cima_eqtl_top.json": "cima_eqtl_top",
    "cima_population_stratification.json": "cima_population_stratification",
    "disease_sankey.json": "disease_sankey",
    "gene_list.json": "gene_list",
    "inflammation_cell_drivers.json": "inflammation_cell_drivers",
    "inflammation_longitudinal.json": "inflammation_longitudinal",
    "treatment_response.json": "treatment_response",
    # Validation tables
    "cima_atlas_validation.json": "cima_atlas_validation",
    "gene_expression_inflammation.json": "gene_expression_inflammation",
    "validation_corr_boxplot.json": "validation_corr_boxplot",
    "validation_summary.json": "validation_summary",
    # Perturbation domain (parse_10M + Tahoe)
    "parse10m_cytokine_heatmap.json": "parse10m_cytokine_heatmap",
    "parse10m_ground_truth.json": "parse10m_ground_truth",
    "parse10m_donor_variability.json": "parse10m_donor_variability",
    "tahoe_drug_sensitivity.json": "tahoe_drug_sensitivity",
    "tahoe_dose_response.json": "tahoe_dose_response",
    "tahoe_pathway_activation.json": "tahoe_pathway_activation",
    # Spatial domain (SpatialCorpus-110M)
    "spatial_tissue_activity.json": "spatial_tissue_activity_viz",
    "spatial_technology_comparison.json": "spatial_technology_comparison_viz",
    "spatial_gene_coverage.json": "spatial_gene_coverage_viz",
    "spatial_dataset_catalog.json": "spatial_dataset_metadata",
}

# Lazy-initialized DuckDB repository singleton
_duckdb_repo = None


def _get_duckdb_repository():
    """Get or create DuckDB repository singleton (lazy init)."""
    global _duckdb_repo
    if _duckdb_repo is None and settings.use_duckdb:
        try:
            from app.repositories.duckdb_repository import DuckDBRepository
            _duckdb_repo = DuckDBRepository(settings.duckdb_atlas_path)
            if _duckdb_repo.available:
                logger.info("DuckDB repository initialized: %s", settings.duckdb_atlas_path)
            else:
                logger.warning("DuckDB file not found, falling back to JSON")
                _duckdb_repo = None
        except Exception as e:
            logger.warning("DuckDB repository unavailable, falling back to JSON: %s", e)
    return _duckdb_repo


class BaseService:
    """Base service with common data loading and caching functionality."""

    def __init__(self, db: AsyncSession | None = None):
        """Initialize service with optional database session."""
        self.db = db
        self._cache = CacheService()
        self._json_repository = JSONRepository()

    @property
    def repository(self):
        """Get repository instance (DuckDB if available, else JSON fallback)."""
        duckdb_repo = _get_duckdb_repository()
        if duckdb_repo is not None:
            return duckdb_repo
        return self._json_repository

    @property
    def viz_data_path(self) -> Path:
        """Get visualization data directory."""
        return settings.viz_data_path

    @property
    def results_path(self) -> Path:
        """Get results directory."""
        return settings.results_base_path

    async def load_json(self, filename: str, subdir: str | None = None) -> Any:
        """
        Load data, preferring DuckDB when available, else JSON file.

        When a DuckDB backend is configured and the requested file has a
        corresponding DuckDB table (see ``_JSON_TO_TABLE``), data is
        fetched from DuckDB via ``get_data()``.  Otherwise falls back to
        reading the JSON file from disk using orjson.

        Args:
            filename: JSON filename
            subdir: Optional subdirectory (bypasses DuckDB for subdir loads)

        Returns:
            Parsed data (list[dict] from DuckDB or raw JSON)

        Raises:
            FileNotFoundError: If file doesn't exist (and no DuckDB fallback)
            ValueError: If path traversal is detected
        """
        # Try DuckDB for top-level viz data files (not subdirectory loads)
        if subdir is None:
            duckdb_repo = _get_duckdb_repository()
            table_name = _JSON_TO_TABLE.get(filename)
            if duckdb_repo is not None and table_name is not None:
                try:
                    return await duckdb_repo.get_data(table_name)
                except Exception as e:
                    logger.debug(
                        "DuckDB query failed for %s (table=%s), "
                        "falling back to JSON: %s",
                        filename, table_name, e,
                    )

        # Fallback: load from JSON file
        if subdir:
            path = self.viz_data_path / subdir / filename
        else:
            path = self.viz_data_path / filename

        # Security check: prevent path traversal
        resolved_path = path.resolve()
        resolved_base = self.viz_data_path.resolve()

        if not resolved_path.is_relative_to(resolved_base):
            raise ValueError(f"Path traversal detected: {filename}")

        if not path.exists():
            raise FileNotFoundError(f"JSON file not found: {path}")

        with open(path, "rb") as f:
            return orjson.loads(f.read())

    async def load_csv(
        self,
        filename: str,
        subdir: str | None = None,
        base_path: Path | None = None,
    ) -> pd.DataFrame:
        """
        Load CSV data from results directory.

        Args:
            filename: CSV filename
            subdir: Optional subdirectory
            base_path: Override base path

        Returns:
            Pandas DataFrame
        """
        base = base_path or self.results_path

        if subdir:
            path = base / subdir / filename
        else:
            path = base / filename

        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")

        return pd.read_csv(path)

    async def get_atlas(self, name: str) -> Atlas | None:
        """Get atlas by name from database."""
        if self.db is None:
            return None

        result = await self.db.execute(select(Atlas).where(Atlas.name == name))
        return result.scalar_one_or_none()

    def filter_by_signature_type(
        self,
        data: list[dict],
        signature_type: str,
        key: str = "signature_type",
    ) -> list[dict]:
        """Filter data by signature type."""
        return [d for d in data if d.get(key) == signature_type]

    def filter_by_cell_type(
        self,
        data: list[dict],
        cell_type: str,
        key: str = "cell_type",
    ) -> list[dict]:
        """Filter data by cell type."""
        return [d for d in data if d.get(key) == cell_type]

    def paginate(
        self,
        data: list[Any],
        offset: int = 0,
        limit: int = 100,
    ) -> tuple[list[Any], int, bool]:
        """
        Paginate a list of data.

        Returns:
            Tuple of (paginated_data, total_count, has_more)
        """
        total = len(data)
        end = offset + limit
        paginated = data[offset:end]
        has_more = end < total

        return paginated, total, has_more

    def round_floats(self, data: list[dict], decimals: int = 4) -> list[dict]:
        """Round float values in list of dicts."""
        result = []
        for item in data:
            rounded = {}
            for k, v in item.items():
                if isinstance(v, float):
                    rounded[k] = round(v, decimals)
                else:
                    rounded[k] = v
            result.append(rounded)
        return result

    def stream_json(
        self,
        data: list[dict],
        format: str = "array",
        etag: str | None = None,
        last_modified: int | None = None,
    ) -> StreamingJSONResponse:
        """
        Create streaming JSON response for large datasets.

        Args:
            data: List of items to stream
            format: "array" (JSON array) or "jsonl" (line-delimited JSON)
            etag: Optional ETag header value
            last_modified: Optional Last-Modified timestamp

        Returns:
            StreamingJSONResponse
        """
        return create_streaming_response(
            items=data,
            format=format,
            etag=etag,
            last_modified=last_modified,
        )


class CachedDataLoader:
    """Utility for loading and caching data files with LRU eviction."""

    _instance: "CachedDataLoader | None" = None
    _data_cache: OrderedDict[str, Any] = OrderedDict()
    _max_cache_entries: int = 50

    def __new__(cls) -> "CachedDataLoader":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._data_cache = OrderedDict()
        return cls._instance

    @cached(prefix="json", ttl=3600)
    async def load_json_cached(self, path: str) -> Any:
        """Load and cache JSON file using orjson."""
        with open(path, "rb") as f:
            return orjson.loads(f.read())

    def load_json_sync(self, path: str) -> Any:
        """Load JSON file synchronously with LRU cache."""
        # Check cache and update LRU order
        if path in self._data_cache:
            self._data_cache.move_to_end(path)
            return self._data_cache[path]

        # Load from disk
        with open(path, "rb") as f:
            data = orjson.loads(f.read())

        # Evict oldest if at capacity
        if len(self._data_cache) >= self._max_cache_entries:
            self._data_cache.popitem(last=False)

        # Add to cache
        self._data_cache[path] = data
        return data

    def clear_cache(self) -> None:
        """Clear in-memory cache."""
        self._data_cache.clear()
