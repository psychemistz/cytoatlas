"""Parquet-based repository implementation with predicate pushdown."""

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from app.config import get_settings
from app.repositories.base import BaseRepository
from app.repositories.json_repository import JSONRepository

settings = get_settings()


class ParquetRepository(BaseRepository):
    """
    Parquet repository with predicate pushdown and memory mapping.

    Falls back to JSON repository if Parquet file not found.
    """

    def __init__(self, parquet_base_path: Path | None = None):
        """
        Initialize Parquet repository.

        Args:
            parquet_base_path: Base path for parquet files
                              (default: {viz_data_path}/parquet_data)
        """
        self._parquet_base_path = (
            parquet_base_path or settings.viz_data_path / "parquet_data"
        )
        self._json_fallback = JSONRepository()

    async def get_activity(
        self,
        atlas: str,
        signature_type: str,
        **filters: Any,
    ) -> list[dict]:
        """Get activity data for an atlas."""
        data_type = f"{atlas}_activity"
        filter_dict = {"signature_type": signature_type, **filters}
        return await self.get_data(data_type, **filter_dict)

    async def get_correlations(
        self,
        atlas: str,
        variable: str,
        **filters: Any,
    ) -> list[dict]:
        """Get correlation data for an atlas."""
        data_type = f"{atlas}_{variable}_correlations"
        return await self.get_data(data_type, **filters)

    async def get_differential(
        self,
        atlas: str,
        comparison: str,
        **filters: Any,
    ) -> list[dict]:
        """Get differential activity data."""
        data_type = f"{atlas}_{comparison}_differential"
        return await self.get_data(data_type, **filters)

    async def get_data(
        self,
        data_type: str,
        **filters: Any,
    ) -> list[dict]:
        """
        Generic method to get any data type.

        Uses predicate pushdown when possible, falls back to JSON.
        """
        # Try Parquet first
        parquet_path = self._get_parquet_path(data_type)

        if parquet_path.exists():
            data = await self._load_parquet(parquet_path, filters)
        else:
            # Fallback to JSON
            data = await self._json_fallback.get_data(data_type, **filters)

        return data

    async def stream_results(
        self,
        data_type: str,
        **filters: Any,
    ) -> AsyncIterator[dict]:
        """Stream results for large datasets."""
        data = await self.get_data(data_type, **filters)
        async for item in self.stream_items(data):
            yield item

    def _get_parquet_path(self, data_type: str) -> Path:
        """
        Get path to Parquet file for data type.

        Partition scheme: parquet_data/{atlas}/{data_type}/data.parquet
        """
        # Extract atlas from data_type if present
        parts = data_type.split("_", 1)
        if len(parts) > 1 and parts[0] in ("cima", "inflammation", "scatlas"):
            atlas = parts[0]
            return self._parquet_base_path / atlas / data_type / "data.parquet"
        else:
            # Generic data type
            return self._parquet_base_path / data_type / "data.parquet"

    async def _load_parquet(
        self,
        path: Path,
        filters: dict[str, Any],
    ) -> list[dict]:
        """
        Load Parquet file with predicate pushdown.

        Args:
            path: Path to Parquet file
            filters: Filters to apply via predicate pushdown

        Returns:
            List of records as dictionaries
        """
        # Build filter expression for predicate pushdown
        filter_expr = self._build_filter_expression(filters)

        # Read with memory mapping and optional filter
        table = pq.read_table(
            path,
            filters=filter_expr,
            memory_map=True,
        )

        # Convert to list of dicts
        return table.to_pylist()

    def _build_filter_expression(
        self,
        filters: dict[str, Any],
    ) -> list[tuple] | None:
        """
        Build PyArrow filter expression from filter dict.

        Args:
            filters: Dictionary of field -> value filters

        Returns:
            Filter expression for PyArrow, or None if no filters
        """
        if not filters:
            return None

        filter_exprs = []

        for key, value in filters.items():
            if value is None:
                continue

            # Handle list values (OR logic)
            if isinstance(value, list):
                # field.isin(values)
                filter_exprs.append((key, "in", value))
            # Handle single value
            else:
                filter_exprs.append((key, "=", value))

        return filter_exprs if filter_exprs else None
