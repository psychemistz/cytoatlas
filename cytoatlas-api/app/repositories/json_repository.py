"""JSON-based repository implementation with LRU eviction.

.. deprecated::
    JSONRepository is retained as a fallback during the DuckDB migration.
    New code should use :class:`~app.repositories.duckdb_repository.DuckDBRepository`
    as the primary data backend.  Once the DuckDB migration is complete and
    ``atlas_data.duckdb`` is generated, this module will be removed.
"""

import logging
import sys
import warnings
from collections import OrderedDict
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import orjson

from app.config import get_settings
from app.repositories.base import BaseRepository

logger = logging.getLogger(__name__)
settings = get_settings()

_DEPRECATION_WARNED = False


class JSONRepository(BaseRepository):
    """
    JSON repository with LRU cache eviction.

    Implements AtlasRepository protocol using JSON files as storage.
    Backward-compatible with existing BaseService.load_json() behavior.

    .. deprecated::
        Prefer DuckDBRepository for new integrations.  This repository
        is kept only as a fallback until the DuckDB migration is complete.
    """

    def __init__(
        self,
        max_cache_entries: int = 50,
        max_cache_bytes: int = 2 * 1024 * 1024 * 1024,  # 2GB
    ):
        """
        Initialize JSON repository.

        Args:
            max_cache_entries: Maximum number of cached files
            max_cache_bytes: Maximum cache size in bytes (default 2GB)
        """
        global _DEPRECATION_WARNED
        if not _DEPRECATION_WARNED:
            warnings.warn(
                "JSONRepository is deprecated and retained only as a fallback. "
                "Use DuckDBRepository as the primary data backend. "
                "Generate atlas_data.duckdb with: python scripts/convert_data_to_duckdb.py --all",
                DeprecationWarning,
                stacklevel=2,
            )
            _DEPRECATION_WARNED = True

        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._cache_sizes: dict[str, int] = {}
        self._max_cache_entries = max_cache_entries
        self._max_cache_bytes = max_cache_bytes
        self._current_cache_bytes = 0

    async def get_activity(
        self,
        atlas: str,
        signature_type: str,
        **filters: Any,
    ) -> list[dict]:
        """Get activity data for an atlas."""
        # Determine file path based on atlas
        if atlas == "cima":
            filename = "cima_activity.json"
        elif atlas == "inflammation":
            filename = "inflammation_activity.json"
        elif atlas == "scatlas":
            filename = "scatlas_activity.json"
        else:
            raise ValueError(f"Unknown atlas: {atlas}")

        data = await self._load_json(filename)

        # Apply filters
        filter_dict = {"signature_type": signature_type, **filters}
        return self.apply_filters(data, filter_dict)

    async def get_correlations(
        self,
        atlas: str,
        variable: str,
        **filters: Any,
    ) -> list[dict]:
        """Get correlation data for an atlas."""
        # Map to appropriate file
        filename = f"{atlas}_{variable}_correlations.json"
        data = await self._load_json(filename)

        return self.apply_filters(data, filters)

    async def get_differential(
        self,
        atlas: str,
        comparison: str,
        **filters: Any,
    ) -> list[dict]:
        """Get differential activity data."""
        filename = f"{atlas}_{comparison}_differential.json"
        data = await self._load_json(filename)

        return self.apply_filters(data, filters)

    async def get_data(
        self,
        data_type: str,
        **filters: Any,
    ) -> list[dict]:
        """Generic method to get any data type."""
        # Try to load from viz_data_path
        data = await self._load_json(f"{data_type}.json")
        return self.apply_filters(data, filters)

    async def stream_results(
        self,
        data_type: str,
        **filters: Any,
    ) -> AsyncIterator[dict]:
        """Stream results for large datasets."""
        data = await self.get_data(data_type, **filters)
        async for item in self.stream_items(data):
            yield item

    async def _load_json(self, filename: str, subdir: str | None = None) -> Any:
        """
        Load JSON file with LRU caching and size-based eviction.

        Args:
            filename: JSON filename
            subdir: Optional subdirectory

        Returns:
            Parsed JSON data
        """
        # Build file path
        if subdir:
            path = settings.viz_data_path / subdir / filename
        else:
            path = settings.viz_data_path / filename

        path_str = str(path)

        # Check cache (and update LRU order)
        if path_str in self._cache:
            # Move to end (most recently used)
            self._cache.move_to_end(path_str)
            return self._cache[path_str]

        # Load from disk
        if not path.exists():
            raise FileNotFoundError(f"JSON file not found: {path}")

        with open(path, "rb") as f:
            data = orjson.loads(f.read())

        # Calculate size
        data_size = sys.getsizeof(data)

        # Evict if necessary (LRU)
        while (
            len(self._cache) >= self._max_cache_entries
            or self._current_cache_bytes + data_size > self._max_cache_bytes
        ):
            if not self._cache:
                break
            # Remove oldest (first item in OrderedDict)
            oldest_key, oldest_value = self._cache.popitem(last=False)
            oldest_size = self._cache_sizes.pop(oldest_key)
            self._current_cache_bytes -= oldest_size

        # Add to cache
        self._cache[path_str] = data
        self._cache_sizes[path_str] = data_size
        self._current_cache_bytes += data_size

        return data

    def clear_cache(self) -> None:
        """Clear the entire cache."""
        self._cache.clear()
        self._cache_sizes.clear()
        self._current_cache_bytes = 0

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return {
            "entries": len(self._cache),
            "max_entries": self._max_cache_entries,
            "bytes": self._current_cache_bytes,
            "max_bytes": self._max_cache_bytes,
            "utilization_pct": (
                (self._current_cache_bytes / self._max_cache_bytes * 100)
                if self._max_cache_bytes > 0
                else 0
            ),
        }
