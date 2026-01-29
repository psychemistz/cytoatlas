"""Base service class with common functionality."""

import json
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import get_settings
from app.core.cache import CacheService, cached
from app.models.atlas import Atlas

settings = get_settings()


class BaseService:
    """Base service with common data loading and caching functionality."""

    def __init__(self, db: AsyncSession | None = None):
        """Initialize service with optional database session."""
        self.db = db
        self._cache = CacheService()

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
        Load JSON data from visualization directory.

        Args:
            filename: JSON filename
            subdir: Optional subdirectory

        Returns:
            Parsed JSON data
        """
        if subdir:
            path = self.viz_data_path / subdir / filename
        else:
            path = self.viz_data_path / filename

        if not path.exists():
            raise FileNotFoundError(f"JSON file not found: {path}")

        with open(path) as f:
            return json.load(f)

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


class CachedDataLoader:
    """Utility for loading and caching data files."""

    _instance: "CachedDataLoader | None" = None
    _data_cache: dict[str, Any] = {}

    def __new__(cls) -> "CachedDataLoader":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._data_cache = {}
        return cls._instance

    @cached(prefix="json", ttl=3600)
    async def load_json_cached(self, path: str) -> Any:
        """Load and cache JSON file."""
        with open(path) as f:
            return json.load(f)

    def load_json_sync(self, path: str) -> Any:
        """Load JSON file synchronously with in-memory cache."""
        if path in self._data_cache:
            return self._data_cache[path]

        with open(path) as f:
            data = json.load(f)
            self._data_cache[path] = data
            return data

    def clear_cache(self) -> None:
        """Clear in-memory cache."""
        self._data_cache.clear()
