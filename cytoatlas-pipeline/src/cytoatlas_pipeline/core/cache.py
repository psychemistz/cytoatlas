"""
Result caching layer.

Provides disk-based caching for pipeline results with TTL and size limits.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar
import pickle

import numpy as np

T = TypeVar("T")


@dataclass
class CacheEntry:
    """Metadata for a cache entry."""

    key: str
    created_at: float
    expires_at: float
    size_bytes: int
    data_path: Path
    metadata: dict[str, Any]


class ResultCache:
    """
    Disk-based result caching with TTL and size management.

    Caches pipeline results to avoid redundant computation.

    Example:
        >>> cache = ResultCache("/path/to/cache", max_size_gb=10, ttl_hours=24)
        >>>
        >>> # Cache expensive computation
        >>> key = cache.make_key(config=config, input_hash=data_hash)
        >>> result = cache.get_or_compute(key, compute_fn, metadata={"step": "activity"})
        >>>
        >>> # Or manual get/set
        >>> if not cache.has(key):
        ...     result = expensive_computation()
        ...     cache.set(key, result)
        ... else:
        ...     result = cache.get(key)
    """

    def __init__(
        self,
        cache_dir: Path | str,
        max_size_gb: float = 10.0,
        ttl_hours: float = 24.0,
        compression: bool = True,
    ):
        """
        Initialize result cache.

        Args:
            cache_dir: Directory for cached data.
            max_size_gb: Maximum cache size in GB.
            ttl_hours: Time-to-live in hours.
            compression: Use compression for cached data.
        """
        self.cache_dir = Path(cache_dir)
        self.max_size_bytes = int(max_size_gb * 1024**3)
        self.ttl_seconds = ttl_hours * 3600
        self.compression = compression

        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Index file for metadata
        self.index_path = self.cache_dir / "index.json"
        self._index: dict[str, dict[str, Any]] = {}
        self._load_index()

    def _load_index(self) -> None:
        """Load cache index from disk."""
        if self.index_path.exists():
            try:
                with open(self.index_path) as f:
                    self._index = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._index = {}
        else:
            self._index = {}

    def _save_index(self) -> None:
        """Save cache index to disk."""
        with open(self.index_path, "w") as f:
            json.dump(self._index, f, indent=2)

    def _get_data_path(self, key: str) -> Path:
        """Get path for cached data."""
        # Use first 2 chars of hash for subdirectory (reduces files per directory)
        subdir = key[:2]
        return self.cache_dir / subdir / f"{key}.pkl"

    @staticmethod
    def make_key(*args, **kwargs) -> str:
        """
        Create cache key from arguments.

        Args:
            *args: Positional arguments to hash.
            **kwargs: Keyword arguments to hash.

        Returns:
            MD5 hash key.
        """
        # Serialize arguments
        key_parts = []
        for arg in args:
            if isinstance(arg, np.ndarray):
                key_parts.append(("array", arg.shape, arg.dtype.str, hash(arg.tobytes())))
            elif isinstance(arg, (dict, list)):
                key_parts.append(json.dumps(arg, sort_keys=True))
            else:
                key_parts.append(str(arg))

        for k, v in sorted(kwargs.items()):
            if isinstance(v, np.ndarray):
                key_parts.append((k, "array", v.shape, v.dtype.str, hash(v.tobytes())))
            elif isinstance(v, (dict, list)):
                key_parts.append((k, json.dumps(v, sort_keys=True)))
            else:
                key_parts.append((k, str(v)))

        key_str = str(key_parts)
        return hashlib.md5(key_str.encode()).hexdigest()

    def has(self, key: str) -> bool:
        """
        Check if key exists and is valid.

        Args:
            key: Cache key.

        Returns:
            True if valid entry exists.
        """
        if key not in self._index:
            return False

        entry = self._index[key]

        # Check expiration
        if time.time() > entry["expires_at"]:
            self._remove_entry(key)
            return False

        # Check file exists
        data_path = Path(entry["data_path"])
        if not data_path.exists():
            self._remove_entry(key)
            return False

        return True

    def get(self, key: str) -> Optional[Any]:
        """
        Get cached value.

        Args:
            key: Cache key.

        Returns:
            Cached value, or None if not found/expired.
        """
        if not self.has(key):
            return None

        data_path = Path(self._index[key]["data_path"])

        try:
            with open(data_path, "rb") as f:
                return pickle.load(f)
        except (pickle.PickleError, IOError):
            self._remove_entry(key)
            return None

    def set(
        self,
        key: str,
        value: Any,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Set cached value.

        Args:
            key: Cache key.
            value: Value to cache.
            metadata: Optional metadata to store.
        """
        # Ensure space
        self._ensure_space()

        data_path = self._get_data_path(key)
        data_path.parent.mkdir(parents=True, exist_ok=True)

        # Serialize
        try:
            with open(data_path, "wb") as f:
                pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
        except (pickle.PickleError, IOError) as e:
            raise RuntimeError(f"Failed to cache value: {e}") from e

        # Update index
        now = time.time()
        self._index[key] = {
            "key": key,
            "created_at": now,
            "expires_at": now + self.ttl_seconds,
            "size_bytes": data_path.stat().st_size,
            "data_path": str(data_path),
            "metadata": metadata or {},
        }
        self._save_index()

    def get_or_compute(
        self,
        key: str,
        compute_fn: Callable[[], T],
        metadata: Optional[dict[str, Any]] = None,
    ) -> T:
        """
        Get cached value or compute and cache.

        Args:
            key: Cache key.
            compute_fn: Function to compute value if not cached.
            metadata: Optional metadata to store.

        Returns:
            Cached or computed value.
        """
        result = self.get(key)
        if result is not None:
            return result

        result = compute_fn()
        self.set(key, result, metadata=metadata)
        return result

    def _remove_entry(self, key: str) -> None:
        """Remove cache entry."""
        if key in self._index:
            data_path = Path(self._index[key]["data_path"])
            if data_path.exists():
                data_path.unlink()
            del self._index[key]
            self._save_index()

    def _get_total_size(self) -> int:
        """Get total cache size in bytes."""
        return sum(entry["size_bytes"] for entry in self._index.values())

    def _ensure_space(self) -> None:
        """Ensure cache has space by removing old entries."""
        # First remove expired
        now = time.time()
        expired = [k for k, v in self._index.items() if v["expires_at"] < now]
        for key in expired:
            self._remove_entry(key)

        # Then remove oldest if still over limit
        while self._get_total_size() > self.max_size_bytes and self._index:
            oldest_key = min(self._index.keys(), key=lambda k: self._index[k]["created_at"])
            self._remove_entry(oldest_key)

    def invalidate(self, key: str) -> None:
        """
        Invalidate a cache entry.

        Args:
            key: Cache key to invalidate.
        """
        self._remove_entry(key)

    def invalidate_pattern(self, pattern: str) -> int:
        """
        Invalidate entries matching pattern in metadata.

        Args:
            pattern: Pattern to match in metadata.

        Returns:
            Number of entries invalidated.
        """
        to_remove = []
        for key, entry in self._index.items():
            metadata_str = json.dumps(entry.get("metadata", {}))
            if pattern in metadata_str:
                to_remove.append(key)

        for key in to_remove:
            self._remove_entry(key)

        return len(to_remove)

    def clear(self) -> None:
        """Clear entire cache."""
        for key in list(self._index.keys()):
            self._remove_entry(key)

        # Also remove any orphaned files
        for subdir in self.cache_dir.iterdir():
            if subdir.is_dir() and subdir.name != "index.json":
                shutil.rmtree(subdir)

    def stats(self) -> dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dict with cache stats.
        """
        total_size = self._get_total_size()
        return {
            "entries": len(self._index),
            "size_bytes": total_size,
            "size_gb": total_size / (1024**3),
            "max_size_gb": self.max_size_bytes / (1024**3),
            "utilization": total_size / self.max_size_bytes if self.max_size_bytes > 0 else 0,
        }
