"""Tiered caching system with L1 (memory) and L2 (filesystem) layers."""

import hashlib
import pickle
import time
from collections import OrderedDict
from functools import wraps
from pathlib import Path
from typing import Any, Callable, ParamSpec, TypeVar

from pydantic import BaseModel

from app.config import get_settings

settings = get_settings()

P = ParamSpec("P")
T = TypeVar("T")


class L1Cache:
    """
    L1 cache: In-memory LRU cache with size-based eviction.

    Fast access, limited size (default 2GB).
    """

    def __init__(self, max_size_bytes: int = 2 * 1024 * 1024 * 1024):
        """
        Initialize L1 cache.

        Args:
            max_size_bytes: Maximum cache size in bytes (default 2GB)
        """
        self._cache: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._sizes: dict[str, int] = {}
        self._max_size_bytes = max_size_bytes
        self._current_size_bytes = 0

    def get(self, key: str) -> Any | None:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        if key not in self._cache:
            return None

        value, expiry = self._cache[key]

        # Check expiry
        if time.time() > expiry:
            self.delete(key)
            return None

        # Update LRU order
        self._cache.move_to_end(key)

        return value

    def set(self, key: str, value: Any, ttl: int = 3600) -> None:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
        """
        # Calculate size
        import sys

        value_size = sys.getsizeof(value)
        expiry = time.time() + ttl

        # Remove existing if present
        if key in self._cache:
            self.delete(key)

        # Evict until there's space
        while self._current_size_bytes + value_size > self._max_size_bytes:
            if not self._cache:
                # Cache empty but value too large
                if value_size > self._max_size_bytes:
                    return  # Don't cache
                break

            # Evict oldest
            oldest_key = next(iter(self._cache))
            self.delete(oldest_key)

        # Add to cache
        self._cache[key] = (value, expiry)
        self._sizes[key] = value_size
        self._current_size_bytes += value_size

    def delete(self, key: str) -> None:
        """Delete key from cache."""
        if key in self._cache:
            del self._cache[key]
            size = self._sizes.pop(key)
            self._current_size_bytes -= size

    def clear(self) -> None:
        """Clear entire cache."""
        self._cache.clear()
        self._sizes.clear()
        self._current_size_bytes = 0

    def stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return {
            "entries": len(self._cache),
            "size_bytes": self._current_size_bytes,
            "max_size_bytes": self._max_size_bytes,
            "utilization_pct": (
                (self._current_size_bytes / self._max_size_bytes * 100)
                if self._max_size_bytes > 0
                else 0
            ),
        }


class L2Cache:
    """
    L2 cache: Filesystem-based cache using pickle.

    Slower than L1 but larger capacity and persistent.
    Falls back gracefully if Redis not available.
    """

    def __init__(self, cache_dir: Path | str | None = None, default_ttl: int = 3600):
        """
        Initialize L2 cache.

        Args:
            cache_dir: Directory for cache files (default: .cache/l2/)
            default_ttl: Default TTL in seconds
        """
        if cache_dir is None:
            self._cache_dir = Path(".cache/l2")
        elif isinstance(cache_dir, str):
            self._cache_dir = Path(cache_dir)
        else:
            self._cache_dir = cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._default_ttl = default_ttl

    def get(self, key: str) -> Any | None:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        cache_file = self._get_cache_file(key)

        if not cache_file.exists():
            return None

        # Check TTL via file modification time
        mtime = cache_file.stat().st_mtime
        if time.time() - mtime > self._default_ttl:
            self.delete(key)
            return None

        # Load from disk
        try:
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        except Exception:
            # Corrupted cache file
            self.delete(key)
            return None

    def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (not enforced, just metadata)
        """
        cache_file = self._get_cache_file(key)

        try:
            with open(cache_file, "wb") as f:
                pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            # Failed to write, skip caching
            pass

    def delete(self, key: str) -> None:
        """Delete key from cache."""
        cache_file = self._get_cache_file(key)
        if cache_file.exists():
            cache_file.unlink()

    def clear(self) -> None:
        """Clear entire cache."""
        for cache_file in self._cache_dir.glob("*.pkl"):
            cache_file.unlink()

    def _get_cache_file(self, key: str) -> Path:
        """Get cache file path for key."""
        # Hash key to get safe filename
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self._cache_dir / f"{key_hash}.pkl"

    def stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        cache_files = list(self._cache_dir.glob("*.pkl"))
        total_size = sum(f.stat().st_size for f in cache_files)

        return {
            "entries": len(cache_files),
            "size_bytes": total_size,
            "size_mb": total_size / 1024 / 1024,
        }


class TieredCache:
    """
    Tiered cache combining L1 (fast/small) and L2 (slower/larger).

    Write-through strategy: writes go to both tiers.
    Read strategy: L1 first, then L2.
    """

    _instance: "TieredCache | None" = None

    def __new__(cls) -> "TieredCache":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        l1_max_size_bytes: int = 2 * 1024 * 1024 * 1024,
        l2_cache_dir: Path | None = None,
    ):
        """Initialize tiered cache."""
        if self._initialized:
            return

        self.l1 = L1Cache(max_size_bytes=l1_max_size_bytes)
        self.l2 = L2Cache(cache_dir=l2_cache_dir)
        self._initialized = True

    def get(self, key: str, tier: str = "both") -> Any | None:
        """
        Get value from cache.

        Args:
            key: Cache key
            tier: Which tier to check ("l1", "l2", "both")

        Returns:
            Cached value or None
        """
        # Try L1 first
        if tier in ("l1", "both"):
            value = self.l1.get(key)
            if value is not None:
                return value

        # Try L2
        if tier in ("l2", "both"):
            value = self.l2.get(key)
            if value is not None:
                # Promote to L1
                self.l1.set(key, value)
                return value

        return None

    def set(
        self,
        key: str,
        value: Any,
        tier: str = "both",
        ttl: int = 3600,
    ) -> None:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            tier: Which tier to write to ("l1", "l2", "both")
            ttl: Time to live in seconds
        """
        if tier in ("l1", "both"):
            self.l1.set(key, value, ttl=ttl)

        if tier in ("l2", "both"):
            self.l2.set(key, value, ttl=ttl)

    def delete(self, key: str, tier: str = "both") -> None:
        """Delete key from cache."""
        if tier in ("l1", "both"):
            self.l1.delete(key)

        if tier in ("l2", "both"):
            self.l2.delete(key)

    def clear(self, tier: str = "both") -> None:
        """Clear cache."""
        if tier in ("l1", "both"):
            self.l1.clear()

        if tier in ("l2", "both"):
            self.l2.clear()

    def stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        return {
            "l1": self.l1.stats(),
            "l2": self.l2.stats(),
        }


def _make_cache_key(prefix: str, func: Callable, args: tuple, kwargs: dict) -> str:
    """Generate cache key from function signature and arguments."""
    key_parts = [prefix, func.__module__, func.__name__]

    # Add args (skip self/cls)
    for arg in args:
        if isinstance(arg, BaseModel):
            key_parts.append(arg.model_dump_json())
        elif hasattr(arg, "__dict__"):
            continue  # Skip objects like self/cls
        else:
            key_parts.append(str(arg))

    # Add sorted kwargs
    for k, v in sorted(kwargs.items()):
        if isinstance(v, BaseModel):
            key_parts.append(f"{k}={v.model_dump_json()}")
        else:
            key_parts.append(f"{k}={v}")

    # Hash for consistent key length
    key_str = ":".join(key_parts)
    key_hash = hashlib.md5(key_str.encode()).hexdigest()[:16]
    return f"{prefix}:{func.__name__}:{key_hash}"


def tiered_cached(
    prefix: str = "cache",
    tier: str = "both",
    ttl: int = 3600,
    key_builder: Callable[..., str] | None = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator for tiered caching.

    Args:
        prefix: Cache key prefix
        tier: Which tier to use ("l1", "l2", "both")
        ttl: Time to live in seconds
        key_builder: Custom function to build cache key

    Usage:
        @tiered_cached(prefix="summary", tier="l1", ttl=600)
        async def get_summary() -> dict:
            ...

        @tiered_cached(prefix="full", tier="l2", ttl=3600)
        async def get_full_data() -> list[dict]:
            ...
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            cache = TieredCache()

            # Build cache key
            if key_builder:
                cache_key = key_builder(*args, **kwargs)
            else:
                cache_key = _make_cache_key(prefix, func, args, kwargs)

            # Try to get from cache
            cached_value = cache.get(cache_key, tier=tier)
            if cached_value is not None:
                return cached_value

            # Execute function
            result = await func(*args, **kwargs)

            # Store in cache
            cache.set(cache_key, result, tier=tier, ttl=ttl)

            return result

        return wrapper

    return decorator


def get_tiered_cache() -> TieredCache:
    """Get global tiered cache instance."""
    return TieredCache()
