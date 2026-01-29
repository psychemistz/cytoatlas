"""Caching layer with Redis or in-memory fallback."""

import hashlib
import json
import time
from collections.abc import Callable
from functools import wraps
from typing import Any, ParamSpec, TypeVar

from pydantic import BaseModel

from app.config import get_settings

settings = get_settings()

P = ParamSpec("P")
T = TypeVar("T")


class InMemoryCache:
    """Simple in-memory cache with TTL support."""

    def __init__(self):
        self._cache: dict[str, tuple[Any, float]] = {}  # key -> (value, expiry_time)

    async def get(self, key: str) -> str | None:
        """Get value from cache."""
        if key in self._cache:
            value, expiry = self._cache[key]
            if time.time() < expiry:
                return value
            del self._cache[key]
        return None

    async def set(self, key: str, value: str, ex: int | None = None) -> None:
        """Set value in cache."""
        ttl = ex or settings.redis_cache_ttl
        expiry = time.time() + ttl
        self._cache[key] = (value, expiry)

    async def delete(self, key: str) -> None:
        """Delete key from cache."""
        self._cache.pop(key, None)

    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        return await self.get(key) is not None

    async def incr(self, key: str) -> int:
        """Increment counter."""
        current = await self.get(key)
        new_val = int(current or 0) + 1
        # Keep existing TTL or set default
        if key in self._cache:
            _, expiry = self._cache[key]
            self._cache[key] = (str(new_val), expiry)
        else:
            await self.set(key, str(new_val))
        return new_val

    async def expire(self, key: str, seconds: int) -> None:
        """Set key expiration."""
        if key in self._cache:
            value, _ = self._cache[key]
            self._cache[key] = (value, time.time() + seconds)

    def ttl(self, key: str) -> int:
        """Get remaining TTL."""
        if key in self._cache:
            _, expiry = self._cache[key]
            return max(0, int(expiry - time.time()))
        return 0

    async def ping(self) -> bool:
        """Health check."""
        return True


class CacheService:
    """Cache service with Redis or in-memory fallback."""

    _instance: "CacheService | None" = None
    _redis: Any = None
    _memory_cache: InMemoryCache | None = None
    _use_redis: bool = False

    def __new__(cls) -> "CacheService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._memory_cache = InMemoryCache()
        return cls._instance

    async def connect(self) -> None:
        """Connect to Redis if available, otherwise use in-memory cache."""
        if settings.use_redis and self._redis is None:
            try:
                import redis.asyncio as redis_async
                self._redis = redis_async.from_url(
                    str(settings.redis_url),
                    encoding="utf-8",
                    decode_responses=True,
                )
                await self._redis.ping()
                self._use_redis = True
                print("Cache: Connected to Redis")
            except Exception as e:
                print(f"Cache: Redis unavailable ({e}), using in-memory cache")
                self._redis = None
                self._use_redis = False
        else:
            print("Cache: Using in-memory cache")

    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        if self._redis is not None:
            await self._redis.close()
            self._redis = None
            self._use_redis = False

    @property
    def redis(self) -> Any:
        """Get cache backend (Redis or in-memory)."""
        if self._use_redis and self._redis is not None:
            return self._redis
        return self._memory_cache

    async def get(self, key: str) -> str | None:
        """Get value from cache."""
        return await self.redis.get(key)

    async def set(
        self,
        key: str,
        value: str | dict | list,
        ttl: int | None = None,
    ) -> None:
        """Set value in cache."""
        if isinstance(value, (dict, list)):
            value = json.dumps(value)
        await self.redis.set(key, value, ex=ttl or settings.redis_cache_ttl)

    async def delete(self, key: str) -> None:
        """Delete key from cache."""
        await self.redis.delete(key)

    async def delete_pattern(self, pattern: str) -> None:
        """Delete all keys matching pattern."""
        if self._use_redis and self._redis is not None:
            async for key in self._redis.scan_iter(match=pattern):
                await self._redis.delete(key)
        # In-memory: simple pattern matching not implemented

    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        return await self.redis.exists(key)

    async def incr(self, key: str) -> int:
        """Increment counter."""
        return await self.redis.incr(key)

    async def expire(self, key: str, seconds: int) -> None:
        """Set key expiration."""
        await self.redis.expire(key, seconds)


def _make_cache_key(prefix: str, func: Callable, args: tuple, kwargs: dict) -> str:
    """Generate cache key from function signature and arguments."""
    # Create a hashable representation of args/kwargs
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


def cached(
    prefix: str = "cache",
    ttl: int | None = None,
    key_builder: Callable[..., str] | None = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator for caching async function results.

    Args:
        prefix: Cache key prefix (e.g., "cima", "inflammation")
        ttl: Time-to-live in seconds (default: settings.redis_cache_ttl)
        key_builder: Custom function to build cache key

    Usage:
        @cached(prefix="cima", ttl=3600)
        async def get_correlations(signature_type: str) -> list[dict]:
            ...
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            cache = CacheService()

            # Build cache key
            if key_builder:
                cache_key = key_builder(*args, **kwargs)
            else:
                cache_key = _make_cache_key(prefix, func, args, kwargs)

            # Try to get from cache
            try:
                cached_value = await cache.get(cache_key)
                if cached_value is not None:
                    return json.loads(cached_value)
            except Exception:
                pass  # Cache miss or error, proceed with function

            # Execute function
            result = await func(*args, **kwargs)

            # Store in cache
            try:
                await cache.set(cache_key, result, ttl=ttl)
            except Exception:
                pass  # Don't fail if caching fails

            return result

        return wrapper

    return decorator


async def get_cache() -> CacheService:
    """FastAPI dependency for cache service."""
    cache = CacheService()
    await cache.connect()
    return cache
