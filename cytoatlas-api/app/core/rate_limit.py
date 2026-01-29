"""Rate limiting middleware and utilities."""

from typing import Annotated

from fastapi import Depends, HTTPException, Request, status

from app.config import get_settings
from app.core.cache import CacheService, get_cache

settings = get_settings()


class RateLimiter:
    """Rate limiter using Redis sliding window."""

    def __init__(
        self,
        requests: int | None = None,
        window: int | None = None,
    ):
        """
        Initialize rate limiter.

        Args:
            requests: Maximum requests allowed in window
            window: Time window in seconds
        """
        self.requests = requests or settings.rate_limit_requests
        self.window = window or settings.rate_limit_window

    async def is_allowed(
        self,
        key: str,
        cache: CacheService,
    ) -> tuple[bool, int, int]:
        """
        Check if request is allowed.

        Returns:
            Tuple of (allowed, remaining, reset_in_seconds)
        """
        cache_key = f"ratelimit:{key}"

        # Get current count
        current = await cache.get(cache_key)

        if current is None:
            # First request in window
            await cache.set(cache_key, "1", ttl=self.window)
            return True, self.requests - 1, self.window

        count = int(current)

        if count >= self.requests:
            # Get TTL for reset time
            ttl = await cache.redis.ttl(cache_key)
            return False, 0, max(ttl, 0)

        # Increment counter
        await cache.incr(cache_key)
        return True, self.requests - count - 1, self.window


class RateLimitMiddleware:
    """FastAPI middleware for rate limiting."""

    def __init__(self, requests: int | None = None, window: int | None = None):
        self.limiter = RateLimiter(requests=requests, window=window)

    async def __call__(
        self,
        request: Request,
        cache: Annotated[CacheService, Depends(get_cache)],
    ) -> None:
        """Check rate limit for request."""
        # Get client identifier (IP or user ID)
        client_ip = request.client.host if request.client else "unknown"

        # Check if authenticated user
        user_id = getattr(request.state, "user_id", None)
        key = f"user:{user_id}" if user_id else f"ip:{client_ip}"

        allowed, remaining, reset = await self.limiter.is_allowed(key, cache)

        # Add rate limit headers to response
        request.state.rate_limit_remaining = remaining
        request.state.rate_limit_reset = reset

        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded. Try again in {reset} seconds.",
                headers={
                    "X-RateLimit-Limit": str(self.limiter.requests),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(reset),
                    "Retry-After": str(reset),
                },
            )


def rate_limit(
    requests: int | None = None,
    window: int | None = None,
) -> RateLimitMiddleware:
    """
    Create rate limit dependency.

    Usage:
        @router.get("/endpoint", dependencies=[Depends(rate_limit(100, 60))])
        async def endpoint():
            ...
    """
    return RateLimitMiddleware(requests=requests, window=window)
