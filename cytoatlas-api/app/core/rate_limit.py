"""Rate limiting middleware and utilities.

Includes defense against X-Forwarded-For header spoofing via trusted_proxies config.
"""

import ipaddress
import logging
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status

from app.config import get_settings
from app.core.cache import CacheService, get_cache

logger = logging.getLogger(__name__)
settings = get_settings()

# Parse trusted proxies at module load time
_trusted_proxies: set[ipaddress.IPv4Network | ipaddress.IPv6Network] = set()


def _parse_trusted_proxies() -> set[ipaddress.IPv4Network | ipaddress.IPv6Network]:
    """Parse trusted_proxies setting into a set of IP networks."""
    proxies = set()
    raw = getattr(settings, "trusted_proxies", "")
    if not raw:
        return proxies
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        try:
            proxies.add(ipaddress.ip_network(entry, strict=False))
        except ValueError:
            logger.warning("Invalid trusted proxy entry: %s", entry)
    return proxies


def get_trusted_proxies() -> set[ipaddress.IPv4Network | ipaddress.IPv6Network]:
    """Get cached trusted proxies set."""
    global _trusted_proxies
    if not _trusted_proxies:
        _trusted_proxies = _parse_trusted_proxies()
    return _trusted_proxies


def get_real_client_ip(request: Request) -> str:
    """Extract the real client IP from a request, safely handling forwarded headers.

    Only trusts X-Forwarded-For and X-Real-IP headers when the immediate
    connection comes from a trusted proxy. This prevents IP spoofing by
    untrusted clients sending fake forwarded headers.

    Args:
        request: The incoming FastAPI request

    Returns:
        The best-effort real client IP address
    """
    # Direct connection IP
    direct_ip = request.client.host if request.client else "unknown"
    if direct_ip == "unknown":
        return direct_ip

    trusted = get_trusted_proxies()

    # If no trusted proxies configured, always use direct connection IP
    # This is the safe default - never trust forwarded headers without config
    if not trusted:
        return direct_ip

    # Check if the direct connection is from a trusted proxy
    try:
        direct_addr = ipaddress.ip_address(direct_ip)
    except ValueError:
        return direct_ip

    is_trusted = any(direct_addr in network for network in trusted)
    if not is_trusted:
        # Connection is not from a trusted proxy - ignore forwarded headers
        return direct_ip

    # Connection is from a trusted proxy - read forwarded headers
    # X-Real-IP takes precedence (set by nginx/reverse proxy)
    x_real_ip = request.headers.get("X-Real-IP")
    if x_real_ip:
        # Validate it looks like an IP
        try:
            ipaddress.ip_address(x_real_ip.strip())
            return x_real_ip.strip()
        except ValueError:
            logger.warning("Invalid X-Real-IP header: %s", x_real_ip)

    # Fall back to X-Forwarded-For (take the leftmost/client IP)
    x_forwarded_for = request.headers.get("X-Forwarded-For")
    if x_forwarded_for:
        # X-Forwarded-For format: client, proxy1, proxy2
        # The leftmost entry is the original client (if we trust the proxy chain)
        ips = [ip.strip() for ip in x_forwarded_for.split(",")]
        if ips:
            try:
                ipaddress.ip_address(ips[0])
                return ips[0]
            except ValueError:
                logger.warning("Invalid X-Forwarded-For first entry: %s", ips[0])

    # Fall through to direct IP
    return direct_ip


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
        """Check rate limit for request.

        Uses secure IP extraction that only trusts forwarded headers
        from configured trusted proxies.
        """
        # Get real client IP (safe against X-Forwarded-For spoofing)
        client_ip = get_real_client_ip(request)

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
