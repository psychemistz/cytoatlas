"""Health check endpoints."""

from fastapi import APIRouter, Depends
from sqlalchemy import text

from app.config import get_settings
from app.core.cache import CacheService, get_cache
from app.core.database import async_session_factory
from app.schemas.common import HealthResponse

router = APIRouter(prefix="/health", tags=["Health"])
settings = get_settings()


@router.get("", response_model=HealthResponse)
async def health_check(
    cache: CacheService = Depends(get_cache),
) -> HealthResponse:
    """
    Check API health status.

    Returns status of API, database, and cache connections.
    """
    # Check database
    db_status = "not configured"
    if settings.use_database and async_session_factory is not None:
        try:
            async with async_session_factory() as session:
                await session.execute(text("SELECT 1"))
                db_status = "connected"
        except Exception:
            db_status = "disconnected"

    # Check cache
    try:
        await cache.redis.ping()
        # Check if using actual Redis or in-memory fallback
        cache_status = "redis" if cache._use_redis else "in-memory"
    except Exception:
        cache_status = "disconnected"

    # Determine overall status
    if db_status == "disconnected":
        status = "degraded"
    elif db_status == "not configured":
        status = "healthy"  # Running without database is OK
    else:
        status = "healthy"

    return HealthResponse(
        status=status,
        version=settings.app_version,
        database=db_status,
        cache=cache_status,
        environment=settings.environment,
    )


@router.get("/ready")
async def readiness_check() -> dict:
    """
    Kubernetes readiness probe.

    Returns 200 if the service is ready to accept traffic.
    """
    # Without database, we're always ready
    if not settings.use_database or async_session_factory is None:
        return {"ready": True}

    try:
        async with async_session_factory() as session:
            await session.execute(text("SELECT 1"))
            return {"ready": True}
    except Exception:
        return {"ready": False}


@router.get("/live")
async def liveness_check() -> dict:
    """
    Kubernetes liveness probe.

    Returns 200 if the service is alive.
    """
    return {"alive": True}


@router.get("/metrics")
async def get_metrics() -> dict:
    """
    Get API metrics summary.

    Returns request counts, response times, and error rates.
    """
    from app.core.logging import metrics

    return metrics.get_summary()
