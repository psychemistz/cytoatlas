"""FastAPI application factory and entry point."""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import get_settings

# Static files directory
STATIC_DIR = Path(__file__).parent.parent / "static"
from app.core.audit import AuditMiddleware, get_audit_logger
from app.core.cache import CacheService
from app.core.logging import MetricsMiddleware, RequestLoggingMiddleware, logger
from app.core.database import close_db, init_db
from app.core.security_headers import SecurityHeadersMiddleware
from app.routers import (
    atlases_router,
    auth_router,
    chat_router,
    cima_router,
    cross_atlas_router,
    export_router,
    gene_router,
    health_router,
    inflammation_router,
    scatlas_router,
    search_router,
    submit_router,
    validation_router,
    websocket_router,
    perturbation_router,
    spatial_router,
)
from app.routers.pipeline import router as pipeline_router

settings = get_settings()


# OpenAPI tags metadata
tags_metadata = [
    {
        "name": "Health",
        "description": "Health check and system status endpoints",
    },
    {
        "name": "Atlas Management",
        "description": "Atlas registry management: list, register, delete, and query atlas metadata.",
    },
    {
        "name": "CIMA Atlas",
        "description": "CIMA-specific endpoints: age/BMI correlations, biochemistry, metabolites, population stratification, eQTL.",
    },
    {
        "name": "Inflammation Atlas",
        "description": "Inflammation Atlas endpoints: disease activity, treatment response, temporal dynamics, severity, cohort validation.",
    },
    {
        "name": "scAtlas",
        "description": "scAtlas endpoints: organ signatures, cancer comparison, immune infiltration, T-cell states, exhaustion, CAF.",
    },
    {
        "name": "Cross-Atlas",
        "description": "Cross-atlas comparison and integration endpoints",
    },
    {
        "name": "Validation",
        "description": "Data validation and credibility assessment",
    },
    {
        "name": "Search",
        "description": "Global search across all atlases",
    },
    {
        "name": "Gene",
        "description": "Gene-centric views and analysis",
    },
    {
        "name": "Export",
        "description": "Data export in various formats",
    },
    {
        "name": "Chat",
        "description": "Claude AI assistant for data exploration",
    },
    {
        "name": "Submit",
        "description": "User dataset submission",
    },
    {
        "name": "Auth",
        "description": "Authentication and authorization",
    },
    {
        "name": "Pipeline",
        "description": "Pipeline status and management",
    },
]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Application lifespan manager."""
    # Startup
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Data path: {settings.viz_data_path}")

    # Security warnings
    if settings.secret_key is None and settings.environment != "production":
        import secrets
        runtime_secret = secrets.token_hex(32)
        # Monkey-patch the settings instance
        object.__setattr__(settings, "secret_key", runtime_secret)
        logger.warning("SECRET_KEY not set - generated random key for this session (will change on restart)")

    if settings.allowed_origins == "*":
        logger.warning("CORS: Allowing all origins (*) - this is insecure in production")

    # Initialize database (if configured)
    if settings.use_database:
        try:
            await init_db()
            logger.info("Database: Connected")
        except Exception as e:
            logger.warning(f"Database: Skipped ({e})")
    else:
        logger.info("Database: Not configured (running without persistence)")

    # Initialize cache
    try:
        cache = CacheService()
        await cache.connect()
        logger.info("Cache: Connected")
    except Exception as e:
        logger.warning(f"Cache: Error ({e})")

    # Initialize audit logger
    if settings.audit_enabled:
        audit_logger = get_audit_logger()
        await audit_logger.start()
        logger.info("Audit: Enabled")

    yield

    # Shutdown
    logger.info("Shutting down...")

    # Stop audit logger
    if settings.audit_enabled:
        try:
            audit_logger = get_audit_logger()
            await audit_logger.stop()
        except Exception:
            pass

    try:
        cache = CacheService()
        await cache.disconnect()
    except Exception:
        pass

    if settings.use_database:
        try:
            await close_db()
        except Exception:
            pass


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    app = FastAPI(
        title=settings.app_name,
        description=(
            "REST API for the CytoAtlas single-cell cytokine activity visualization. "
            "Provides access to computed cytokine and secreted protein activity signatures "
            "across three major human immune cell atlases: CIMA, Inflammation Atlas, and scAtlas."
        ),
        version=settings.app_version,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        openapi_tags=tags_metadata,
        lifespan=lifespan,
    )

    # Add API version header and caching middleware
    @app.middleware("http")
    async def add_api_version_and_cache_headers(request: Request, call_next):
        """Add API version header and cache control to all responses."""
        response = await call_next(request)
        response.headers["X-API-Version"] = "1.0"

        # Add cache headers for data endpoints (GET only)
        if request.method == "GET" and request.url.path.startswith(settings.api_v1_prefix):
            # Don't cache health checks or websocket connections
            if not request.url.path.endswith("/health") and "/ws" not in request.url.path:
                response.headers["Cache-Control"] = "public, max-age=3600"

        return response

    # Middleware (NOTE: FastAPI processes in REVERSE order of add_middleware calls)
    # Execution order: GZip -> SecurityHeaders -> CORS -> Audit -> RequestLogging -> Metrics
    # So add in reverse: Metrics first, GZip last
    app.add_middleware(MetricsMiddleware)
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(AuditMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-API-Key", "X-Request-ID"],
    )
    app.add_middleware(SecurityHeadersMiddleware)

    # Add GZip compression (minimum 1000 bytes)
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """Handle uncaught exceptions.

        SECURITY: Never expose stack traces, internal paths, or
        implementation details to clients regardless of debug mode.
        Debug details are logged server-side only.
        """
        # Log the actual error server-side (with full details)
        logger.error(f"Unhandled exception: {exc}", exc_info=True)

        # In development debug mode, include the exception type (but not
        # full traceback or file paths) to aid local development.
        if settings.debug and settings.environment == "development":
            detail = f"{type(exc).__name__}: {str(exc)}"
            # Strip any file paths from the message
            import re
            detail = re.sub(r"(/[^\s:]+)+", "[path]", detail)
        else:
            detail = "An unexpected error occurred"

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "Internal server error",
                "detail": detail,
            },
        )

    # Include routers
    api_prefix = settings.api_v1_prefix

    app.include_router(health_router, prefix=api_prefix)
    app.include_router(auth_router, prefix=api_prefix)
    app.include_router(atlases_router, prefix=api_prefix)  # Atlas management
    app.include_router(cima_router, prefix=api_prefix)
    app.include_router(inflammation_router, prefix=api_prefix)
    app.include_router(scatlas_router, prefix=api_prefix)
    app.include_router(cross_atlas_router, prefix=api_prefix)
    app.include_router(validation_router, prefix=api_prefix)
    app.include_router(export_router, prefix=api_prefix)

    # New modules
    app.include_router(search_router, prefix=api_prefix)
    app.include_router(gene_router, prefix=api_prefix)  # Gene-centric views
    app.include_router(submit_router, prefix=api_prefix)
    app.include_router(chat_router, prefix=api_prefix)
    app.include_router(websocket_router, prefix=api_prefix)
    app.include_router(pipeline_router, prefix=api_prefix)  # Pipeline management
    app.include_router(perturbation_router, prefix=api_prefix)  # Perturbation (parse_10M + Tahoe)
    app.include_router(spatial_router, prefix=api_prefix)  # Spatial (SpatialCorpus-110M)

    # Mount static files (CSS, JS, assets)
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
        logger.info(f"Static files: Serving from {STATIC_DIR}")

    # Root endpoint - serve frontend HTML
    @app.get("/")
    async def root():
        """Serve the frontend application."""
        index_path = STATIC_DIR / "index.html"
        if index_path.exists():
            return FileResponse(index_path)
        # Fallback to API info if no frontend
        return {
            "name": settings.app_name,
            "version": settings.app_version,
            "docs": "/docs",
            "api": settings.api_v1_prefix,
        }

    # SPA catch-all route for client-side routing
    @app.get("/{path:path}")
    async def spa_catch_all(path: str):
        """Catch-all route for SPA client-side routing."""
        # Don't catch API routes or static files
        if path.startswith("api/") or path.startswith("static/") or path in ["docs", "redoc", "openapi.json"]:
            return JSONResponse(status_code=404, content={"error": "Not found"})

        index_path = STATIC_DIR / "index.html"
        if index_path.exists():
            return FileResponse(index_path)
        return JSONResponse(status_code=404, content={"error": "Not found"})

    return app


# Create app instance
app = create_app()


def main() -> None:
    """Run the application with uvicorn."""
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        workers=1 if settings.debug else 4,
    )


if __name__ == "__main__":
    main()
