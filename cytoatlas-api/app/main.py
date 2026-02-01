"""FastAPI application factory and entry point."""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import get_settings

# Static files directory
STATIC_DIR = Path(__file__).parent.parent / "static"
from app.core.cache import CacheService
from app.core.logging import MetricsMiddleware, RequestLoggingMiddleware, logger
from app.core.database import close_db, init_db
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
)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Application lifespan manager."""
    # Startup
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Data path: {settings.viz_data_path}")

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

    yield

    # Shutdown
    logger.info("Shutting down...")
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
        lifespan=lifespan,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request logging and metrics middleware
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(MetricsMiddleware)

    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        """Handle uncaught exceptions."""
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "Internal server error",
                "detail": str(exc) if settings.debug else None,
            },
        )

    # Include routers
    api_prefix = settings.api_v1_prefix

    app.include_router(health_router, prefix=api_prefix)
    app.include_router(auth_router, prefix=api_prefix)
    app.include_router(atlases_router, prefix=api_prefix)  # Unified dynamic API
    app.include_router(cima_router, prefix=api_prefix)      # Legacy CIMA-specific
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
