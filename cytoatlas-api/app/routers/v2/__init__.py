"""API v2 routers — unified atlas endpoints with DuckDB backend."""

from app.routers.v2.atlases import router as v2_atlases_router
from app.routers.v2.validation import router as v2_validation_router

__all__ = [
    "v2_atlases_router",
    "v2_validation_router",
]
