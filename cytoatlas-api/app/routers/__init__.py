"""API routers."""

from app.routers.auth import router as auth_router
from app.routers.cima import router as cima_router
from app.routers.cross_atlas import router as cross_atlas_router
from app.routers.export import router as export_router
from app.routers.health import router as health_router
from app.routers.inflammation import router as inflammation_router
from app.routers.scatlas import router as scatlas_router
from app.routers.validation import router as validation_router

__all__ = [
    "auth_router",
    "cima_router",
    "cross_atlas_router",
    "export_router",
    "health_router",
    "inflammation_router",
    "scatlas_router",
    "validation_router",
]
