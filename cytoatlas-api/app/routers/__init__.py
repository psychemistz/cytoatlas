"""API routers."""

from app.routers.atlases import router as atlases_router
from app.routers.auth import router as auth_router
from app.routers.chat import router as chat_router
from app.routers.cima import router as cima_router
from app.routers.cross_atlas import router as cross_atlas_router
from app.routers.export import router as export_router
from app.routers.health import router as health_router
from app.routers.inflammation import router as inflammation_router
from app.routers.scatlas import router as scatlas_router
from app.routers.search import router as search_router
from app.routers.submit import router as submit_router
from app.routers.validation import router as validation_router
from app.routers.websocket import router as websocket_router

__all__ = [
    "atlases_router",
    "auth_router",
    "chat_router",
    "cima_router",
    "cross_atlas_router",
    "export_router",
    "health_router",
    "inflammation_router",
    "scatlas_router",
    "search_router",
    "submit_router",
    "validation_router",
    "websocket_router",
]
