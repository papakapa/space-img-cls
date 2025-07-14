from fastapi import APIRouter

from app.routes.ai import ai_router
from app.routes.health import health_router
from app.routes.data import data_router

main_router = APIRouter()

main_router.include_router(health_router, prefix="/health", tags=["Health"])
main_router.include_router(ai_router, prefix="/ai", tags=["AI"])
main_router.include_router(data_router, prefix='/data', tags=["Data"])
