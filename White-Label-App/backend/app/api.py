"""Versioned API router assembly.

Every module's router is registered here and nowhere else, so there is one
place to read to know what the API exposes — and one place a feature flag will
eventually gate a whole module from.
"""

from fastapi import APIRouter

from app.modules.meta.router import router as meta_router

api_v1 = APIRouter()
api_v1.include_router(meta_router)

# Modules land here as they are built:
#   api_v1.include_router(auth_router)
#   api_v1.include_router(users_router)
#   api_v1.include_router(admin_router)
