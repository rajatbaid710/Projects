"""Deployment metadata for clients.

This is the first white-label seam. The web (and later mobile) client asks the
API who it is rather than hardcoding a product name, so rebranding a deployment
is an environment change.

It will grow into the full theme/flags endpoints from Phase 4 —
`GET /config/theme` and `GET /config/flags` — at which point the response here
stays the cheap, unauthenticated "which product am I talking to?" call.
"""

from fastapi import APIRouter
from pydantic import BaseModel

from app.core.deps import SettingsDep

router = APIRouter(prefix="/meta", tags=["meta"])


class MetaResponse(BaseModel):
    product_name: str
    version: str
    environment: str


@router.get("", response_model=MetaResponse, summary="Deployment identity")
async def read_meta(settings: SettingsDep) -> MetaResponse:
    return MetaResponse(
        product_name=settings.product_name,
        version=settings.version,
        environment=settings.environment,
    )
