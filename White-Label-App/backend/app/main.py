"""FastAPI application factory."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import api_v1
from app.core.clients import close_clients
from app.core.config import Settings, get_settings
from app.core.db import get_engine
from app.core.errors import register_exception_handlers
from app.core.logging import setup_logging
from app.core.middleware import RequestContextMiddleware
from app.modules.health.router import router as health_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    logger.info(
        "Starting %s v%s (%s)",
        settings.product_name,
        settings.version,
        settings.environment,
    )
    yield
    # Connection pools must be released explicitly or a reload leaks sockets.
    await close_clients()
    await get_engine().dispose()
    logger.info("Shutdown complete")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    # Human-readable logs locally, JSON everywhere a log aggregator is reading.
    setup_logging(debug=settings.debug, json_output=not settings.is_local)

    app = FastAPI(
        title=settings.product_name,
        version=settings.version,
        # Interactive docs are a discovery surface; keep them out of production.
        docs_url="/docs" if settings.is_local else None,
        redoc_url=None,
        openapi_url="/openapi.json" if settings.is_local else None,
        lifespan=lifespan,
    )
    app.state.settings = settings

    app.add_middleware(RequestContextMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,  # explicit list, never "*"
        allow_credentials=True,  # required for the httpOnly cookie auth in Phase 1
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID"],
    )

    register_exception_handlers(app)

    # Health sits outside /api/v1: probes are infrastructure, not product API,
    # and their paths must stay stable across API versions.
    app.include_router(health_router)
    app.include_router(api_v1, prefix=settings.api_v1_prefix)

    return app


app = create_app()
