"""Shared test fixtures.

The client fixture drives the ASGI app in-process — no server, no sockets. Note
that `ASGITransport` does not run the lifespan, which is intentional: the tests
in this phase exercise routes that touch no backing service, so the suite stays
runnable with nothing installed but Python.
"""

from collections.abc import AsyncIterator

import pytest
from httpx import ASGITransport, AsyncClient

from app.core.config import Settings
from app.main import create_app


@pytest.fixture
def settings() -> Settings:
    """Explicit settings so the suite never depends on a developer's .env."""
    return Settings(
        product_name="Test Product",
        version="0.0.0-test",
        environment="local",
        debug=False,
        database_url="postgresql+asyncpg://app:app@localhost:5432/app_test",
        redis_url="redis://localhost:6379/1",
        qdrant_url="http://localhost:6333",
        cors_origins=["http://localhost:3000"],
    )


@pytest.fixture
async def client(settings: Settings) -> AsyncIterator[AsyncClient]:
    app = create_app(settings)
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as async_client:
        yield async_client
