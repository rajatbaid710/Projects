"""Shared clients for the non-SQL backing services.

Cached per process so we hold one connection pool each, not one per request.
The AI layer will reuse `get_qdrant()` for vector search; for now only the
readiness check touches them.
"""

from functools import lru_cache

from qdrant_client import AsyncQdrantClient
from redis.asyncio import Redis

from app.core.config import get_settings


@lru_cache
def get_redis() -> Redis:
    return Redis.from_url(get_settings().redis_url, decode_responses=True)


@lru_cache
def get_qdrant() -> AsyncQdrantClient:
    return AsyncQdrantClient(url=get_settings().qdrant_url)


async def close_clients() -> None:
    """Close whatever was actually opened. Called from the app lifespan.

    The `currsize` guards matter: calling the getters here would *create* a
    client just to close it.
    """
    if get_redis.cache_info().currsize:
        await get_redis().aclose()
        get_redis.cache_clear()
    if get_qdrant.cache_info().currsize:
        await get_qdrant().close()
        get_qdrant.cache_clear()
