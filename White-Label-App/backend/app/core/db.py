"""Database engine, session factory, and the declarative base.

The engine and sessionmaker are built lazily and cached, so importing this
module never opens a connection — which keeps Alembic, tests, and the app all
able to import it freely.
"""

from collections.abc import AsyncGenerator
from functools import lru_cache

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.core.config import get_settings


class Base(DeclarativeBase):
    """Declarative base for every model in the project.

    Models must be imported before `Base.metadata` is used for autogenerate;
    `app/core/models.py` is the single place that does those imports.
    """


@lru_cache
def get_engine() -> AsyncEngine:
    settings = get_settings()
    return create_async_engine(
        settings.database_url,
        echo=settings.debug,
        pool_pre_ping=True,  # survives Postgres restarts and idle disconnects
    )


@lru_cache
def get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(
        bind=get_engine(),
        expire_on_commit=False,  # lets response serialization read attrs post-commit
        autoflush=False,
    )


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency yielding a session scoped to one request.

    Commit explicitly in the service layer. Any exception rolls back before the
    session is returned to the pool.
    """
    async with get_sessionmaker()() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
