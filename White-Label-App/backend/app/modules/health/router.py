"""Liveness and readiness endpoints.

Two endpoints, deliberately different:

* `/health` touches nothing. It answers "is this process alive?" — the question
  an orchestrator asks before deciding to restart the container. If it checked
  the database, a brief Postgres blip would get the API killed and restarted,
  which helps nobody.
* `/health/ready` checks every backing service and reports each one separately.
  It answers "should this instance receive traffic?" and is what you look at
  first when something is broken.
"""

import asyncio
import time
from typing import Literal

from fastapi import APIRouter, Response, status
from pydantic import BaseModel
from sqlalchemy import text

from app.core.clients import get_qdrant, get_redis
from app.core.db import get_engine
from app.core.deps import SettingsDep

router = APIRouter(tags=["health"])


class DependencyStatus(BaseModel):
    name: str
    ok: bool
    latency_ms: float
    detail: str | None = None


class LivenessResponse(BaseModel):
    status: Literal["ok"]
    environment: str
    version: str


class ReadinessResponse(BaseModel):
    status: Literal["ready", "degraded"]
    dependencies: list[DependencyStatus]


@router.get("/health", response_model=LivenessResponse, summary="Liveness probe")
async def liveness(settings: SettingsDep) -> LivenessResponse:
    return LivenessResponse(
        status="ok",
        environment=settings.environment,
        version=settings.version,
    )


async def _timed(name: str, probe: object) -> DependencyStatus:
    """Run one awaitable probe, recording latency and swallowing its failure.

    A readiness check must never itself raise — a broken dependency is data to
    report, not a 500.
    """
    started = time.perf_counter()
    try:
        await probe  # type: ignore[misc]
        return DependencyStatus(
            name=name,
            ok=True,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
        )
    except Exception as exc:
        return DependencyStatus(
            name=name,
            ok=False,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
            detail=f"{type(exc).__name__}: {exc}",
        )


async def _probe_postgres() -> None:
    async with get_engine().connect() as conn:
        await conn.execute(text("SELECT 1"))


async def _probe_redis() -> None:
    await get_redis().ping()


async def _probe_qdrant() -> None:
    await get_qdrant().get_collections()


@router.get("/health/ready", response_model=ReadinessResponse, summary="Readiness probe")
async def readiness(response: Response) -> ReadinessResponse:
    # Probed concurrently: three sequential timeouts would make a readiness
    # check slower than the timeout of whatever is polling it.
    dependencies = await asyncio.gather(
        _timed("postgres", _probe_postgres()),
        _timed("redis", _probe_redis()),
        _timed("qdrant", _probe_qdrant()),
    )

    all_ok = all(dep.ok for dep in dependencies)
    if not all_ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return ReadinessResponse(
        status="ready" if all_ok else "degraded",
        dependencies=list(dependencies),
    )
