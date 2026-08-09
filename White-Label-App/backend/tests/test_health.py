"""Tests for the Phase 0 wiring: probes, metadata, and the error contract."""

import pytest
from httpx import AsyncClient


async def test_liveness_reports_ok_without_touching_dependencies(client: AsyncClient) -> None:
    # No services are running in the unit suite; liveness must still answer.
    response = await client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["version"] == "0.0.0-test"


async def test_meta_serves_configured_product_name(client: AsyncClient) -> None:
    # The white-label contract: branding comes from config, not from code.
    response = await client.get("/api/v1/meta")

    assert response.status_code == 200
    assert response.json() == {
        "product_name": "Test Product",
        "version": "0.0.0-test",
        "environment": "local",
    }


async def test_every_response_carries_a_request_id(client: AsyncClient) -> None:
    response = await client.get("/health")

    assert response.headers["X-Request-ID"]


async def test_inbound_request_id_is_preserved(client: AsyncClient) -> None:
    # A trace started by a client or proxy must survive the hop.
    response = await client.get("/health", headers={"X-Request-ID": "trace-abc"})

    assert response.headers["X-Request-ID"] == "trace-abc"


async def test_errors_use_the_standard_envelope(client: AsyncClient) -> None:
    response = await client.get("/api/v1/does-not-exist")

    assert response.status_code == 404
    error = response.json()["error"]
    assert error["code"] == "http_404"
    assert error["request_id"]


@pytest.mark.integration
async def test_readiness_reports_all_dependencies(client: AsyncClient) -> None:
    """Requires Postgres, Redis, and Qdrant to be up.

    Run with: pytest -m integration
    """
    response = await client.get("/health/ready")

    assert response.status_code == 200, response.json()
    body = response.json()
    assert body["status"] == "ready"
    assert {dep["name"] for dep in body["dependencies"]} == {"postgres", "redis", "qdrant"}
    assert all(dep["ok"] for dep in body["dependencies"])
