"""Application settings.

Every value comes from the environment (or an `.env` file in local dev) —
nothing product-specific is hardcoded, because this backend is meant to be
cloned and rebranded. See `infra/.env.example` for the full list.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Identity of this deployment ───────────────────────────────────────────
    # `product_name` is the first white-label knob: it is served to clients via
    # /api/v1/meta so the UI can brand itself without a rebuild.
    product_name: str = "White Label App"
    version: str = "0.1.0"
    environment: Literal["local", "staging", "production"] = "local"
    debug: bool = False

    api_v1_prefix: str = "/api/v1"

    # ── Backing services ──────────────────────────────────────────────────────
    database_url: str = Field(
        default="postgresql+asyncpg://app:app@localhost:5432/app",
        description="SQLAlchemy async DSN. Must use the +asyncpg driver.",
    )
    redis_url: str = Field(default="redis://localhost:6379/0")
    qdrant_url: str = Field(default="http://localhost:6333")

    # ── HTTP ──────────────────────────────────────────────────────────────────
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, value: object) -> object:
        """Accept a comma-separated string so CORS_ORIGINS reads naturally in
        docker-compose and shell env, not just as a JSON array."""
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value

    @field_validator("database_url")
    @classmethod
    def _require_async_driver(cls, value: str) -> str:
        # Caught here rather than as an opaque SQLAlchemy error six frames deep.
        if "+asyncpg" not in value:
            raise ValueError("DATABASE_URL must use the postgresql+asyncpg:// driver")
        return value

    @property
    def is_local(self) -> bool:
        return self.environment == "local"


@lru_cache
def get_settings() -> Settings:
    """Cached so settings are parsed and validated exactly once per process."""
    return Settings()
