"""Shared FastAPI dependencies.

This is where request-scoped injection lives. Phase 1 adds `get_current_user`
here and Phase 2 adds `require_permission`, so every route inherits auth and
RBAC the same way.

On settings: handlers read them through `SettingsDep`, which resolves from
`request.app.state` rather than the module-level cache. That is what makes
`create_app(settings)` actually govern behaviour — tests can build an app with
explicit settings, and two apps can coexist in one process.

The boundary worth knowing: process-level singletons (the SQLAlchemy engine, the
Redis and Qdrant clients) are built from the module-level `get_settings()`, since
they outlive any single request. Injected settings therefore change request
handling, not which database the process is connected to.
"""

from typing import Annotated

from fastapi import Depends, Request

from app.core.config import Settings


def get_request_settings(request: Request) -> Settings:
    return request.app.state.settings  # type: ignore[no-any-return]


SettingsDep = Annotated[Settings, Depends(get_request_settings)]
