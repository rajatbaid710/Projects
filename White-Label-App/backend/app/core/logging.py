"""Structured logging with a per-request correlation ID.

Correlation IDs are cheap to add now and painful to retrofit: once every log
line carries one, tracing a single request across API, worker, and error
tracker is trivial. The ID is stored in a ContextVar so any code — service
layer, task, exception handler — can read it without it being threaded through
every function signature.
"""

import json
import logging
import sys
from contextvars import ContextVar
from typing import Any

_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)


def set_request_id(request_id: str) -> None:
    _request_id.set(request_id)


def get_request_id() -> str | None:
    return _request_id.get()


class _RequestIdFilter(logging.Filter):
    """Makes `%(request_id)s` available to every formatter."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = get_request_id() or "-"
        return True


class _JsonFormatter(logging.Formatter):
    """One JSON object per line — what log aggregators want in staging/prod."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", "-"),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        # Anything passed as `extra={...}` rides along without special-casing.
        for key, value in record.__dict__.items():
            if key.startswith("ctx_"):
                payload[key.removeprefix("ctx_")] = value
        return json.dumps(payload, default=str)


def setup_logging(*, debug: bool = False, json_output: bool = True) -> None:
    """Install a single stdout handler. Idempotent — safe under uvicorn reload."""
    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(_RequestIdFilter())
    if json_output:
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(levelname)-8s [%(request_id)s] %(name)s: %(message)s")
        )

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.DEBUG if debug else logging.INFO)

    # uvicorn installs its own handlers; defer to ours so output stays uniform.
    for name in ("uvicorn", "uvicorn.access", "uvicorn.error"):
        uvicorn_logger = logging.getLogger(name)
        uvicorn_logger.handlers.clear()
        uvicorn_logger.propagate = True
