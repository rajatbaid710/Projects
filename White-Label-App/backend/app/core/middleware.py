"""Cross-cutting HTTP middleware."""

import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from app.core.logging import set_request_id

logger = logging.getLogger("app.access")

REQUEST_ID_HEADER = "X-Request-ID"


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Assigns a correlation ID to every request and logs one access line.

    An inbound `X-Request-ID` is honoured so a trace started by a client or an
    upstream proxy carries through; otherwise one is generated. The ID is echoed
    back in the response header, which is what makes a user-reported error
    traceable in the logs.
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = request.headers.get(REQUEST_ID_HEADER) or str(uuid.uuid4())
        set_request_id(request_id)

        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            # The exception handler builds the response; we just record timing
            # so failed requests appear in the access log like any other.
            duration_ms = (time.perf_counter() - started) * 1000
            logger.exception(
                "%s %s -> unhandled",
                request.method,
                request.url.path,
                extra={"ctx_duration_ms": round(duration_ms, 2)},
            )
            raise

        duration_ms = (time.perf_counter() - started) * 1000
        response.headers[REQUEST_ID_HEADER] = request_id
        logger.info(
            "%s %s -> %s",
            request.method,
            request.url.path,
            response.status_code,
            extra={
                "ctx_method": request.method,
                "ctx_path": request.url.path,
                "ctx_status": response.status_code,
                "ctx_duration_ms": round(duration_ms, 2),
            },
        )
        return response
