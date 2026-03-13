"""Global exception handlers for the FastAPI application.

FastAPI allows registering exception handlers at the app level — they act as
a last-resort catch for any exception that was not handled by an endpoint.

Two handlers are defined:
  - unhandled_exception_handler: catches any uncaught Exception, logs the full
    traceback, and returns a clean 500 JSON response. This prevents Python
    tracebacks from leaking to the client while ensuring the error is recorded.

Registration (in app/main.py):
    app.add_exception_handler(Exception, unhandled_exception_handler)

Why a global handler instead of try/except in every endpoint?
  Endpoint-level try/except should only handle *expected*, *recoverable* errors
  (e.g. ValidationError → 422). Unexpected errors (disk full, OOM, bug) should
  propagate naturally and be caught here once, keeping endpoint code clean.
"""

import logging

from fastapi import Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all handler for any exception not handled by an endpoint.

    Logs the full traceback at ERROR level so it appears in structured logs,
    then returns a 500 JSON response with a safe, user-friendly message — no
    internal details or Python tracebacks are sent to the client.

    Args:
        request: The incoming FastAPI request (used for logging context).
        exc:     The unhandled exception.

    Returns:
        A 500 JSON response with a generic error message.
    """
    logger.exception(
        "Unhandled exception",
        extra={"method": request.method, "path": str(request.url.path)},
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected server error occurred. Check the server logs for details."},
    )
