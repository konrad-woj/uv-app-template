"""Churn App — FastAPI application entry point.

This file wires together all the components:
  - lifespan: startup/shutdown logic (logging, model load).
  - v1 router: mounts all endpoint groups under /api/v1.
  - Global exception handler: structured 500 responses for unexpected errors.
  - Root endpoint: welcome / entry point.
  - /health: liveness probe — always 200 while the process is running.
  - /ready: readiness probe — 503 when no model is loaded.

Application lifecycle (lifespan):
  startup  → configure JSON logging, then load pipeline.joblib if CHURN_MODEL_PATH is set.
  shutdown → release app.state (no teardown needed for this app).

Model loading is performed in a thread (asyncio.to_thread) because
ChurnPipeline.load() calls joblib which does blocking I/O. Running it directly
in the async lifespan would block the event loop and delay the app becoming
ready to serve other endpoints (e.g. /docs, /api/v1/greetings/hello).

The model lives on app.state.pipeline so endpoint handlers can access it via
the get_pipeline() dependency (app/core/dependencies.py) without any module-
level globals or import-time side effects. FastAPI guarantees that app.state is
shared across all requests in a single process.

Horizontal scaling note — hot-reload and multiple workers:
  app.state is process-local. When you run multiple uvicorn workers (--workers N)
  or multiple replicas behind a load balancer, each process has its own
  app.state.pipeline. A hot-reload triggered by one replica does NOT propagate
  to others. Two production-grade solutions:
    1. Use a shared model registry (MLflow, S3): on each POST /train (or a
       dedicated reload endpoint), upload the new model, then broadcast a
       reload signal via Redis pub/sub or an internal POST /reload that each
       replica calls.
    2. Separate training and inference services: the trainer service writes to
       a shared store; inference replicas load from it at startup and on a
       configurable refresh interval.
  Both patterns keep the HTTP contract (submit → train → predict) identical;
  only the model storage and propagation layer changes.

Environment variables (see app/core/config.py for the full list):
    CHURN_MODEL_PATH          Path to a trained pipeline.joblib to load at startup.
    CHURN_MLFLOW_TRACKING_URI MLflow server URI for experiment tracking.
    CHURN_DEFAULT_THRESHOLD   Default prediction threshold (default: 0.5).
    CHURN_LOG_LEVEL           Logging verbosity — DEBUG, INFO, WARNING, ERROR (default: INFO).

Quick start:
    uv run uvicorn app.main:app --reload
    # → visit http://localhost:8000/docs
"""

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime

from fastapi import FastAPI, HTTPException, Request, status
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from app.api.v1.router import router as v1_router
from app.core.config import settings
from app.core.errors import unhandled_exception_handler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request ID middleware
# ---------------------------------------------------------------------------


class RequestIdMiddleware(BaseHTTPMiddleware):
    """Attach a unique X-Request-ID header to every request and response.

    This is the foundation of distributed tracing: when every log line and
    every response carries the same ID, you can reconstruct the full lifecycle
    of any request by filtering logs on that ID — across replicas, retries,
    and async background tasks.

    Behaviour:
      - If the client already sent an X-Request-ID (e.g. forwarded from an
        API gateway), that value is reused — preserving the upstream trace ID.
      - Otherwise, a fresh UUID4 is generated.
      - The ID is stored on request.state.request_id so endpoint handlers and
        service functions can include it in structured log extras.
      - The same ID is echoed back in the response header so clients can
        correlate their logs with server logs.

    Usage in an endpoint:

        @router.post("/predict")
        async def predict(request: Request, ...):
            logger.info("Scoring batch", extra={"request_id": request.state.request_id})
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = request_id
        response: Response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Configure logging and load the model at startup; clean up at shutdown.

    FastAPI calls this generator once: everything before `yield` runs at startup,
    everything after runs at shutdown. The app does not accept requests until
    the startup block completes, so heavy operations here delay the first request
    — keep them bounded (model load is typically < 5 s for XGBoost).
    """
    # ── Startup ──────────────────────────────────────────────────────────────

    # Configure structured JSON logging using the same formatter as churn-lib
    # so all log output — from this app and from the library — is consistent.
    # Do this first so all subsequent startup log lines are already structured.
    from churn_lib._logging import configure_cli_logging

    configure_cli_logging(level=settings.log_level)

    logger.info("Churn App starting", extra={"log_level": settings.log_level, "version": "0.1.0"})

    app.state.pipeline = None

    if settings.model_path:
        if not settings.model_path.exists():
            # Log a warning but do not abort startup — the app can still serve
            # non-model endpoints (e.g. /docs, /api/v1/greetings/hello) and
            # accept training requests when churn-lib[train] is installed.
            logger.warning(
                "CHURN_MODEL_PATH is set but the file does not exist — starting without a model.",
                extra={"model_path": str(settings.model_path)},
            )
        else:
            # ChurnPipeline.load() uses joblib (blocking I/O).
            # asyncio.to_thread() runs it in a thread-pool so the event loop
            # stays responsive during the load (typically 1–5 s for XGBoost).
            from churn_lib import ChurnPipeline

            t0 = time.perf_counter()
            try:
                app.state.pipeline = await asyncio.to_thread(ChurnPipeline.load, settings.model_path)
                elapsed = round((time.perf_counter() - t0) * 1000)
                logger.info(
                    "Model loaded at startup",
                    extra={
                        "model_path": str(settings.model_path),
                        "elapsed_ms": elapsed,
                        "model_version": app.state.pipeline.config.model_card.version,
                    },
                )
            except Exception:
                # Corrupt or incompatible file — log and proceed without a model.
                logger.exception(
                    "Failed to load model at startup — starting without a model.",
                    extra={"model_path": str(settings.model_path)},
                )
    else:
        logger.info(
            "No CHURN_MODEL_PATH configured — starting without a model. "
            "POST /api/v1/churn/train to train one, or set CHURN_MODEL_PATH and restart."
        )
    # ─────────────────────────────────────────────────────────────────────────

    yield  # Application runs here.

    # ── Shutdown ─────────────────────────────────────────────────────────────
    logger.info("Churn App shutting down")
    app.state.pipeline = None
    # ─────────────────────────────────────────────────────────────────────────


app = FastAPI(
    title="Churn App API",
    description=(
        "Production-ready ML microservice demonstrating best-practice FastAPI structure.\n\n"
        "**Project layout:**\n"
        "- `app/core/` — settings, shared dependencies, and error handlers.\n"
        "- `app/schemas/` — Pydantic request/response models.\n"
        "- `app/services/` — business logic; no HTTP concepts here.\n"
        "- `app/api/v1/endpoints/` — thin endpoint handlers per domain.\n\n"
        "**Quick start:**\n"
        "1. `POST /api/v1/churn/train` — train a model (requires `uv sync --extra train`).\n"
        "2. `GET  /api/v1/churn/schema` — inspect feature constraints.\n"
        "3. `POST /api/v1/churn/predict` — score customer records.\n"
        "4. `POST /api/v1/churn/drift` — monitor for distribution shift.\n\n"
        "Or load a pre-trained model at startup: `CHURN_MODEL_PATH=./models/.../pipeline.joblib`.\n\n"
        "**Operational endpoints:**\n"
        "- `GET /health` — liveness probe (always 200 while the process is alive).\n"
        "- `GET /ready`  — readiness probe (503 until a model is loaded)."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# Attach a unique X-Request-ID to every request/response for log correlation.
app.add_middleware(RequestIdMiddleware)

app.include_router(v1_router)

# Global fallback: log unexpected exceptions and return a clean 500 response.
# Endpoint handlers should only catch *expected* errors (e.g. ValidationError → 422).
# Everything else propagates here so endpoint code stays clean.
app.add_exception_handler(Exception, unhandled_exception_handler)


@app.get("/", tags=["meta"])
async def root():
    """API entry point — links to docs and reports server time.

    Not a health probe: this endpoint does not reflect model readiness.
    Use GET /health (liveness) or GET /ready (readiness) for orchestration checks.
    """
    return {
        "message": "Welcome to Churn App API",
        "docs": "/docs",
        "openapi": "/openapi.json",
        "server_time": datetime.now(UTC).isoformat(),
    }


@app.get("/health", tags=["meta"])
async def health():
    """Liveness probe — returns 200 as long as the process is running.

    Kubernetes and most container orchestrators call this endpoint to decide
    whether to *restart* a container. It should only fail if the process itself
    is broken (e.g. deadlocked event loop). It must NOT check external
    dependencies like the database or the model — that is the job of /ready.

    Configure your liveness probe to tolerate a slow start (initialDelaySeconds)
    so the container is not restarted before the model finishes loading.

    Returns:
        {"status": "ok"} with HTTP 200.
    """
    return {"status": "ok"}


@app.get("/ready", tags=["meta"])
async def ready(request: Request):
    """Readiness probe — returns 503 until a model is loaded.

    Kubernetes calls this endpoint to decide whether to *route traffic* to a
    replica. A replica that returns 503 here is temporarily removed from the
    load-balancer pool without being restarted.

    This is the correct place to check soft dependencies — the ML model must
    be loaded before the service can usefully serve /predict or /drift.
    Returning 503 here until that condition is met prevents clients from
    receiving misleading 503 errors from /predict.

    Typical lifecycle in a rolling deployment:
      1. New replica starts → model loads (takes 1–5 s).
      2. /ready returns 503 during this window → no traffic is routed here.
      3. Model finishes loading → /ready returns 200 → traffic starts flowing.
      4. Old replica is terminated after the new one is ready.

    Returns:
        {"status": "ready", "model_version": "..."} with HTTP 200 when a
        model is loaded; HTTP 503 otherwise.

    Raises:
        HTTPException(503): When no model is currently loaded.
    """
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Service not ready: no model is loaded. Set CHURN_MODEL_PATH and restart, or POST /api/v1/churn/train."
            ),
        )
    return {
        "status": "ready",
        "model_version": pipeline.config.model_card.version,
    }
