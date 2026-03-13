FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

# ---------------------------------------------------------------------------
# Non-root user — security best practice for containerised services.
#
# Running as root is unnecessary and grants any exploited process full
# container-level privileges. Creating a dedicated system user limits the
# blast radius: the process can only read files it owns, and it cannot
# install packages, modify system config, or escape via privilege escalation.
# ---------------------------------------------------------------------------
RUN addgroup --system --gid 1001 appgroup \
    && adduser --system --uid 1001 --gid 1001 --no-create-home appuser

COPY churn-lib/ /churn-lib/

COPY ml-app/pyproject.toml ml-app/uv.lock ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --no-dev

COPY ml-app/ .

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Switch to the non-root user before any runtime instruction.
# All files installed above are owned by root (readable by all) which is fine;
# the running process only needs read access to the virtualenv and app code.
RUN chown -R appuser:appgroup /app
USER appuser

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

# ---------------------------------------------------------------------------
# Workers vs hot-reload — choose the right mode for your deployment:
#
# Development / single replica:
#   CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
#   One event loop, one process. POST /train hot-reloads the model in-place.
#
# Production / high-throughput inference (recommended):
#   CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
#   Multiple worker processes maximise CPU utilisation across cores.
#   IMPORTANT: each worker has its own app.state — hot-reload after POST /train
#   only updates the worker that handled the request. The others still serve
#   the old model. Solutions:
#     1. Restart the service after training (simple, acceptable downtime).
#     2. Use a shared model store (S3, MLflow registry) + a /reload endpoint
#        that every worker calls on a schedule or via a signal.
#     3. Deploy training and inference as separate services sharing a
#        model store — the inference service never trains.
#
# Override at runtime:  docker run -e UVICORN_WORKERS=4 ...
# ---------------------------------------------------------------------------
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
