# ml-app

A production-ready FastAPI microservice for churn prediction, built on top of **churn-lib**.

This package is designed as a teaching reference for **how to structure ML microservices correctly**:
clean layered architecture, optional dependency gating, structured JSON logging, dependency
injection, and proper error handling — all in one runnable example.

---

## Architecture

```
HTTP Request
     │
     ▼
┌─────────────────────────────────────────────────────────────┐
│  app/main.py  —  FastAPI app, lifespan, exception handler   │
└──────────────────────┬──────────────────────────────────────┘
                       │ includes
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  app/api/v1/router.py  —  aggregates all domain routers     │
│  app/api/v1/endpoints/                                      │
│    greetings/hello.py       — minimal worked example        │
│    churn/predict.py         — POST /predict, GET /schema    │
│    churn/train.py           — POST /train (sync + async)    │  ← [train] extra only
│    churn/drift.py           — POST /drift                   │
└──────────────────────┬──────────────────────────────────────┘
                       │ calls
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  app/services/  —  business logic; no HTTP concepts here    │
│    greeter.py                                               │
│    churn/inference.py  ──────────────────────────────────── │ ──► churn-lib[predict]
│    churn/training.py  (imported only with [train] extra)    │ ──► churn-lib[train]
│    churn/drift.py  ───────────────────────────────────────- │ ──► churn-lib[predict]
└──────────────────────┬──────────────────────────────────────┘
                       │ injected via
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  app/core/                                                  │
│    config.py — Settings (pydantic-settings, env var prefix) │
│    dependencies.py — get_pipeline(), get_settings()         │
│    errors.py       — global unhandled exception handler     │
└─────────────────────────────────────────────────────────────┘
```

### Key design principles

| Principle | Where to look |
|---|---|
| **Service layer** — no HTTP in business logic | `app/services/` |
| **Dependency injection** — testable, overridable | `app/core/dependencies.py` |
| **Optional extras** — inference stays lean | `pyproject.toml`, `api/v1/router.py` |
| **Structured logging** — JSON lines with `timestamp`, `level`, `request_id` | `app/main.py` lifespan |
| **Request ID middleware** — log correlation across replicas | `app/main.py` `RequestIdMiddleware` |
| **Global error handler** — no tracebacks to clients | `app/core/errors.py` |
| **Async model load** — event loop stays responsive | `app/main.py` lifespan |
| **Model hot-reload** — no restart after training | `app/services/churn/training.py` |
| **Schema-first docs** — auto-generated from Pydantic | `app/schemas/churn.py` |
| **Batch size cap** — OOM protection on /predict | `app/schemas/churn.py` `max_length=1000` |

---

## Project structure

```
ml-app/
├── pyproject.toml                    # deps, optional extras, ruff, pyright, pytest
├── Dockerfile                        # multi-stage uv build, non-root user
├── docker-compose.yml                # local dev; healthcheck targets /health
├── app/
│   ├── main.py                       # FastAPI app, lifespan, RequestIdMiddleware,
│   │                                 #   GET / (entry point)
│   │                                 #   GET /health (liveness probe)
│   │                                 #   GET /ready  (readiness probe)
│   ├── core/
│   │   ├── config.py                 # Settings (env vars with CHURN_ prefix)
│   │   ├── dependencies.py           # get_pipeline(), get_settings()
│   │   └── errors.py                 # global exception handler
│   ├── schemas/
│   │   └── churn.py                  # all Pydantic request/response models
│   ├── services/
│   │   ├── greeter.py                # greeting business logic
│   │   └── churn/
│   │       ├── inference.py          # wraps churn-lib predict_batch
│   │       ├── training.py           # wraps churn-lib train (heavy deps)
│   │       └── drift.py              # wraps churn-lib check_drift
│   └── api/
│       └── v1/
│           ├── router.py             # aggregates all domain routers
│           └── endpoints/
│               ├── greetings/
│               │   └── hello.py      # minimal example endpoints
│               └── churn/
│                   ├── predict.py    # POST /predict, GET /schema
│                   ├── train.py      # POST /train (sync + async), GET /jobs/{id}
│                   └── drift.py      # POST /drift
└── tests/
    ├── conftest.py                   # shared fixtures (mock_pipeline, client, client_no_model)
    ├── test_health.py                # GET /, GET /health, GET /ready
    ├── test_predict.py               # POST /predict, GET /schema
    ├── test_drift.py                 # POST /drift
    ├── test_train.py                 # POST /train, POST /train/async, GET /jobs/{id}
    ├── test_inference_service.py     # run_prediction() unit tests
    └── test_dependencies.py          # get_pipeline() unit tests
```

---

## Running locally

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager

### Inference only (lightweight)

```bash
cd ml-app
uv sync                              # inference deps only (no mlflow/matplotlib)
uv run uvicorn app.main:app --reload
```

### With training endpoints

```bash
uv sync --extra train                # also installs churn-lib[train]
uv run uvicorn app.main:app --reload
```

Visit **http://localhost:8000/docs** for the interactive API.

---

## Environment variables

All variables use the `CHURN_` prefix. Create a `.env` file in `ml-app/` for local development:

```env
# Load a pre-trained model at startup (optional)
CHURN_MODEL_PATH=./models/20240101_120000/pipeline.joblib

# MLflow tracking server (optional — defaults to ./mlruns if unset)
CHURN_MLFLOW_TRACKING_URI=http://localhost:5000

# Decision threshold applied when the request does not specify one (default: 0.5)
CHURN_DEFAULT_THRESHOLD=0.4

# Logging verbosity: DEBUG | INFO | WARNING | ERROR (default: INFO)
CHURN_LOG_LEVEL=INFO
```

---

## API reference

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `GET` | `/` | — | Entry point; returns server time and links to docs |
| `GET` | `/health` | — | **Liveness probe** — always 200 while process is alive |
| `GET` | `/ready` | — | **Readiness probe** — 503 until a model is loaded |
| `GET` | `/api/v1/greetings/hello` | — | Generic greeting (minimal example) |
| `GET` | `/api/v1/greetings/greet?name=` | — | Personalised greeting |
| `POST` | `/api/v1/greetings/greet-and-return?name=` | — | Greeting + echo request body |
| `GET` | `/api/v1/churn/schema` | model | Feature constraints for the loaded model |
| `POST` | `/api/v1/churn/predict` | model | Score 1–1000 customer records per request |
| `POST` | `/api/v1/churn/drift` | model | PSI drift check (reference vs serving) |
| `POST` | `/api/v1/churn/train` | `[train]` | Train a model (sync — blocks until done) |
| `POST` | `/api/v1/churn/train/async` | `[train]` | Train a model (async — returns a job ID) |
| `GET` | `/api/v1/churn/train/jobs/{id}` | `[train]` | Poll background training job status |

**model** = requires a loaded pipeline (503 if none is loaded).
**[train]** = requires `uv sync --extra train`.

> **Operational probes:** Use `/health` as the liveness probe and `/ready` as the readiness probe in
> Kubernetes, ECS, or docker-compose. Do not use `/` for either — it returns 200 regardless of model state.

---

## Quick start workflow

> **Before you start:** steps 1 and 4 require the `[train]` extra.
> Run `uv sync --extra train` once before following this workflow.

### 1 — Train a model

```bash
curl -X POST http://localhost:8000/api/v1/churn/train \
  -H "Content-Type: application/json" \
  -d '{}'
```

The response includes `optimal_threshold` — the threshold calibrated to maximise F1
on the hold-out set. Use it in `/predict` instead of the default 0.5.

```json
{
  "status": "ok",
  "model_path": "/abs/path/to/models/20240101_120000/pipeline.joblib",
  "optimal_threshold": 0.35,
  "metrics": {"accuracy": 0.89, "macro_f1": 0.87, ...},
  "summary": "Run saved to: ..."
}
```

Alternatively, train from the CLI (no server needed):

```bash
uv run python -m churn_lib.trainer --n-samples 5000 --output-dir models/
```

### 2 — Inspect the feature schema

```bash
curl http://localhost:8000/api/v1/churn/schema
```

Returns the feature names, valid ranges, and allowed categories for the loaded model.

### 3 — Score customers

```bash
curl -X POST http://localhost:8000/api/v1/churn/predict \
  -H "Content-Type: application/json" \
  -d '{
    "customers": [{
      "tenure_months": 3,
      "monthly_charges": 95.0,
      "num_products": 1,
      "support_calls_last_6m": 5,
      "age": 34,
      "contract_type": "month-to-month",
      "payment_method": "electronic_check",
      "has_internet": true,
      "has_streaming": false
    }],
    "threshold": 0.35
  }'
```

```json
{
  "predictions": [{"prediction": 1, "label": "1", "probabilities": {"0": 0.28, "1": 0.72}}],
  "model_version": "1.0.0",
  "threshold_used": 0.35
}
```

### 4 — Detect drift (weekly monitoring)

```bash
curl -X POST http://localhost:8000/api/v1/churn/drift \
  -H "Content-Type: application/json" \
  -d '{
    "reference": [{"tenure_months": 24, "monthly_charges": 45.0, ...}],
    "serving":   [{"tenure_months": 3,  "monthly_charges": 95.0, ...}]
  }'
```

```json
{
  "overall_status": "major",
  "drifted_features": ["tenure_months", "monthly_charges"],
  "reference_samples": 1,
  "serving_samples": 1,
  "features": [...]
}
```

When `overall_status` is `"major"`, retrain.

---

## Async training workflow

For production integrations that cannot block an HTTP connection:

```python
import time, httpx

# 1. Submit the job
resp = httpx.post("http://localhost:8000/api/v1/churn/train/async", json={})
poll_url = resp.json()["poll_url"]

# 2. Poll until done
while True:
    job = httpx.get(f"http://localhost:8000{poll_url}").json()
    if job["status"] == "done":
        print("Optimal threshold:", job["result"]["optimal_threshold"])
        break
    if job["status"] == "failed":
        print("Training failed:", job["error"])
        break
    time.sleep(5)
```

**Job lifecycle:** `pending` → `running` → `done` | `failed`

> **Production note:** The in-memory job store (`_jobs` dict in `train.py`) resets on
> restart. Replace it with Redis, PostgreSQL, or a task queue (Celery, ARQ) to persist
> job state across deployments — the endpoint contract stays the same.

---

## Loading a pre-trained model at startup

```bash
CHURN_MODEL_PATH=./models/20240101_120000/pipeline.joblib \
  uv run uvicorn app.main:app
```

Model loading is non-blocking (`asyncio.to_thread`) so the app accepts requests on
non-model endpoints (docs, greetings) immediately while the model loads in the background.
If the file is missing or corrupt, the app starts without a model and logs a warning —
it does not crash.

---

## Extending the API

To add a new ML domain (e.g. "customer segmentation"):

1. **Schema** — add `app/schemas/segmentation.py` with request/response Pydantic models.
2. **Service** — add `app/services/segmentation/` with pure-Python functions; no HTTP here.
3. **Endpoints** — add `app/api/v1/endpoints/segmentation/` with a router and thin handlers.
4. **Router** — import and include the new router in `app/api/v1/router.py`.

That's it. Each domain is isolated and independently testable.

---

## Development

```bash
# Install dev dependencies (test runner, linters)
uv sync --group dev

# Lint, format, and type-check
uv run task precommit

# Run tests — inference-only installation
uv run task test
# 45 tests run; 12 skipped (training endpoint tests require [train] extra — see below)

# Run all tests including training endpoints
uv sync --extra train --group dev
uv run task test
# 57 tests run, 0 skipped

# Run tests with coverage
uv run task test-cov

# Start MLflow tracking UI (requires Docker — run from churn-lib/ directory)
# cd ../churn-lib && docker compose up -d
```

### Why some tests are skipped

The training endpoints (`POST /train`, `POST /train/async`, `GET /train/jobs/{id}`) are only
registered when `churn-lib[train]` is installed. The test suite mirrors this: `test_train.py`
skips itself if the import fails, the same way the router skips registering the routes.
To run all 57 tests, install the `[train]` extra before running the suite.

---

## Docker

Build from the **repository root** (needed to include `churn-lib`):

```bash
# From repo root
docker build -f ml-app/Dockerfile -t ml-app:latest .

# Inference-only (default)
docker run -p 8000:8000 ml-app:latest

# With training enabled and a pre-trained model
docker run \
  -p 8000:8000 \
  -e CHURN_MODEL_PATH=/models/pipeline.joblib \
  -v /path/to/local/models:/models \
  ml-app:latest
```

---

The architecture is intentionally simple so each layer can be studied and extended independently.
Start with `app/api/v1/endpoints/greetings/hello.py` as the minimal worked example, then follow
the same pattern to add new domains under `app/services/` and `app/api/v1/endpoints/`.

