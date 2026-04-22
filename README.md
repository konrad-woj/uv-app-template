# uv-app-template

A collection of best practices and patterns for Python ML projects — designed to learn from or use as a starting point.

The repo is a monorepo with two packages:

| Package | Description |
|---------|-------------|
| [`churn-lib/`](churn-lib/README.md) | ML library — training, inference, validation, and drift detection for churn prediction |
| [`ml-app/`](ml-app/README.md) | FastAPI microservice — production-ready HTTP API built on top of churn-lib |

## Highlights

- **Layered architecture** — endpoints → services → library; no HTTP leaking into business logic
- **Optional extras** — inference stays lightweight; heavy training deps (`mlflow`, `matplotlib`) are opt-in
- **Dependency injection** — `get_pipeline()` FastAPI dependency; overridable in tests
- **Structured JSON logging** with request ID correlation
- **Async + sync training** endpoints with in-memory job tracking
- **MLflow experiment tracking** — every training run logged with params, metrics, and artifacts; local file store or Docker-based server
- **Drift detection** via Population Stability Index (PSI)
- **Threshold calibration** — find the F1/recall/precision operating point that fits your business constraint
- **Docker** multi-stage build from repo root; non-root user
