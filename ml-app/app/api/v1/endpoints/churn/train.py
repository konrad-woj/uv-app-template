"""Churn model training endpoints.

This module is only imported when churn-lib[train] is installed (adds mlflow
and matplotlib). The router is registered in app/api/v1/router.py via a
guarded try/except — if the extra is missing, these routes simply don't exist
and the inference process never imports the heavy deps.

Routes:
    POST /api/v1/churn/train              — synchronous (blocks until done)
    POST /api/v1/churn/train/async        — asynchronous (returns a job ID)
    GET  /api/v1/churn/train/jobs/{id}    — poll job status

Why two training modes?
  Synchronous is simpler: great for development, curl/Postman, or demos where
  waiting 10–30 s is fine. It shows the full result in one response.

  Asynchronous is what production services use: a long-running operation should
  never hold an HTTP connection open. Clients submit a job, poll until "done",
  then read the result. This pattern is used by every major ML platform API
  (OpenAI fine-tuning, Vertex AI, SageMaker Training Jobs).

In-memory job store:
  The _jobs dict below is process-local and resets on restart. It is
  intentionally simple so students can focus on the async pattern itself.
  In production, replace it with Redis, PostgreSQL, or a task queue such as
  Celery or ARQ — the endpoint contract (submit → poll → result) stays the same.
"""

import asyncio
import logging
import uuid
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request, status

from app.schemas.churn import JobStatusResponse, TrainJobResponse, TrainRequest, TrainResponse
from app.services.churn.training import run_training

logger = logging.getLogger(__name__)
router = APIRouter()

# ---------------------------------------------------------------------------
# In-memory job store
# process-local; resets on restart — use Redis/DB in production.
# ---------------------------------------------------------------------------
_jobs: dict[str, dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Synchronous training
# ---------------------------------------------------------------------------


@router.post(
    "/train",
    response_model=TrainResponse,
    summary="Train a new churn model (synchronous — blocks until done)",
    status_code=status.HTTP_200_OK,
)
async def train_sync(body: TrainRequest, request: Request) -> TrainResponse:
    """Train a ChurnPipeline and hot-reload it into memory. Blocks until complete.

    **Use this for:**
    - Local development and experimentation.
    - CLI / curl / Postman calls where you want the result immediately.
    - Demos and notebooks where waiting 10–30 s per run is acceptable.

    **For production**, prefer `POST /api/v1/churn/train/async` to avoid holding
    the HTTP connection open while training runs.

    **What happens:**
    1. Generates `n_samples` synthetic customer records via churn-lib.
    2. Trains an XGBoost pipeline with stratified train/test split.
    3. Calibrates the decision threshold to maximise F1 on the hold-out set.
    4. Saves all artifacts (pipeline.joblib, metrics.json, plots) to
       `output_dir/<timestamp>/`.
    5. Hot-reloads the new model into memory — `/predict` uses it immediately,
       no restart required.

    **Example request:**
    ```json
    {}
    ```
    An empty body uses all defaults and returns a trained model in ~20 s.

    **Using the result:**
    Pass `optimal_threshold` from this response as the `threshold` field in
    `/predict` to use the calibrated operating point instead of the default 0.5.
    """
    try:
        return await asyncio.to_thread(
            run_training,
            request.app,
            body.n_samples,
            body.output_dir,
            body.test_size,
            body.random_seed,
        )
    except Exception as exc:
        logger.exception("Synchronous training run failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Training failed: {exc}",
        ) from exc


# ---------------------------------------------------------------------------
# Asynchronous training
# ---------------------------------------------------------------------------


@router.post(
    "/train/async",
    response_model=TrainJobResponse,
    summary="Submit a training job (asynchronous — returns immediately)",
    status_code=status.HTTP_202_ACCEPTED,
)
async def train_async(
    body: TrainRequest,
    request: Request,
    background_tasks: BackgroundTasks,
) -> TrainJobResponse:
    """Submit a training job and return immediately with a job ID.

    **Use this for:**
    - Production integrations where you cannot block an HTTP connection.
    - Any client that needs to remain responsive while training runs.

    **Workflow:**
    1. `POST /api/v1/churn/train/async` → receive `{"job_id": "...", "poll_url": "..."}`.
    2. `GET <poll_url>` every few seconds (e.g. every 5 s).
    3. When `status` is `"done"`, read `result` for metrics and model path.
       The new model is already live — start using `/predict` immediately.
    4. If `status` is `"failed"`, read `error` for the failure reason.

    **Job lifecycle:**
    ```
    pending → running → done
                     ↘ failed
    ```
    """
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"status": "pending", "result": None, "error": None}
    background_tasks.add_task(_run_training_job, job_id, request.app, body)

    poll_url = f"/api/v1/churn/train/jobs/{job_id}"
    return TrainJobResponse(
        job_id=job_id,
        status="pending",
        message=f"Training job submitted. Poll {poll_url} for status and results.",
        poll_url=poll_url,
    )


@router.get(
    "/train/jobs/{job_id}",
    response_model=JobStatusResponse,
    summary="Poll the status of a background training job",
)
async def get_job_status(job_id: str) -> JobStatusResponse:
    """Return the current status of an async training job.

    Poll this endpoint after `POST /api/v1/churn/train/async`.

    **Status values:**
    - `pending`  — job is queued but has not started yet.
    - `running`  — training is in progress.
    - `done`     — training succeeded; see `result` for metrics and model path.
    - `failed`   — training failed; see `error` for the reason.

    **Example — polling loop (Python):**
    ```python
    import time, httpx

    resp = httpx.post("http://localhost:8000/api/v1/churn/train/async", json={})
    poll_url = resp.json()["poll_url"]

    while True:
        status = httpx.get(f"http://localhost:8000{poll_url}").json()
        if status["status"] == "done":
            print("Optimal threshold:", status["result"]["optimal_threshold"])
            break
        if status["status"] == "failed":
            print("Training failed:", status["error"])
            break
        time.sleep(5)
    ```
    """
    if job_id not in _jobs:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' not found.",
        )
    job = _jobs[job_id]
    return JobStatusResponse(
        job_id=job_id,
        status=job["status"],
        result=job["result"],
        error=job["error"],
    )


# ---------------------------------------------------------------------------
# Background task helper (private)
# ---------------------------------------------------------------------------


async def _run_training_job(job_id: str, app: Any, body: TrainRequest) -> None:
    """Background task: run training in a thread and update the job store."""
    _jobs[job_id]["status"] = "running"
    logger.info("Background training job started", extra={"job_id": job_id})
    try:
        result = await asyncio.to_thread(
            run_training,
            app,
            body.n_samples,
            body.output_dir,
            body.test_size,
            body.random_seed,
        )
        _jobs[job_id]["status"] = "done"
        _jobs[job_id]["result"] = result
        logger.info("Background training job completed", extra={"job_id": job_id})
    except Exception as exc:
        _jobs[job_id]["status"] = "failed"
        _jobs[job_id]["error"] = str(exc)
        logger.exception("Background training job failed", extra={"job_id": job_id})
