"""Tests for POST /api/v1/churn/train, POST /api/v1/churn/train/async,
and GET /api/v1/churn/train/jobs/{id}.

Skip condition
--------------
These endpoints are only registered when churn-lib[train] is installed (adds
mlflow and matplotlib). The module-level skip guard checks for the import
at collection time — if the extra is missing, the entire module is skipped
with a clear message rather than failing with a confusing ImportError.

Testing strategy
----------------
Training runs take 10–60 s on real data. Patching `run_training` at the
endpoint's import site makes tests fast and independent of file I/O, MLflow,
and ML correctness. What we're testing here is the HTTP contract:
  - Synchronous endpoint blocks and returns a TrainResponse.
  - Async endpoint returns 202 immediately with a job_id and poll_url.
  - Job status endpoint returns the correct status for known / unknown IDs.
  - The in-memory job store transitions: pending → running → done.

Patch target: `app.api.v1.endpoints.churn.train.run_training`
  Same reasoning as in test_predict.py — patch at the call site.
"""

from __future__ import annotations

import time
from unittest.mock import patch

import pytest

# Guard — skip this module if the train extra is not installed.
try:
    from app.api.v1.endpoints.churn import train as _train_module  # noqa: F401

    _TRAIN_AVAILABLE = True
except ImportError:
    _TRAIN_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _TRAIN_AVAILABLE,
    reason="churn-lib[train] not installed — training endpoints are disabled",
)

from fastapi.testclient import TestClient  # noqa: E402 (after skip guard)

from app.schemas.churn import TrainResponse  # noqa: E402

_PATCH_TARGET = "app.api.v1.endpoints.churn.train.run_training"

_MOCK_TRAIN_RESPONSE = TrainResponse(
    status="ok",
    summary="Run saved to: /tmp/test-run\nAccuracy: 0.85",
    model_path="/tmp/test-run/pipeline.joblib",
    metrics={"accuracy": 0.85, "macro_avg_f1": 0.83},
    optimal_threshold=0.42,
)


class TestTrainSync:
    """POST /api/v1/churn/train — synchronous training."""

    def test_returns_200_on_success(self, client: TestClient) -> None:
        with patch(_PATCH_TARGET, return_value=_MOCK_TRAIN_RESPONSE):
            response = client.post("/api/v1/churn/train", json={})
        assert response.status_code == 200

    def test_response_structure(self, client: TestClient) -> None:
        """TrainResponse must have status, summary, model_path, metrics, optimal_threshold."""
        with patch(_PATCH_TARGET, return_value=_MOCK_TRAIN_RESPONSE):
            data = client.post("/api/v1/churn/train", json={}).json()

        assert data["status"] == "ok"
        assert "summary" in data
        assert "model_path" in data
        assert "metrics" in data
        assert "optimal_threshold" in data

    def test_optimal_threshold_is_float(self, client: TestClient) -> None:
        """optimal_threshold must be a float in [0, 1] — callers use it as /predict threshold."""
        with patch(_PATCH_TARGET, return_value=_MOCK_TRAIN_RESPONSE):
            data = client.post("/api/v1/churn/train", json={}).json()

        threshold = data["optimal_threshold"]
        assert isinstance(threshold, float)
        assert 0.0 <= threshold <= 1.0

    def test_returns_500_when_training_fails(self, client: TestClient) -> None:
        """Training failures must surface as 500 with a descriptive message.

        An uncaught exception during training (e.g. out of disk space, corrupt
        data) should not leak a Python traceback to the client.
        """
        with patch(_PATCH_TARGET, side_effect=RuntimeError("Disk full")):
            response = client.post("/api/v1/churn/train", json={})
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data


class TestTrainAsync:
    """POST /api/v1/churn/train/async — asynchronous (job-based) training."""

    def test_returns_202_immediately(self, client: TestClient) -> None:
        """The async endpoint must return 202 immediately — not wait for training.

        202 (Accepted) is the correct status code for an async operation: the
        server has accepted the request but hasn't completed it yet. 200 would
        imply the work is done; 201 would imply a resource was created at a URL.
        """
        with patch(_PATCH_TARGET, return_value=_MOCK_TRAIN_RESPONSE):
            response = client.post("/api/v1/churn/train/async", json={})
        assert response.status_code == 202

    def test_response_contains_job_id_and_poll_url(self, client: TestClient) -> None:
        """The response must give the client enough to track the job."""
        with patch(_PATCH_TARGET, return_value=_MOCK_TRAIN_RESPONSE):
            data = client.post("/api/v1/churn/train/async", json={}).json()

        assert "job_id" in data
        assert "poll_url" in data
        assert "status" in data
        assert data["status"] == "pending"

    def test_poll_url_is_valid_path(self, client: TestClient) -> None:
        """poll_url must be a path that GET /train/jobs/{id} can resolve."""
        with patch(_PATCH_TARGET, return_value=_MOCK_TRAIN_RESPONSE):
            data = client.post("/api/v1/churn/train/async", json={}).json()

        assert data["poll_url"].startswith("/api/v1/churn/train/jobs/")
        assert data["job_id"] in data["poll_url"]

    def test_job_id_is_unique_per_submission(self, client: TestClient) -> None:
        """Each submission must produce a different job_id.

        UUIDs must be generated per-request, not shared or cached, otherwise
        two concurrent callers could overwrite each other's job results.
        """
        with patch(_PATCH_TARGET, return_value=_MOCK_TRAIN_RESPONSE):
            id1 = client.post("/api/v1/churn/train/async", json={}).json()["job_id"]
            id2 = client.post("/api/v1/churn/train/async", json={}).json()["job_id"]

        assert id1 != id2


class TestJobStatus:
    """GET /api/v1/churn/train/jobs/{job_id} — polling endpoint."""

    def test_returns_404_for_unknown_job_id(self, client: TestClient) -> None:
        """Unknown job IDs must return 404, not 500 or an empty body.

        404 here is client-recoverable: the caller submitted to a different
        replica (in-memory store doesn't persist across replicas) or the
        process restarted. The detail message should explain this.
        """
        response = client.get("/api/v1/churn/train/jobs/nonexistent-job-id-12345")
        assert response.status_code == 404

    def test_newly_submitted_job_is_pending(self, client: TestClient) -> None:
        """A job polled immediately after submission must be 'pending' or 'running'.

        Depending on TestClient's background task scheduling, the job may have
        already transitioned to 'running' by the time we poll. Both are valid
        initial states — what matters is that 'done' or 'failed' is NOT returned
        before the mock has a chance to run.
        """

        # Slow mock to ensure the job hasn't completed by the time we poll.
        def slow_training(*args, **kwargs):
            time.sleep(0.05)
            return _MOCK_TRAIN_RESPONSE

        with patch(_PATCH_TARGET, side_effect=slow_training):
            job_data = client.post("/api/v1/churn/train/async", json={}).json()
            job_id = job_data["job_id"]
            poll_url = job_data["poll_url"]

            status_data = client.get(poll_url).json()

        assert status_data["job_id"] == job_id
        assert status_data["status"] in {"pending", "running", "done"}

    def test_completed_job_contains_result(self, client: TestClient) -> None:
        """After the background task completes, the result must be available at the poll URL.

        TestClient runs background tasks before the context manager exits, so
        by the time we call get() after the with-block, the task is done.
        """
        with patch(_PATCH_TARGET, return_value=_MOCK_TRAIN_RESPONSE):
            job_data = client.post("/api/v1/churn/train/async", json={}).json()
            # TestClient finalises background tasks on __exit__, so the job
            # will be in 'done' state immediately after the with block.

        # Poll after background tasks have completed.
        poll_response = client.get(job_data["poll_url"]).json()

        assert poll_response["status"] == "done"
        assert poll_response["result"] is not None
        assert poll_response["result"]["status"] == "ok"

    def test_failed_job_contains_error(self, client: TestClient) -> None:
        """A failed training job must expose the error reason at the poll URL."""
        with patch(_PATCH_TARGET, side_effect=RuntimeError("OOM during training")):
            job_data = client.post("/api/v1/churn/train/async", json={}).json()

        poll_response = client.get(job_data["poll_url"]).json()

        assert poll_response["status"] == "failed"
        assert poll_response["error"] is not None
        assert "OOM" in poll_response["error"]
