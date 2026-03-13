"""Tests for operational / meta endpoints: GET /, GET /health, GET /ready.

Why these tests matter
----------------------
Health and readiness probes are the interface between the application and its
orchestration layer (Kubernetes, ECS, docker-compose). If they misbehave:
  - A buggy liveness probe that returns 503 will cause endless container
    restarts (crash-loop), making the service unavailable.
  - A readiness probe that always returns 200 routes traffic to a replica
    before it can serve real requests, causing client-visible errors.

These tests lock in the contract that orchestration systems depend on.
"""

from __future__ import annotations

from fastapi.testclient import TestClient


class TestRoot:
    """GET / — entry point, not a probe."""

    def test_returns_200(self, client: TestClient) -> None:
        response = client.get("/")
        assert response.status_code == 200

    def test_response_body_has_expected_keys(self, client: TestClient) -> None:
        data = client.get("/").json()
        assert "message" in data
        assert "docs" in data
        assert "server_time" in data

    def test_always_200_even_without_model(self, client_no_model: TestClient) -> None:
        """Root must never return 503 — it is not a readiness probe."""
        response = client_no_model.get("/")
        assert response.status_code == 200


class TestHealth:
    """GET /health — liveness probe.

    The liveness probe must always return 200 while the process is alive,
    regardless of whether a model is loaded. An orchestrator that gets 503
    here will restart the container — not what we want during model loading.
    """

    def test_returns_200_with_model(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 200

    def test_returns_200_without_model(self, client_no_model: TestClient) -> None:
        """Liveness must be independent of model state."""
        response = client_no_model.get("/health")
        assert response.status_code == 200

    def test_response_body(self, client: TestClient) -> None:
        data = client.get("/health").json()
        assert data == {"status": "ok"}


class TestReady:
    """GET /ready — readiness probe.

    The readiness probe controls whether an orchestrator routes traffic to
    this replica. It returns 200 only when the model is loaded and the
    service can usefully serve /predict requests.
    """

    def test_returns_200_when_model_loaded(self, client: TestClient) -> None:
        """A replica with a model loaded is ready to serve traffic."""
        response = client.get("/ready")
        assert response.status_code == 200

    def test_response_body_includes_model_version(self, client: TestClient) -> None:
        data = client.get("/ready").json()
        assert data["status"] == "ready"
        assert "model_version" in data
        assert isinstance(data["model_version"], str)

    def test_returns_503_without_model(self, client_no_model: TestClient) -> None:
        """A replica with no model should not receive traffic — readiness must fail.

        This is the critical difference between /health and /ready: /health
        is about process liveness; /ready is about serving capability.
        A 503 here prevents the orchestrator from routing requests to this
        replica until the model is available.
        """
        response = client_no_model.get("/ready")
        assert response.status_code == 503

    def test_503_body_contains_actionable_detail(self, client_no_model: TestClient) -> None:
        """Error messages in operational probes should tell operators what to do."""
        data = client_no_model.get("/ready").json()
        assert "detail" in data
        # The detail should mention *how* to fix the situation.
        assert "model" in data["detail"].lower()
