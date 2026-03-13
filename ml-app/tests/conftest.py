"""Shared test fixtures for the ml-app test suite.

Fixture design principles
-------------------------
1. Mock at the right boundary.
   Endpoint tests use `TestClient` + FastAPI dependency overrides — they test
   the full HTTP layer (routing, validation, serialisation, error mapping)
   without running real ML inference. Service unit tests mock the churn-lib
   public API so they test the service logic in isolation.

2. Use dependency overrides, not `app.state` patching.
   `app.dependency_overrides[get_pipeline] = lambda: mock_pipeline` is the
   FastAPI-idiomatic way to inject test doubles. It does not touch globals or
   module state, is automatically scoped to the duration of the override, and
   is easy to reset.

3. Session-scoped config, function-scoped everything else.
   PipelineConfig.from_yaml() is cheap (reads a YAML file), but creating a
   real trained pipeline would be slow. The mock_pipeline fixture gives
   endpoints a MagicMock with the real config attached so /schema and /drift
   work correctly, while /predict is tested via service-layer mocking.

4. Always clear overrides after each test.
   The fixtures use try/finally (via pytest fixtures with yield) to ensure
   `app.dependency_overrides.clear()` runs even if the test fails.

Client fixtures summary
-----------------------
  client            — TestClient with get_pipeline overridden to return mock_pipeline.
                      Use for /predict, /schema, /drift happy-path tests.
  client_no_model   — TestClient with no override; app.state.pipeline is None
                      (no CHURN_MODEL_PATH set in test env). Use for 503 tests.

Sample data
-----------
  VALID_CUSTOMER    — a single valid CustomerFeatures dict (all fields in range).
  MOCK_PREDICTION   — a PredictionOut dict for use as a service-layer return value.
"""

from __future__ import annotations

from collections.abc import Generator
from unittest.mock import MagicMock

import pytest
from churn_lib import ChurnPipeline, PipelineConfig
from fastapi.testclient import TestClient

from app.core.dependencies import get_pipeline
from app.main import app

# ---------------------------------------------------------------------------
# Canonical sample data — copy in tests, never mutate these
# ---------------------------------------------------------------------------

VALID_CUSTOMER: dict = {
    "tenure_months": 12,
    "monthly_charges": 50.0,
    "num_products": 2,
    "support_calls_last_6m": 1,
    "age": 35,
    "contract_type": "month-to-month",
    "payment_method": "credit_card",
    "has_internet": True,
    "has_streaming": False,
}

MOCK_PREDICTION: dict = {
    "prediction": 1,
    "label": "1",
    "probabilities": {"0": 0.3, "1": 0.7},
}


# ---------------------------------------------------------------------------
# Session-scoped — loaded once per worker, shared across all tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def config() -> PipelineConfig:
    """Real PipelineConfig from the default YAML — no ML, no disk writes.

    Shared across the session because from_yaml() is deterministic and cheap.
    Having the real config available means /schema and /drift tests can
    exercise real schema lookups without a fitted model.
    """
    return PipelineConfig.from_yaml()


# ---------------------------------------------------------------------------
# Function-scoped fixtures — fresh per test to prevent mutation side-effects
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_pipeline(config: PipelineConfig) -> MagicMock:
    """MagicMock that satisfies the ChurnPipeline interface for HTTP-layer tests.

    Uses `spec=ChurnPipeline` so attribute access raises AttributeError for
    anything not on the real class — catching interface drift early.
    Attaches the real PipelineConfig so /schema and /drift work correctly.

    Inference behaviour (predict_proba etc.) is not configured here — patch
    the relevant service function at the test level instead of configuring
    the mock, which keeps individual tests self-contained.
    """
    pipeline = MagicMock(spec=ChurnPipeline)
    pipeline.config = config
    return pipeline


@pytest.fixture
def client(mock_pipeline: MagicMock) -> Generator[TestClient, None, None]:
    """TestClient with a mock model pre-loaded via two mechanisms:

    1. Dependency override: replaces get_pipeline() for all domain endpoints
       (/predict, /schema, /drift) so they receive the mock without touching
       module globals. FastAPI resolves Depends(get_pipeline) before calling
       the handler, so the handler sees the mock directly.

    2. app.state.pipeline: set manually after lifespan startup so that meta
       endpoints that bypass the DI system (i.e. GET /ready) also see a model.
       /ready reads app.state.pipeline directly because it is an operational
       probe that must not be routed through the same DI chain as ML endpoints.

    Both mechanisms are cleaned up in the finally block even if the test fails.
    """
    app.dependency_overrides[get_pipeline] = lambda: mock_pipeline
    try:
        with TestClient(app, raise_server_exceptions=False) as c:
            # Lifespan runs at __enter__ and sets app.state.pipeline = None
            # (no CHURN_MODEL_PATH in test env). Set it on the module-level app
            # so /ready sees a loaded model via request.app.state.pipeline.
            app.state.pipeline = mock_pipeline
            yield c
    finally:
        app.dependency_overrides.clear()
        app.state.pipeline = None


@pytest.fixture
def client_no_model() -> Generator[TestClient, None, None]:
    """TestClient with no model loaded — get_pipeline() will return 503.

    No dependency override is set, so get_pipeline() falls through to its
    real implementation, which reads app.state.pipeline. Since no
    CHURN_MODEL_PATH is set in the test environment, the lifespan sets
    app.state.pipeline = None, causing get_pipeline() to raise HTTP 503.

    Use this to verify that model-dependent endpoints behave correctly
    when no model is available.
    """
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c
