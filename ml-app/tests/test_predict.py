"""Tests for POST /api/v1/churn/predict and GET /api/v1/churn/schema.

Testing strategy
----------------
Endpoint tests validate the HTTP contract: routing, request validation,
response shape, status codes, and error mapping. They do NOT test ML
correctness — that is churn-lib's responsibility and is covered in churn-lib's
own test suite.

To keep these tests fast and deterministic, `run_prediction` is patched at
the endpoint's import site. This means tests run without a GPU, without
XGBoost inference, and without loading a real model file. The mock_pipeline
fixture (from conftest.py) provides a real PipelineConfig so /schema works
correctly without any additional mocking.

Patch target: `app.api.v1.endpoints.churn.predict.run_prediction`
  Why this path? Python resolves names at the call site. The endpoint imported
  `run_prediction` from `app.services.churn.inference`, so patching the
  service module directly would have no effect — the endpoint already holds a
  reference to the original function. Patching at the import site ensures the
  endpoint sees the mock.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.schemas.churn import PredictionOut
from tests.conftest import MOCK_PREDICTION, VALID_CUSTOMER

# The target for patching run_prediction in the predict endpoint module.
_PATCH_TARGET = "app.api.v1.endpoints.churn.predict.run_prediction"


class TestPredictHappyPath:
    """POST /predict — successful inference requests."""

    def test_returns_200(self, client: TestClient) -> None:
        with patch(_PATCH_TARGET, return_value=[PredictionOut(**MOCK_PREDICTION)]):
            response = client.post(
                "/api/v1/churn/predict",
                json={"customers": [VALID_CUSTOMER]},
            )
        assert response.status_code == 200

    def test_response_has_correct_structure(self, client: TestClient) -> None:
        """The response must contain predictions, model_version, and threshold_used."""
        with patch(_PATCH_TARGET, return_value=[PredictionOut(**MOCK_PREDICTION)]):
            data = client.post(
                "/api/v1/churn/predict",
                json={"customers": [VALID_CUSTOMER]},
            ).json()

        assert "predictions" in data
        assert "model_version" in data
        assert "threshold_used" in data

    def test_prediction_count_matches_input_count(self, client: TestClient) -> None:
        """One prediction per input customer — order and count must be preserved.

        This is a fundamental contract: callers identify which prediction
        belongs to which customer by index. A mismatch here silently assigns
        wrong churn scores to customers.
        """
        two_customers = [VALID_CUSTOMER, VALID_CUSTOMER.copy()]
        two_predictions = [PredictionOut(**MOCK_PREDICTION), PredictionOut(**MOCK_PREDICTION)]

        with patch(_PATCH_TARGET, return_value=two_predictions):
            data = client.post(
                "/api/v1/churn/predict",
                json={"customers": two_customers},
            ).json()

        assert len(data["predictions"]) == 2

    def test_threshold_used_matches_request_value(self, client: TestClient) -> None:
        """The threshold sent in the request must be echoed in the response.

        threshold_used in the response lets callers audit which operating
        point was applied, especially when the default changes server-side.
        """
        with patch(_PATCH_TARGET, return_value=[PredictionOut(**MOCK_PREDICTION)]):
            data = client.post(
                "/api/v1/churn/predict",
                json={"customers": [VALID_CUSTOMER], "threshold": 0.3},
            ).json()

        assert data["threshold_used"] == pytest.approx(0.3)

    def test_default_threshold_applied_when_not_specified(self, client: TestClient) -> None:
        """Omitting threshold in the request should use the server default (0.5)."""
        with patch(_PATCH_TARGET, return_value=[PredictionOut(**MOCK_PREDICTION)]):
            data = client.post(
                "/api/v1/churn/predict",
                json={"customers": [VALID_CUSTOMER]},
            ).json()

        assert data["threshold_used"] == pytest.approx(0.5)

    def test_prediction_fields_are_present(self, client: TestClient) -> None:
        """Every PredictionOut must have prediction, label, and probabilities."""
        with patch(_PATCH_TARGET, return_value=[PredictionOut(**MOCK_PREDICTION)]):
            data = client.post(
                "/api/v1/churn/predict",
                json={"customers": [VALID_CUSTOMER]},
            ).json()

        pred = data["predictions"][0]
        assert "prediction" in pred
        assert "label" in pred
        assert "probabilities" in pred


class TestPredictErrorHandling:
    """POST /predict — error conditions."""

    def test_returns_503_when_no_model_loaded(self, client_no_model: TestClient) -> None:
        """Requests to /predict without a loaded model must return 503.

        503 (Service Unavailable) signals a temporary condition — the model
        is not yet loaded. Clients should retry after a delay rather than
        treating this as a permanent error. This is different from 404 (not
        found) or 500 (unexpected error).
        """
        response = client_no_model.post(
            "/api/v1/churn/predict",
            json={"customers": [VALID_CUSTOMER]},
        )
        assert response.status_code == 503

    def test_returns_422_for_invalid_feature_value(self, client: TestClient) -> None:
        """Out-of-range feature values must be rejected with 422 before reaching the model.

        422 (Unprocessable Entity) tells the caller that the request was
        syntactically valid JSON but semantically invalid — a recoverable
        client error. The client must fix the data before retrying.

        This test exercises the churn-lib ValidationError → 422 mapping in
        the endpoint handler.
        """
        from churn_lib import ValidationError

        with patch(_PATCH_TARGET, side_effect=ValidationError("tenure_months out of range")):
            response = client.post(
                "/api/v1/churn/predict",
                json={"customers": [VALID_CUSTOMER]},
            )
        assert response.status_code == 422

    def test_returns_422_for_pydantic_validation_error(self, client: TestClient) -> None:
        """Missing required fields must be rejected by Pydantic before the handler runs.

        FastAPI validates the request body against CustomerFeatures before
        calling the endpoint. This test ensures that schema-level validation
        works (no tenure_months field).
        """
        invalid = {k: v for k, v in VALID_CUSTOMER.items() if k != "tenure_months"}
        response = client.post(
            "/api/v1/churn/predict",
            json={"customers": [invalid]},
        )
        assert response.status_code == 422

    def test_returns_422_for_empty_customers_list(self, client: TestClient) -> None:
        """An empty customers list must be rejected — min_length=1 is enforced."""
        response = client.post(
            "/api/v1/churn/predict",
            json={"customers": []},
        )
        assert response.status_code == 422

    def test_returns_422_for_oversized_batch(self, client: TestClient) -> None:
        """Batches exceeding 1000 customers must be rejected — max_length=1000 is enforced.

        Without this limit, a malicious or buggy client could send 1M records
        in a single request, causing the server to OOM. Rejecting at the Pydantic
        layer happens before any processing, keeping the failure cheap.
        """
        oversized = [VALID_CUSTOMER] * 1001
        response = client.post(
            "/api/v1/churn/predict",
            json={"customers": oversized},
        )
        assert response.status_code == 422


class TestSchema:
    """GET /api/v1/churn/schema — feature schema endpoint."""

    def test_returns_200(self, client: TestClient) -> None:
        response = client.get("/api/v1/churn/schema")
        assert response.status_code == 200

    def test_response_structure(self, client: TestClient) -> None:
        """The schema response must expose model card and all three feature groups."""
        data = client.get("/api/v1/churn/schema").json()

        assert "default_threshold" in data
        assert "model" in data
        assert "features" in data

        model = data["model"]
        assert "name" in model
        assert "version" in model
        assert "description" in model
        assert "intended_use" in model
        assert "limitations" in model

        features = data["features"]
        assert "numeric" in features
        assert "categorical" in features
        assert "binary" in features

    def test_numeric_features_have_bounds(self, client: TestClient) -> None:
        """Numeric feature schemas must carry min/max so clients can validate locally."""
        features = client.get("/api/v1/churn/schema").json()["features"]
        for schema in features["numeric"]:
            assert "name" in schema
            assert "min" in schema
            assert "max" in schema

    def test_categorical_features_have_categories(self, client: TestClient) -> None:
        """Categorical feature schemas must list all valid values."""
        features = client.get("/api/v1/churn/schema").json()["features"]
        for schema in features["categorical"]:
            assert "categories" in schema
            assert isinstance(schema["categories"], list)
            assert len(schema["categories"]) > 0

    def test_returns_503_when_no_model_loaded(self, client_no_model: TestClient) -> None:
        """Schema requires the model's config — 503 when no model is loaded."""
        response = client_no_model.get("/api/v1/churn/schema")
        assert response.status_code == 503
