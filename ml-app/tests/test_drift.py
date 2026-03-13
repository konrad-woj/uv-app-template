"""Tests for POST /api/v1/churn/drift.

Testing strategy
----------------
Unlike the predict endpoint, drift detection does NOT run ML inference — it
only reads the PipelineConfig from the loaded pipeline and applies PSI
statistics to the raw feature data. This means we can run the full drift
code path in tests without mocking at the service layer.

The mock_pipeline fixture (from conftest.py) attaches a real PipelineConfig,
so these tests exercise the actual PSI computation end-to-end. This is more
valuable than mocking run_drift_check, because it tests that:
  1. The endpoint correctly routes to the drift service.
  2. The service converts list[dict] to DataFrames correctly.
  3. The PSI library returns the expected shape of results.
  4. The endpoint serialises DriftReport → DriftResponse correctly.

Only the 503 tests use client_no_model, because that is the only error path
that depends on the mock/no-model distinction.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from tests.conftest import VALID_CUSTOMER

# Two samples that are identical — PSI will be exactly 0 (or near-zero) for
# all features, so overall_status must be "stable".
_IDENTICAL_REFERENCE = [VALID_CUSTOMER] * 20
_IDENTICAL_SERVING = [VALID_CUSTOMER] * 20


class TestDriftHappyPath:
    """POST /drift — successful drift detection requests."""

    def test_returns_200(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/churn/drift",
            json={"reference": _IDENTICAL_REFERENCE, "serving": _IDENTICAL_SERVING},
        )
        assert response.status_code == 200

    def test_response_structure(self, client: TestClient) -> None:
        """The drift response must expose overall_status and per-feature details."""
        data = client.post(
            "/api/v1/churn/drift",
            json={"reference": _IDENTICAL_REFERENCE, "serving": _IDENTICAL_SERVING},
        ).json()

        assert "overall_status" in data
        assert "drifted_features" in data
        assert "reference_samples" in data
        assert "serving_samples" in data
        assert "features" in data

    def test_sample_counts_match_input(self, client: TestClient) -> None:
        """reference_samples and serving_samples must reflect the input sizes."""
        data = client.post(
            "/api/v1/churn/drift",
            json={"reference": _IDENTICAL_REFERENCE, "serving": _IDENTICAL_SERVING},
        ).json()

        assert data["reference_samples"] == len(_IDENTICAL_REFERENCE)
        assert data["serving_samples"] == len(_IDENTICAL_SERVING)

    def test_identical_distributions_are_stable(self, client: TestClient) -> None:
        """Identical reference and serving data must produce 'stable' overall status.

        This is the fundamental PSI invariant: PSI(P, P) = 0, which maps to
        'stable'. If this fails, the PSI computation or status mapping is broken.
        """
        data = client.post(
            "/api/v1/churn/drift",
            json={"reference": _IDENTICAL_REFERENCE, "serving": _IDENTICAL_SERVING},
        ).json()

        assert data["overall_status"] == "stable"
        assert data["drifted_features"] == []

    def test_feature_results_include_psi_and_status(self, client: TestClient) -> None:
        """Every feature in the response must have a PSI score and a status label."""
        data = client.post(
            "/api/v1/churn/drift",
            json={"reference": _IDENTICAL_REFERENCE, "serving": _IDENTICAL_SERVING},
        ).json()

        for feature in data["features"]:
            assert "feature" in feature
            assert "feature_type" in feature
            assert "psi" in feature
            assert "status" in feature
            assert isinstance(feature["psi"], float)
            assert feature["status"] in {"stable", "moderate", "major"}

    def test_feature_types_match_config(self, client: TestClient, config) -> None:
        """feature_type in each result must match the PipelineConfig classification.

        This verifies that the config-driven schema lookup is correct — a
        numeric feature must not be reported as categorical and vice versa.
        """
        data = client.post(
            "/api/v1/churn/drift",
            json={"reference": _IDENTICAL_REFERENCE, "serving": _IDENTICAL_SERVING},
        ).json()

        numeric_names = {s.name for s in config.numeric_schemas}
        categorical_names = {s.name for s in config.categorical_schemas}
        binary_names = {s.name for s in config.binary_schemas}

        for feature in data["features"]:
            name = feature["feature"]
            if name in numeric_names:
                assert feature["feature_type"] == "numeric"
            elif name in categorical_names:
                assert feature["feature_type"] == "categorical"
            elif name in binary_names:
                assert feature["feature_type"] == "binary"


class TestDriftErrorHandling:
    """POST /drift — error conditions."""

    def test_returns_503_when_no_model_loaded(self, client_no_model: TestClient) -> None:
        """Drift detection requires the pipeline's config — 503 without a model.

        The pipeline config defines which features to check and their expected
        types. Without it, the drift check cannot know what to measure.
        """
        response = client_no_model.post(
            "/api/v1/churn/drift",
            json={"reference": _IDENTICAL_REFERENCE, "serving": _IDENTICAL_SERVING},
        )
        assert response.status_code == 503

    def test_returns_422_for_empty_reference(self, client: TestClient) -> None:
        """An empty reference list must be rejected — min_length=1 is enforced."""
        response = client.post(
            "/api/v1/churn/drift",
            json={"reference": [], "serving": _IDENTICAL_SERVING},
        )
        assert response.status_code == 422

    def test_returns_422_for_empty_serving(self, client: TestClient) -> None:
        """An empty serving list must be rejected — min_length=1 is enforced."""
        response = client.post(
            "/api/v1/churn/drift",
            json={"reference": _IDENTICAL_REFERENCE, "serving": []},
        )
        assert response.status_code == 422
