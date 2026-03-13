"""Unit tests for app/services/churn/inference.py.

Testing strategy
----------------
The inference service is the translation layer between the web layer's Pydantic
models and churn-lib's plain-dict API. It is small but critical: if it drops
fields, reorders customers, or ignores the threshold, predictions will be
silently wrong.

These tests exercise the service in isolation — no HTTP, no running app,
no real ML model. `predict_batch` from churn-lib is mocked at the service
module's import site so we test only the service logic:
  - CustomerFeatures → model_dump() → predict_batch (correct argument passing)
  - predict_batch result → PredictionOut (correct field mapping)
  - Batch order is preserved (index N in input → index N in output)
  - ValidationError from predict_batch propagates unchanged

Why test the service separately from the endpoint?
  Endpoint tests (test_predict.py) mock run_prediction entirely, which means
  they don't exercise any service code. These tests catch bugs in the service
  layer that would otherwise only surface in integration tests.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from churn_lib import ValidationError

from app.schemas.churn import CustomerFeatures
from app.services.churn.inference import run_prediction
from tests.conftest import VALID_CUSTOMER

_PATCH_TARGET = "app.services.churn.inference.predict_batch"

_RAW_PREDICTION = {
    "prediction": 1,
    "label": "1",
    "probabilities": {"0": 0.3, "1": 0.7},
}


def _make_customers(*overrides: dict) -> list[CustomerFeatures]:
    """Build a list of CustomerFeatures, optionally overriding fields."""
    return [CustomerFeatures(**{**VALID_CUSTOMER, **ov}) for ov in overrides]


class TestRunPrediction:
    """run_prediction() — service layer unit tests."""

    def test_returns_one_prediction_per_customer(self) -> None:
        """Output length must equal input length — this is the ordering contract."""
        pipeline = MagicMock()
        customers = _make_customers({}, {})  # two customers

        with patch(_PATCH_TARGET, return_value=[_RAW_PREDICTION, _RAW_PREDICTION]):
            results = run_prediction(pipeline, customers, threshold=0.5)

        assert len(results) == 2

    def test_prediction_fields_are_mapped_correctly(self) -> None:
        """Every field in churn-lib's PredictionResult must appear in PredictionOut."""
        pipeline = MagicMock()
        customers = _make_customers({})

        with patch(_PATCH_TARGET, return_value=[_RAW_PREDICTION]):
            result = run_prediction(pipeline, customers, threshold=0.5)[0]

        assert result.prediction == 1
        assert result.label == "1"
        assert result.probabilities == {"0": 0.3, "1": 0.7}

    def test_passes_threshold_to_predict_batch(self) -> None:
        """The threshold must be forwarded to predict_batch — not silently ignored.

        predict_batch uses the threshold to convert probabilities to binary
        predictions. If the service drops it, callers cannot change the
        operating point.
        """
        pipeline = MagicMock()
        customers = _make_customers({})

        with patch(_PATCH_TARGET, return_value=[_RAW_PREDICTION]) as mock_batch:
            run_prediction(pipeline, customers, threshold=0.3)

        # Verify the call was made with threshold=0.3 as a keyword argument.
        _, kwargs = mock_batch.call_args
        assert kwargs.get("threshold") == pytest.approx(0.3)

    def test_passes_pipeline_to_predict_batch(self) -> None:
        """The same pipeline object must be forwarded — not a copy or None."""
        pipeline = MagicMock()
        customers = _make_customers({})

        with patch(_PATCH_TARGET, return_value=[_RAW_PREDICTION]) as mock_batch:
            run_prediction(pipeline, customers, threshold=0.5)

        args, _ = mock_batch.call_args
        assert args[0] is pipeline

    def test_customers_converted_to_plain_dicts_for_predict_batch(self) -> None:
        """predict_batch expects plain dicts, not Pydantic models.

        churn-lib's validate_batch() uses dict-style access. Passing Pydantic
        models directly would cause AttributeError in validate_batch().
        """
        pipeline = MagicMock()
        customers = _make_customers({})

        with patch(_PATCH_TARGET, return_value=[_RAW_PREDICTION]) as mock_batch:
            run_prediction(pipeline, customers, threshold=0.5)

        args, _ = mock_batch.call_args
        passed_samples = args[1]
        assert isinstance(passed_samples, list)
        assert isinstance(passed_samples[0], dict)

    def test_validation_error_propagates_unchanged(self) -> None:
        """ValidationError from churn-lib must not be wrapped or swallowed.

        The endpoint catches ValidationError and maps it to HTTP 422. If the
        service wraps it in a different exception type, the endpoint's except
        clause won't catch it and the client receives a 500 instead of 422.
        """
        pipeline = MagicMock()
        customers = _make_customers({})

        with patch(_PATCH_TARGET, side_effect=ValidationError("tenure_months=-5 out of range")):
            with pytest.raises(ValidationError):
                run_prediction(pipeline, customers, threshold=0.5)
