"""Tests for churn_lib.validate — ValidationError paths and validate_sample delegation."""

from __future__ import annotations

import pytest

import churn_lib.validate as validate_module
from churn_lib import PipelineConfig, ValidationError, validate_batch, validate_sample


class TestValidateBatch:
    def test_valid_batch_passes(self, valid_batch, config: PipelineConfig):
        validate_batch(valid_batch, config)  # must not raise

    def test_empty_list_raises(self, config: PipelineConfig):
        with pytest.raises(ValidationError, match="non-empty"):
            validate_batch([], config)

    @pytest.mark.parametrize(
        "col",
        [
            "tenure_months",  # numeric
            "contract_type",  # categorical
            "has_internet",  # binary
        ],
    )
    def test_missing_column_raises(self, col, valid_sample, config: PipelineConfig):
        del valid_sample[col]
        with pytest.raises(ValidationError, match=col):
            validate_batch([valid_sample], config)

    @pytest.mark.parametrize(
        "col,bad_value",
        [
            ("tenure_months", None),  # null
            ("tenure_months", "twelve"),  # string
            ("tenure_months", True),  # bool is rejected for numeric (Python gotcha)
        ],
    )
    def test_numeric_wrong_type_raises(self, col, bad_value, valid_sample, config: PipelineConfig):
        valid_sample[col] = bad_value
        with pytest.raises(ValidationError, match=col):
            validate_batch([valid_sample], config)

    @pytest.mark.parametrize(
        "col,bad_value,direction",
        [
            ("tenure_months", -1, "minimum"),
            ("tenure_months", 121, "maximum"),
            ("monthly_charges", -0.01, "minimum"),
            ("age", 101, "maximum"),
        ],
    )
    def test_numeric_out_of_range_raises(self, col, bad_value, direction, valid_sample, config: PipelineConfig):
        valid_sample[col] = bad_value
        with pytest.raises(ValidationError, match=direction):
            validate_batch([valid_sample], config)

    @pytest.mark.parametrize(
        "col,bad_value",
        [
            ("contract_type", "weekly"),
            ("payment_method", "bitcoin"),
        ],
    )
    def test_categorical_invalid_value_raises(self, col, bad_value, valid_sample, config: PipelineConfig):
        valid_sample[col] = bad_value
        with pytest.raises(ValidationError, match="categories"):
            validate_batch([valid_sample], config)

    @pytest.mark.parametrize(
        "col,bad_value",
        [
            ("has_internet", "yes"),  # string
            ("has_internet", 2),  # int outside {0, 1}
            ("has_internet", None),  # null
        ],
    )
    def test_binary_invalid_value_raises(self, col, bad_value, valid_sample, config: PipelineConfig):
        valid_sample[col] = bad_value
        with pytest.raises(ValidationError, match=col):
            validate_batch([valid_sample], config)

    @pytest.mark.parametrize("valid_value", [True, False, 0, 1])
    def test_binary_valid_representations_pass(self, valid_value, valid_sample, config: PipelineConfig):
        valid_sample["has_internet"] = valid_value
        validate_batch([valid_sample], config)  # must not raise

    def test_multiple_errors_all_reported(self, valid_sample, config: PipelineConfig):
        """Validation collects all errors before raising, not just the first."""
        valid_sample["tenure_months"] = "bad_type"
        valid_sample["monthly_charges"] = -999.0
        valid_sample["contract_type"] = "invalid_category"
        with pytest.raises(ValidationError) as exc_info:
            validate_batch([valid_sample], config)
        msg = str(exc_info.value)
        assert "tenure_months" in msg
        assert "monthly_charges" in msg
        assert "contract_type" in msg

    def test_non_dict_sample_raises(self, config: PipelineConfig):
        with pytest.raises(ValidationError, match="expected dict"):
            validate_batch(["not_a_dict"], config)


class TestValidateSample:
    def test_delegates_to_validate_batch(self, valid_sample, config: PipelineConfig, monkeypatch):
        """validate_sample must wrap the input in a list and call validate_batch."""
        calls: list[list[dict]] = []
        real = validate_module.validate_batch

        def spy(samples, cfg):
            calls.append(list(samples))
            return real(samples, cfg)

        monkeypatch.setattr(validate_module, "validate_batch", spy)
        validate_sample(valid_sample, config)
        assert calls == [[valid_sample]]
