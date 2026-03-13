"""Tests for churn_lib.data_generator."""

from __future__ import annotations

import pandas as pd
import pytest

from churn_lib.data_generator import generate_prediction_data, generate_training_data

EXPECTED_FEATURE_COLUMNS = [
    "tenure_months",
    "monthly_charges",
    "num_products",
    "support_calls_last_6m",
    "age",
    "contract_type",
    "payment_method",
    "has_internet",
    "has_streaming",
]


class TestGenerateTrainingData:
    def test_churn_column_is_binary(self):
        df = generate_training_data(500, 42)
        assert df["churn"].isin([0, 1]).all()
        assert df["churn"].nunique() == 2

    def test_feature_columns_present(self):
        df = generate_training_data(100, 0)
        for col in EXPECTED_FEATURE_COLUMNS:
            assert col in df.columns, f"Missing column: {col}"

    def test_no_nulls(self):
        df = generate_training_data(200, 1)
        assert df.isnull().sum().sum() == 0

    def test_reproducibility(self):
        df1 = generate_training_data(200, 99)
        df2 = generate_training_data(200, 99)
        pd.testing.assert_frame_equal(df1, df2)

    @pytest.mark.parametrize("n_samples", [1, 500])
    def test_n_samples_respected(self, n_samples):
        assert len(generate_training_data(n_samples, 0)) == n_samples

    def test_numeric_columns_in_expected_ranges(self):
        df = generate_training_data(500, 42)
        assert df["tenure_months"].between(1, 72).all()
        assert df["monthly_charges"].between(20.0, 120.0).all()
        assert df["num_products"].between(1, 5).all()
        assert df["support_calls_last_6m"].between(0, 10).all()
        assert df["age"].between(18, 80).all()

    def test_categorical_columns_valid_values(self):
        df = generate_training_data(300, 7)
        assert set(df["contract_type"].unique()).issubset({"month-to-month", "one-year", "two-year"})
        assert set(df["payment_method"].unique()).issubset(
            {"credit_card", "bank_transfer", "electronic_check", "mailed_check"}
        )


class TestGeneratePredictionData:
    def test_no_churn_column(self):
        assert "churn" not in generate_prediction_data(10, 0).columns

    def test_feature_columns_match_training(self):
        df = generate_prediction_data(20, 0)
        for col in EXPECTED_FEATURE_COLUMNS:
            assert col in df.columns
