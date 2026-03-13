"""Tests for churn_lib.drift — check_drift() and PSI threshold logic."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from churn_lib import PipelineConfig, check_drift
from churn_lib.data_generator import generate_training_data
from churn_lib.drift import _psi_status


class TestCheckDriftStable:
    def test_identical_data_is_stable(self, training_df: pd.DataFrame, config: PipelineConfig):
        report = check_drift(training_df, training_df, config)
        assert report["overall_status"] == "stable"
        assert report["drifted_features"] == []

    def test_report_has_required_keys(self, training_df: pd.DataFrame, config: PipelineConfig):
        report = check_drift(training_df, training_df, config)
        assert set(report.keys()) >= {
            "reference_samples",
            "serving_samples",
            "features",
            "drifted_features",
            "overall_status",
        }

    def test_features_list_covers_all_columns(self, training_df: pd.DataFrame, config: PipelineConfig):
        report = check_drift(training_df, training_df, config)
        assert len(report["features"]) == len(config.feature_columns)

    def test_feature_types_match_config(self, training_df: pd.DataFrame, config: PipelineConfig):
        report = check_drift(training_df, training_df, config)
        type_map = {r["feature"]: r["feature_type"] for r in report["features"]}
        for name in config.numeric_features:
            assert type_map[name] == "numeric"
        for name in config.categorical_features:
            assert type_map[name] == "categorical"
        for name in config.binary_features:
            assert type_map[name] == "binary"


class TestCheckDriftDetection:
    def test_constant_numeric_feature_causes_drift(self, training_df: pd.DataFrame, config: PipelineConfig):
        serving_df = training_df.copy()
        serving_df["tenure_months"] = 72  # constant at max → extreme distribution shift
        report = check_drift(training_df, serving_df, config)
        assert "tenure_months" in report["drifted_features"]

    def test_constant_categorical_feature_causes_drift(self, training_df: pd.DataFrame, config: PipelineConfig):
        serving_df = training_df.copy()
        serving_df["contract_type"] = "month-to-month"  # 100% one category
        report = check_drift(training_df, serving_df, config)
        assert "contract_type" in report["drifted_features"]

    def test_constant_binary_feature_causes_drift(self, training_df: pd.DataFrame, config: PipelineConfig):
        serving_df = training_df.copy()
        serving_df["has_internet"] = True  # 100% True vs ~75% in reference
        report = check_drift(training_df, serving_df, config)
        assert "has_internet" in report["drifted_features"]

    def test_overall_status_reflects_worst_feature(self, training_df: pd.DataFrame, config: PipelineConfig):
        serving_df = training_df.copy()
        serving_df["tenure_months"] = 72
        report = check_drift(training_df, serving_df, config)
        statuses = {r["status"] for r in report["features"]}
        worst = "major" if "major" in statuses else ("moderate" if "moderate" in statuses else "stable")
        assert report["overall_status"] == worst


class TestCheckDriftOutput:
    def test_output_path_writes_json(self, training_df: pd.DataFrame, config: PipelineConfig, tmp_path):
        output_path = str(tmp_path / "drift.json")
        report = check_drift(training_df, training_df, config, output_path=output_path)
        loaded = json.loads((tmp_path / "drift.json").read_text())
        assert loaded["overall_status"] == report["overall_status"]


class TestCheckDriftEdgeCases:
    def test_missing_feature_in_serving_does_not_raise(self, training_df: pd.DataFrame, config: PipelineConfig):
        serving_df = training_df.drop(columns=["tenure_months"])
        check_drift(training_df, serving_df, config)  # must not raise

    def test_asymmetric_sample_sizes(self, config: PipelineConfig):
        ref = generate_training_data(50, 1)
        serv = generate_training_data(300, 2)
        report = check_drift(ref, serv, config)
        assert report["reference_samples"] == 50
        assert report["serving_samples"] == 300


class TestPsiStatus:
    @pytest.mark.parametrize(
        "psi,expected_status",
        [
            (0.099, "stable"),  # just below lower boundary
            (0.10, "moderate"),  # lower boundary of moderate
            (0.249, "moderate"),  # just below upper boundary
            (0.25, "major"),  # lower boundary of major
            (1.0, "major"),
        ],
    )
    def test_psi_status_thresholds(self, psi, expected_status):
        assert _psi_status(psi) == expected_status
