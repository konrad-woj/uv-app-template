"""Tests for churn_lib.trainer — train() artifacts and find_threshold() logic.

MLflow is redirected to an isolated tmp directory for this entire module via the
module-scoped autouse fixture `redirect_mlflow`. The expensive `train()` call is
cached in `train_result` (module scope) so all artifact tests share one run.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from churn_lib import ChurnPipeline, PipelineConfig
from churn_lib.trainer import find_threshold, train

# ---------------------------------------------------------------------------
# Module-scoped fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", autouse=True)
def redirect_mlflow(tmp_path_factory):
    """Redirect MLflow tracking to an isolated tmp dir for all trainer tests."""
    import mlflow

    original_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(f"file://{tmp_path_factory.mktemp('mlflow')}")
    yield
    mlflow.set_tracking_uri(original_uri)


@pytest.fixture(scope="module")
def train_result(redirect_mlflow, config: PipelineConfig, training_df: pd.DataFrame, tmp_path_factory):
    """Run train() once for the module; all artifact tests share the output."""
    output_dir = tmp_path_factory.mktemp("runs")
    pipeline = ChurnPipeline(config)
    summary = train(pipeline, training_df, output_dir=output_dir, test_size=0.2, random_seed=42)
    run_dir = sorted(output_dir.iterdir())[0]
    return summary, run_dir


@pytest.fixture(scope="module")
def X_test_y_test(config: PipelineConfig, training_df: pd.DataFrame):
    """Hold-out split matching the parameters used inside train()."""
    X = training_df[config.feature_columns]
    y = training_df[config.target]
    splits = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    return splits[1], splits[3]  # X_test, y_test


# ---------------------------------------------------------------------------
# train() summary and artifacts
# ---------------------------------------------------------------------------


class TestTrain:
    def test_summary_contains_run_dir(self, train_result):
        summary, _ = train_result
        assert "Run saved to:" in summary

    @pytest.mark.parametrize(
        "filename",
        [
            "pipeline.joblib",
            "metrics.json",
            "config.json",
            "confusion_matrix.png",
            "feature_importance.png",
            "threshold_curve.json",
        ],
    )
    def test_artifact_exists(self, filename, train_result):
        _, run_dir = train_result
        assert (run_dir / filename).exists(), f"Missing artifact: {filename}"

    def test_metrics_json_structure(self, train_result):
        _, run_dir = train_result
        data = json.loads((run_dir / "metrics.json").read_text())
        for key in ("accuracy", "macro avg", "weighted avg", "optimal_threshold"):
            assert key in data
        assert 0.0 <= data["accuracy"] <= 1.0
        assert 0.0 <= data["optimal_threshold"] <= 1.0

    def test_pipeline_joblib_is_loadable(self, train_result):
        _, run_dir = train_result
        loaded = ChurnPipeline.load(run_dir / "pipeline.joblib")
        assert loaded.label_names_ == ["0", "1"]

    def test_threshold_curve_record_keys(self, train_result):
        _, run_dir = train_result
        records = json.loads((run_dir / "threshold_curve.json").read_text())
        assert len(records) > 0
        for row in records:
            assert {"threshold", "precision", "recall", "f1"}.issubset(row.keys())


# ---------------------------------------------------------------------------
# MLflow logging
# ---------------------------------------------------------------------------


class TestTrainMlflow:
    def test_mlflow_run_created(self, train_result):
        import mlflow

        runs = mlflow.search_runs(experiment_names=["churn-prediction"])
        assert len(runs) >= 1

    def test_mlflow_params_logged(self, train_result):
        import mlflow

        row = mlflow.search_runs(experiment_names=["churn-prediction"]).iloc[0]
        assert "params.test_size" in row.index
        assert "params.random_seed" in row.index
        assert "params.n_features" in row.index

    def test_mlflow_metrics_logged(self, train_result):
        import mlflow

        row = mlflow.search_runs(experiment_names=["churn-prediction"]).iloc[0]
        assert "metrics.accuracy" in row.index
        assert "metrics.macro_f1" in row.index


# ---------------------------------------------------------------------------
# find_threshold()
# ---------------------------------------------------------------------------


class TestFindThreshold:
    def test_threshold_is_float_in_sweep_range(self, fitted_pipeline: ChurnPipeline, X_test_y_test):
        X_test, y_test = X_test_y_test
        threshold, _ = find_threshold(fitted_pipeline, X_test, y_test)
        assert isinstance(threshold, float)
        assert 0.05 <= threshold <= 0.95

    def test_curve_df_columns(self, fitted_pipeline: ChurnPipeline, X_test_y_test):
        X_test, y_test = X_test_y_test
        _, curve_df = find_threshold(fitted_pipeline, X_test, y_test)
        assert list(curve_df.columns) == ["threshold", "precision", "recall", "f1"]

    @pytest.mark.parametrize("optimize_for", ["f1", "precision", "recall"])
    def test_optimize_for_modes(self, optimize_for, fitted_pipeline: ChurnPipeline, X_test_y_test):
        X_test, y_test = X_test_y_test
        threshold, _ = find_threshold(fitted_pipeline, X_test, y_test, optimize_for=optimize_for)
        assert 0.05 <= threshold <= 0.95

    def test_min_recall_constraint_satisfied(self, fitted_pipeline: ChurnPipeline, X_test_y_test):
        X_test, y_test = X_test_y_test
        threshold, curve_df = find_threshold(fitted_pipeline, X_test, y_test, min_recall=0.1)
        row = curve_df[np.isclose(curve_df["threshold"], threshold)].iloc[0]
        assert row["recall"] >= 0.1

    def test_min_precision_constraint_satisfied(self, fitted_pipeline: ChurnPipeline, X_test_y_test):
        X_test, y_test = X_test_y_test
        threshold, curve_df = find_threshold(fitted_pipeline, X_test, y_test, min_precision=0.1)
        row = curve_df[np.isclose(curve_df["threshold"], threshold)].iloc[0]
        assert row["precision"] >= 0.1

    @pytest.mark.parametrize(
        "kwargs,match",
        [
            ({"min_recall": 0.5, "min_precision": 0.9}, "at most one"),
            ({"min_recall": 1.01}, "No threshold achieves recall"),
            ({"min_precision": 1.01}, "No threshold achieves precision"),
        ],
    )
    def test_invalid_constraints_raise(self, kwargs, match, fitted_pipeline: ChurnPipeline, X_test_y_test):
        X_test, y_test = X_test_y_test
        with pytest.raises(ValueError, match=match):
            find_threshold(fitted_pipeline, X_test, y_test, **kwargs)

    def test_multiclass_pipeline_returns_default(self, config: PipelineConfig, training_df: pd.DataFrame):
        """Multi-class pipelines are not supported — returns (0.5, empty DataFrame)."""
        pipeline = ChurnPipeline(config)
        X = training_df[config.feature_columns]
        y = pd.Series([i % 3 for i in range(len(training_df))], name="churn")
        pipeline.fit(X, y)
        threshold, curve_df = find_threshold(pipeline, X, y)
        assert threshold == 0.5
        assert curve_df.empty
