"""Tests for churn_lib.inference — predict_single and predict_batch."""

from __future__ import annotations

import pytest

from churn_lib import ChurnPipeline, ValidationError, predict_batch, predict_single


class TestPredictSingle:
    def test_returns_required_keys(self, fitted_pipeline: ChurnPipeline, valid_sample):
        result = predict_single(fitted_pipeline, valid_sample)
        assert set(result.keys()) == {"prediction", "label", "probabilities"}

    def test_probabilities_sum_to_one(self, fitted_pipeline: ChurnPipeline, valid_sample):
        result = predict_single(fitted_pipeline, valid_sample)
        assert abs(sum(result["probabilities"].values()) - 1.0) < 1e-6

    def test_label_matches_prediction(self, fitted_pipeline: ChurnPipeline, valid_sample):
        result = predict_single(fitted_pipeline, valid_sample)
        assert result["label"] == str(result["prediction"])

    def test_low_threshold_flags_high_risk_as_churn(self, fitted_pipeline: ChurnPipeline, high_risk_sample):
        result = predict_single(fitted_pipeline, high_risk_sample, threshold=0.01)
        assert result["prediction"] == 1

    def test_high_threshold_keeps_low_risk_as_no_churn(self, fitted_pipeline: ChurnPipeline, low_risk_sample):
        result = predict_single(fitted_pipeline, low_risk_sample, threshold=0.99)
        assert result["prediction"] == 0

    def test_missing_feature_raises_validation_error(self, fitted_pipeline: ChurnPipeline, valid_sample):
        del valid_sample["age"]
        with pytest.raises(ValidationError):
            predict_single(fitted_pipeline, valid_sample)


class TestPredictBatch:
    def test_length_matches_input(self, fitted_pipeline: ChurnPipeline, valid_batch):
        assert len(predict_batch(fitted_pipeline, valid_batch)) == len(valid_batch)

    def test_order_preserved(self, fitted_pipeline: ChurnPipeline, high_risk_sample, low_risk_sample):
        """Result index must correspond to input index."""
        results = predict_batch(fitted_pipeline, [high_risk_sample, low_risk_sample])
        assert results[0]["probabilities"]["1"] > results[1]["probabilities"]["1"]

    def test_low_threshold_predicts_more_positives_than_high(self, fitted_pipeline: ChurnPipeline, prediction_df):
        samples = prediction_df.to_dict(orient="records")
        low_pos = sum(r["prediction"] == 1 for r in predict_batch(fitted_pipeline, samples, threshold=0.1))
        high_pos = sum(r["prediction"] == 1 for r in predict_batch(fitted_pipeline, samples, threshold=0.9))
        assert low_pos >= high_pos

    def test_empty_batch_raises_validation_error(self, fitted_pipeline: ChurnPipeline):
        with pytest.raises(ValidationError):
            predict_batch(fitted_pipeline, [])
