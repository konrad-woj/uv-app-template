"""Tests for churn_lib.pipeline — PipelineConfig and ChurnPipeline."""

from __future__ import annotations

import textwrap

import numpy as np
import pytest

from churn_lib import ChurnPipeline, PipelineConfig


class TestPipelineConfig:
    def test_loads_default_config(self, config: PipelineConfig):
        assert config.target == "churn"
        assert len(config.numeric_features) == 5
        assert len(config.categorical_features) == 2
        assert len(config.binary_features) == 2

    def test_feature_columns_order(self, config: PipelineConfig):
        # ColumnTransformer depends on this exact ordering
        expected = config.numeric_features + config.categorical_features + config.binary_features
        assert config.feature_columns == expected

    def test_custom_yaml_path(self, tmp_path):
        yaml_text = textwrap.dedent("""\
            model:
              n_estimators: 50
              max_depth: 3
              learning_rate: 0.05
              random_state: 7
              use_class_weights: false
            features:
              numeric:
                - name: tenure_months
                  min: 0
                  max: 120
              categorical:
                - name: contract_type
                  categories: [month-to-month, one-year]
              binary:
                - name: has_internet
            target: churn
        """)
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml_text)
        cfg = PipelineConfig.from_yaml(config_path)
        assert cfg.target == "churn"
        assert cfg.n_estimators == 50
        assert cfg.use_class_weights is False

    def test_yaml_missing_target_raises(self, tmp_path):
        yaml_text = "features:\n  numeric:\n    - name: tenure_months\n"
        (tmp_path / "bad.yaml").write_text(yaml_text)
        with pytest.raises(KeyError):
            PipelineConfig.from_yaml(tmp_path / "bad.yaml")

    def test_numeric_schemas_have_min_max(self, config: PipelineConfig):
        for schema in config.numeric_schemas:
            assert schema.min is not None, f"{schema.name}: missing min"
            assert schema.max is not None, f"{schema.name}: missing max"
            assert schema.min < schema.max

    def test_categorical_schemas_have_categories(self, config: PipelineConfig):
        for schema in config.categorical_schemas:
            assert schema.categories, f"{schema.name}: missing categories"


class TestChurnPipelineFit:
    def test_classes_and_labels_set_after_fit(self, fitted_pipeline: ChurnPipeline):
        assert set(fitted_pipeline.classes_) == {0, 1}
        assert fitted_pipeline.label_names_ == ["0", "1"]

    def test_fit_with_sample_weight(self, config: PipelineConfig, training_df):
        pipeline = ChurnPipeline(config)
        X = training_df[config.feature_columns]
        y = training_df[config.target]
        pipeline.fit(X, y, sample_weight=np.ones(len(y)))
        assert pipeline.classes_ is not None


class TestChurnPipelinePredict:
    def test_predict_shape_and_values(self, fitted_pipeline: ChurnPipeline, training_df, config):
        X = training_df[config.feature_columns].head(20)
        preds = fitted_pipeline.predict(X)
        assert preds.shape == (20,)
        assert set(preds).issubset({0, 1})

    def test_predict_proba_shape_and_sums(self, fitted_pipeline: ChurnPipeline, training_df, config):
        X = training_df[config.feature_columns].head(30)
        proba = fitted_pipeline.predict_proba(X)
        assert proba.shape == (30, 2)
        assert np.allclose(proba.sum(axis=1), 1.0)


class TestChurnPipelineSaveLoad:
    def test_load_roundtrip_predictions_identical(self, fitted_pipeline: ChurnPipeline, tmp_path, training_df, config):
        path = tmp_path / "model.joblib"
        fitted_pipeline.save(path)
        loaded = ChurnPipeline.load(path)
        X = training_df[config.feature_columns].head(10)
        np.testing.assert_array_equal(fitted_pipeline.predict(X), loaded.predict(X))
        assert loaded.label_names_ == ["0", "1"]

    def test_save_creates_parent_directories(self, fitted_pipeline: ChurnPipeline, tmp_path):
        path = tmp_path / "nested" / "dirs" / "model.joblib"
        fitted_pipeline.save(path)
        assert path.exists()
