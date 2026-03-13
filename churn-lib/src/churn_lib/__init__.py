"""churn-lib — churn prediction library.

Import from here in scripts, notebooks, CLI tools, or services:

  ChurnPipeline    — train, save, load, and run a pipeline
  PipelineConfig   — build or inspect pipeline configuration
  FeatureSchema    — per-feature name, description, valid ranges/categories
  ModelCard        — model name, version, intended use, and limitations
  predict_single   — score one customer record (with optional threshold)
  predict_batch    — score many records efficiently (with optional threshold)
  PredictionResult — TypedDict describing the prediction output schema
  check_drift      — compare serving data against the training distribution
  DriftReport      — TypedDict describing drift check results
  validate_sample  — validate a single input dict against the feature schema
  validate_batch   — validate a list of input dicts (called automatically by predict_batch)
  ValidationError  — raised when validation fails; subclass of ValueError
"""

from churn_lib.drift import DriftReport, check_drift
from churn_lib.inference import PredictionResult, predict_batch, predict_single
from churn_lib.pipeline import ChurnPipeline, FeatureSchema, ModelCard, PipelineConfig
from churn_lib.validate import ValidationError, validate_batch, validate_sample

__all__ = [
    "ChurnPipeline",
    "DriftReport",
    "FeatureSchema",
    "ModelCard",
    "PipelineConfig",
    "PredictionResult",
    "ValidationError",
    "check_drift",
    "predict_batch",
    "predict_single",
    "validate_batch",
    "validate_sample",
]
