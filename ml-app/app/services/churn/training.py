"""Churn training service — orchestrates a full training run via churn-lib.

Called by both the synchronous and asynchronous training endpoints so that
both share exactly the same training path. Keeping orchestration here (not in
the endpoint) means:
  - Endpoint handlers stay thin.
  - The training workflow can be tested in isolation.
  - Switching to a different data source or model library only requires
    changing this file.

Heavy imports (mlflow, matplotlib) are only triggered when this module is
imported. This module is only imported when the training router is registered,
which only happens when churn-lib[train] is installed (see api/v1/router.py).
The inference-only process never touches these imports.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from churn_lib import ChurnPipeline, PipelineConfig
from churn_lib.data_generator import generate_training_data
from churn_lib.trainer import train

from app.core.config import settings
from app.schemas.churn import TrainResponse

if TYPE_CHECKING:
    from fastapi import FastAPI

logger = logging.getLogger(__name__)


def run_training(
    app: FastAPI,
    n_samples: int,
    output_dir: str,
    test_size: float,
    random_seed: int,
) -> TrainResponse:
    """Execute a full training run and hot-reload the new model into app.state.

    Steps:
        1. Configure MLflow tracking URI if CHURN_MLFLOW_TRACKING_URI is set.
        2. Generate synthetic training data via churn-lib.
        3. Build a ChurnPipeline from the default config.
        4. Train, evaluate, and save artifacts (delegates to churn_lib.trainer.train).
        5. Parse metrics from the run directory written by the trainer.
        6. Hot-reload: attach the new pipeline to app.state so subsequent
           /predict calls use the freshly trained model immediately — no restart needed.

    Args:
        app:        The FastAPI application instance (for hot-reloading app.state).
        n_samples:  Number of synthetic training samples to generate.
        output_dir: Parent directory; the trainer creates a timestamped subdirectory.
        test_size:  Fraction of data held out for evaluation.
        random_seed: Reproducibility seed.

    Returns:
        TrainResponse with the classification report, key metrics, model path,
        and the calibrated decision threshold.
    """
    # MLflow tracking URI — set as env var so the trainer picks it up without
    # any coupling between this service and the mlflow import in the trainer.
    if settings.mlflow_tracking_uri:
        os.environ.setdefault("MLFLOW_TRACKING_URI", settings.mlflow_tracking_uri)
        logger.info("MLflow tracking URI set", extra={"uri": settings.mlflow_tracking_uri})

    logger.info(
        "Training run started",
        extra={"n_samples": n_samples, "output_dir": output_dir, "test_size": test_size},
    )

    cfg = PipelineConfig.from_yaml()
    df = generate_training_data(n_samples=n_samples, random_seed=random_seed)
    pipeline = ChurnPipeline(cfg)

    # train() fits the pipeline, evaluates on the hold-out set, saves all
    # artifacts to a timestamped subdirectory, and returns a summary string
    # whose first line is "Run saved to: <run_dir>".
    summary = train(pipeline, df, output_dir=output_dir, test_size=test_size, random_seed=random_seed)

    # Parse the run directory from the first line of the summary string, e.g.:
    #   "Run saved to: ./models/20240101_120000"
    # Using the summary rather than scanning the directory makes this safe for
    # concurrent training runs that write to the same output_dir.
    first_line = summary.split("\n")[0]
    if not first_line.startswith("Run saved to: "):
        raise RuntimeError(
            f"Unexpected trainer output format — could not locate run directory. First line was: {first_line!r}"
        )
    run_dir = Path(first_line.removeprefix("Run saved to: ").strip())
    model_path = run_dir / "pipeline.joblib"

    # Read the metrics.json saved by the trainer (classification report + threshold).
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise RuntimeError(
            f"metrics.json not found in run directory {run_dir}. "
            "The training run may have failed before saving artifacts."
        )
    raw: dict[str, Any] = json.loads(metrics_path.read_text())
    optimal_threshold = float(raw.get("optimal_threshold", 0.5))

    # Extract the flat numeric metrics from the nested classification report.
    # The report has keys like "accuracy" (float) and "macro avg" (dict).
    metrics: dict[str, float] = {}
    for key, value in raw.items():
        if isinstance(value, (int, float)) and key != "optimal_threshold":
            metrics[key] = float(value)
        elif isinstance(value, dict) and "f1-score" in value:
            # Class-level or average rows: flatten as "<key>_f1", etc.
            label = key.replace(" ", "_")
            metrics[f"{label}_precision"] = float(value.get("precision", 0.0))
            metrics[f"{label}_recall"] = float(value.get("recall", 0.0))
            metrics[f"{label}_f1"] = float(value.get("f1-score", 0.0))

    # Hot-reload: the new model is live for all subsequent requests.
    # No restart required — FastAPI stores it on app.state which is shared
    # across all requests in this process.
    app.state.pipeline = pipeline
    logger.info("Model hot-reloaded into app.state", extra={"model_path": str(model_path)})

    return TrainResponse(
        status="ok",
        summary=summary,
        model_path=str(model_path.resolve()),
        metrics=metrics,
        optimal_threshold=optimal_threshold,
    )
