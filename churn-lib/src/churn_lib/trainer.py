"""Offline training: split → weight → fit → evaluate → save artifacts.

Every run writes to a fresh timestamped subdirectory under output_dir so
no two runs overwrite each other. Load the resulting pipeline.joblib to serve.

CLI usage:
    uv run python -m churn_lib.trainer --help
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight

from churn_lib.pipeline import ChurnPipeline, PipelineConfig

MLFLOW_EXPERIMENT = "churn-prediction"

logger = logging.getLogger(__name__)


def train(
    pipeline: ChurnPipeline,
    df: pd.DataFrame,
    output_dir: str | Path = "eval_results",
    test_size: float = 0.2,
    random_seed: int = 42,
) -> str:
    """Fit the pipeline on df and persist evaluation artifacts to a timestamped directory.

    Args:
        pipeline:    An unfitted ChurnPipeline.
        df:          Labelled DataFrame — must contain feature columns and the target.
        output_dir:  Parent directory for run artifacts.
        test_size:   Fraction held out for evaluation.
        random_seed: Seed for the train/test split.

    Returns:
        Human-readable summary string with the classification report.
    """
    cfg = pipeline.config
    run_dir = Path(output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "Training run started",
        extra={
            "model": cfg.model_card.name,
            "version": cfg.model_card.version,
            "run_dir": str(run_dir),
        },
    )

    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name=f"{cfg.model_card.name}-v{cfg.model_card.version}"):
        mlflow.log_params(
            {
                "model": cfg.model_card.name,
                "version": cfg.model_card.version,
                "n_features": len(cfg.feature_columns),
                "test_size": test_size,
                "random_seed": random_seed,
                "use_class_weights": cfg.use_class_weights,
            }
        )

        # Step 1: split
        X, y = df[cfg.feature_columns], df[cfg.target]
        class_distribution = {str(cls): int((y == cls).sum()) for cls in np.unique(y)}
        logger.info(
            "Dataset loaded",
            extra={
                "total_samples": len(df),
                "n_features": len(cfg.feature_columns),
                "target": cfg.target,
                "class_distribution": class_distribution,
            },
        )

        # train_test_split stubs return list[Any]; cast each split to its concrete type.
        splits = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_seed)
        X_train = cast(pd.DataFrame, splits[0])
        X_test = cast(pd.DataFrame, splits[1])
        y_train = cast(pd.Series, splits[2])
        y_test = cast(pd.Series, splits[3])
        logger.info(
            "Train/test split complete",
            extra={
                "n_train": len(y_train),
                "n_test": len(y_test),
                "test_size": test_size,
                "stratified": True,
                "random_seed": random_seed,
            },
        )

        # Step 2: class imbalance — weights applied to train set only so the test
        # set keeps the real distribution for honest evaluation.
        sample_weight: np.ndarray | None = None
        if cfg.use_class_weights:
            weights: np.ndarray = compute_sample_weight("balanced", y_train)
            sample_weight = weights
            logger.info(
                "Sample weights computed",
                extra={
                    "strategy": "balanced",
                    "min_weight": round(float(weights.min()), 4),
                    "max_weight": round(float(weights.max()), 4),
                },
            )

        # Step 3: fit
        pipeline.fit(X_train, y_train, sample_weight=sample_weight)

        # Step 4: evaluate
        y_pred = pipeline.predict(X_test)
        labels = pipeline.label_names_ or [str(c) for c in np.unique(y_test)]

        # classification_report stubs return str | dict; cast to dict when output_dict=True.
        report_dict = cast(
            dict[str, Any],
            classification_report(y_test, y_pred, target_names=labels, output_dict=True),
        )
        report_str = cast(str, classification_report(y_test, y_pred, target_names=labels))

        overall: dict[str, Any] = report_dict
        metrics: dict[str, float] = {
            "accuracy": round(float(overall["accuracy"]), 4),
            "macro_f1": round(float(overall["macro avg"]["f1-score"]), 4),
            "weighted_f1": round(float(overall["weighted avg"]["f1-score"]), 4),
        }
        for label in labels:
            if label in report_dict:
                m: dict[str, float] = report_dict[label]
                logger.info(
                    "Class metrics",
                    extra={
                        "class": label,
                        "precision": round(m["precision"], 4),
                        "recall": round(m["recall"], 4),
                        "f1_score": round(m["f1-score"], 4),
                        "support": int(m["support"]),
                    },
                )
                metrics[f"{label}_precision"] = round(m["precision"], 4)
                metrics[f"{label}_recall"] = round(m["recall"], 4)
                metrics[f"{label}_f1"] = round(m["f1-score"], 4)
        logger.info(
            "Overall metrics",
            extra={
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
            },
        )

        # Step 5: threshold calibration (binary only)
        optimal_threshold = 0.5
        if pipeline.classes_ is not None and len(pipeline.classes_) == 2:
            optimal_threshold, threshold_curve = find_threshold(pipeline, X_test, y_test)
            _save_json(threshold_curve.to_dict(orient="records"), run_dir / "threshold_curve.json")  # type: ignore[arg-type]
            logger.info(
                "Threshold calibration complete",
                extra={"optimal_threshold": optimal_threshold, "objective": "f1"},
            )
        metrics["optimal_threshold"] = optimal_threshold
        mlflow.log_metrics(metrics)

        # Step 6: save artifacts
        report_dict["optimal_threshold"] = optimal_threshold
        _save_json(report_dict, run_dir / "metrics.json")
        _save_json(cfg.to_dict(), run_dir / "config.json")
        _save_confusion_matrix(y_test, y_pred, labels, run_dir / "confusion_matrix.png")
        _save_feature_importance(pipeline, run_dir / "feature_importance.png")
        pipeline.save(run_dir / "pipeline.joblib")
        mlflow.log_artifacts(str(run_dir))

        logger.info("All artifacts saved", extra={"run_dir": str(run_dir)})

    return (
        f"Run saved to: {run_dir}\n"
        f"Train samples: {len(y_train)} | Test samples: {len(y_test)}\n"
        f"Classes: {labels} | Optimal threshold (F1): {optimal_threshold}\n\n"
        f"{report_str}"
    )


# ---------------------------------------------------------------------------
# Threshold calibration
# ---------------------------------------------------------------------------


def find_threshold(
    pipeline: ChurnPipeline,
    X: pd.DataFrame,
    y: pd.Series,
    optimize_for: str = "f1",
    min_recall: float | None = None,
    min_precision: float | None = None,
) -> tuple[float, pd.DataFrame]:
    """Sweep decision thresholds and return the operating point that fits your business objective.

    In churn use cases the default 0.5 threshold is rarely optimal — and the
    right operating point depends on costs, not just raw metrics. Two common
    business questions this function can answer:

    - "Catch at least 80% of churners, then make the retention team as precise
      as possible" → pass ``min_recall=0.8`` (maximises precision subject to
      recall >= 0.8).
    - "Only send offers when we're 90% sure the customer will churn, then
      maximise coverage" → pass ``min_precision=0.9`` (maximises recall subject
      to precision >= 0.9).
    - "Just give me the best F1" → leave both as None (default behaviour).

    The full precision/recall/F1 curve is always returned so you can explore
    every possible operating point yourself.

    Only meaningful for binary classifiers; returns (0.5, empty DataFrame) for
    multi-class pipelines where per-class thresholds would be needed.

    Args:
        pipeline:      A fitted ChurnPipeline.
        X:             Feature DataFrame (typically the held-out test set).
        y:             True labels.
        optimize_for:  Metric to maximise when no constraint is given —
                       "f1", "precision", or "recall".
        min_recall:    If set, only consider thresholds where recall >= this
                       value, then maximise precision among those.
        min_precision: If set, only consider thresholds where precision >= this
                       value, then maximise recall among those.

    Returns:
        Tuple of (best_threshold, curve_df) where curve_df has columns
        [threshold, precision, recall, f1].

    Raises:
        ValueError: If both min_recall and min_precision are specified, or if
            the constraint cannot be satisfied by any threshold in the sweep.
    """
    from sklearn.metrics import f1_score, precision_score, recall_score

    if min_recall is not None and min_precision is not None:
        raise ValueError("Specify at most one of min_recall or min_precision, not both.")

    if pipeline.classes_ is None or len(pipeline.classes_) != 2:
        logger.warning("find_threshold requires a binary pipeline — returning default 0.5")
        return 0.5, pd.DataFrame()

    pos_proba: np.ndarray = pipeline.predict_proba(X)[:, 1]
    thresholds = np.round(np.arange(0.05, 0.96, 0.05), 2)

    rows = []
    for t in thresholds:
        preds = (pos_proba >= t).astype(int)
        rows.append(
            {
                "threshold": float(t),
                # zero_division=0: extreme thresholds may predict no positives → metric
                # is undefined (0/0). We treat it as 0.0 silently; no warning needed
                # because we intentionally sweep to extremes.
                "precision": round(float(precision_score(y, preds, zero_division=0)), 4),  # type: ignore[call-overload]
                "recall": round(float(recall_score(y, preds, zero_division=0)), 4),  # type: ignore[call-overload]
                "f1": round(float(f1_score(y, preds, zero_division=0)), 4),  # type: ignore[call-overload]
            }
        )

    curve_df = pd.DataFrame(rows)

    if min_recall is not None:
        candidates = cast(pd.DataFrame, curve_df[curve_df["recall"] >= min_recall])
        if candidates.empty:
            raise ValueError(
                f"No threshold achieves recall >= {min_recall}. Max recall in sweep: {curve_df['recall'].max():.4f}."
            )
        best_idx = int(cast(int, cast(pd.Series, candidates["precision"]).idxmax()))
        objective = f"precision (min_recall={min_recall})"
    elif min_precision is not None:
        candidates = cast(pd.DataFrame, curve_df[curve_df["precision"] >= min_precision])
        if candidates.empty:
            raise ValueError(
                f"No threshold achieves precision >= {min_precision}. "
                f"Max precision in sweep: {curve_df['precision'].max():.4f}."
            )
        best_idx = int(cast(int, cast(pd.Series, candidates["recall"]).idxmax()))
        objective = f"recall (min_precision={min_precision})"
    else:
        best_idx = int(cast(int, curve_df[optimize_for].idxmax()))
        objective = optimize_for

    best_threshold = float(cast(float, curve_df.loc[best_idx, "threshold"]))

    logger.info(
        "Threshold sweep complete",
        extra={
            "objective": objective,
            "optimal_threshold": best_threshold,
            "precision_at_optimal": float(cast(float, curve_df.loc[best_idx, "precision"])),
            "recall_at_optimal": float(cast(float, curve_df.loc[best_idx, "recall"])),
            "f1_at_optimal": float(cast(float, curve_df.loc[best_idx, "f1"])),
        },
    )
    return best_threshold, curve_df


# ---------------------------------------------------------------------------
# Private artifact helpers
# ---------------------------------------------------------------------------


def _save_json(data: dict[str, Any], path: Path) -> None:
    """Serialise data to an indented JSON file at path."""
    path.write_text(json.dumps(data, indent=2, default=str))


def _save_confusion_matrix(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    path: Path,
) -> None:
    """Plot the confusion matrix for y_true vs y_pred and save to path."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels).plot(ax=ax, colorbar=False)
    ax.set_title("Confusion Matrix — test set")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _save_feature_importance(pipeline: ChurnPipeline, path: Path) -> None:
    """Bar-chart of top-20 XGBoost feature importances, saved to path.

    Reconstructs feature names from the ColumnTransformer to account for
    one-hot expanded categorical columns.
    """
    model = pipeline._pipeline.named_steps["model"]
    preprocessor = pipeline._pipeline.named_steps["preprocessor"]

    feature_names: list[str] = []
    for name, transformer, cols in preprocessor.transformers_:
        if name == "cat":
            feature_names.extend(transformer.named_steps["encoder"].get_feature_names_out(cols).tolist())
        else:
            feature_names.extend(cols if isinstance(cols, list) else list(cols))

    importances = model.feature_importances_
    if len(importances) != len(feature_names):
        logger.warning("Feature name/importance length mismatch — skipping plot")
        return

    top_n = 20
    sorted_idx = np.argsort(importances)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(
        [feature_names[i] for i in sorted_idx][::-1],
        importances[sorted_idx][::-1],
    )
    ax.set_xlabel("Importance (gain)")
    ax.set_title(f"Top-{top_n} Feature Importances")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    """Train with defaults or CLI overrides; argparse is imported here only."""
    import argparse

    from churn_lib._logging import configure_cli_logging
    from churn_lib.data_generator import generate_training_data

    parser = argparse.ArgumentParser(
        prog="python -m churn_lib.trainer",
        description="Train a ChurnPipeline on generated data and save evaluation artifacts.",
    )
    parser.add_argument(
        "--config",
        metavar="PATH",
        default=None,
        help="Path to a YAML config file (default: built-in default_config.yaml).",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=5_000,
        metavar="N",
        help="Number of training samples to generate (default: 5000).",
    )
    parser.add_argument(
        "--output-dir",
        default="eval_results",
        metavar="DIR",
        help="Parent directory for run artifacts (default: eval_results).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        metavar="FRAC",
        help="Fraction of data held out for evaluation (default: 0.2).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        metavar="INT",
        help="Random seed for data generation and train/test split (default: 42).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    args = parser.parse_args()

    configure_cli_logging(args.log_level)

    cfg = PipelineConfig.from_yaml(args.config) if args.config else PipelineConfig.from_yaml()
    df = generate_training_data(n_samples=args.n_samples, random_seed=args.seed)
    pipeline = ChurnPipeline(cfg)
    summary = train(
        pipeline,
        df,
        output_dir=args.output_dir,
        test_size=args.test_size,
        random_seed=args.seed,
    )
    print(summary)


if __name__ == "__main__":
    main()
