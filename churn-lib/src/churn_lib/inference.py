"""Score customer records with a fitted ChurnPipeline.

predict_single() is for one record at a time; predict_batch() scores many
records in a single vectorised pass — use it whenever throughput matters.

Both functions accept plain dicts, so they work from any calling context:
REPL, CLI, notebook, or service.

Example
-------
    pipeline = ChurnPipeline.load("eval_results/.../pipeline.joblib")

    result = predict_single(pipeline, {
        "tenure_months": 3, "monthly_charges": 95.0, "num_products": 1,
        "support_calls_last_6m": 5, "age": 34,
        "contract_type": "month-to-month", "payment_method": "electronic_check",
        "has_internet": True, "has_streaming": False,
    })
    # {"prediction": 1, "label": "1", "probabilities": {"0": 0.18, "1": 0.82}}
"""

from __future__ import annotations

import logging
import time
from typing import TypedDict

import numpy as np
import pandas as pd

from churn_lib.pipeline import ChurnPipeline
from churn_lib.validate import validate_batch

logger = logging.getLogger(__name__)


class PredictionResult(TypedDict):
    """A single model prediction, JSON-serialisable as-is.

    prediction:    Raw class value returned by the model (int).
    label:         String label from pipeline.label_names_.
    probabilities: Per-class probabilities keyed by label; values sum to 1.0.
    """

    prediction: int
    label: str
    probabilities: dict[str, float]


def predict_single(
    pipeline: ChurnPipeline,
    sample: dict,
    threshold: float = 0.5,
) -> PredictionResult:
    """Predict churn for a single customer record.

    A convenience wrapper around predict_batch — all logic lives there.

    Args:
        pipeline:  A fitted ChurnPipeline.
        sample:    Feature values as a flat dict.
        threshold: Decision threshold for binary models. Ignored for multi-class.
    """
    logger.debug("predict_single called", extra={"features": list(sample.keys()), "threshold": threshold})
    return predict_batch(pipeline, [sample], threshold=threshold)[0]


def predict_batch(
    pipeline: ChurnPipeline,
    samples: list[dict],
    threshold: float = 0.5,
) -> list[PredictionResult]:
    """Predict churn for a batch of customer records.

    Converts the list of dicts to a DataFrame in one shot so XGBoost can score
    the whole batch in a single vectorised call. Handles any number of classes:
    class values from pipeline.predict() are mapped to labels via a lookup built
    from pipeline.classes_, so the result is correct even when classes are
    non-contiguous (e.g. [1, 2, 3]).

    For binary classification a custom ``threshold`` overrides the default 0.5
    decision boundary. Use trainer.find_threshold() to find the operating point
    that satisfies a business constraint — e.g. maximise precision where
    recall >= 0.8, or maximise recall where precision >= 0.9.
    For multi-class models threshold is ignored and argmax is used; per-class
    thresholds would require a separate calibration step outside this function.

    Args:
        pipeline:  A fitted ChurnPipeline.
        samples:   Feature dicts — all must have the same keys.
        threshold: Decision threshold for the positive class (binary only).

    Returns:
        One PredictionResult per input sample, in the same order.
    """
    validate_batch(samples, pipeline.config)

    classes = pipeline.classes_ if pipeline.classes_ is not None else np.array([0, 1])
    is_binary = len(classes) == 2

    logger.debug(
        "predict_batch started",
        extra={"batch_size": len(samples), "n_classes": len(classes), "threshold": threshold},
    )

    t0 = time.perf_counter()
    df = pd.DataFrame(samples)
    probabilities: np.ndarray = pipeline.predict_proba(df)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    labels: list[str] = pipeline.label_names_ or [str(c) for c in classes]
    class_to_label = dict(zip(classes, labels, strict=False))

    # Apply custom threshold for binary models; fall back to argmax for multi-class.
    if is_binary and threshold != 0.5:
        # probabilities[:, 1] is the positive-class probability
        raw_preds = (probabilities[:, 1] >= threshold).astype(int)
        predictions: np.ndarray = np.array([classes[p] for p in raw_preds])
        if threshold != 0.5:
            logger.debug(
                "Custom threshold applied",
                extra={"threshold": threshold, "binary": True},
            )
    else:
        predictions = pipeline.predict(df)

    results = [
        PredictionResult(
            prediction=int(pred),
            label=class_to_label[pred],
            probabilities={lbl: float(p) for lbl, p in zip(labels, proba, strict=False)},
        )
        for pred, proba in zip(predictions, probabilities, strict=False)
    ]

    class_counts = {class_to_label[cls]: int((predictions == cls).sum()) for cls in classes}
    logger.info(
        "predict_batch complete",
        extra={
            "batch_size": len(samples),
            "n_classes": len(classes),
            "threshold": threshold,
            "elapsed_ms": round(elapsed_ms, 2),
            "throughput_per_sec": (round(len(samples) / (elapsed_ms / 1000), 1) if elapsed_ms > 0 else None),
            "class_distribution": class_counts,
        },
    )
    return results


# ---------------------------------------------------------------------------
# CLI entrypoint — argparse is imported inside main() so it is never loaded
# when this module is imported as a library.
# ---------------------------------------------------------------------------


def main() -> None:
    """Score customer records from the command line using a fitted pipeline."""
    import argparse
    import json
    from pathlib import Path

    from churn_lib._logging import configure_cli_logging
    from churn_lib.data_generator import generate_prediction_data

    parser = argparse.ArgumentParser(
        prog="python -m churn_lib.inference",
        description="Run churn inference on a fitted pipeline.",
    )
    parser.add_argument(
        "--model",
        required=True,
        metavar="PATH",
        help="Path to a fitted pipeline.joblib file.",
    )
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--input",
        metavar="PATH",
        help="CSV or JSON file with customer records to score (no target column).",
    )
    input_group.add_argument(
        "--n-samples",
        type=int,
        default=10,
        metavar="N",
        help="Generate N synthetic samples when no --input is given (default: 10).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        metavar="INT",
        help="Random seed for synthetic sample generation (default: 123).",
    )
    parser.add_argument(
        "--output",
        metavar="PATH",
        default=None,
        help="Write results as JSON to this file instead of stdout.",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: WARNING — set INFO to see inference metrics).",
    )
    args = parser.parse_args()

    configure_cli_logging(args.log_level)

    pipeline = ChurnPipeline.load(args.model)

    if args.input:
        path = Path(args.input)
        df = pd.read_csv(path) if path.suffix == ".csv" else pd.read_json(path)
        samples = df.to_dict(orient="records")
        logger.info("Input loaded", extra={"source": str(path), "n_samples": len(samples)})
    else:
        df = generate_prediction_data(n_samples=args.n_samples, random_seed=args.seed)
        samples = df.to_dict(orient="records")
        logger.info(
            "Synthetic input generated",
            extra={"n_samples": len(samples), "seed": args.seed},
        )

    results = predict_batch(pipeline, samples)
    output = json.dumps(results, indent=2)

    if args.output:
        Path(args.output).write_text(output)
        print(f"Results written to {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
