"""Feature drift detection using Population Stability Index (PSI).

MLOps principle: a model is only as good as the data it scores. Customer
behaviour shifts over time — new acquisition channels bring different segments,
seasonal patterns change charge amounts, support tooling changes call volumes.
Detecting those shifts early lets you retrigger retraining before model
performance silently degrades.

PSI thresholds (industry standard):
  PSI < 0.10    → stable      (no action needed)
  PSI 0.10-0.25 -> moderate    (investigate; schedule retraining review)
  PSI > 0.25    → major shift (model likely stale; retrain soon)

Usage:
    from churn_lib.drift import check_drift
    from churn_lib.pipeline import PipelineConfig

    cfg = PipelineConfig.from_yaml()
    report = check_drift(reference_df, serving_df, cfg)
    print(report["overall_status"])   # "stable" | "moderate" | "major"

CLI:
    uv run python -m churn_lib.drift --reference train.csv --serving live.csv
"""

from __future__ import annotations

import logging
from typing import TypedDict, cast

import numpy as np
import pandas as pd

from churn_lib.pipeline import PipelineConfig

logger = logging.getLogger(__name__)

_PSI_STABLE = 0.10
_PSI_MODERATE = 0.25
_N_BINS = 10  # number of bins for numeric PSI; coarser = more stable, finer = more sensitive


class FeatureDriftResult(TypedDict):
    """PSI drift statistics for a single feature."""

    feature: str
    feature_type: str  # "numeric" | "categorical" | "binary"
    psi: float
    status: str  # "stable" | "moderate" | "major"


class DriftReport(TypedDict):
    """Full drift report comparing a reference distribution to a serving distribution."""

    reference_samples: int
    serving_samples: int
    features: list[FeatureDriftResult]
    drifted_features: list[str]  # features with status != "stable"
    overall_status: str  # worst-case status across all features


def check_drift(
    reference_df: pd.DataFrame,
    serving_df: pd.DataFrame,
    config: PipelineConfig,
    output_path: str | None = None,
) -> DriftReport:
    """Compute PSI for every feature column and return a structured drift report.

    The reference dataset defines the expected distribution (typically the
    training set). The serving dataset is what the model is currently scoring.

    Args:
        reference_df: The training dataset — defines expected distributions.
        serving_df:   Current scoring data — compared against reference.
        config:       PipelineConfig that names the columns to check.
        output_path:  Optional path to write the report as JSON.

    Returns:
        DriftReport with per-feature PSI scores and an overall status.
    """
    results: list[FeatureDriftResult] = []

    for feature in config.numeric_features:
        if feature not in reference_df.columns or feature not in serving_df.columns:
            logger.warning("Feature missing from one dataset — skipping", extra={"feature": feature})
            continue
        ref_col = cast(pd.Series, reference_df[feature]).dropna()
        serv_col = cast(pd.Series, serving_df[feature]).dropna()
        psi = _psi_numeric(ref_col, serv_col)
        status = _psi_status(psi)
        results.append(FeatureDriftResult(feature=feature, feature_type="numeric", psi=round(psi, 4), status=status))
        logger.debug("Numeric drift", extra={"feature": feature, "psi": round(psi, 4), "status": status})

    for feature in config.categorical_features:
        if feature not in reference_df.columns or feature not in serving_df.columns:
            logger.warning("Feature missing from one dataset — skipping", extra={"feature": feature})
            continue
        ref_col = cast(pd.Series, reference_df[feature]).dropna()
        serv_col = cast(pd.Series, serving_df[feature]).dropna()
        psi = _psi_categorical(ref_col, serv_col)
        status = _psi_status(psi)
        results.append(
            FeatureDriftResult(feature=feature, feature_type="categorical", psi=round(psi, 4), status=status)
        )
        logger.debug("Categorical drift", extra={"feature": feature, "psi": round(psi, 4), "status": status})

    for feature in config.binary_features:
        if feature not in reference_df.columns or feature not in serving_df.columns:
            logger.warning("Feature missing from one dataset — skipping", extra={"feature": feature})
            continue
        # Treat booleans as categorical with two categories
        ref_col = cast(pd.Series, reference_df[feature]).astype(str).dropna()
        serv_col = cast(pd.Series, serving_df[feature]).astype(str).dropna()
        psi = _psi_categorical(ref_col, serv_col)
        status = _psi_status(psi)
        results.append(FeatureDriftResult(feature=feature, feature_type="binary", psi=round(psi, 4), status=status))
        logger.debug("Binary drift", extra={"feature": feature, "psi": round(psi, 4), "status": status})

    drifted = [r["feature"] for r in results if r["status"] != "stable"]
    statuses = [r["status"] for r in results]
    overall = "major" if "major" in statuses else ("moderate" if "moderate" in statuses else "stable")

    report = DriftReport(
        reference_samples=len(reference_df),
        serving_samples=len(serving_df),
        features=results,
        drifted_features=drifted,
        overall_status=overall,
    )

    logger.info(
        "Drift check complete",
        extra={
            "reference_samples": len(reference_df),
            "serving_samples": len(serving_df),
            "n_features_checked": len(results),
            "n_drifted": len(drifted),
            "overall_status": overall,
            "drifted_features": drifted,
        },
    )

    if output_path:
        import json
        from pathlib import Path

        Path(output_path).write_text(json.dumps(report, indent=2, default=str))
        logger.info("Drift report saved", extra={"path": output_path})

    return report


# ---------------------------------------------------------------------------
# PSI computation helpers
# ---------------------------------------------------------------------------


def _psi(ref_probs: np.ndarray, serv_probs: np.ndarray) -> float:
    """PSI = sum((serving% - reference%) * log(serving% / reference%))."""
    eps = 1e-6
    ref_probs = np.clip(ref_probs, eps, None)
    serv_probs = np.clip(serv_probs, eps, None)
    return float(np.sum((serv_probs - ref_probs) * np.log(serv_probs / ref_probs)))


def _psi_numeric(reference: pd.Series, serving: pd.Series, n_bins: int = _N_BINS) -> float:
    """PSI for a numeric feature — bins defined by the reference distribution."""
    _, bin_edges = np.histogram(reference, bins=n_bins)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf
    ref_counts, _ = np.histogram(reference, bins=bin_edges)
    serv_counts, _ = np.histogram(serving, bins=bin_edges)
    return _psi(ref_counts / max(len(reference), 1), serv_counts / max(len(serving), 1))


def _psi_categorical(reference: pd.Series, serving: pd.Series) -> float:
    """PSI for a categorical feature across the union of observed categories."""
    categories = set(reference.unique()) | set(serving.unique())
    ref_freq = reference.value_counts(normalize=True)
    serv_freq = serving.value_counts(normalize=True)
    ref_probs = np.array([ref_freq.get(c, 0.0) for c in categories])
    serv_probs = np.array([serv_freq.get(c, 0.0) for c in categories])
    return _psi(ref_probs, serv_probs)


def _psi_status(psi: float) -> str:
    """Map a PSI value to a human-readable status."""
    if psi < _PSI_STABLE:
        return "stable"
    if psi < _PSI_MODERATE:
        return "moderate"
    return "major"


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    """Compare two CSV/JSON datasets for feature drift from the command line."""
    import argparse
    import json

    from churn_lib._logging import configure_cli_logging

    parser = argparse.ArgumentParser(
        prog="python -m churn_lib.drift",
        description="Compute feature drift (PSI) between a reference and serving dataset.",
    )
    parser.add_argument("--reference", required=True, metavar="PATH", help="Reference (training) dataset CSV.")
    parser.add_argument("--serving", required=True, metavar="PATH", help="Serving (production) dataset CSV.")
    parser.add_argument("--config", metavar="PATH", default=None, help="YAML config (default: built-in).")
    parser.add_argument("--output", metavar="PATH", default=None, help="Save drift report as JSON.")
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: WARNING — set INFO to see per-feature PSI).",
    )
    args = parser.parse_args()

    configure_cli_logging(args.log_level)

    cfg = PipelineConfig.from_yaml(args.config) if args.config else PipelineConfig.from_yaml()
    reference_df = pd.read_csv(args.reference)
    serving_df = pd.read_csv(args.serving)

    report = check_drift(reference_df, serving_df, cfg, output_path=args.output)
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
