"""Churn drift detection service — bridge between the endpoint and churn-lib.

Converts list[CustomerFeatures] inputs to DataFrames (what churn-lib expects)
and forwards the PipelineConfig from the loaded pipeline so drift detection is
always consistent with the features the model was trained on.

Only churn-lib[predict] deps are used here (pandas, numpy). This module is
safe for the inference-only installation.
"""

import pandas as pd
from churn_lib import ChurnPipeline, DriftReport, check_drift

from app.schemas.churn import CustomerFeatures


def run_drift_check(
    pipeline: ChurnPipeline,
    reference: list[CustomerFeatures],
    serving: list[CustomerFeatures],
) -> DriftReport:
    """Compute PSI drift between reference and serving feature distributions.

    The PipelineConfig from the loaded pipeline defines which features to check
    and their expected types — ensuring drift detection always matches what the
    model was trained on, with no manual schema duplication.

    Accepts validated CustomerFeatures objects (same schema as /predict) and
    converts them to plain dicts before building DataFrames. This ensures that
    only well-formed, range-checked feature vectors reach the PSI computation.

    Args:
        pipeline:  Loaded ChurnPipeline (provides the feature schema).
        reference: Training-time distribution as validated customer feature objects.
        serving:   Recent scoring data as validated customer feature objects.

    Returns:
        DriftReport TypedDict from churn-lib with per-feature PSI scores and
        an overall status: 'stable' | 'moderate' | 'major'.
    """
    reference_df = pd.DataFrame([c.model_dump() for c in reference])
    serving_df = pd.DataFrame([c.model_dump() for c in serving])
    return check_drift(reference_df, serving_df, pipeline.config)
