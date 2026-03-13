"""Churn drift detection service — bridge between the endpoint and churn-lib.

Converts list[dict] inputs to DataFrames (what churn-lib expects) and forwards
the PipelineConfig from the loaded pipeline so drift detection is always
consistent with the features the model was trained on.

Only churn-lib[predict] deps are used here (pandas, numpy). This module is
safe for the inference-only installation.
"""

import pandas as pd
from churn_lib import ChurnPipeline, DriftReport, check_drift


def run_drift_check(
    pipeline: ChurnPipeline,
    reference: list[dict],
    serving: list[dict],
) -> DriftReport:
    """Compute PSI drift between reference and serving feature distributions.

    The PipelineConfig from the loaded pipeline defines which features to check
    and their expected types — ensuring drift detection always matches what the
    model was trained on, with no manual schema duplication.

    Args:
        pipeline:  Loaded ChurnPipeline (provides the feature schema).
        reference: Training-time distribution as a list of feature dicts.
        serving:   Recent scoring data as a list of feature dicts.

    Returns:
        DriftReport TypedDict from churn-lib with per-feature PSI scores and
        an overall status: 'stable' | 'moderate' | 'major'.
    """
    reference_df = pd.DataFrame(reference)
    serving_df = pd.DataFrame(serving)
    return check_drift(reference_df, serving_df, pipeline.config)
