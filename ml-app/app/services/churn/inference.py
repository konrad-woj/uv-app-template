"""Churn inference service — bridge between HTTP schemas and churn-lib.

The service layer translates between the web layer's Pydantic models and the
library's plain-dict API. Keeping this translation out of endpoint handlers
means:
  - Endpoint handlers stay thin and readable (no model_dump() noise there).
  - This file can be unit-tested without a running FastAPI app.
  - Swapping churn-lib for a different inference backend only requires
    changing this file, not the endpoint.

Only churn-lib[predict] deps are used here (pandas, numpy, sklearn, xgboost).
No mlflow, no matplotlib — this module is safe for production inference.
"""

from churn_lib import ChurnPipeline, predict_batch

from app.schemas.churn import CustomerFeatures, PredictionOut


def run_prediction(
    pipeline: ChurnPipeline,
    customers: list[CustomerFeatures],
    threshold: float,
) -> list[PredictionOut]:
    """Score a list of customers and return structured prediction results.

    Converts Pydantic models → plain dicts (what churn-lib expects),
    calls predict_batch (which also runs validate_batch internally),
    then wraps results back into PredictionOut objects.

    Args:
        pipeline:  Loaded ChurnPipeline from app.state.
        customers: Validated customer records from the request body.
        threshold: Decision threshold for the positive (churn) class.

    Returns:
        One PredictionOut per input customer, in the same order as the input.

    Raises:
        ValidationError: If any feature value falls outside its expected range.
                         The caller (endpoint) converts this to a 422 response.
    """
    samples = [c.model_dump() for c in customers]
    results = predict_batch(pipeline, samples, threshold=threshold)
    return [PredictionOut(**r) for r in results]
