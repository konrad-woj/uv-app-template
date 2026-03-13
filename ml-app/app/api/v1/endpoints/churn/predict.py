"""Churn prediction endpoints.

Routes:
    POST /api/v1/churn/predict  — score one or more customer records
    GET  /api/v1/churn/schema   — return the feature schema for the loaded model

Design notes:
  - The pipeline is injected via Depends(get_pipeline) — it is never
    constructed inside the handler, making the function thin and testable.
  - ValidationError from churn-lib surfaces as a 422 with a descriptive message;
    it is logged at WARNING (client error, not our bug).
  - All other exceptions propagate to the global handler in app/core/errors.py
    which logs them at ERROR and returns a clean 500 response.
  - Both endpoints require a loaded model. If none is loaded, get_pipeline()
    raises a 503 before the handler is called.
"""

import logging

from churn_lib import ChurnPipeline, ValidationError
from fastapi import APIRouter, Depends, HTTPException, status

from app.core.config import Settings
from app.core.dependencies import get_pipeline, get_settings
from app.schemas.churn import (
    FeatureSchemaOut,
    FeaturesOut,
    ModelCardOut,
    PredictRequest,
    PredictResponse,
    SchemaResponse,
)
from app.services.churn.inference import run_prediction

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/predict",
    response_model=PredictResponse,
    summary="Score customer records for churn probability",
)
async def predict(
    body: PredictRequest,
    pipeline: ChurnPipeline = Depends(get_pipeline),
    settings: Settings = Depends(get_settings),
) -> PredictResponse:
    """Score one or more customer records and return churn probabilities.

    All records are scored in a single vectorised pass — send large batches
    rather than making one request per customer.

    **Example request:**
    ```json
    {
      "customers": [
        {
          "tenure_months": 3,
          "monthly_charges": 95.0,
          "num_products": 1,
          "support_calls_last_6m": 5,
          "age": 34,
          "contract_type": "month-to-month",
          "payment_method": "electronic_check",
          "has_internet": true,
          "has_streaming": false
        }
      ],
      "threshold": 0.4
    }
    ```

    **Response fields:**
    - `prediction`: 0 (no churn) or 1 (churn).
    - `label`: string label from the model ("0" or "1").
    - `probabilities`: per-class probability dict, e.g. `{"0": 0.32, "1": 0.68}`.
    - `threshold_used`: the threshold that was applied to convert probabilities
      to binary predictions — useful for auditing.

    **Threshold guidance:**
    The server default threshold is set via `CHURN_DEFAULT_THRESHOLD` (default: 0.5).
    Override it per-request via the `threshold` field. Use `optimal_threshold` from a
    training run (`POST /api/v1/churn/train`) for a data-driven operating point.
    """
    threshold = body.threshold if body.threshold is not None else settings.default_threshold
    try:
        predictions = run_prediction(pipeline, body.customers, threshold)
    except ValidationError as exc:
        logger.warning("Prediction validation failed", extra={"error": str(exc), "batch_size": len(body.customers)})
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=str(exc),
        ) from exc

    return PredictResponse(
        predictions=predictions,
        model_version=pipeline.config.model_card.version,
        threshold_used=threshold,
    )


@router.get(
    "/schema",
    response_model=SchemaResponse,
    summary="Return the feature schema for the loaded model",
)
async def feature_schema(
    pipeline: ChurnPipeline = Depends(get_pipeline),
    settings: Settings = Depends(get_settings),
) -> SchemaResponse:
    """Return the feature constraints (names, types, valid ranges/categories).

    Use this to understand exactly what the `/predict` endpoint expects without
    reading the source code. The schema travels with the model — it reflects
    exactly what the loaded pipeline was trained on.

    **Example response:**
    ```json
    {
      "default_threshold": 0.5,
      "model": {"name": "Churn Prediction Model", "version": "1.0.0", ...},
      "features": {
        "numeric": [{"name": "tenure_months", "min": 0, "max": 120, ...}],
        "categorical": [{"name": "contract_type", "categories": [...], ...}],
        "binary": [{"name": "has_internet", ...}]
      }
    }
    ```
    """
    cfg = pipeline.config
    return SchemaResponse(
        default_threshold=settings.default_threshold,
        model=ModelCardOut(
            name=cfg.model_card.name,
            version=cfg.model_card.version,
            description=cfg.model_card.description,
            intended_use=cfg.model_card.intended_use,
            limitations=cfg.model_card.limitations,
        ),
        features=FeaturesOut(
            numeric=[
                FeatureSchemaOut(name=s.name, description=s.description, min=s.min, max=s.max)
                for s in cfg.numeric_schemas
            ],
            categorical=[
                FeatureSchemaOut(name=s.name, description=s.description, categories=s.categories)
                for s in cfg.categorical_schemas
            ],
            binary=[FeatureSchemaOut(name=s.name, description=s.description) for s in cfg.binary_schemas],
        ),
    )
