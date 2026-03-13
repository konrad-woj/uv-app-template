"""Churn drift detection endpoint.

Route:
    POST /api/v1/churn/drift — compare reference vs serving feature distributions

Drift detection is a critical MLOps practice. Models degrade silently when the
real-world data they score diverges from the distribution they were trained on.
This endpoint wraps churn-lib's PSI (Population Stability Index) check, which
flags features whose distribution has shifted beyond configurable thresholds.

PSI thresholds (industry standard):
  PSI < 0.10    → stable      (no action needed)
  PSI 0.10–0.25 → moderate    (investigate; schedule retraining review)
  PSI > 0.25    → major shift (model likely stale — retrain soon)

When to integrate this endpoint:
  Schedule it in your monitoring pipeline (e.g. weekly cron). Pass a sample of
  the original training data as `reference` and recent scoring data as `serving`.
  Alert when `overall_status` is "moderate" or "major".
"""

from churn_lib import ChurnPipeline
from fastapi import APIRouter, Depends

from app.core.dependencies import get_pipeline
from app.schemas.churn import DriftRequest, DriftResponse, FeatureDriftOut
from app.services.churn.drift import run_drift_check

router = APIRouter()


@router.post(
    "/drift",
    response_model=DriftResponse,
    summary="Detect feature distribution shift between reference and serving data (PSI)",
)
async def detect_drift(
    body: DriftRequest,
    pipeline: ChurnPipeline = Depends(get_pipeline),
) -> DriftResponse:
    """Compare a reference dataset against recent serving data for distribution shift.

    The feature schema comes from the loaded pipeline, so drift detection is
    always consistent with what the model was trained on — no manual schema
    duplication required.

    **PSI interpretation:**
    | PSI range    | Status   | Recommended action                        |
    |------------- |----------|-------------------------------------------|
    | < 0.10       | stable   | No action needed                          |
    | 0.10 – 0.25  | moderate | Investigate; schedule retraining review   |
    | > 0.25       | major    | Model likely stale — retrain soon         |

    **Typical monitoring workflow:**
    1. Store a sample of your training data (or generate it via churn-lib's
       `generate_training_data()`).
    2. Collect a rolling window of recent scoring inputs (last 7 or 30 days).
    3. POST both here weekly; alert on `overall_status` != "stable".
    4. When drift is detected, POST `/api/v1/churn/train` to retrain.

    **Example request:**
    ```json
    {
      "reference": [
        {
          "tenure_months": 24, "monthly_charges": 45.0, "num_products": 2,
          "support_calls_last_6m": 1, "age": 42,
          "contract_type": "two_year", "payment_method": "bank_transfer",
          "has_internet": true, "has_streaming": false
        }
      ],
      "serving": [
        {
          "tenure_months": 3, "monthly_charges": 95.0, "num_products": 1,
          "support_calls_last_6m": 7, "age": 29,
          "contract_type": "month-to-month", "payment_method": "electronic_check",
          "has_internet": true, "has_streaming": true
        }
      ]
    }
    ```
    """
    report = run_drift_check(pipeline, body.reference, body.serving)

    return DriftResponse(
        overall_status=report["overall_status"],
        drifted_features=report["drifted_features"],
        reference_samples=report["reference_samples"],
        serving_samples=report["serving_samples"],
        features=[FeatureDriftOut(**f) for f in report["features"]],
    )
