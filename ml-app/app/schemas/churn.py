"""Pydantic schemas for all churn-related endpoints.

Keeping schemas here — separate from endpoint handlers — means:
  - They can be imported by tests without loading a running FastAPI app.
  - Multiple endpoint files can share the same schema without circular imports.
  - The full data contract is visible in one place.

Naming convention:
  *Request  — what the client sends in the request body.
  *Response — what the service returns in the response body.
  *Out      — a single item inside a list response.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------


class CustomerFeatures(BaseModel):
    """Input features for a single customer record.

    Field ranges mirror the constraints defined in churn-lib's
    default_config.yaml. Out-of-range or missing values are rejected by
    churn-lib's validate_batch() before they reach the model, so you get
    a clear error message rather than a silent wrong prediction.
    """

    tenure_months: Annotated[float, Field(ge=0, le=120, description="Months since subscription start.")]
    monthly_charges: Annotated[float, Field(ge=0, le=200, description="Current monthly bill in USD.")]
    num_products: Annotated[int, Field(ge=1, le=10, description="Number of active products/services.")]
    support_calls_last_6m: Annotated[int, Field(ge=0, le=20, description="Support calls in the last 6 months.")]
    age: Annotated[int, Field(ge=18, le=100, description="Customer age in years.")]
    contract_type: Annotated[
        str,
        Field(description="Contract type: 'month-to-month', 'one_year', or 'two_year'."),
    ]
    payment_method: Annotated[
        str,
        Field(description=("Payment method: 'electronic_check', 'mailed_check', 'bank_transfer', or 'credit_card'.")),
    ]
    has_internet: Annotated[bool, Field(description="Whether the customer has an internet subscription.")]
    has_streaming: Annotated[bool, Field(description="Whether the customer has a streaming add-on.")]


class PredictRequest(BaseModel):
    """Batch prediction request — score one or more customer records in one call."""

    customers: Annotated[
        list[CustomerFeatures],
        Field(
            min_length=1,
            max_length=1000,
            description=(
                "Customer records to score. At least one required; at most 1000 per request. "
                "All records are scored in a single vectorised pass — larger batches are more "
                "efficient than many small requests. For datasets exceeding 1000 records, "
                "split into chunks of ≤1000 and call /predict for each chunk."
            ),
        ),
    ]
    threshold: Annotated[
        float,
        Field(
            default=0.5,
            ge=0.0,
            le=1.0,
            description=(
                "Decision threshold for the positive (churn) class. "
                "Lower values catch more churners (higher recall, more false positives). "
                "Higher values are more conservative (higher precision, fewer false alarms). "
                "Use `optimal_threshold` from a training run for a data-driven operating point."
            ),
        ),
    ] = 0.5


class PredictionOut(BaseModel):
    """Prediction result for a single customer."""

    prediction: int = Field(description="Predicted class value (0 = no churn, 1 = churn).")
    label: str = Field(description="String label corresponding to the prediction.")
    probabilities: dict[str, float] = Field(description="Per-class probabilities keyed by label. Values sum to 1.0.")


class PredictResponse(BaseModel):
    """Batch prediction response."""

    predictions: list[PredictionOut]
    model_version: str | None = Field(
        default=None,
        description="Model card version from the loaded pipeline.",
    )
    threshold_used: float = Field(description="The decision threshold applied to this request.")


# ---------------------------------------------------------------------------
# Train (synchronous)
# ---------------------------------------------------------------------------


class TrainRequest(BaseModel):
    """Training run configuration.

    All fields are optional — the defaults reproduce the churn-lib baseline
    exactly, so posting an empty body `{}` immediately yields a trained model.
    """

    n_samples: Annotated[
        int,
        Field(default=5_000, ge=100, description="Synthetic training samples to generate."),
    ] = 5_000
    output_dir: Annotated[
        str,
        Field(
            default="models",
            description=(
                "Parent directory for run artifacts. "
                "The trainer creates a timestamped subdirectory per run "
                "so no two runs overwrite each other."
            ),
        ),
    ] = "models"
    test_size: Annotated[
        float,
        Field(default=0.2, gt=0.0, lt=1.0, description="Fraction of data held out for evaluation."),
    ] = 0.2
    random_seed: Annotated[
        int,
        Field(default=42, description="Reproducibility seed for data generation and train/test split."),
    ] = 42


class TrainResponse(BaseModel):
    """Result of a completed training run."""

    status: str = Field(description="'ok' on success.")
    summary: str = Field(description="Human-readable classification report from churn-lib.")
    model_path: str = Field(description="Absolute path to the saved pipeline.joblib.")
    metrics: dict[str, float] = Field(description="Key evaluation metrics (accuracy, F1, threshold).")
    optimal_threshold: float = Field(
        description=(
            "Decision threshold that maximises F1 on the hold-out set. "
            "Pass this as `threshold` in /predict to use the calibrated operating point "
            "rather than the default 0.5."
        )
    )


# ---------------------------------------------------------------------------
# Train (asynchronous) — job tracking
# ---------------------------------------------------------------------------


class TrainJobResponse(BaseModel):
    """Acknowledgement returned immediately when an async training job is submitted."""

    job_id: str = Field(description="UUID identifying this background training run.")
    status: str = Field(description="Initial status — always 'pending' at submission time.")
    message: str = Field(description="Human-readable description of next steps.")
    poll_url: str = Field(description="GET this URL to check job status and retrieve results.")


class JobStatusResponse(BaseModel):
    """Current state of a background training job."""

    job_id: str = Field(description="UUID identifying this background training run.")
    status: str = Field(description="'pending' | 'running' | 'done' | 'failed'")
    result: TrainResponse | None = Field(
        default=None,
        description="Populated when status is 'done'. Contains metrics and model path.",
    )
    error: str | None = Field(
        default=None,
        description="Error message when status is 'failed'.",
    )


# ---------------------------------------------------------------------------
# Drift
# ---------------------------------------------------------------------------


class FeatureDriftOut(BaseModel):
    """PSI drift result for a single feature."""

    feature: str = Field(description="Feature column name.")
    feature_type: str = Field(description="'numeric' | 'categorical' | 'binary'")
    psi: float = Field(description="Population Stability Index score.")
    status: str = Field(description="'stable' (PSI < 0.10) | 'moderate' (0.10–0.25) | 'major' (> 0.25)")


class DriftRequest(BaseModel):
    """Drift detection request — two samples of customer feature data."""

    reference: Annotated[
        list[dict],
        Field(
            min_length=1,
            description=(
                "Reference (training-time) distribution. "
                "Typically a sample from the dataset used to train the current model."
            ),
        ),
    ]
    serving: Annotated[
        list[dict],
        Field(
            min_length=1,
            description="Recent scoring data to compare against the reference.",
        ),
    ]


class DriftResponse(BaseModel):
    """Full PSI drift report comparing reference vs serving distributions."""

    overall_status: str = Field(description="Worst-case status across all features: 'stable' | 'moderate' | 'major'")
    drifted_features: list[str] = Field(description="Names of features with status != 'stable'.")
    reference_samples: int = Field(description="Number of records in the reference dataset.")
    serving_samples: int = Field(description="Number of records in the serving dataset.")
    features: list[FeatureDriftOut]


# ---------------------------------------------------------------------------
# Schema (GET /api/v1/churn/schema)
# ---------------------------------------------------------------------------


class FeatureSchemaOut(BaseModel):
    """Schema for a single input feature — name, type constraints, and description."""

    name: str = Field(description="Column name expected in the request body.")
    description: str = Field(description="Human-readable explanation of what the feature measures.")
    min: float | None = Field(default=None, description="Inclusive lower bound for numeric features.")
    max: float | None = Field(default=None, description="Inclusive upper bound for numeric features.")
    categories: list[str] | None = Field(
        default=None,
        description="Exhaustive list of valid string values for categorical features.",
    )


class ModelCardOut(BaseModel):
    """High-level documentation about the loaded model."""

    name: str = Field(description="Display name of the model.")
    version: str = Field(description="Semantic version — bumped on every retrain.")
    description: str = Field(description="What the model predicts and how.")
    intended_use: str = Field(description="The specific business problem it solves.")
    limitations: str = Field(description="Known failure modes, data assumptions, or restrictions.")


class FeaturesOut(BaseModel):
    """All feature schemas, grouped by type."""

    numeric: list[FeatureSchemaOut] = Field(description="Continuous numeric features (min/max bounded).")
    categorical: list[FeatureSchemaOut] = Field(description="Categorical features with a fixed set of allowed values.")
    binary: list[FeatureSchemaOut] = Field(description="Boolean features (true/false).")


class SchemaResponse(BaseModel):
    """Feature schema and model card for the currently loaded pipeline.

    Use this response to understand exactly what /predict expects, without
    reading the source code. The schema travels with the model artifact —
    it always reflects the features the loaded pipeline was trained on.

    Example use cases:
      - Validate client-side feature engineering against the live schema.
      - Auto-generate prediction request payloads in downstream services.
      - Build UI forms for manual customer record entry.
    """

    default_threshold: float = Field(
        description=(
            "Server-side default decision threshold (CHURN_DEFAULT_THRESHOLD env var). "
            "Per-request overrides via the `threshold` field in /predict."
        )
    )
    model: ModelCardOut
    features: FeaturesOut
