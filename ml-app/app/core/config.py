"""Application-wide settings, loaded from environment variables or a .env file.

Pydantic-settings reads env vars with the CHURN_ prefix automatically. You can
also create a .env file next to pyproject.toml for local development — it is
gitignored by convention.

Example .env:
    CHURN_MODEL_PATH=./models/20240101_120000/pipeline.joblib
    CHURN_MLFLOW_TRACKING_URI=http://localhost:5000
    CHURN_DEFAULT_THRESHOLD=0.4
    CHURN_LOG_LEVEL=DEBUG
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CHURN_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    model_path: Path | None = Field(
        default=None,
        description=(
            "Path to a fitted pipeline.joblib. "
            "When set, the model is loaded into memory at startup. "
            "If unset, the app starts without a model — call POST /api/v1/churn/train first."
        ),
    )
    mlflow_tracking_uri: str | None = Field(
        default=None,
        description=(
            "MLflow tracking server URI (e.g. http://localhost:5000). "
            "If unset, MLflow logs to the local ./mlruns directory. "
            "Only relevant when the [train] extra is installed."
        ),
    )
    log_level: str = Field(
        default="INFO",
        description=(
            "Logging verbosity for the application. "
            "Accepts standard Python level names: DEBUG, INFO, WARNING, ERROR. "
            "Logs are emitted as structured JSON using churn-lib's JsonFormatter."
        ),
    )
    default_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Default decision threshold for binary churn classification. "
            "Overridable per-request via the `threshold` field in /predict."
        ),
    )


# Module-level singleton — import this wherever settings are needed.
# FastAPI's dependency injection is not required for config; it's simpler to
# import the singleton directly in service modules.
settings = Settings()
