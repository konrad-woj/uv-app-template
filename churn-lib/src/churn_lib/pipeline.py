"""ChurnPipeline: config, preprocessing, and model in one serialisable object.

Sklearn's Pipeline keeps preprocessing and model together so they travel as
one joblib file — eliminating any risk of train/serve skew. PipelineConfig is
a plain dataclass so it can be diffed, versioned, and logged without ceremony.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = Path(__file__).parent / "default_config.yaml"


# ---------------------------------------------------------------------------
# Feature schema
# ---------------------------------------------------------------------------


@dataclass
class FeatureSchema:
    """Schema for a single input feature — name, description, and constraints.

    Constraints are enforced by validate_batch() before scoring, so invalid
    inputs are rejected with a clear message rather than silently producing
    wrong predictions.

    Attributes:
        name:        Column name expected in input dicts / DataFrames.
        description: Human-readable explanation of what the feature measures.
        min:         Inclusive lower bound for numeric features (None = unconstrained).
        max:         Inclusive upper bound for numeric features (None = unconstrained).
        categories:  Exhaustive list of valid values for categorical features
                     (None = any non-null value accepted).
    """

    name: str
    description: str = ""
    min: float | None = None
    max: float | None = None
    categories: list[str] | None = None


# ---------------------------------------------------------------------------
# Model card
# ---------------------------------------------------------------------------


@dataclass
class ModelCard:
    """High-level documentation that travels with every saved pipeline.

    A model card is the standard way to communicate what a model does, who
    it's for, and where it should NOT be used — a concept introduced by Google
    in 2019 and now standard practice in production ML.

    Attributes:
        name:         Short display name for the model.
        version:      Semantic version string — bump on every retrain.
        description:  What the model predicts and how.
        intended_use: The specific business problem it solves.
        limitations:  Known failure modes, data assumptions, or restrictions.
    """

    name: str = "Churn Prediction Model"
    version: str = "1.0.0"
    description: str = ""
    intended_use: str = ""
    limitations: str = ""


# ---------------------------------------------------------------------------
# Pipeline config
# ---------------------------------------------------------------------------


@dataclass
class PipelineConfig:
    """All parameters that define a ChurnPipeline.

    Keeping config separate from the fitted object makes it easy to log,
    serialise to JSON, and compare across experiments. Feature schemas carry
    validation constraints (ranges, allowed categories) that are enforced
    before any data touches the model.
    """

    numeric_schemas: list[FeatureSchema]
    categorical_schemas: list[FeatureSchema]
    binary_schemas: list[FeatureSchema]
    target: str
    model_card: ModelCard = field(default_factory=ModelCard)
    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1
    random_state: int = 42
    use_class_weights: bool = True

    # ------------------------------------------------------------------
    # Convenience properties — rest of codebase uses these string lists
    # ------------------------------------------------------------------

    @property
    def numeric_features(self) -> list[str]:
        """Ordered list of numeric column names."""
        return [s.name for s in self.numeric_schemas]

    @property
    def categorical_features(self) -> list[str]:
        """Ordered list of categorical column names."""
        return [s.name for s in self.categorical_schemas]

    @property
    def binary_features(self) -> list[str]:
        """Ordered list of binary column names."""
        return [s.name for s in self.binary_schemas]

    @property
    def feature_columns(self) -> list[str]:
        """All input columns in the fixed order expected by the ColumnTransformer."""
        return self.numeric_features + self.categorical_features + self.binary_features

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable representation of the full config including schemas."""
        return asdict(self)

    @classmethod
    def from_yaml(cls, path: str | Path = _DEFAULT_CONFIG_PATH) -> PipelineConfig:
        """Load config from a YAML file, falling back to dataclass defaults for missing keys.

        Feature entries may be plain strings (legacy) or dicts with name,
        description, min, max, categories fields.
        """
        with open(path) as f:
            raw = yaml.safe_load(f)

        features = raw.get("features", {})
        model = raw.get("model", {})
        card_raw = raw.get("model_card", {})

        return cls(
            numeric_schemas=_parse_schemas(features.get("numeric", [])),
            categorical_schemas=_parse_schemas(features.get("categorical", [])),
            binary_schemas=_parse_schemas(features.get("binary", [])),
            target=raw["target"],
            model_card=ModelCard(
                name=card_raw.get("name", "Churn Prediction Model"),
                version=str(card_raw.get("version", "1.0.0")),
                description=card_raw.get("description", "").strip(),
                intended_use=card_raw.get("intended_use", "").strip(),
                limitations=card_raw.get("limitations", "").strip(),
            ),
            n_estimators=model.get("n_estimators", 100),
            max_depth=model.get("max_depth", 6),
            learning_rate=model.get("learning_rate", 0.1),
            random_state=model.get("random_state", 42),
            use_class_weights=model.get("use_class_weights", True),
        )


def _parse_schemas(raw: list[Any]) -> list[FeatureSchema]:
    """Parse a YAML feature list into FeatureSchema objects.

    Accepts both plain strings (legacy: ``- tenure_months``) and dicts
    (new: ``- name: tenure_months\n  min: 0\n  max: 120``).
    """
    schemas = []
    for item in raw:
        if isinstance(item, str):
            schemas.append(FeatureSchema(name=item))
        else:
            schemas.append(
                FeatureSchema(
                    name=item["name"],
                    description=item.get("description", ""),
                    min=item.get("min"),
                    max=item.get("max"),
                    categories=item.get("categories"),
                )
            )
    return schemas


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class ChurnPipeline:
    """Preprocessing + XGBoost classifier, packaged as one serialisable object.

    Example
    -------
    >>> cfg = PipelineConfig.from_yaml()
    >>> pipeline = ChurnPipeline(cfg)
    >>> pipeline.fit(X_train, y_train)
    >>> preds = pipeline.predict(X_test)
    >>> pipeline.save("model.joblib")
    >>> loaded = ChurnPipeline.load("model.joblib")
    """

    def __init__(self, config: PipelineConfig) -> None:
        """Build the unfitted sklearn Pipeline from config. No data required yet."""
        self.config = config
        self._pipeline: Pipeline = self._build()
        self.classes_: np.ndarray | None = None  # set by fit()
        self.label_names_: list[str] | None = None  # string versions of classes_

    def _build(self) -> Pipeline:
        """Construct the ColumnTransformer + XGBClassifier sklearn Pipeline."""
        cfg = self.config
        logger.debug(
            "Building pipeline",
            extra={
                "n_numeric": len(cfg.numeric_features),
                "n_categorical": len(cfg.categorical_features),
                "n_binary": len(cfg.binary_features),
                "n_estimators": cfg.n_estimators,
                "max_depth": cfg.max_depth,
                "learning_rate": cfg.learning_rate,
            },
        )

        numeric_transformer = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        categorical_transformer = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
            ]
        )
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, cfg.numeric_features),
                ("cat", categorical_transformer, cfg.categorical_features),
                ("bin", "passthrough", cfg.binary_features),  # booleans → XGBoost as-is
            ],
            remainder="drop",
        )
        model = XGBClassifier(
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            learning_rate=cfg.learning_rate,
            eval_metric="logloss",
            random_state=cfg.random_state,
            verbosity=0,
        )
        return Pipeline([("preprocessor", preprocessor), ("model", model)])

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: np.ndarray | None = None,
    ) -> ChurnPipeline:
        """Fit preprocessing and model end-to-end.

        Args:
            X: Feature DataFrame — must contain all columns in config.feature_columns.
            y: Target Series — any number of classes is supported.
            sample_weight: Optional per-sample weights (e.g. from compute_sample_weight).

        Returns:
            self, so calls can be chained.
        """
        self.classes_ = np.unique(y)
        self.label_names_ = [str(c) for c in self.classes_]

        class_counts = {str(cls): int((y == cls).sum()) for cls in self.classes_}
        logger.info(
            "Fitting pipeline",
            extra={
                "n_samples": len(y),
                "n_features": len(self.config.feature_columns),
                "n_classes": len(self.classes_),
                "classes": self.label_names_,
                "class_counts": class_counts,
                "weighted": sample_weight is not None,
            },
        )

        fit_params: dict[str, Any] = {}
        if sample_weight is not None:
            fit_params["model__sample_weight"] = sample_weight

        self._pipeline.fit(X[self.config.feature_columns], y, **fit_params)
        logger.info(
            "Pipeline fit complete",
            extra={"n_samples": len(y), "classes": self.label_names_},
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return the predicted class label for each row."""
        return self._pipeline.predict(X[self.config.feature_columns])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return class probabilities, shape (n_samples, n_classes)."""
        return self._pipeline.predict_proba(X[self.config.feature_columns])

    def save(self, path: str | Path) -> None:
        """Pickle the entire ChurnPipeline (config + fitted sklearn Pipeline) to disk.

        Loading the file with ChurnPipeline.load() is all that's needed to serve
        predictions — no separate config file required.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        size_kb = round(path.stat().st_size / 1024, 1)
        logger.info(
            "Pipeline saved",
            extra={"path": str(path), "size_kb": size_kb, "classes": self.label_names_},
        )

    @classmethod
    def load(cls, path: str | Path) -> ChurnPipeline:
        """Restore a ChurnPipeline saved with save()."""
        path = Path(path)
        pipeline: ChurnPipeline = joblib.load(path)
        logger.info(
            "Pipeline loaded",
            extra={
                "path": str(path),
                "classes": pipeline.label_names_,
                "n_features": len(pipeline.config.feature_columns),
            },
        )
        return pipeline
