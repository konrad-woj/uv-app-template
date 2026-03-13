"""Shared fixtures for the churn-lib test suite.

Session-scoped fixtures (config, training_df, fitted_pipeline) are expensive
to create and shared across all test modules. pytest-xdist creates one session
per worker, so each worker fits the pipeline once — acceptable given the small
500-sample dataset used in tests.

Function-scoped fixtures (valid_sample, valid_batch, high_risk_sample,
low_risk_sample) return fresh copies so tests cannot mutate shared state.
"""

from __future__ import annotations

import pandas as pd
import pytest

from churn_lib import ChurnPipeline, PipelineConfig
from churn_lib.data_generator import generate_prediction_data, generate_training_data

# ---------------------------------------------------------------------------
# Canonical sample dicts — copy in fixtures, never mutate these directly
# ---------------------------------------------------------------------------

_VALID_SAMPLE: dict = {
    "tenure_months": 12,
    "monthly_charges": 50.0,
    "num_products": 2,
    "support_calls_last_6m": 1,
    "age": 35,
    "contract_type": "month-to-month",
    "payment_method": "credit_card",
    "has_internet": True,
    "has_streaming": False,
}

# High churn risk: month-to-month, very new, expensive, many calls, risky payment
_HIGH_RISK_SAMPLE: dict = {
    "tenure_months": 1,
    "monthly_charges": 119.0,
    "num_products": 1,
    "support_calls_last_6m": 10,
    "age": 25,
    "contract_type": "month-to-month",
    "payment_method": "electronic_check",
    "has_internet": True,
    "has_streaming": True,
}

# Low churn risk: two-year contract, long tenure, cheap, no calls, stable payment
_LOW_RISK_SAMPLE: dict = {
    "tenure_months": 60,
    "monthly_charges": 25.0,
    "num_products": 4,
    "support_calls_last_6m": 0,
    "age": 55,
    "contract_type": "two-year",
    "payment_method": "bank_transfer",
    "has_internet": False,
    "has_streaming": False,
}


# ---------------------------------------------------------------------------
# Session-scoped (expensive) fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def config() -> PipelineConfig:
    return PipelineConfig.from_yaml()


@pytest.fixture(scope="session")
def training_df(config: PipelineConfig) -> pd.DataFrame:
    return generate_training_data(n_samples=500, random_seed=42)


@pytest.fixture(scope="session")
def prediction_df() -> pd.DataFrame:
    return generate_prediction_data(n_samples=50, random_seed=123)


@pytest.fixture(scope="session")
def fitted_pipeline(config: PipelineConfig, training_df: pd.DataFrame) -> ChurnPipeline:
    """Fit a ChurnPipeline once per worker session and share across all tests."""
    pipeline = ChurnPipeline(config)
    X = training_df[config.feature_columns]
    y = training_df[config.target]
    pipeline.fit(X, y)
    return pipeline


# ---------------------------------------------------------------------------
# Function-scoped (cheap) fixtures — always fresh copies
# ---------------------------------------------------------------------------


@pytest.fixture
def valid_sample() -> dict:
    return _VALID_SAMPLE.copy()


@pytest.fixture
def valid_batch() -> list[dict]:
    return [_VALID_SAMPLE.copy(), _VALID_SAMPLE.copy()]


@pytest.fixture
def high_risk_sample() -> dict:
    return _HIGH_RISK_SAMPLE.copy()


@pytest.fixture
def low_risk_sample() -> dict:
    return _LOW_RISK_SAMPLE.copy()
