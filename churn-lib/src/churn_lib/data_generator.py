"""Synthetic churn dataset generator with realistic feature-to-label correlations.

The churn signal is rule-based (not random), so a model trained on this data
can actually learn. Key drivers: contract type, tenure, monthly charges,
support-call frequency, and payment method.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_CONTRACT_TYPES = ["month-to-month", "one-year", "two-year"]
_PAYMENT_METHODS = ["credit_card", "bank_transfer", "electronic_check", "mailed_check"]
_CONTRACT_PROBS = [
    0.50,
    0.30,
    0.20,
]  # skewed toward month-to-month for richer churn signal


def _churn_probability(row: pd.Series) -> float:
    """Compute churn probability for one customer row via business-rule heuristics.

    Mirrors real-world drivers: short-tenure + expensive plan, frequent support
    calls, and month-to-month contracts all raise churn risk; long tenure and
    multi-product subscriptions reduce it.
    """
    prob = 0.08  # baseline churn rate

    prob += {"month-to-month": 0.35, "one-year": 0.10, "two-year": 0.02}[str(row["contract_type"])]

    if row["tenure_months"] < 6:
        prob += 0.20
    elif row["tenure_months"] < 12 and row["monthly_charges"] > 70:
        prob += 0.15  # early-life, high-cost — classic at-risk segment

    prob += row["support_calls_last_6m"] * 0.04  # each call is a frustration signal

    if row["payment_method"] == "electronic_check":
        prob += 0.08  # correlated with lower commitment in real datasets

    if row["tenure_months"] > 48:
        prob -= 0.15  # long-tenure customers are sticky

    if row["num_products"] >= 3:
        prob -= 0.05  # embedded customers churn less

    return float(np.clip(prob, 0.02, 0.97))


def _base_features(rng: np.random.Generator, n_samples: int) -> pd.DataFrame:
    """Generate the shared feature schema used by both public functions."""
    return pd.DataFrame(
        {
            "tenure_months": rng.integers(1, 73, n_samples),
            "monthly_charges": rng.uniform(20.0, 120.0, n_samples).round(2),
            "num_products": rng.integers(1, 6, n_samples),
            "support_calls_last_6m": rng.integers(0, 11, n_samples),
            "age": rng.integers(18, 81, n_samples),
            "contract_type": rng.choice(_CONTRACT_TYPES, n_samples, p=_CONTRACT_PROBS),
            "payment_method": rng.choice(_PAYMENT_METHODS, n_samples),
            "has_internet": rng.choice([True, False], n_samples, p=[0.75, 0.25]),
            "has_streaming": rng.choice([True, False], n_samples, p=[0.50, 0.50]),
        }
    )


def generate_training_data(n_samples: int = 5_000, random_seed: int = 42) -> pd.DataFrame:
    """Generate a labelled DataFrame ready for train/test split.

    Each row's ``churn`` label (0/1) is sampled against a rule-derived probability,
    introducing realistic noise without making the task trivially learnable.

    Args:
        n_samples:   Number of synthetic customer records.
        random_seed: Seed for full reproducibility.
    """
    rng = np.random.default_rng(random_seed)
    df = _base_features(rng, n_samples)
    churn_probs = df.apply(_churn_probability, axis=1).to_numpy()
    df["churn"] = (rng.uniform(0.0, 1.0, n_samples) < churn_probs).astype(int)

    churn_rate = float(df["churn"].mean())
    logger.info(
        "Training data generated",
        extra={
            "n_samples": n_samples,
            "random_seed": random_seed,
            "churn_rate": round(churn_rate, 4),
            "n_churned": int(df["churn"].sum()),
            "n_retained": int((df["churn"] == 0).sum()),
            "features": list(df.columns[:-1]),
        },
    )
    return df


def generate_prediction_data(n_samples: int = 100, random_seed: int = 123) -> pd.DataFrame:
    """Generate an unlabelled DataFrame for inference demos.

    No ``churn`` column — matches what a real scoring service receives.

    Args:
        n_samples:   Number of synthetic customer records.
        random_seed: Seed for reproducibility.
    """
    rng = np.random.default_rng(random_seed)
    df = _base_features(rng, n_samples)
    logger.info(
        "Prediction data generated",
        extra={
            "n_samples": n_samples,
            "random_seed": random_seed,
            "features": list(df.columns),
        },
    )
    return df
