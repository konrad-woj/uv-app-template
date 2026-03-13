"""Input validation for churn prediction requests.

Real services fail loudly at the boundary, not silently inside the model.
Validating here means you get a clear error message pointing to the exact
problem — missing column, out-of-range value, unknown category — rather than
a cryptic sklearn or XGBoost traceback that hides the root cause.

Constraints come directly from the feature schemas defined in config YAML,
so validation stays in sync with the model card automatically.

This module has no external dependencies: pure stdlib + pipeline config.

Example
-------
    from churn_lib.validate import validate_batch
    from churn_lib.pipeline import PipelineConfig

    cfg = PipelineConfig.from_yaml()
    validate_batch(samples, cfg)  # raises ValidationError if invalid
"""

from __future__ import annotations

from churn_lib.pipeline import FeatureSchema, PipelineConfig


class ValidationError(ValueError):
    """Raised when one or more input samples fail schema validation.

    Inherits from ValueError so callers that already catch ValueError will
    handle it — no extra import needed just to catch the exception type.
    """


def validate_batch(samples: list[dict], config: PipelineConfig) -> None:
    """Validate a list of sample dicts against the pipeline's feature schema.

    Checks every sample for:
    - Missing required feature columns
    - Wrong Python type for numeric features (must be int or float, not None)
    - Value out of the [min, max] range declared in the feature schema
    - Wrong Python type for binary features (must be bool or 0/1 int)
    - Unknown category for categorical features that declare a categories list

    All errors across all samples are collected before raising, so you see
    the full picture in one go rather than fixing one error at a time.

    Args:
        samples: List of feature dicts, one per customer record.
        config:  PipelineConfig whose feature schemas define the constraints.

    Raises:
        ValidationError: If any sample is invalid. The message lists every
            offending sample index and field.
    """
    if not samples:
        raise ValidationError("samples must be a non-empty list")

    errors: list[str] = []

    required = set(config.feature_columns)
    numeric_schemas = {s.name: s for s in config.numeric_schemas}
    categorical_schemas = {s.name: s for s in config.categorical_schemas}
    binary_cols = set(config.binary_features)

    for i, sample in enumerate(samples):
        if not isinstance(sample, dict):
            errors.append(f"  sample[{i}]: expected dict, got {type(sample).__name__}")
            continue

        present = set(sample.keys())
        missing = required - present
        if missing:
            errors.append(f"  sample[{i}]: missing columns: {sorted(missing)}")

        for col, schema in numeric_schemas.items():
            if col not in present:
                continue
            val = sample[col]
            if val is None or not isinstance(val, (int, float)) or isinstance(val, bool):
                errors.append(
                    f"  sample[{i}][{col!r}]: expected numeric (int/float), got {type(val).__name__!r} = {val!r}"
                )
                continue
            errors.extend(_check_range(i, col, float(val), schema))

        for col, schema in categorical_schemas.items():
            if col not in present:
                continue
            val = sample[col]
            if val is None:
                errors.append(f"  sample[{i}][{col!r}]: value must not be None")
                continue
            if schema.categories is not None and val not in schema.categories:
                errors.append(f"  sample[{i}][{col!r}]: {val!r} not in allowed categories {schema.categories}")

        for col in binary_cols:
            if col not in present:
                continue
            val = sample[col]
            if not isinstance(val, bool) and val not in (0, 1):
                errors.append(f"  sample[{i}][{col!r}]: expected bool or 0/1, got {type(val).__name__!r} = {val!r}")

    if errors:
        raise ValidationError("Input validation failed:\n" + "\n".join(errors))


def validate_sample(sample: dict, config: PipelineConfig) -> None:
    """Validate a single sample dict. Convenience wrapper around validate_batch.

    Args:
        sample: Feature values as a flat dict.
        config: PipelineConfig whose feature schemas define the constraints.

    Raises:
        ValidationError: If the sample is invalid.
    """
    validate_batch([sample], config)


def _check_range(i: int, col: str, val: float, schema: FeatureSchema) -> list[str]:
    """Return range violation messages for a numeric value, or an empty list."""
    errs: list[str] = []
    if schema.min is not None and val < schema.min:
        errs.append(f"  sample[{i}][{col!r}]: {val} is below minimum ({schema.min})")
    if schema.max is not None and val > schema.max:
        errs.append(f"  sample[{i}][{col!r}]: {val} is above maximum ({schema.max})")
    return errs
