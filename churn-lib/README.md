# churn-lib

ML library for churn prediction — training, inference, validation, and drift detection.

## Installation

churn-lib has no mandatory runtime dependencies. Install only what you need via extras:

```bash
# Inference only (load a saved pipeline and score records)
uv add "churn-lib[predict]"

# Training + inference (fit, evaluate, save pipelines)
uv add "churn-lib[train]"

# Full development environment (training + tests + linting)
cd churn-lib
uv sync --group dev
```

**Extras at a glance:**

| Extra | Installs | Use when |
|-------|----------|----------|
| `predict` | pandas, numpy, scikit-learn, xgboost, pyyaml | Loading a saved pipeline and running inference |
| `train` | `predict` + matplotlib, mlflow | Training, evaluating, and saving a pipeline |
| `dev` (group) | `train` + pytest, ruff, taskipy | Local development |

## Getting started

Train a model with one command — no data or config required:

```bash
cd churn-lib
uv sync --group dev
uv run trainer
```

This generates 5,000 synthetic customer records, trains an XGBoost pipeline, evaluates it on a held-out test set, and writes all artifacts to `eval_results/<timestamp>/`.

## Usage examples

### Training

**CLI — quickstart with defaults:**

```bash
uv run trainer
```

**CLI — custom options:**

```bash
uv run trainer \
  --config path/to/config.yaml \
  --n-samples 20000 \
  --output-dir runs \
  --test-size 0.25 \
  --seed 7 \
  --log-level INFO
```

**Python API:**

```python
from churn_lib import ChurnPipeline, PipelineConfig
from churn_lib.data_generator import generate_training_data
from churn_lib.trainer import train

cfg = PipelineConfig.from_yaml()               # uses built-in default_config.yaml
df  = generate_training_data(n_samples=10_000)
pipeline = ChurnPipeline(cfg)

summary = train(pipeline, df, output_dir="runs")
print(summary)
# Run saved to: runs/20250308_120000
# Train samples: 8000 | Test samples: 2000
# Classes: ['0', '1'] | Optimal threshold (F1): 0.35
# ...classification report...
```

### Inference

**Python API — single record:**

```python
from churn_lib import ChurnPipeline, predict_single

pipeline = ChurnPipeline.load("eval_results/20250308_120000/pipeline.joblib")

result = predict_single(pipeline, {
    "tenure_months": 3,
    "monthly_charges": 95.0,
    "num_products": 1,
    "support_calls_last_6m": 5,
    "age": 34,
    "contract_type": "month-to-month",
    "payment_method": "electronic_check",
    "has_internet": True,
    "has_streaming": False,
})
# {"prediction": 1, "label": "1", "probabilities": {"0": 0.18, "1": 0.82}}
```

**Python API — batch (preferred for throughput):**

```python
from churn_lib import predict_batch

results = predict_batch(pipeline, records, threshold=0.35)
for r in results:
    print(r["label"], r["probabilities"])
```

The `threshold` parameter overrides the default 0.5 decision boundary for binary models. Use `trainer.find_threshold()` to pick the right value (see below).

**CLI — score synthetic or CSV/JSON data:**

```bash
# Score 20 synthetic records
uv run python -m churn_lib.inference --model eval_results/.../pipeline.joblib --n-samples 20

# Score from a CSV file and write results to a file
uv run python -m churn_lib.inference \
  --model eval_results/.../pipeline.joblib \
  --input customers.csv \
  --output predictions.json
```

### Threshold calibration

The default 0.5 threshold is rarely optimal for churn. Use `find_threshold()` to pick an operating point that matches your business constraint:

```python
from churn_lib.trainer import find_threshold

# Maximise F1 (default)
threshold, curve_df = find_threshold(pipeline, X_test, y_test)

# Catch at least 80% of churners, then be as precise as possible
threshold, curve_df = find_threshold(pipeline, X_test, y_test, min_recall=0.8)

# Only send offers when 90% confident, then maximise coverage
threshold, curve_df = find_threshold(pipeline, X_test, y_test, min_precision=0.9)

print(threshold)     # e.g. 0.35
print(curve_df)      # DataFrame: threshold | precision | recall | f1
```

### Drift detection

Monitor serving data against the training distribution using Population Stability Index (PSI):

| PSI | Status | Action |
|-----|--------|--------|
| < 0.10 | stable | No action |
| 0.10 – 0.25 | moderate | Investigate; schedule retrain review |
| > 0.25 | major | Model likely stale — retrain soon |

**Python API:**

```python
from churn_lib import check_drift, PipelineConfig
import pandas as pd

cfg          = PipelineConfig.from_yaml()
reference_df = pd.read_csv("train.csv")
serving_df   = pd.read_csv("live.csv")

report = check_drift(reference_df, serving_df, cfg, output_path="drift_report.json")
print(report["overall_status"])     # "stable" | "moderate" | "major"
print(report["drifted_features"])   # list of feature names with non-stable PSI
```

**CLI:**

```bash
uv run python -m churn_lib.drift \
  --reference train.csv \
  --serving live.csv \
  --output drift_report.json \
  --log-level INFO
```

### Input validation

Validation runs automatically inside `predict_batch`. You can also call it directly:

```python
from churn_lib import validate_sample, validate_batch, ValidationError

try:
    validate_sample(record, pipeline.config)
except ValidationError as e:
    print(e)  # lists every offending field across all samples
```

Checks enforced: missing columns, wrong type, out-of-range numeric values, unknown categories, and invalid binary values.

## Feature schema

The default model uses these features (defined in [src/churn_lib/default_config.yaml](src/churn_lib/default_config.yaml)):

| Feature | Type | Range / Categories |
|---------|------|--------------------|
| `tenure_months` | numeric | 0 – 120 |
| `monthly_charges` | numeric | 0.0 – 500.0 |
| `num_products` | numeric | 1 – 10 |
| `support_calls_last_6m` | numeric | 0 – 50 |
| `age` | numeric | 18 – 100 |
| `contract_type` | categorical | `month-to-month`, `one-year`, `two-year` |
| `payment_method` | categorical | `electronic_check`, `mailed_check`, `bank_transfer`, `credit_card` |
| `has_internet` | binary | `True` / `False` |
| `has_streaming` | binary | `True` / `False` |

To use a custom schema, copy `default_config.yaml`, edit the feature list, and pass `--config path/to/config.yaml` to the trainer CLI or `PipelineConfig.from_yaml("path/to/config.yaml")` in Python.

## Where to find results

### Local artifact directory

Each training run writes to a **timestamped subdirectory** under `eval_results/` (or your `--output-dir`):

```
eval_results/
└── 20250308_120000/
    ├── pipeline.joblib       # fitted ChurnPipeline — load this to serve predictions
    ├── metrics.json          # accuracy, F1, per-class precision/recall, optimal threshold
    ├── config.json           # full PipelineConfig snapshot for this run
    ├── confusion_matrix.png  # test-set confusion matrix
    ├── feature_importance.png # top-20 XGBoost feature importances (gain)
    └── threshold_curve.json  # precision/recall/F1 at every threshold from 0.05→0.95
```

**How to read the key files:**

- **`metrics.json`** — start here. Contains `accuracy`, `macro_f1`, `weighted_f1`, per-class precision/recall/F1, and `optimal_threshold`. The threshold is the F1-maximising decision boundary found during training.
- **`confusion_matrix.png`** — rows are true labels, columns are predicted labels. For churn the bottom-right cell (true positives — actual churners correctly flagged) is the most business-critical.
- **`feature_importance.png`** — XGBoost gain-based importances. Features at the top drive the most information gain in the tree splits.
- **`threshold_curve.json`** — array of `{threshold, precision, recall, f1}` objects. Use this to pick a different operating point after training without re-running `find_threshold()`.
- **`pipeline.joblib`** — the only file you need for inference. Load it with `ChurnPipeline.load(path)`.

### MLflow experiment tracking

Every run is also logged to the `churn-prediction` MLflow experiment. Logged per run: params (model, version, features, seed), metrics (accuracy, F1, per-class precision/recall, optimal threshold), and all artifacts.

**Option 1: Local file store (no server required)**

The default. Runs are written to `./mlruns` in your working directory.

```bash
uv run mlflow ui --port 5000
# Open http://localhost:5000
```

**Option 2: Local tracking server (Docker)**

Starts a persistent MLflow server with a separate artifact store. Results survive across terminal sessions and are accessible at a stable URL.

```bash
# Start the server
uv run task mlflow
# MLflow UI: http://localhost:5000

# Point your runs at the server
export MLFLOW_TRACKING_URI=http://localhost:5000
uv run trainer

# Stop the server
uv run task mlflow-stop
```

Data is persisted in `./mlruns` (metadata) and `./mlartifacts` (artifact files).

**Comparing runs in the UI:** open http://localhost:5000, select the `churn-prediction` experiment, and use the table or chart view to compare metrics across runs. Artifacts (confusion matrix, feature importance, pipeline) are accessible from each run's detail page.
