# Model Construction

## Overview

This project trains three machine learning models organized into two tasks:

| Task           | Models                                      | Purpose                                        |
|----------------|---------------------------------------------|------------------------------------------------|
| Classification | Logistic Regression, Random Forest          | Predict if an asset outperforms SPY next month |
| Regression     | Linear Regression                           | Predict expected next-month return per asset   |

All models are built as scikit-learn `Pipeline` objects and trained using a
**time-based split** to respect the chronological order of financial data.

---

## Feature Columns (12 features)

All models share the same feature matrix `X`:

| Feature                  | Description                                  |
|--------------------------|----------------------------------------------|
| `return_1m`              | 1-month return                               |
| `return_3m`              | 3-month cumulative return                    |
| `return_6m`              | 6-month cumulative return                    |
| `return_12m`             | 12-month cumulative return                   |
| `volatility_3m`          | Rolling 3-month volatility                   |
| `volatility_6m`          | Rolling 6-month volatility                   |
| `drawdown_3m`            | Rolling 3-month max drawdown                 |
| `drawdown_6m`            | Rolling 6-month max drawdown                 |
| `sharpe_3m`              | Rolling 3-month Sharpe ratio                 |
| `sharpe_6m`              | Rolling 6-month Sharpe ratio                 |
| `relative_strength_3m`   | Return vs SPY over 3 months                  |
| `relative_strength_6m`   | Return vs SPY over 6 months                  |

---

## Train / Test Split

The dataset is split by **time order**, not randomly, to avoid data leakage:

```python
TEST_SIZE = 0.20
```

Unique dates are sorted chronologically. The last 20% of months form the
test set and the first 80% form the training set.
Timeline ──────────────────────────────────────────►
│◄──────────── 80% Train ────────────►│◄─ 20% Test ─►│

This is implemented in:

```python
from src.models.train_classification_models import split_dataset_by_time
```

---

## Classification Models

### Target Variable

```python
CLASSIFICATION_TARGET_COLUMN = "target_outperform_next_month"
# 1 → asset outperforms SPY next month
# 0 → asset underperforms SPY next month
```

### Logistic Regression Pipeline

```python
Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
    ("model",   LogisticRegression(max_iter=1000, random_state=42)),
])
```

### Random Forest Pipeline

```python
Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model",   RandomForestClassifier(
                    n_estimators=300,
                    max_depth=5,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1,
                )),
])
```

> Random Forest does not require StandardScaler since it is tree-based.

### Time Series Cross-Validation

Classification models are evaluated with `TimeSeriesSplit` before final training:

```python
N_SPLITS = 5
```

The number of splits is capped at `min(5, len(X_train) // 30)` to ensure
each fold has sufficient data.

Cross-validation scoring:

| CV Metric       | Description                          |
|-----------------|--------------------------------------|
| `cv_precision`  | Mean precision across folds          |
| `cv_recall`     | Mean recall across folds             |
| `cv_f1`         | Mean F1 score across folds           |
| `cv_roc_auc`    | Mean ROC AUC across folds            |

### Classification Evaluation Metrics

After training on the full training set, models are evaluated on the test set:

| Metric       | Description                                       |
|--------------|---------------------------------------------------|
| `precision`  | Ratio of correct outperform predictions           |
| `recall`     | Coverage of actual outperform cases               |
| `f1_score`   | Harmonic mean of precision and recall             |
| `roc_auc`    | Area under the ROC curve                          |

### Classification Artifacts

For each classification model:

```text
reports/figures/
├── confusion_matrix_logistic_regression.png
├── confusion_matrix_random_forest.png
├── roc_curve_logistic_regression.png
└── roc_curve_random_forest.png
```

---

## Regression Model

### Purpose

Linear Regression predicts the **expected next-month return** per asset.
These predictions are used as inputs for portfolio optimization.

### Regression Target

The target `y_regression` is computed in memory as the **next month's
`return_1m`** per asset using a `groupby + shift(-1)` operation:

```python
dataframe["y_regression"] = (
    dataframe.groupby("asset")["return_1m"].shift(-1)
)
```

> This target is never persisted in `model_dataset.csv`.

### Linear Regression Pipeline

```python
Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
    ("model",   LinearRegression()),
])
```

### Regression Evaluation Metrics

| Metric   | Description                                    |
|----------|------------------------------------------------|
| `mae`    | Mean Absolute Error                            |
| `rmse`   | Root Mean Squared Error                        |
| `r2`     | Coefficient of determination                   |

---

## Output Artifacts

All model metrics are consolidated in a single JSON file:

```text
reports/metrics/model_metrics.json
```

Structure:

```json
{
    "logistic_regression": {
        "cv_precision": ...,
        "cv_recall": ...,
        "cv_f1": ...,
        "cv_roc_auc": ...,
        "precision": ...,
        "recall": ...,
        "f1_score": ...,
        "roc_auc": ...
    },
    "random_forest": { ... },
    "linear_regression": {
        "mae": ...,
        "rmse": ...,
        "r2": ...
    }
}
```

---

## Training Entry Point

All three models are trained in a single script:

```bash
python scripts/03_train_models.py
```

Or as part of the full pipeline:

```bash
python scripts/05_run_full_pipeline.py --amount 10000
```
