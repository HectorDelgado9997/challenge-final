# Project Architecture

## Overview

This project follows a modular, layered architecture that separates data
extraction, feature engineering, model training, portfolio optimization,
and MLOps tracking into independent, testable modules. The full pipeline
is orchestrated by `scripts/05_run_full_pipeline.py`.

---

## Directory Structure

```text
challenge-final/
├── data/
│   ├── raw/
│   │   ├── asset_details.csv          ← Asset universe definition
│   │   └── monthly_prices.csv         ← Raw monthly prices from yfinance
│   ├── processed/
│   │   └── model_dataset.csv          ← Feature-engineered dataset
│   └── outputs/
│       └── recommended_allocation.csv ← Final portfolio allocation
│
├── docs/
│   ├── architecture.md
│   ├── dataset_extraction.md
│   ├── execution_guide.md
│   ├── mlops.md
│   └── model_construction.md
│
├── notebooks/
│   └── exploratory_analysis.ipynb     ← EDA
│
├── reports/
│   ├── figures/
│   │   ├── confusion_matrix_*.png     ← Per-model confusion matrices
│   │   └── roc_curve_*.png            ← Per-model ROC curves
│   └── metrics/
│       └── model_metrics.json         ← Consolidated model metrics
│
├── scripts/
│   ├── 01_extract_data.py             ← yfinance extraction
│   ├── 02_build_dataset.py            ← Feature engineering
│   ├── 03_train_models.py             ← Model training + MLflow
│   ├── 04_optimize_portfolio.py       ← Portfolio optimization
│   └── 05_run_full_pipeline.py        ← Full pipeline entry point
│
├── src/
│   ├── config/
│   │   └── settings.py                ← All project constants and paths
│   ├── data/
│   │   ├── extract_data.py            ← run_data_extraction()
│   │   ├── build_dataset.py           ← run_dataset_building()
│   │   └── validate_data.py           ← validate_*() functions
│   ├── features/
│   │   └── feature_engineering.py     ← Technical indicator computation
│   ├── models/
│   │   ├── train_classification_models.py ← Logistic Regression, Random Forest
│   │   ├── train_regression_model.py      ← Linear Regression
│   │   ├── predict_expected_returns.py    ← Inference for portfolio input
│   │   └── evaluate_models.py             ← Metrics + plots + JSON export
│   ├── portfolio/
│   │   └── optimize_portfolio.py      ← PyPortfolioOpt integration
│   ├── mlops/
│   │   └── mlflow_tracking.py         ← configure_mlflow(), log_model_run()
│   └── utils/
│       ├── logger.py                  ← get_logger(__name__)
│       └── exceptions.py             ← Custom exception classes
│
├── tests/                             ← pytest test suite
├── .gitignore
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Layer Diagram
┌──────────────────────────────────────────────────────────────┐
│              scripts/05_run_full_pipeline.py                 │
│                  (Orchestration Layer)                       │
└───────────────────────────┬──────────────────────────────────┘
│
┌────────────────────┼──────────────────────┐
▼                    ▼                       ▼
┌─────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Config Layer│    │   Data Layer    │    │  Utils Layer    │
│ settings.py │    │ extract_data    │    │ logger.py       │
│ ALL paths   │    │ build_dataset   │    │ exceptions.py   │
│ ALL params  │    │ validate_data   │    │                 │
└─────────────┘    └───────┬─────────┘    └─────────────────┘
│
▼
┌────────────────────────┐
│    Features Layer      │
│  feature_engineering   │
│  returns, volatility,  │
│  drawdown, sharpe,     │
│  relative_strength     │
└────────────┬───────────┘
│
┌────────────┴───────────┐
▼                        ▼
┌─────────────────────────┐  ┌──────────────────────────┐
│   Classification Layer  │  │    Regression Layer      │
│  train_classification   │  │  train_regression_model  │
│  Logistic Regression    │  │  Linear Regression       │
│  Random Forest          │  │  target: next return_1m  │
│  TimeSeriesSplit CV     │  │  time-based split        │
└────────────┬────────────┘  └────────────┬─────────────┘
│                            │
└──────────────┬─────────────┘
▼
┌─────────────────────────┐
│    Evaluation Layer     │
│  evaluate_models.py     │
│  confusion matrix       │
│  ROC curve              │
│  metrics → JSON         │
└─────────────┬───────────┘
│
┌─────────────┴──────────────┐
▼                            ▼
┌─────────────────────────┐  ┌────────────────────────────┐
│     MLOps Layer         │  │    Portfolio Layer         │
│  mlflow_tracking.py     │  │  predict_expected_returns  │
│  configure_mlflow()     │  │  build_monthly_returns     │
│  log_model_run()        │  │  calculate_covariance      │
│  params + metrics +     │  │  optimize_weights()        │
│  artifacts + model      │  │  EfficientFrontier         │
└─────────────────────────┘  │  max_sharpe / min_vol      │
│  build_allocation_df       │
└────────────────────────────┘

---

## Full Data Flow
asset_details.csv
│
▼
run_data_extraction(start_date, end_date)
│  yfinance → daily prices → resample monthly → adjusted_close
▼
data/raw/monthly_prices.csv
│
▼
run_dataset_building()
│  feature_engineering → returns, volatility, drawdown,
│  sharpe, relative_strength, target_outperform_next_month
▼
data/processed/model_dataset.csv
│
├──────────────────────────────────────────────┐
▼                                              ▼
Classification (80% train / 20% test)         Regression
│                                              │
├── Logistic Regression Pipeline          create_regression_target()
│   Imputer → Scaler → LogisticRegression      │  y = next return_1m (in memory)
│                                              │
├── Random Forest Pipeline             Linear Regression Pipeline
│   Imputer → RandomForestClassifier       Imputer → Scaler → LinearRegression
│                                              │
├── TimeSeriesSplit CV (up to 5 folds)         │
│                                              ▼
├── evaluate_classification_model()    predict_expected_returns()
│   precision, recall, F1, ROC AUC         latest features per asset
│                                              │
├── save_confusion_matrix_plot()              ▼
├── save_roc_curve_plot()            ┌─────────────────────────┐
└── log_model_run() → MLflow        │  Portfolio Optimization  │
│  expected_returns (μ)    │
│  covariance_matrix (Σ)   │
│  EfficientFrontier       │
│  max_sharpe → weights    │
│  fallback: min_volatility│
│  weight bounds [0, 0.35] │
└────────────┬────────────┘
│
▼
data/outputs/
recommended_allocation.csv
(asset, weight,
expected_return,
allocated_amount)

---

## Design Principles

| Principle              | Implementation                                                  |
|------------------------|-----------------------------------------------------------------|
| Single responsibility  | Each module handles exactly one concern                         |
| Time-aware splitting   | `split_dataset_by_time()` prevents data leakage                |
| In-memory regression target | `y_regression` never persists to disk                  |
| Fallback optimization  | `max_sharpe` falls back to `min_volatility` automatically      |
| Custom exceptions      | `DataExtractionError`, `DataValidationError`, `ModelTrainingError`, `PortfolioOptimizationError` |
| Unified logging        | `get_logger(__name__)` across all modules                       |
| Config as single source| All paths, params and constants live in `settings.py`          |
| Numbered scripts       | `01_` to `05_` enforce execution order and independent stages   |
