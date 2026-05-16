# Architecture

## Objective

This document describes the technical architecture of the Monthly Investment Recommendation project.

The project is designed as a modular local pipeline that combines data extraction, feature engineering, supervised learning, MLOps tracking, and portfolio optimization.

## High-Level Architecture

```text
asset_details.csv
      │
      ▼
yfinance data extraction
      │
      ▼
monthly_prices.csv
      │
      ▼
feature engineering
      │
      ▼
model_dataset.csv
      │
      ├───────────────┐
      ▼               ▼
classification       regression
models               model
      │               │
      ▼               ▼
classification       expected
metrics              returns
      │               │
      └──────┬────────┘
             ▼
       MLflow tracking
             │
             ▼
portfolio optimization
             │
             ▼
recommended_allocation.csv

Pipeline Stages
1. Data Extraction

Implemented in:

src/data/extract_data.py
scripts/01_extract_data.py

Responsibilities:

Load asset universe
Validate asset details
Download daily price data from yfinance
Extract adjusted close prices
Convert daily prices to monthly prices
Save monthly_prices.csv

Output:

data/raw/monthly_prices.csv
2. Feature Engineering

Implemented in:

src/features/feature_engineering.py

Responsibilities:

Calculate return features
Calculate volatility features
Calculate drawdown features
Calculate Sharpe ratio features
Calculate relative strength features
Create classification target
Clean final feature dataset

Output columns:

date
asset
return_1m
return_3m
return_6m
return_12m
volatility_3m
volatility_6m
drawdown_3m
drawdown_6m
sharpe_3m
sharpe_6m
relative_strength_3m
relative_strength_6m
target_outperform_next_month
3. Dataset Construction

Implemented in:

src/data/build_dataset.py
scripts/02_build_dataset.py

Responsibilities:

Load monthly_prices.csv
Apply feature engineering
Validate final model dataset
Save model_dataset.csv

Output:

data/processed/model_dataset.csv
4. Model Training

Implemented in:

src/models/train_classification_models.py
src/models/train_regression_model.py
scripts/03_train_models.py

Classification models:

Logistic Regression
Random Forest Classifier

Regression model:

Linear Regression

Responsibilities:

Apply time-aware train-test split
Train models
Evaluate models
Generate plots
Save metrics

Outputs:

reports/metrics/model_metrics.json
reports/figures/*.png
5. MLOps Tracking

Implemented in:

src/mlops/mlflow_tracking.py

Responsibilities:

Configure MLflow tracking URI
Configure MLflow experiment
Log parameters
Log metrics
Log artifacts
Log model signatures
Log input examples
Log trained models

Tracking URI:

http://127.0.0.1:5000
6. Expected Return Prediction

Implemented in:

src/models/predict_expected_returns.py

Responsibilities:

Train expected return model
Select latest feature row per asset
Predict expected monthly returns
Return expected returns indexed by asset
7. Portfolio Optimization

Implemented in:

src/portfolio/optimize_portfolio.py
scripts/04_optimize_portfolio.py

Responsibilities:

Build monthly returns matrix
Calculate covariance matrix
Use predicted expected returns
Optimize portfolio weights
Calculate allocated amount
Save recommendation

Output:

data/outputs/recommended_allocation.csv
8. Full Pipeline

Implemented in:

scripts/05_run_full_pipeline.py

Responsibilities:

Run data extraction
Build model dataset
Train models
Log MLflow runs
Optimize portfolio
Generate final allocation
Configuration Layer

Central configuration is defined in:

src/config/settings.py

This file contains:

Project paths
Input and output files
Expected columns
Feature columns
Model parameters
Portfolio constraints
MLflow configuration
Default dates
Default investment amount
Validation Layer

Validation is implemented in:

src/data/validate_data.py

Validation includes:

File existence
Required columns
Date format
Asset list integrity
Null checks
Infinite value checks
Target value checks
Monthly investment amount checks
Exception Handling

Custom exceptions are defined in:

src/utils/exceptions.py

Main exception types:

DataValidationError
DataExtractionError
FeatureEngineeringError
ModelTrainingError
PortfolioOptimizationError
Logging

Reusable logging is implemented in:

src/utils/logger.py

The logger provides consistent messages across the pipeline.

Design Principles

The architecture follows these principles:

Modular code
Reusable functions
Clear separation of responsibilities
Local reproducibility
Input validation
Exception handling
Logging
No unnecessary hardcoding
GitHub-ready structure
MLflow experiment tracking
Local Execution Model

The project is designed to run locally from the repository root.

Main command:

python scripts/05_run_full_pipeline.py --amount 10000

MLflow command:

mlflow ui --host 127.0.0.1 --port 5000
Final Output

The final output is:

data/outputs/recommended_allocation.csv

It contains:

asset
weight
expected_return
allocated_amount

---