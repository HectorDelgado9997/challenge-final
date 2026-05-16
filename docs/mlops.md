# MLOps

## Objective

The objective of the MLOps stage is to track machine learning experiments locally using MLflow.

The project logs model parameters, metrics, artifacts, model signatures, input examples, and trained models.

## Tool

The project uses:

```text
MLflow

Tracking URI

The MLflow tracking URI is configured as:

http://127.0.0.1:5000

The configuration is defined in:

src/config/settings.py
MLflow Module

MLflow helper functions are implemented in:

src/mlops/mlflow_tracking.py

The main functions are:

configure_mlflow
log_model_run
Start MLflow UI

From the project root, run:

mlflow ui --host 127.0.0.1 --port 5000

Then open:

http://127.0.0.1:5000
Experiment Name

The experiment name is:

monthly-investment-recommendation
Logged Runs

The project logs three model runs:

logistic-regression-classifier
random-forest-classifier
linear-regression-expected-returns
Logged Parameters

Each run logs relevant model parameters.

Examples:

model_type
max_iter
random_state
n_estimators
max_depth
min_samples_split
min_samples_leaf
n_jobs
Logged Metrics
Logistic Regression
precision
recall
f1_score
roc_auc
cv_precision
cv_recall
cv_f1
cv_roc_auc
Random Forest Classifier
precision
recall
f1_score
roc_auc
cv_precision
cv_recall
cv_f1
cv_roc_auc
Linear Regression
mae
rmse
r2
Logged Artifacts

For classification models, the project logs:

confusion_matrix
roc_curve
model

The plots are generated in:

reports/figures/
Model Signatures

The project uses:

mlflow.models.infer_signature

This records the expected input and output structure of each model.

Input Examples

Each model run logs an input example based on the first rows of the feature matrix.

The feature columns are:

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
Model Registry

This project focuses on local experiment tracking. It does not require promotion to staging or production through the MLflow Model Registry.

How to Generate Runs

Start MLflow:

mlflow ui --host 127.0.0.1 --port 5000

Run model training:

python scripts/03_train_models.py

Or run the full pipeline:

python scripts/05_run_full_pipeline.py --amount 10000
Expected Result

The MLflow UI should show:

Experiment: monthly-investment-recommendation
Runs:
  logistic-regression-classifier
  random-forest-classifier
  linear-regression-expected-returns

Each run should contain:

Parameters
Metrics
Artifacts
Model
Signature
Input example
Local Files Ignored by Git

The following MLflow-related files and folders are ignored by Git:

mlruns/
mlartifacts/
mlflow.db

These files are local execution artifacts and should not be committed to GitHub.