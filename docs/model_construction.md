# Model Construction

## Objective

The objective of the model construction stage is to train supervised machine learning models that support monthly investment recommendations.

The project uses:

```text
Logistic Regression
Random Forest Classifier
Linear Regression

Final Modeling Dataset

The final dataset is generated at:

data/processed/model_dataset.csv

It contains the following columns:

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

No additional columns are persisted in the final modeling dataset.

Feature Engineering

Feature engineering is implemented in:

src/features/feature_engineering.py

The features are calculated per asset and per month.

Return Features
return_1m
return_3m
return_6m
return_12m

These represent monthly and rolling cumulative returns.

Volatility Features
volatility_3m
volatility_6m

These are rolling standard deviations of monthly returns.

Drawdown Features
drawdown_3m
drawdown_6m

Drawdown measures the percentage decline from the rolling maximum price.

Sharpe Ratio Features
sharpe_3m
sharpe_6m

The project uses a simplified Sharpe ratio assuming a zero risk-free rate.

This is a design decision for academic simplicity.

Relative Strength Features
relative_strength_3m
relative_strength_6m

Relative strength is calculated against the benchmark asset.

The benchmark asset is:

SPY
Classification Target

The classification target is:

target_outperform_next_month

It is defined as:

1 if the asset outperforms SPY in the next month
0 otherwise
Classification Models

Classification models estimate whether an asset is likely to outperform the benchmark in the next month.

Logistic Regression

Logistic Regression is used as an interpretable baseline model.

Pipeline:

SimpleImputer
StandardScaler
LogisticRegression
Random Forest Classifier

Random Forest is used to capture non-linear relationships and feature interactions.

Pipeline:

SimpleImputer
RandomForestClassifier
Regression Model

Linear Regression is used to estimate expected next-month returns.

The regression target is created in memory:

y_regression = next month's return_1m by asset

This target is not saved in model_dataset.csv.

Pipeline:

SimpleImputer
StandardScaler
LinearRegression
Train-Test Split

The project uses a time-aware train-test split.

The dataset is sorted by date and divided as:

Train: oldest 80% of dates
Test: most recent 20% of dates

This avoids temporal leakage, which is especially important in financial modeling.

Cross-Validation

Classification models use:

TimeSeriesSplit

This preserves temporal order during validation.

Classification Metrics

The classification models are evaluated with:

Precision
Recall
F1-score
ROC-AUC
Confusion Matrix
ROC Curve

The confusion matrix and ROC curve are saved as artifacts in:

reports/figures/
Regression Metrics

The regression model is evaluated with:

MAE
RMSE
R2

These metrics measure how accurately the model predicts next-month returns.

Model Outputs

Training generates:

reports/metrics/model_metrics.json
reports/figures/confusion_matrix_logistic_regression.png
reports/figures/confusion_matrix_random_forest.png
reports/figures/roc_curve_logistic_regression.png
reports/figures/roc_curve_random_forest.png
Expected Returns

The expected monthly returns are predicted by the Linear Regression model and used as inputs for portfolio optimization.

The prediction logic is implemented in:

src/models/predict_expected_returns.py
Known Limitations
Financial returns are noisy and difficult to predict.
The project uses only historical price-based features.
The risk-free rate is assumed to be zero.
The model does not include transaction costs.
The model does not include taxes.
The model does not include macroeconomic variables.
Disclaimer

This project is academic and technical. It is not financial advice.