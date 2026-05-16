# Monthly Investment Recommendation

## Project Overview

Monthly Investment Recommendation is a local machine learning and portfolio optimization project designed to recommend a monthly capital allocation across a selected universe of financial assets.

The system downloads historical financial data from `yfinance`, builds a monthly modeling dataset, trains supervised machine learning models, tracks experiments with MLflow, and generates a recommended portfolio allocation using Modern Portfolio Theory through `PyPortfolioOpt`.

This project is part of the final machine learning challenge and follows a modular software engineering structure suitable for local execution in VS Code and GitHub-based collaboration.

## Objective

The objective is to build a system that recommends how to allocate a monthly investment amount across financial assets such as:

```text
SPY, QQQ, BTC-USD, XLE, GRID, FLKR, GLD, EWW, EWZ

The project applies:

Logistic Regression
Random Forest Classifier
Linear Regression
Modern Portfolio Theory
PyPortfolioOpt
MLflow for local experiment tracking

Technical Stack

Python
pandas
numpy
scikit-learn
yfinance
PyPortfolioOpt
MLflow
matplotlib
joblib
Git
GitHub
VS Code

Project Structure

monthly-investment-recommendation/
│
├── data/
│   ├── raw/
│   │   ├── asset_details.csv
│   │   └── monthly_prices.csv
│   │
│   ├── processed/
│   │   └── model_dataset.csv
│   │
│   └── outputs/
│       └── recommended_allocation.csv
│
├── docs/
│   ├── dataset_extraction.md
│   ├── model_construction.md
│   ├── mlops.md
│   ├── execution_guide.md
│   └── architecture.md
│
├── notebooks/
│   └── exploratory_analysis.ipynb
│
├── reports/
│   ├── figures/
│   └── metrics/
│
├── scripts/
│   ├── 01_extract_data.py
│   ├── 02_build_dataset.py
│   ├── 03_train_models.py
│   ├── 04_optimize_portfolio.py
│   └── 05_run_full_pipeline.py
│
├── src/
│   ├── config/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── portfolio/
│   ├── mlops/
│   └── utils/
│
├── tests/
├── .gitignore
├── requirements.txt
├── pyproject.toml
└── README.md

Dataset

The project uses historical financial prices downloaded from yfinance.

The asset universe is defined in:
data/raw/asset_details.csv

The final modeling dataset is generated in:
data/processed/model_dataset.csv

Models
Classification Models

The classification task predicts whether an asset will outperform the benchmark asset in the next month.

The benchmark asset is:

SPY

Models used:

Logistic Regression
Random Forest Classifier

Classification metrics:

Precision
Recall
F1-score
ROC-AUC
Confusion Matrix
ROC Curve

Regression Model

The regression task predicts the expected next-month return for each asset.

Model used:

Linear Regression

Regression metrics:

MAE
RMSE
R2

The predicted expected returns are used as inputs for portfolio optimization.

Portfolio Optimization

Portfolio optimization is implemented with PyPortfolioOpt.

The optimizer uses:

Expected returns predicted by Linear Regression
Monthly covariance matrix based on historical monthly returns
Maximum Sharpe Ratio optimization
Long-only constraints
Maximum weight per asset

The final output is saved in:

data/outputs/recommended_allocation.csv

The output contains:

asset
weight
expected_return
allocated_amount

MLflow Tracking

MLflow is used for local experiment tracking.

Start the MLflow UI with:

mlflow ui --host 127.0.0.1 --port 5000

Open:

http://127.0.0.1:5000

The project logs:

Model parameters
Model metrics
Model artifacts
Confusion matrix plots
ROC curve plots
Trained models
Model signatures
Input examples
Local Execution
1. Clone the repository
git clone git@github.com:HectorDelgado9997/challenge-final.git
cd challenge-final
2. Create virtual environment
python -m venv .venv

Activate it with Git Bash:

source .venv/Scripts/activate
3. Install dependencies
pip install -r requirements.txt
4. Run the full pipeline

Start MLflow first:

mlflow ui --host 127.0.0.1 --port 5000

In another terminal:

python scripts/05_run_full_pipeline.py --amount 10000

Optional custom start date:

python scripts/05_run_full_pipeline.py --amount 10000 --start-date 2015-01-01
Individual Script Execution
python scripts/01_extract_data.py
python scripts/02_build_dataset.py
python scripts/03_train_models.py
python scripts/04_optimize_portfolio.py --amount 10000
python scripts/05_run_full_pipeline.py --amount 10000
Expected Outputs
data/raw/monthly_prices.csv
data/processed/model_dataset.csv
data/outputs/recommended_allocation.csv
reports/figures/*.png
reports/metrics/model_metrics.json
MLflow experiment runs
Important Disclaimer

This project is for academic and technical purposes only. It is not financial advice. Historical returns do not guarantee future performance.