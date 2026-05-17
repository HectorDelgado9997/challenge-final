# Technical Execution Guide

## Prerequisites

| Tool       | Version recommended |
|------------|---------------------|
| Python     | 3.9+                |
| Git        | Any recent version  |
| Git Bash   | Windows users       |

---

## 1. Clone the Repository

```bash
git clone git@github.com:HectorDelgado9997/challenge-final.git
cd challenge-final
```

---

## 2. Create and Activate Virtual Environment

```bash
# Create the environment
python -m venv .venv

# Activate — Windows Git Bash
source .venv/Scripts/activate

# Activate — Linux / Mac
source .venv/bin/activate
```

> You should see `(.venv)` at the start of your terminal prompt.

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

| Package          | Purpose                              |
|------------------|--------------------------------------|
| pandas           | Data manipulation                    |
| numpy            | Numerical operations                 |
| scikit-learn     | Model training and evaluation        |
| yfinance         | Financial data extraction            |
| PyPortfolioOpt   | Portfolio optimization               |
| mlflow           | Experiment tracking                  |
| matplotlib       | Plots and visualizations             |
| seaborn          | Statistical visualizations           |
| joblib           | Model serialization                  |
| pydantic         | Data validation                      |
| python-dotenv    | Environment variable loading         |
| pytest           | Unit testing                         |

---

## 4. Prepare the Asset Details File

Make sure the asset universe file exists at:

```text
data/raw/asset_details.csv
```

Expected format:

```csv
asset,domain
SPY,US Equity
QQQ,US Technology
BTC-USD,Cryptocurrency
XLE,US Energy
GRID,Clean Energy
FLKR,Frontier Markets
GLD,Commodities
EWW,Mexico Equity
EWZ,Brazil Equity
```

---

## 5. Start the MLflow Server

Open a **first terminal** and run:

```bash
mlflow ui --host 127.0.0.1 --port 5000
```

> Keep this terminal open during the entire pipeline execution.
> Open `http://127.0.0.1:5000` to monitor experiments in real time.

---

## 6. Run the Full Pipeline

Open a **second terminal**, activate the virtual environment and run:

```bash
python scripts/05_run_full_pipeline.py --amount 10000
```

With a custom start date:

```bash
python scripts/05_run_full_pipeline.py --amount 10000 --start-date 2015-01-01
```

### What the full pipeline executes
Validate arguments (amount, date range)
│
▼
run_data_extraction()       → data/raw/monthly_prices.csv
│
▼
run_dataset_building()      → data/processed/model_dataset.csv
│
▼
train_and_evaluate_classifier()   [Logistic Regression]
train_and_evaluate_classifier()   [Random Forest]
train_and_evaluate_regression_model()  [Linear Regression]
│
▼
log_model_run() × 3         → MLflow experiment runs
save_metrics_to_json()      → reports/metrics/model_metrics.json
│
▼
run_portfolio_optimization() → data/outputs/recommended_allocation.csv

---

## 7. Run Scripts Individually

Each stage can also be executed independently:

```bash
# Step 1 — Extract financial data from yfinance
python scripts/01_extract_data.py

# Step 2 — Build model dataset with features and target
python scripts/02_build_dataset.py

# Step 3 — Train and evaluate all models
python scripts/03_train_models.py

# Step 4 — Run portfolio optimization
python scripts/04_optimize_portfolio.py --amount 10000
```

---

## 8. Check the Outputs

After a successful run the following files are generated:

```text
data/
├── raw/
│   └── monthly_prices.csv              ← Raw monthly prices from yfinance
├── processed/
│   └── model_dataset.csv               ← Feature-engineered dataset
└── outputs/
    └── recommended_allocation.csv      ← Final portfolio allocation

reports/
├── figures/
│   ├── confusion_matrix_logistic_regression.png
│   ├── confusion_matrix_random_forest.png
│   ├── roc_curve_logistic_regression.png
│   └── roc_curve_random_forest.png
└── metrics/
    └── model_metrics.json              ← All model metrics consolidated
```

---

## 9. Recommended Allocation Output

The final output `recommended_allocation.csv` contains one row per
allocated asset:

| Column             | Description                              |
|--------------------|------------------------------------------|
| `asset`            | Asset ticker                             |
| `weight`           | Portfolio weight (0.0 to 0.35)           |
| `expected_return`  | Predicted next-month return              |
| `allocated_amount` | Capital allocated (weight × amount)      |

Example for `--amount 10000`:

```csv
asset,weight,expected_return,allocated_amount
SPY,0.35,0.021,3500.0
GLD,0.30,0.018,3000.0
QQQ,0.20,0.015,2000.0
EWW,0.15,0.012,1500.0
```

---

## 10. Run the Tests

```bash
pytest -v
```

---

## Common Errors

| Error                              | Likely cause                          | Fix                                    |
|------------------------------------|---------------------------------------|----------------------------------------|
| `ModuleNotFoundError`              | Virtual env not activated             | Run `source .venv/Scripts/activate`    |
| `FileNotFoundError` on CSV        | asset_details.csv missing             | Add the file to `data/raw/`            |
| `DataExtractionError`             | No internet / yfinance rate limit     | Wait and retry                         |
| `PortfolioOptimizationError`      | max_sharpe failed                     | Fallback to min_volatility (automatic) |
| `mlflow.exceptions`               | MLflow server not running             | Run `mlflow ui --host 127.0.0.1 --port 5000` first |
| Port 5000 in use                  | Another process on port 5000          | Run `mlflow ui --port 5001`            |
| `Not enough unique dates`         | Dataset too short for time split      | Use `--start-date 2015-01-01`          |
