# Execution Guide

## Objective

This guide explains how to clone, configure, run, and validate the Monthly Investment Recommendation project locally.

The project is designed to run in VS Code using a local Python virtual environment.

## 1. Clone the Repository

```bash
git clone git@github.com:HectorDelgado9997/challenge-final.git
cd challenge-final

If using HTTPS:

git clone https://github.com/HectorDelgado9997/challenge-final.git
cd challenge-final
2. Open the Project in VS Code
code .
3. Create a Virtual Environment
python -m venv .venv
4. Activate the Virtual Environment

Using Git Bash on Windows:

source .venv/Scripts/activate

Using PowerShell on Windows:

.venv\Scripts\Activate.ps1
5. Install Dependencies
pip install -r requirements.txt
6. Validate Project Structure

The repository should contain:

data/
docs/
notebooks/
reports/
scripts/
src/
tests/
README.md
requirements.txt
pyproject.toml
7. Run Data Extraction
python scripts/01_extract_data.py

Expected output:

data/raw/monthly_prices.csv

Validate:

python -c "import pandas as pd; df = pd.read_csv('data/raw/monthly_prices.csv'); print(df.head()); print(df.shape)"
8. Build the Model Dataset
python scripts/02_build_dataset.py

Expected output:

data/processed/model_dataset.csv

Validate:

python -c "import pandas as pd; df = pd.read_csv('data/processed/model_dataset.csv'); print(df.head()); print(df.shape); print(df.columns.tolist())"
9. Start MLflow

Open a terminal and run:

mlflow ui --host 127.0.0.1 --port 5000

Keep this terminal open.

Open in the browser:

http://127.0.0.1:5000
10. Train Models

In a second terminal, activate the virtual environment:

source .venv/Scripts/activate

Run:

python scripts/03_train_models.py

Expected outputs:

reports/metrics/model_metrics.json
reports/figures/confusion_matrix_logistic_regression.png
reports/figures/confusion_matrix_random_forest.png
reports/figures/roc_curve_logistic_regression.png
reports/figures/roc_curve_random_forest.png
MLflow runs

Validate:

cat reports/metrics/model_metrics.json
11. Optimize Portfolio
python scripts/04_optimize_portfolio.py --amount 10000

Expected output:

data/outputs/recommended_allocation.csv

Validate:

python -c "import pandas as pd; df = pd.read_csv('data/outputs/recommended_allocation.csv'); print(df)"
12. Run the Full Pipeline

Start MLflow first:

mlflow ui --host 127.0.0.1 --port 5000

In another terminal:

python scripts/05_run_full_pipeline.py --amount 10000

Optional:

python scripts/05_run_full_pipeline.py --amount 10000 --start-date 2015-01-01
13. Expected Final Outputs
data/raw/monthly_prices.csv
data/processed/model_dataset.csv
data/outputs/recommended_allocation.csv
reports/metrics/model_metrics.json
reports/figures/*.png
MLflow experiment runs