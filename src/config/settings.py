from pathlib import Path
from typing import Tuple


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
OUTPUT_DATA_DIR = DATA_DIR / "outputs"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"

MODELS_DIR = PROJECT_ROOT / "models"


# ============================================================
# Input and output files
# ============================================================

ASSET_DETAILS_PATH = RAW_DATA_DIR / "asset_details.csv"
MONTHLY_PRICES_PATH = RAW_DATA_DIR / "monthly_prices.csv"
MODEL_DATASET_PATH = PROCESSED_DATA_DIR / "model_dataset.csv"
RECOMMENDED_ALLOCATION_PATH = OUTPUT_DATA_DIR / "recommended_allocation.csv"

MODEL_METRICS_PATH = METRICS_DIR / "model_metrics.json"


# ============================================================
# Dataset configuration
# ============================================================

DATE_COLUMN = "date"
ASSET_COLUMN = "asset"
BENCHMARK_ASSET = "SPY"

EXPECTED_ASSET_DETAILS_COLUMNS = [
    "asset",
    "domain",
]

EXPECTED_MODEL_COLUMNS = [
    "date",
    "asset",
    "return_1m",
    "return_3m",
    "return_6m",
    "return_12m",
    "volatility_3m",
    "volatility_6m",
    "drawdown_3m",
    "drawdown_6m",
    "sharpe_3m",
    "sharpe_6m",
    "relative_strength_3m",
    "relative_strength_6m",
    "target_outperform_next_month",
]

FEATURE_COLUMNS = [
    "return_1m",
    "return_3m",
    "return_6m",
    "return_12m",
    "volatility_3m",
    "volatility_6m",
    "drawdown_3m",
    "drawdown_6m",
    "sharpe_3m",
    "sharpe_6m",
    "relative_strength_3m",
    "relative_strength_6m",
]

CLASSIFICATION_TARGET_COLUMN = "target_outperform_next_month"


# ============================================================
# Data extraction configuration
# ============================================================

DEFAULT_START_DATE = "2015-01-01"
DEFAULT_END_DATE = None
YFINANCE_INTERVAL = "1d"
MONTHLY_PRICE_COLUMN = "adjusted_close"


# ============================================================
# Train/test configuration
# ============================================================

TEST_SIZE = 0.20
N_SPLITS = 5
RANDOM_STATE = 42


# ============================================================
# Model configuration
# ============================================================

LOGISTIC_REGRESSION_PARAMS = {
    "max_iter": 1000,
    "random_state": RANDOM_STATE,
}

RANDOM_FOREST_PARAMS = {
    "n_estimators": 300,
    "max_depth": 5,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

LINEAR_REGRESSION_PARAMS = {}


# ============================================================
# Portfolio optimization configuration
# ============================================================

DEFAULT_MONTHLY_INVESTMENT_AMOUNT = 10000.0
MIN_ASSET_WEIGHT = 0.00
MAX_ASSET_WEIGHT = 0.35

WEIGHT_BOUNDS: Tuple[float, float] = (
    MIN_ASSET_WEIGHT,
    MAX_ASSET_WEIGHT,
)

RISK_FREE_RATE = 0.0


# ============================================================
# MLflow configuration
# ============================================================

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
MLFLOW_EXPERIMENT_NAME = "monthly-investment-recommendation"

LOGISTIC_REGRESSION_RUN_NAME = "logistic-regression-classifier"
RANDOM_FOREST_RUN_NAME = "random-forest-classifier"
LINEAR_REGRESSION_RUN_NAME = "linear-regression-expected-returns"
PORTFOLIO_OPTIMIZATION_RUN_NAME = "portfolio-optimization"


# ============================================================
# Runtime folders
# ============================================================

REQUIRED_DIRECTORIES = [
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    OUTPUT_DATA_DIR,
    FIGURES_DIR,
    METRICS_DIR,
]


def create_required_directories() -> None:
    """
    Create required project directories if they do not exist.

    Returns
    -------
    None
    """
    for directory in REQUIRED_DIRECTORIES:
        directory.mkdir(parents=True, exist_ok=True)