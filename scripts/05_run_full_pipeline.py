import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))


import pandas as pd

from src.config.settings import (
    DEFAULT_END_DATE,
    DEFAULT_MONTHLY_INVESTMENT_AMOUNT,
    DEFAULT_START_DATE,
    FEATURE_COLUMNS,
    LINEAR_REGRESSION_PARAMS,
    LINEAR_REGRESSION_RUN_NAME,
    LOGISTIC_REGRESSION_PARAMS,
    LOGISTIC_REGRESSION_RUN_NAME,
    MLFLOW_TRACKING_URI,
    RANDOM_FOREST_PARAMS,
    RANDOM_FOREST_RUN_NAME,
    create_required_directories,
)
from src.data.build_dataset import run_dataset_building
from src.data.extract_data import run_data_extraction
from src.data.validate_data import (
    validate_date_range,
    validate_model_dataset,
    validate_monthly_investment_amount,
)
from src.mlops.mlflow_tracking import configure_mlflow, log_model_run
from src.models.evaluate_models import save_metrics_to_json
from src.models.train_classification_models import (
    build_logistic_regression_pipeline,
    build_random_forest_pipeline,
    train_and_evaluate_classifier,
)
from src.models.train_regression_model import train_and_evaluate_regression_model
from src.portfolio.optimize_portfolio import run_portfolio_optimization
from src.utils.exceptions import (
    DataExtractionError,
    DataValidationError,
    ModelTrainingError,
    PortfolioOptimizationError,
)
from src.utils.logger import get_logger


logger = get_logger(__name__)


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for the full pipeline.

    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run the complete Monthly Investment Recommendation pipeline."
    )

    parser.add_argument(
        "--amount",
        type=float,
        default=DEFAULT_MONTHLY_INVESTMENT_AMOUNT,
        help="Monthly investment amount used to calculate allocated capital.",
    )

    parser.add_argument(
        "--start-date",
        type=str,
        default=DEFAULT_START_DATE,
        help="Start date for yfinance extraction. Format: YYYY-MM-DD.",
    )

    parser.add_argument(
        "--end-date",
        type=str,
        default=DEFAULT_END_DATE,
        help="End date for yfinance extraction. Format: YYYY-MM-DD. Defaults to latest available.",
    )

    return parser.parse_args()


def train_models_with_mlflow(dataframe: pd.DataFrame) -> dict:
    """
    Train classification and regression models and log them into MLflow.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Final model dataset.

    Returns
    -------
    dict
        Dictionary containing model metrics.
    """
    try:
        logger.info("Starting model training block")

        configure_mlflow()

        input_example = dataframe[FEATURE_COLUMNS].head(5)

        logistic_regression_result = train_and_evaluate_classifier(
            dataframe=dataframe,
            model_name="logistic_regression",
            model=build_logistic_regression_pipeline(),
        )

        log_model_run(
            run_name=LOGISTIC_REGRESSION_RUN_NAME,
            model=logistic_regression_result["model"],
            model_type="classification",
            metrics=logistic_regression_result["metrics"],
            params=LOGISTIC_REGRESSION_PARAMS,
            artifacts=logistic_regression_result["artifacts"],
            input_example=input_example,
        )

        random_forest_result = train_and_evaluate_classifier(
            dataframe=dataframe,
            model_name="random_forest",
            model=build_random_forest_pipeline(),
        )

        log_model_run(
            run_name=RANDOM_FOREST_RUN_NAME,
            model=random_forest_result["model"],
            model_type="classification",
            metrics=random_forest_result["metrics"],
            params=RANDOM_FOREST_PARAMS,
            artifacts=random_forest_result["artifacts"],
            input_example=input_example,
        )

        linear_regression_result = train_and_evaluate_regression_model(
            dataframe=dataframe,
        )

        log_model_run(
            run_name=LINEAR_REGRESSION_RUN_NAME,
            model=linear_regression_result["model"],
            model_type="regression",
            metrics=linear_regression_result["metrics"],
            params=LINEAR_REGRESSION_PARAMS,
            artifacts=None,
            input_example=input_example,
        )

        metrics = {
            "logistic_regression": logistic_regression_result["metrics"],
            "random_forest": random_forest_result["metrics"],
            "linear_regression": linear_regression_result["metrics"],
        }

        save_metrics_to_json(metrics)

        logger.info("Model training block completed successfully")

        return metrics

    except Exception as error:
        logger.exception("Model training block failed")
        raise ModelTrainingError("Model training block failed.") from error


def main() -> None:
    """
    Run the complete project pipeline.

    Returns
    -------
    None
    """
    args = parse_arguments()

    try:
        validate_monthly_investment_amount(args.amount)
        validate_date_range(start_date=args.start_date, end_date=args.end_date)

        create_required_directories()

        logger.info("Starting full pipeline")
        logger.info("Monthly investment amount: %s", args.amount)
        logger.info("Start date: %s", args.start_date)
        logger.info("End date: %s", args.end_date)

        monthly_prices = run_data_extraction(
            start_date=args.start_date,
            end_date=args.end_date,
        )

        logger.info("Monthly prices generated. Shape=%s", monthly_prices.shape)

        model_dataset = run_dataset_building()

        validate_model_dataset(model_dataset)

        logger.info("Model dataset generated. Shape=%s", model_dataset.shape)

        metrics = train_models_with_mlflow(dataframe=model_dataset)

        allocation, portfolio_summary = run_portfolio_optimization(
            dataframe=model_dataset,
            monthly_investment_amount=args.amount,
        )

        logger.info("Full pipeline completed successfully")
        logger.info("MLflow UI: %s", MLFLOW_TRACKING_URI)
        logger.info("Model metrics: %s", metrics)
        logger.info("Recommended allocation:")
        logger.info("\n%s", allocation.to_string(index=False))
        logger.info("Portfolio summary: %s", portfolio_summary)

    except (
        DataExtractionError,
        DataValidationError,
        ModelTrainingError,
        PortfolioOptimizationError,
    ):
        logger.exception("Full pipeline failed because of a known project error")
        raise

    except Exception as error:
        logger.exception("Full pipeline failed because of an unexpected error")
        raise RuntimeError("Full pipeline failed unexpectedly.") from error


if __name__ == "__main__":
    main()