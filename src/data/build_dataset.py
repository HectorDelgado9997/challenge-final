from pathlib import Path

import pandas as pd

from src.config.settings import (
    BENCHMARK_ASSET,
    MODEL_DATASET_PATH,
    MONTHLY_PRICES_PATH,
    PROCESSED_DATA_DIR,
)
from src.data.validate_data import (
    validate_file_exists,
    validate_model_dataset,
)
from src.features.feature_engineering import build_features
from src.utils.exceptions import DataValidationError, FeatureEngineeringError
from src.utils.logger import get_logger


logger = get_logger(__name__)


def load_monthly_prices(monthly_prices_path: Path = MONTHLY_PRICES_PATH) -> pd.DataFrame:
    """
    Load monthly prices dataset.

    Parameters
    ----------
    monthly_prices_path : Path
        Path to monthly_prices.csv.

    Returns
    -------
    pd.DataFrame
        Monthly prices DataFrame.
    """
    try:
        validate_file_exists(monthly_prices_path)

        monthly_prices = pd.read_csv(monthly_prices_path)

        logger.info(
            "Monthly prices loaded successfully from %s. Shape=%s",
            monthly_prices_path,
            monthly_prices.shape,
        )

        return monthly_prices

    except Exception as error:
        logger.exception("Failed to load monthly prices")
        raise DataValidationError("Failed to load monthly prices.") from error


def save_model_dataset(
    model_dataset: pd.DataFrame,
    output_path: Path = MODEL_DATASET_PATH,
) -> None:
    """
    Save model dataset to CSV.

    Parameters
    ----------
    model_dataset : pd.DataFrame
        Final model dataset.
    output_path : Path
        Output path.

    Returns
    -------
    None
    """
    try:
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

        model_dataset.to_csv(output_path, index=False)

        logger.info("Model dataset saved successfully to %s", output_path)

    except Exception as error:
        logger.exception("Failed to save model dataset")
        raise DataValidationError("Failed to save model dataset.") from error


def run_dataset_building(
    monthly_prices_path: Path = MONTHLY_PRICES_PATH,
    output_path: Path = MODEL_DATASET_PATH,
    benchmark_asset: str = BENCHMARK_ASSET,
) -> pd.DataFrame:
    """
    Run the complete dataset building pipeline.

    Parameters
    ----------
    monthly_prices_path : Path
        Path to monthly_prices.csv.
    output_path : Path
        Output path for model_dataset.csv.
    benchmark_asset : str
        Benchmark asset ticker.

    Returns
    -------
    pd.DataFrame
        Final model dataset.
    """
    try:
        logger.info("Starting model dataset construction pipeline")

        monthly_prices = load_monthly_prices(monthly_prices_path=monthly_prices_path)

        model_dataset = build_features(
            monthly_prices=monthly_prices,
            benchmark_asset=benchmark_asset,
        )

        validate_model_dataset(model_dataset)

        save_model_dataset(
            model_dataset=model_dataset,
            output_path=output_path,
        )

        logger.info(
            "Model dataset construction completed successfully. Shape=%s",
            model_dataset.shape,
        )

        return model_dataset

    except (DataValidationError, FeatureEngineeringError):
        logger.exception("Known error during dataset construction")
        raise

    except Exception as error:
        logger.exception("Unexpected error during dataset construction")
        raise DataValidationError("Dataset construction failed unexpectedly.") from error