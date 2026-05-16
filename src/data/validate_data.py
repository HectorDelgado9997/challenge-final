from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.config.settings import (
    ASSET_COLUMN,
    DATE_COLUMN,
    EXPECTED_ASSET_DETAILS_COLUMNS,
    EXPECTED_MODEL_COLUMNS,
)
from src.utils.exceptions import DataValidationError
from src.utils.logger import get_logger


logger = get_logger(__name__)


def validate_file_exists(file_path: Path) -> None:
    """
    Validate that a file exists.

    Parameters
    ----------
    file_path : Path
        File path to validate.

    Raises
    ------
    DataValidationError
        If the file does not exist.
    """
    if not isinstance(file_path, Path):
        raise DataValidationError("file_path must be a pathlib.Path object.")

    if not file_path.exists():
        raise DataValidationError(f"File does not exist: {file_path}")

    if not file_path.is_file():
        raise DataValidationError(f"Path is not a file: {file_path}")


def validate_required_columns(
    dataframe: pd.DataFrame,
    expected_columns: list[str],
    dataset_name: str,
) -> None:
    """
    Validate that a DataFrame contains the expected columns.

    Parameters
    ----------
    dataframe : pd.DataFrame
        DataFrame to validate.
    expected_columns : list[str]
        Required column names.
    dataset_name : str
        Human-readable dataset name.

    Raises
    ------
    DataValidationError
        If required columns are missing.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise DataValidationError(f"{dataset_name} must be a pandas DataFrame.")

    missing_columns = set(expected_columns) - set(dataframe.columns)

    if missing_columns:
        raise DataValidationError(
            f"{dataset_name} is missing required columns: {sorted(missing_columns)}"
        )


def validate_asset_details(dataframe: pd.DataFrame) -> None:
    """
    Validate asset details dataset.

    Expected columns:
    - asset
    - domain

    Parameters
    ----------
    dataframe : pd.DataFrame
        Asset details DataFrame.

    Raises
    ------
    DataValidationError
        If the dataset does not meet the required structure.
    """
    logger.info("Validating asset details dataset")

    validate_required_columns(
        dataframe=dataframe,
        expected_columns=EXPECTED_ASSET_DETAILS_COLUMNS,
        dataset_name="asset_details",
    )

    if dataframe.empty:
        raise DataValidationError("asset_details dataset is empty.")

    if dataframe[ASSET_COLUMN].isna().any():
        raise DataValidationError("asset_details contains null values in asset column.")

    if dataframe["domain"].isna().any():
        raise DataValidationError("asset_details contains null values in domain column.")

    if dataframe[ASSET_COLUMN].duplicated().any():
        duplicated_assets = dataframe.loc[
            dataframe[ASSET_COLUMN].duplicated(), ASSET_COLUMN
        ].tolist()
        raise DataValidationError(
            f"asset_details contains duplicated assets: {duplicated_assets}"
        )

    if not dataframe[ASSET_COLUMN].map(lambda value: isinstance(value, str)).all():
        raise DataValidationError("All asset values must be strings.")

    if not dataframe["domain"].map(lambda value: isinstance(value, str)).all():
        raise DataValidationError("All domain values must be strings.")

    logger.info("Asset details validation completed successfully")


def validate_date_column(dataframe: pd.DataFrame, dataset_name: str) -> None:
    """
    Validate that the date column exists and can be converted to datetime.

    Parameters
    ----------
    dataframe : pd.DataFrame
        DataFrame to validate.
    dataset_name : str
        Human-readable dataset name.

    Raises
    ------
    DataValidationError
        If date column is missing or invalid.
    """
    if DATE_COLUMN not in dataframe.columns:
        raise DataValidationError(f"{dataset_name} does not contain date column.")

    try:
        pd.to_datetime(dataframe[DATE_COLUMN])
    except Exception as error:
        raise DataValidationError(
            f"{dataset_name} contains invalid date values."
        ) from error


def validate_no_infinite_values(
    dataframe: pd.DataFrame,
    dataset_name: str,
) -> None:
    """
    Validate that numeric columns do not contain infinite values.

    Parameters
    ----------
    dataframe : pd.DataFrame
        DataFrame to validate.
    dataset_name : str
        Human-readable dataset name.

    Raises
    ------
    DataValidationError
        If infinite values are found.
    """
    numeric_dataframe = dataframe.select_dtypes(include=[np.number])

    if np.isinf(numeric_dataframe.to_numpy()).any():
        raise DataValidationError(f"{dataset_name} contains infinite numeric values.")


def validate_model_dataset(dataframe: pd.DataFrame) -> None:
    """
    Validate final model dataset.

    The final dataset must contain only the expected project columns.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Model dataset.

    Raises
    ------
    DataValidationError
        If the dataset does not meet the required structure.
    """
    logger.info("Validating model dataset")

    validate_required_columns(
        dataframe=dataframe,
        expected_columns=EXPECTED_MODEL_COLUMNS,
        dataset_name="model_dataset",
    )

    unexpected_columns = set(dataframe.columns) - set(EXPECTED_MODEL_COLUMNS)
    if unexpected_columns:
        raise DataValidationError(
            f"model_dataset contains unexpected columns: {sorted(unexpected_columns)}"
        )

    if dataframe.empty:
        raise DataValidationError("model_dataset is empty.")

    validate_date_column(dataframe=dataframe, dataset_name="model_dataset")

    if dataframe[ASSET_COLUMN].isna().any():
        raise DataValidationError("model_dataset contains null values in asset column.")

    duplicated_rows = dataframe.duplicated(subset=[DATE_COLUMN, ASSET_COLUMN])
    if duplicated_rows.any():
        raise DataValidationError(
            "model_dataset contains duplicated rows by date and asset."
        )

    numeric_columns = [
        column
        for column in EXPECTED_MODEL_COLUMNS
        if column not in [DATE_COLUMN, ASSET_COLUMN]
    ]

    for column in numeric_columns:
        if not pd.api.types.is_numeric_dtype(dataframe[column]):
            raise DataValidationError(f"Column {column} must be numeric.")

    validate_no_infinite_values(
        dataframe=dataframe,
        dataset_name="model_dataset",
    )

    target_values = set(dataframe["target_outperform_next_month"].dropna().unique())

    if not target_values.issubset({0, 1}):
        raise DataValidationError(
            "target_outperform_next_month must contain only 0 and 1."
        )

    if dataframe["target_outperform_next_month"].isna().any():
        raise DataValidationError(
            "target_outperform_next_month contains null values."
        )

    logger.info("Model dataset validation completed successfully")


def validate_date_range(
    start_date: str,
    end_date: Optional[str] = None,
) -> None:
    """
    Validate start and end dates.

    Parameters
    ----------
    start_date : str
        Start date in YYYY-MM-DD format.
    end_date : Optional[str]
        End date in YYYY-MM-DD format or None.

    Raises
    ------
    DataValidationError
        If dates are invalid or start_date is after end_date.
    """
    try:
        parsed_start_date = pd.to_datetime(start_date)
    except Exception as error:
        raise DataValidationError(
            f"Invalid start_date: {start_date}. Expected format: YYYY-MM-DD."
        ) from error

    if end_date is None:
        return

    try:
        parsed_end_date = pd.to_datetime(end_date)
    except Exception as error:
        raise DataValidationError(
            f"Invalid end_date: {end_date}. Expected format: YYYY-MM-DD."
        ) from error

    if parsed_start_date >= parsed_end_date:
        raise DataValidationError("start_date must be earlier than end_date.")


def validate_monthly_investment_amount(amount: float) -> None:
    """
    Validate monthly investment amount.

    Parameters
    ----------
    amount : float
        Monthly investment amount.

    Raises
    ------
    DataValidationError
        If amount is not positive.
    """
    if not isinstance(amount, (int, float)):
        raise DataValidationError("Monthly investment amount must be numeric.")

    if amount <= 0:
        raise DataValidationError("Monthly investment amount must be greater than zero.")


def validate_non_empty_asset_list(assets: list[str]) -> None:
    """
    Validate asset list.

    Parameters
    ----------
    assets : list[str]
        List of asset tickers.

    Raises
    ------
    DataValidationError
        If assets list is empty or invalid.
    """
    if not isinstance(assets, list):
        raise DataValidationError("Assets must be provided as a list.")

    if not assets:
        raise DataValidationError("Assets list cannot be empty.")

    invalid_assets = [
        asset for asset in assets if not isinstance(asset, str) or not asset.strip()
    ]

    if invalid_assets:
        raise DataValidationError(f"Invalid asset values found: {invalid_assets}")