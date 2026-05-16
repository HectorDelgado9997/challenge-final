from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

from src.config.settings import (
    ASSET_COLUMN,
    ASSET_DETAILS_PATH,
    DEFAULT_END_DATE,
    DEFAULT_START_DATE,
    MONTHLY_PRICE_COLUMN,
    MONTHLY_PRICES_PATH,
    RAW_DATA_DIR,
    YFINANCE_INTERVAL,
)
from src.data.validate_data import (
    validate_asset_details,
    validate_date_range,
    validate_file_exists,
    validate_non_empty_asset_list,
)
from src.utils.exceptions import DataExtractionError
from src.utils.logger import get_logger


logger = get_logger(__name__)


def load_asset_details(asset_details_path: Path = ASSET_DETAILS_PATH) -> pd.DataFrame:
    """
    Load and validate the asset details file.

    Parameters
    ----------
    asset_details_path : Path
        Path to asset_details.csv.

    Returns
    -------
    pd.DataFrame
        Validated asset details DataFrame.
    """
    try:
        validate_file_exists(asset_details_path)

        asset_details = pd.read_csv(asset_details_path)

        validate_asset_details(asset_details)

        logger.info("Asset details loaded successfully from %s", asset_details_path)

        return asset_details

    except Exception as error:
        logger.exception("Failed to load asset details")
        raise DataExtractionError("Failed to load asset details.") from error


def get_asset_list(asset_details: pd.DataFrame) -> list[str]:
    """
    Extract asset tickers from asset details DataFrame.

    Parameters
    ----------
    asset_details : pd.DataFrame
        Asset details DataFrame.

    Returns
    -------
    list[str]
        List of asset tickers.
    """
    try:
        assets = asset_details[ASSET_COLUMN].dropna().astype(str).str.strip().tolist()

        validate_non_empty_asset_list(assets)

        logger.info("Asset list extracted: %s", assets)

        return assets

    except Exception as error:
        logger.exception("Failed to extract asset list")
        raise DataExtractionError("Failed to extract asset list.") from error


def download_daily_prices(
    assets: list[str],
    start_date: str = DEFAULT_START_DATE,
    end_date: Optional[str] = DEFAULT_END_DATE,
    interval: str = YFINANCE_INTERVAL,
) -> pd.DataFrame:
    """
    Download daily adjusted close prices from yfinance.

    Parameters
    ----------
    assets : list[str]
        List of asset tickers.
    start_date : str
        Start date in YYYY-MM-DD format.
    end_date : Optional[str]
        End date in YYYY-MM-DD format. If None, yfinance downloads until latest available date.
    interval : str
        yfinance interval.

    Returns
    -------
    pd.DataFrame
        Daily adjusted close prices in long format with columns:
        date, asset, adjusted_close.
    """
    try:
        validate_non_empty_asset_list(assets)
        validate_date_range(start_date=start_date, end_date=end_date)

        logger.info(
            "Downloading daily prices from yfinance. Assets=%s | Start=%s | End=%s",
            assets,
            start_date,
            end_date,
        )

        raw_prices = yf.download(
            tickers=assets,
            start=start_date,
            end=end_date,
            interval=interval,
            auto_adjust=False,
            progress=False,
            group_by="column",
            threads=True,
        )

        if raw_prices.empty:
            raise DataExtractionError("yfinance returned an empty DataFrame.")

        adjusted_close = _extract_adjusted_close(raw_prices=raw_prices, assets=assets)

        daily_prices = (
            adjusted_close.reset_index()
            .melt(
                id_vars="Date",
                var_name=ASSET_COLUMN,
                value_name=MONTHLY_PRICE_COLUMN,
            )
            .rename(columns={"Date": "date"})
        )

        daily_prices["date"] = pd.to_datetime(daily_prices["date"])
        daily_prices = daily_prices.dropna(subset=[MONTHLY_PRICE_COLUMN])
        daily_prices = daily_prices.sort_values(["asset", "date"]).reset_index(drop=True)

        if daily_prices.empty:
            raise DataExtractionError("Daily prices dataset is empty after cleaning.")

        logger.info("Daily prices downloaded successfully. Shape=%s", daily_prices.shape)

        return daily_prices

    except Exception as error:
        logger.exception("Failed to download daily prices")
        raise DataExtractionError("Failed to download daily prices from yfinance.") from error


def _extract_adjusted_close(raw_prices: pd.DataFrame, assets: list[str]) -> pd.DataFrame:
    """
    Extract adjusted close prices from yfinance output.

    Parameters
    ----------
    raw_prices : pd.DataFrame
        Raw yfinance output.
    assets : list[str]
        List of asset tickers.

    Returns
    -------
    pd.DataFrame
        Wide DataFrame with Date index and assets as columns.
    """
    if isinstance(raw_prices.columns, pd.MultiIndex):
        if "Adj Close" not in raw_prices.columns.get_level_values(0):
            raise DataExtractionError("Adj Close column not found in yfinance output.")

        adjusted_close = raw_prices["Adj Close"]

    else:
        if "Adj Close" not in raw_prices.columns:
            raise DataExtractionError("Adj Close column not found in yfinance output.")

        adjusted_close = raw_prices[["Adj Close"]]
        adjusted_close.columns = assets

    adjusted_close = adjusted_close.copy()

    missing_assets = set(assets) - set(adjusted_close.columns)

    if missing_assets:
        logger.warning("Some assets were not returned by yfinance: %s", missing_assets)

    return adjusted_close


def convert_daily_to_monthly_prices(daily_prices: pd.DataFrame) -> pd.DataFrame:
    """
    Convert daily prices to monthly prices using the last available price of each month.

    Parameters
    ----------
    daily_prices : pd.DataFrame
        Daily prices DataFrame with date, asset and adjusted_close columns.

    Returns
    -------
    pd.DataFrame
        Monthly prices DataFrame with date, asset and adjusted_close columns.
    """
    try:
        required_columns = {"date", ASSET_COLUMN, MONTHLY_PRICE_COLUMN}
        missing_columns = required_columns - set(daily_prices.columns)

        if missing_columns:
            raise DataExtractionError(
                f"Daily prices missing required columns: {sorted(missing_columns)}"
            )

        monthly_prices = daily_prices.copy()
        monthly_prices["date"] = pd.to_datetime(monthly_prices["date"])

        monthly_prices = (
            monthly_prices.set_index("date")
            .groupby(ASSET_COLUMN)[MONTHLY_PRICE_COLUMN]
            .resample("ME")
            .last()
            .reset_index()
        )

        monthly_prices = monthly_prices.dropna(subset=[MONTHLY_PRICE_COLUMN])
        monthly_prices = monthly_prices.sort_values(["asset", "date"]).reset_index(drop=True)

        if monthly_prices.empty:
            raise DataExtractionError("Monthly prices dataset is empty.")

        logger.info("Monthly prices generated successfully. Shape=%s", monthly_prices.shape)

        return monthly_prices

    except Exception as error:
        logger.exception("Failed to convert daily prices to monthly prices")
        raise DataExtractionError("Failed to convert daily prices to monthly prices.") from error


def save_monthly_prices(
    monthly_prices: pd.DataFrame,
    output_path: Path = MONTHLY_PRICES_PATH,
) -> None:
    """
    Save monthly prices dataset to CSV.

    Parameters
    ----------
    monthly_prices : pd.DataFrame
        Monthly prices DataFrame.
    output_path : Path
        Output CSV path.

    Returns
    -------
    None
    """
    try:
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

        monthly_prices.to_csv(output_path, index=False)

        logger.info("Monthly prices saved successfully to %s", output_path)

    except Exception as error:
        logger.exception("Failed to save monthly prices")
        raise DataExtractionError("Failed to save monthly prices.") from error


def run_data_extraction(
    asset_details_path: Path = ASSET_DETAILS_PATH,
    output_path: Path = MONTHLY_PRICES_PATH,
    start_date: str = DEFAULT_START_DATE,
    end_date: Optional[str] = DEFAULT_END_DATE,
) -> pd.DataFrame:
    """
    Run the complete data extraction pipeline.

    Parameters
    ----------
    asset_details_path : Path
        Path to asset_details.csv.
    output_path : Path
        Path where monthly_prices.csv will be saved.
    start_date : str
        Start date in YYYY-MM-DD format.
    end_date : Optional[str]
        End date in YYYY-MM-DD format. If None, latest available date is used.

    Returns
    -------
    pd.DataFrame
        Monthly prices DataFrame.
    """
    try:
        logger.info("Starting data extraction pipeline")

        asset_details = load_asset_details(asset_details_path=asset_details_path)

        assets = get_asset_list(asset_details=asset_details)

        daily_prices = download_daily_prices(
            assets=assets,
            start_date=start_date,
            end_date=end_date,
        )

        monthly_prices = convert_daily_to_monthly_prices(daily_prices=daily_prices)

        save_monthly_prices(
            monthly_prices=monthly_prices,
            output_path=output_path,
        )

        logger.info("Data extraction pipeline completed successfully")

        return monthly_prices

    except Exception as error:
        logger.exception("Data extraction pipeline failed")
        raise DataExtractionError("Data extraction pipeline failed.") from error