import numpy as np
import pandas as pd

from src.config.settings import (
    ASSET_COLUMN,
    BENCHMARK_ASSET,
    CLASSIFICATION_TARGET_COLUMN,
    DATE_COLUMN,
    EXPECTED_MODEL_COLUMNS,
    FEATURE_COLUMNS,
    MONTHLY_PRICE_COLUMN,
)
from src.utils.exceptions import FeatureEngineeringError
from src.utils.logger import get_logger


logger = get_logger(__name__)


def validate_monthly_prices_input(monthly_prices: pd.DataFrame) -> None:
    """
    Validate monthly prices input.

    Parameters
    ----------
    monthly_prices : pd.DataFrame
        Monthly prices DataFrame.

    Raises
    ------
    FeatureEngineeringError
        If required columns are missing or invalid.
    """
    required_columns = {DATE_COLUMN, ASSET_COLUMN, MONTHLY_PRICE_COLUMN}
    missing_columns = required_columns - set(monthly_prices.columns)

    if missing_columns:
        raise FeatureEngineeringError(
            f"monthly_prices is missing required columns: {sorted(missing_columns)}"
        )

    if monthly_prices.empty:
        raise FeatureEngineeringError("monthly_prices dataset is empty.")

    if monthly_prices[DATE_COLUMN].isna().any():
        raise FeatureEngineeringError("monthly_prices contains null dates.")

    if monthly_prices[ASSET_COLUMN].isna().any():
        raise FeatureEngineeringError("monthly_prices contains null assets.")

    if monthly_prices[MONTHLY_PRICE_COLUMN].isna().any():
        raise FeatureEngineeringError("monthly_prices contains null adjusted prices.")

    if not pd.api.types.is_numeric_dtype(monthly_prices[MONTHLY_PRICE_COLUMN]):
        raise FeatureEngineeringError(
            f"{MONTHLY_PRICE_COLUMN} must be a numeric column."
        )


def calculate_returns(monthly_prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate monthly and rolling cumulative returns.

    Parameters
    ----------
    monthly_prices : pd.DataFrame
        Monthly prices DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with return features.
    """
    try:
        logger.info("Calculating return features")

        dataframe = monthly_prices.copy()
        dataframe[DATE_COLUMN] = pd.to_datetime(dataframe[DATE_COLUMN])
        dataframe = dataframe.sort_values([ASSET_COLUMN, DATE_COLUMN])

        grouped_prices = dataframe.groupby(ASSET_COLUMN)[MONTHLY_PRICE_COLUMN]

        dataframe["return_1m"] = grouped_prices.pct_change(periods=1)
        dataframe["return_3m"] = grouped_prices.pct_change(periods=3)
        dataframe["return_6m"] = grouped_prices.pct_change(periods=6)
        dataframe["return_12m"] = grouped_prices.pct_change(periods=12)

        return dataframe

    except Exception as error:
        logger.exception("Failed to calculate returns")
        raise FeatureEngineeringError("Failed to calculate returns.") from error


def calculate_volatility(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate rolling volatility features.

    Parameters
    ----------
    dataframe : pd.DataFrame
        DataFrame containing return_1m.

    Returns
    -------
    pd.DataFrame
        DataFrame with volatility features.
    """
    try:
        logger.info("Calculating volatility features")

        dataframe = dataframe.copy()
        grouped_returns = dataframe.groupby(ASSET_COLUMN)["return_1m"]

        dataframe["volatility_3m"] = grouped_returns.transform(
            lambda series: series.rolling(window=3, min_periods=3).std()
        )

        dataframe["volatility_6m"] = grouped_returns.transform(
            lambda series: series.rolling(window=6, min_periods=6).std()
        )

        return dataframe

    except Exception as error:
        logger.exception("Failed to calculate volatility")
        raise FeatureEngineeringError("Failed to calculate volatility.") from error


def calculate_drawdown(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate rolling drawdown features.

    Drawdown is calculated as the percentage decline from the rolling maximum price.

    Parameters
    ----------
    dataframe : pd.DataFrame
        DataFrame containing adjusted_close.

    Returns
    -------
    pd.DataFrame
        DataFrame with drawdown features.
    """
    try:
        logger.info("Calculating drawdown features")

        dataframe = dataframe.copy()

        def rolling_drawdown(price_series: pd.Series, window: int) -> pd.Series:
            rolling_max = price_series.rolling(window=window, min_periods=window).max()
            return (price_series / rolling_max) - 1.0

        dataframe["drawdown_3m"] = dataframe.groupby(ASSET_COLUMN)[
            MONTHLY_PRICE_COLUMN
        ].transform(lambda series: rolling_drawdown(series, window=3))

        dataframe["drawdown_6m"] = dataframe.groupby(ASSET_COLUMN)[
            MONTHLY_PRICE_COLUMN
        ].transform(lambda series: rolling_drawdown(series, window=6))

        return dataframe

    except Exception as error:
        logger.exception("Failed to calculate drawdown")
        raise FeatureEngineeringError("Failed to calculate drawdown.") from error


def calculate_sharpe_ratio(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate simplified rolling Sharpe ratio features.

    A zero risk-free rate is assumed for this academic project.

    Parameters
    ----------
    dataframe : pd.DataFrame
        DataFrame containing return and volatility features.

    Returns
    -------
    pd.DataFrame
        DataFrame with Sharpe ratio features.
    """
    try:
        logger.info("Calculating Sharpe ratio features")

        dataframe = dataframe.copy()

        dataframe["sharpe_3m"] = dataframe["return_3m"] / dataframe["volatility_3m"]
        dataframe["sharpe_6m"] = dataframe["return_6m"] / dataframe["volatility_6m"]

        dataframe["sharpe_3m"] = dataframe["sharpe_3m"].replace(
            [np.inf, -np.inf],
            np.nan,
        )
        dataframe["sharpe_6m"] = dataframe["sharpe_6m"].replace(
            [np.inf, -np.inf],
            np.nan,
        )

        return dataframe

    except Exception as error:
        logger.exception("Failed to calculate Sharpe ratios")
        raise FeatureEngineeringError("Failed to calculate Sharpe ratios.") from error


def calculate_relative_strength(
    dataframe: pd.DataFrame,
    benchmark_asset: str = BENCHMARK_ASSET,
) -> pd.DataFrame:
    """
    Calculate relative strength versus benchmark.

    Parameters
    ----------
    dataframe : pd.DataFrame
        DataFrame containing return_3m and return_6m.
    benchmark_asset : str
        Benchmark asset ticker.

    Returns
    -------
    pd.DataFrame
        DataFrame with relative strength features.
    """
    try:
        logger.info("Calculating relative strength features")

        dataframe = dataframe.copy()

        benchmark_returns = dataframe.loc[
            dataframe[ASSET_COLUMN] == benchmark_asset,
            [DATE_COLUMN, "return_3m", "return_6m"],
        ].rename(
            columns={
                "return_3m": "benchmark_return_3m",
                "return_6m": "benchmark_return_6m",
            }
        )

        if benchmark_returns.empty:
            raise FeatureEngineeringError(
                f"Benchmark asset {benchmark_asset} was not found in the dataset."
            )

        dataframe = dataframe.merge(
            benchmark_returns,
            on=DATE_COLUMN,
            how="left",
        )

        dataframe["relative_strength_3m"] = (
            dataframe["return_3m"] - dataframe["benchmark_return_3m"]
        )
        dataframe["relative_strength_6m"] = (
            dataframe["return_6m"] - dataframe["benchmark_return_6m"]
        )

        dataframe = dataframe.drop(
            columns=["benchmark_return_3m", "benchmark_return_6m"]
        )

        return dataframe

    except Exception as error:
        logger.exception("Failed to calculate relative strength")
        raise FeatureEngineeringError(
            "Failed to calculate relative strength."
        ) from error


def create_classification_target(
    dataframe: pd.DataFrame,
    benchmark_asset: str = BENCHMARK_ASSET,
) -> pd.DataFrame:
    """
    Create binary target indicating whether an asset outperforms the benchmark next month.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Feature DataFrame.
    benchmark_asset : str
        Benchmark asset ticker.

    Returns
    -------
    pd.DataFrame
        DataFrame with target_outperform_next_month.
    """
    try:
        logger.info("Creating classification target")

        dataframe = dataframe.copy()
        dataframe = dataframe.sort_values([ASSET_COLUMN, DATE_COLUMN])

        dataframe["next_month_return"] = dataframe.groupby(ASSET_COLUMN)[
            "return_1m"
        ].shift(-1)

        benchmark_next_return = dataframe.loc[
            dataframe[ASSET_COLUMN] == benchmark_asset,
            [DATE_COLUMN, "next_month_return"],
        ].rename(columns={"next_month_return": "benchmark_next_month_return"})

        if benchmark_next_return.empty:
            raise FeatureEngineeringError(
                f"Benchmark asset {benchmark_asset} was not found in the dataset."
            )

        dataframe = dataframe.merge(
            benchmark_next_return,
            on=DATE_COLUMN,
            how="left",
        )

        dataframe[CLASSIFICATION_TARGET_COLUMN] = (
            dataframe["next_month_return"] > dataframe["benchmark_next_month_return"]
        ).astype(int)

        dataframe = dataframe.drop(
            columns=["next_month_return", "benchmark_next_month_return"]
        )

        return dataframe

    except Exception as error:
        logger.exception("Failed to create classification target")
        raise FeatureEngineeringError(
            "Failed to create classification target."
        ) from error


def clean_feature_dataset(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Clean feature dataset and keep only the expected model columns.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Feature DataFrame.

    Returns
    -------
    pd.DataFrame
        Clean model dataset.
    """
    try:
        logger.info("Cleaning feature dataset")

        dataframe = dataframe.copy()

        dataframe = dataframe.replace([np.inf, -np.inf], np.nan)

        final_columns = EXPECTED_MODEL_COLUMNS

        missing_columns = set(final_columns) - set(dataframe.columns)
        if missing_columns:
            raise FeatureEngineeringError(
                f"Feature dataset is missing final columns: {sorted(missing_columns)}"
            )

        dataframe = dataframe[final_columns]

        required_non_null_columns = FEATURE_COLUMNS + [CLASSIFICATION_TARGET_COLUMN]
        dataframe = dataframe.dropna(subset=required_non_null_columns)

        dataframe[CLASSIFICATION_TARGET_COLUMN] = dataframe[
            CLASSIFICATION_TARGET_COLUMN
        ].astype(int)

        dataframe = dataframe.sort_values([DATE_COLUMN, ASSET_COLUMN]).reset_index(
            drop=True
        )

        if dataframe.empty:
            raise FeatureEngineeringError(
                "Model dataset is empty after feature cleaning."
            )

        return dataframe

    except Exception as error:
        logger.exception("Failed to clean feature dataset")
        raise FeatureEngineeringError("Failed to clean feature dataset.") from error


def build_features(
    monthly_prices: pd.DataFrame,
    benchmark_asset: str = BENCHMARK_ASSET,
) -> pd.DataFrame:
    """
    Build the final feature dataset for modeling.

    Parameters
    ----------
    monthly_prices : pd.DataFrame
        Monthly prices DataFrame.
    benchmark_asset : str
        Benchmark asset ticker.

    Returns
    -------
    pd.DataFrame
        Final model dataset.
    """
    try:
        logger.info("Starting feature engineering pipeline")

        validate_monthly_prices_input(monthly_prices)

        dataframe = calculate_returns(monthly_prices)
        dataframe = calculate_volatility(dataframe)
        dataframe = calculate_drawdown(dataframe)
        dataframe = calculate_sharpe_ratio(dataframe)
        dataframe = calculate_relative_strength(
            dataframe=dataframe,
            benchmark_asset=benchmark_asset,
        )
        dataframe = create_classification_target(
            dataframe=dataframe,
            benchmark_asset=benchmark_asset,
        )
        dataframe = clean_feature_dataset(dataframe)

        logger.info("Feature engineering pipeline completed. Shape=%s", dataframe.shape)

        return dataframe

    except Exception as error:
        logger.exception("Feature engineering pipeline failed")
        raise FeatureEngineeringError("Feature engineering pipeline failed.") from error