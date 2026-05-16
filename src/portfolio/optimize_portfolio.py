from pathlib import Path

import pandas as pd
from pypfopt import EfficientFrontier

from src.config.settings import (
    ASSET_COLUMN,
    DATE_COLUMN,
    DEFAULT_MONTHLY_INVESTMENT_AMOUNT,
    MAX_ASSET_WEIGHT,
    MIN_ASSET_WEIGHT,
    RECOMMENDED_ALLOCATION_PATH,
    RISK_FREE_RATE,
    WEIGHT_BOUNDS,
)
from src.data.validate_data import validate_monthly_investment_amount
from src.models.predict_expected_returns import predict_expected_returns
from src.utils.exceptions import PortfolioOptimizationError
from src.utils.logger import get_logger


logger = get_logger(__name__)


def build_monthly_returns_matrix(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Build monthly returns matrix with dates as rows and assets as columns.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Model dataset containing date, asset and return_1m.

    Returns
    -------
    pd.DataFrame
        Monthly returns matrix.
    """
    try:
        required_columns = {DATE_COLUMN, ASSET_COLUMN, "return_1m"}
        missing_columns = required_columns - set(dataframe.columns)

        if missing_columns:
            raise PortfolioOptimizationError(
                f"Dataset is missing required columns: {sorted(missing_columns)}"
            )

        returns_matrix = dataframe.copy()
        returns_matrix[DATE_COLUMN] = pd.to_datetime(returns_matrix[DATE_COLUMN])

        returns_matrix = returns_matrix.pivot(
            index=DATE_COLUMN,
            columns=ASSET_COLUMN,
            values="return_1m",
        ).sort_index()

        returns_matrix = returns_matrix.dropna(axis=0, how="all")
        returns_matrix = returns_matrix.dropna(axis=1, how="all")

        if returns_matrix.empty:
            raise PortfolioOptimizationError("Monthly returns matrix is empty.")

        return returns_matrix

    except Exception as error:
        logger.exception("Failed to build monthly returns matrix")
        raise PortfolioOptimizationError(
            "Failed to build monthly returns matrix."
        ) from error


def calculate_covariance_matrix(returns_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate monthly covariance matrix.

    Parameters
    ----------
    returns_matrix : pd.DataFrame
        Monthly returns matrix.

    Returns
    -------
    pd.DataFrame
        Covariance matrix.
    """
    try:
        covariance_matrix = returns_matrix.cov()

        if covariance_matrix.empty:
            raise PortfolioOptimizationError("Covariance matrix is empty.")

        return covariance_matrix

    except Exception as error:
        logger.exception("Failed to calculate covariance matrix")
        raise PortfolioOptimizationError(
            "Failed to calculate covariance matrix."
        ) from error


def optimize_weights(
    expected_returns: pd.Series,
    covariance_matrix: pd.DataFrame,
) -> dict[str, float]:
    """
    Optimize portfolio weights using PyPortfolioOpt.

    Parameters
    ----------
    expected_returns : pd.Series
        Expected monthly returns indexed by asset.
    covariance_matrix : pd.DataFrame
        Monthly covariance matrix.

    Returns
    -------
    dict[str, float]
        Optimized clean weights.
    """
    try:
        common_assets = expected_returns.index.intersection(covariance_matrix.index)

        if len(common_assets) < 2:
            raise PortfolioOptimizationError(
                "At least two common assets are required for optimization."
            )

        expected_returns = expected_returns.loc[common_assets]
        covariance_matrix = covariance_matrix.loc[common_assets, common_assets]

        logger.info("Optimizing portfolio weights")
        logger.info("Weight bounds: min=%s max=%s", MIN_ASSET_WEIGHT, MAX_ASSET_WEIGHT)

        efficient_frontier = EfficientFrontier(
            expected_returns=expected_returns,
            cov_matrix=covariance_matrix,
            weight_bounds=WEIGHT_BOUNDS,
        )

        if expected_returns.max() > RISK_FREE_RATE:
            try:
                efficient_frontier.max_sharpe(risk_free_rate=RISK_FREE_RATE)
            except Exception:
                logger.warning("max_sharpe failed. Falling back to min_volatility.")
                efficient_frontier.min_volatility()
        else:
            logger.warning(
                "All expected returns are below or equal to risk-free rate. "
                "Using min_volatility."
            )
            efficient_frontier.min_volatility()

        clean_weights = efficient_frontier.clean_weights()

        if not clean_weights:
            raise PortfolioOptimizationError("Optimization returned empty weights.")

        logger.info("Optimized weights: %s", clean_weights)

        return clean_weights

    except Exception as error:
        logger.exception("Failed to optimize portfolio weights")
        raise PortfolioOptimizationError(
            "Failed to optimize portfolio weights."
        ) from error


def build_allocation_dataframe(
    weights: dict[str, float],
    expected_returns: pd.Series,
    monthly_investment_amount: float,
) -> pd.DataFrame:
    """
    Build allocation DataFrame.

    Parameters
    ----------
    weights : dict[str, float]
        Optimized asset weights.
    expected_returns : pd.Series
        Expected monthly returns indexed by asset.
    monthly_investment_amount : float
        Amount to invest.

    Returns
    -------
    pd.DataFrame
        Allocation recommendation.
    """
    try:
        validate_monthly_investment_amount(monthly_investment_amount)

        allocation = pd.DataFrame(
            {
                "asset": list(weights.keys()),
                "weight": list(weights.values()),
            }
        )

        allocation["expected_return"] = allocation["asset"].map(expected_returns)
        allocation["allocated_amount"] = (
            allocation["weight"] * monthly_investment_amount
        )

        allocation = allocation.sort_values("weight", ascending=False).reset_index(
            drop=True
        )

        allocation = allocation[allocation["weight"] > 0]

        if allocation.empty:
            raise PortfolioOptimizationError("Allocation DataFrame is empty.")

        return allocation

    except Exception as error:
        logger.exception("Failed to build allocation DataFrame")
        raise PortfolioOptimizationError(
            "Failed to build allocation DataFrame."
        ) from error


def calculate_portfolio_summary(
    allocation: pd.DataFrame,
    covariance_matrix: pd.DataFrame,
) -> dict[str, float]:
    """
    Calculate basic portfolio summary.

    Parameters
    ----------
    allocation : pd.DataFrame
        Allocation DataFrame.
    covariance_matrix : pd.DataFrame
        Monthly covariance matrix.

    Returns
    -------
    dict[str, float]
        Portfolio summary metrics.
    """
    try:
        assets = allocation["asset"].tolist()
        weights = allocation.set_index("asset")["weight"]

        selected_covariance = covariance_matrix.loc[assets, assets]

        expected_portfolio_return = (
            allocation["weight"] * allocation["expected_return"]
        ).sum()

        portfolio_variance = weights.T @ selected_covariance @ weights
        portfolio_volatility = portfolio_variance ** 0.5

        if portfolio_volatility == 0:
            portfolio_sharpe = 0.0
        else:
            portfolio_sharpe = (
                expected_portfolio_return - RISK_FREE_RATE
            ) / portfolio_volatility

        summary = {
            "expected_portfolio_return": float(expected_portfolio_return),
            "portfolio_volatility": float(portfolio_volatility),
            "portfolio_sharpe": float(portfolio_sharpe),
            "number_of_assets": float(len(assets)),
        }

        return summary

    except Exception as error:
        logger.exception("Failed to calculate portfolio summary")
        raise PortfolioOptimizationError(
            "Failed to calculate portfolio summary."
        ) from error


def save_allocation(
    allocation: pd.DataFrame,
    output_path: Path = RECOMMENDED_ALLOCATION_PATH,
) -> None:
    """
    Save allocation recommendation to CSV.

    Parameters
    ----------
    allocation : pd.DataFrame
        Allocation DataFrame.
    output_path : Path
        Output path.

    Returns
    -------
    None
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        allocation.to_csv(output_path, index=False)

        logger.info("Recommended allocation saved to %s", output_path)

    except Exception as error:
        logger.exception("Failed to save recommended allocation")
        raise PortfolioOptimizationError(
            "Failed to save recommended allocation."
        ) from error


def run_portfolio_optimization(
    dataframe: pd.DataFrame,
    monthly_investment_amount: float = DEFAULT_MONTHLY_INVESTMENT_AMOUNT,
    output_path: Path = RECOMMENDED_ALLOCATION_PATH,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """
    Run complete portfolio optimization pipeline.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Model dataset.
    monthly_investment_amount : float
        Amount to invest.
    output_path : Path
        Output path for allocation CSV.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, float]]
        Allocation DataFrame and portfolio summary.
    """
    try:
        logger.info("Starting portfolio optimization pipeline")

        validate_monthly_investment_amount(monthly_investment_amount)

        expected_returns = predict_expected_returns(dataframe)

        returns_matrix = build_monthly_returns_matrix(dataframe)

        covariance_matrix = calculate_covariance_matrix(returns_matrix)

        weights = optimize_weights(
            expected_returns=expected_returns,
            covariance_matrix=covariance_matrix,
        )

        allocation = build_allocation_dataframe(
            weights=weights,
            expected_returns=expected_returns,
            monthly_investment_amount=monthly_investment_amount,
        )

        summary = calculate_portfolio_summary(
            allocation=allocation,
            covariance_matrix=covariance_matrix,
        )

        save_allocation(
            allocation=allocation,
            output_path=output_path,
        )

        logger.info("Portfolio optimization completed successfully")
        logger.info("Portfolio summary: %s", summary)

        return allocation, summary

    except Exception as error:
        logger.exception("Portfolio optimization pipeline failed")
        raise PortfolioOptimizationError(
            "Portfolio optimization pipeline failed."
        ) from error