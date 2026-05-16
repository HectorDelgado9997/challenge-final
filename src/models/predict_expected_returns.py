from typing import Any

import pandas as pd

from src.config.settings import (
    ASSET_COLUMN,
    DATE_COLUMN,
    FEATURE_COLUMNS,
)
from src.models.train_regression_model import (
    build_linear_regression_pipeline,
    create_regression_target,
    get_regression_features_and_target,
)
from src.utils.exceptions import ModelTrainingError
from src.utils.logger import get_logger


logger = get_logger(__name__)


def get_latest_features_by_asset(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Get the latest available feature row for each asset.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Model dataset.

    Returns
    -------
    pd.DataFrame
        Latest feature row by asset.
    """
    try:
        dataframe = dataframe.copy()
        dataframe[DATE_COLUMN] = pd.to_datetime(dataframe[DATE_COLUMN])

        latest_features = (
            dataframe.sort_values([ASSET_COLUMN, DATE_COLUMN])
            .groupby(ASSET_COLUMN)
            .tail(1)
            .reset_index(drop=True)
        )

        if latest_features.empty:
            raise ModelTrainingError("Latest features dataset is empty.")

        return latest_features

    except Exception as error:
        logger.exception("Failed to get latest features by asset")
        raise ModelTrainingError("Failed to get latest features by asset.") from error


def train_expected_return_model(dataframe: pd.DataFrame) -> Any:
    """
    Train Linear Regression model using next-month return as temporary target.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Model dataset.

    Returns
    -------
    Any
        Trained scikit-learn pipeline.
    """
    try:
        logger.info("Training expected return model")

        regression_dataframe = create_regression_target(dataframe)

        X, y = get_regression_features_and_target(regression_dataframe)

        model = build_linear_regression_pipeline()
        model.fit(X, y)

        logger.info("Expected return model trained successfully")

        return model

    except Exception as error:
        logger.exception("Failed to train expected return model")
        raise ModelTrainingError("Failed to train expected return model.") from error


def predict_expected_returns(dataframe: pd.DataFrame) -> pd.Series:
    """
    Predict expected monthly returns for the latest available month.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Model dataset.

    Returns
    -------
    pd.Series
        Predicted expected returns indexed by asset.
    """
    try:
        logger.info("Predicting expected monthly returns")

        model = train_expected_return_model(dataframe)

        latest_features = get_latest_features_by_asset(dataframe)

        X_latest = latest_features[FEATURE_COLUMNS]

        predicted_returns = model.predict(X_latest)

        expected_returns = pd.Series(
            data=predicted_returns,
            index=latest_features[ASSET_COLUMN],
            name="expected_return",
        )

        expected_returns = expected_returns.sort_index()

        if expected_returns.empty:
            raise ModelTrainingError("Expected returns prediction is empty.")

        logger.info("Expected returns predicted successfully")
        logger.info("Expected returns: %s", expected_returns.to_dict())

        return expected_returns

    except Exception as error:
        logger.exception("Failed to predict expected returns")
        raise ModelTrainingError("Failed to predict expected returns.") from error