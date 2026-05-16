from typing import Any

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config.settings import (
    ASSET_COLUMN,
    DATE_COLUMN,
    FEATURE_COLUMNS,
    LINEAR_REGRESSION_PARAMS,
)
from src.models.evaluate_models import evaluate_regression_model
from src.models.train_classification_models import split_dataset_by_time
from src.utils.exceptions import ModelTrainingError
from src.utils.logger import get_logger


logger = get_logger(__name__)


def create_regression_target(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Create regression target in memory.

    The target is next month's return_1m by asset.
    This target is not persisted in model_dataset.csv.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Model dataset.

    Returns
    -------
    pd.DataFrame
        DataFrame with temporary y_regression column.
    """
    try:
        dataframe = dataframe.copy()
        dataframe[DATE_COLUMN] = pd.to_datetime(dataframe[DATE_COLUMN])
        dataframe = dataframe.sort_values([ASSET_COLUMN, DATE_COLUMN])

        dataframe["y_regression"] = dataframe.groupby(ASSET_COLUMN)["return_1m"].shift(
            -1
        )

        dataframe = dataframe.dropna(subset=["y_regression"])

        if dataframe.empty:
            raise ModelTrainingError(
                "Regression dataset is empty after creating target."
            )

        return dataframe

    except Exception as error:
        logger.exception("Failed to create regression target")
        raise ModelTrainingError("Failed to create regression target.") from error


def build_linear_regression_pipeline() -> Pipeline:
    """
    Build Linear Regression pipeline.

    Returns
    -------
    Pipeline
        Scikit-learn pipeline.
    """
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LinearRegression(**LINEAR_REGRESSION_PARAMS)),
        ]
    )


def get_regression_features_and_target(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separate regression features and target.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Dataset with temporary y_regression column.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        Feature matrix and regression target.
    """
    missing_features = set(FEATURE_COLUMNS) - set(dataframe.columns)
    if missing_features:
        raise ModelTrainingError(f"Missing feature columns: {sorted(missing_features)}")

    if "y_regression" not in dataframe.columns:
        raise ModelTrainingError("Missing temporary regression target y_regression.")

    X = dataframe[FEATURE_COLUMNS]
    y = dataframe["y_regression"]

    return X, y


def train_and_evaluate_regression_model(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    """
    Train and evaluate Linear Regression model.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Model dataset.

    Returns
    -------
    dict[str, Any]
        Trained model and metrics.
    """
    try:
        logger.info("Training Linear Regression model")

        regression_dataframe = create_regression_target(dataframe)

        train_dataframe, test_dataframe = split_dataset_by_time(regression_dataframe)

        X_train, y_train = get_regression_features_and_target(train_dataframe)
        X_test, y_test = get_regression_features_and_target(test_dataframe)

        model = build_linear_regression_pipeline()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        metrics = evaluate_regression_model(
            y_true=y_test,
            y_pred=y_pred,
        )

        logger.info("Linear Regression metrics: %s", metrics)

        return {
            "model_name": "linear_regression",
            "model": model,
            "metrics": metrics,
        }

    except Exception as error:
        logger.exception("Failed to train Linear Regression model")
        raise ModelTrainingError("Failed to train Linear Regression model.") from error