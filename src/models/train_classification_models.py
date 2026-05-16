from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config.settings import (
    CLASSIFICATION_TARGET_COLUMN,
    DATE_COLUMN,
    FEATURE_COLUMNS,
    LOGISTIC_REGRESSION_PARAMS,
    N_SPLITS,
    RANDOM_FOREST_PARAMS,
    TEST_SIZE,
)
from src.models.evaluate_models import (
    evaluate_classification_model,
    save_confusion_matrix_plot,
    save_roc_curve_plot,
)
from src.utils.exceptions import ModelTrainingError
from src.utils.logger import get_logger


logger = get_logger(__name__)


def split_dataset_by_time(
    dataframe: pd.DataFrame,
    test_size: float = TEST_SIZE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into train and test sets using time order.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Model dataset.
    test_size : float
        Test set proportion.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Train and test DataFrames.
    """
    if dataframe.empty:
        raise ModelTrainingError("Input dataframe is empty.")

    if not 0 < test_size < 1:
        raise ModelTrainingError("test_size must be between 0 and 1.")

    dataframe = dataframe.copy()
    dataframe[DATE_COLUMN] = pd.to_datetime(dataframe[DATE_COLUMN])

    unique_dates = sorted(dataframe[DATE_COLUMN].unique())

    if len(unique_dates) < 10:
        raise ModelTrainingError("Not enough unique dates to perform time split.")

    split_index = int(len(unique_dates) * (1 - test_size))
    train_dates = unique_dates[:split_index]
    test_dates = unique_dates[split_index:]

    train_dataframe = dataframe[dataframe[DATE_COLUMN].isin(train_dates)].copy()
    test_dataframe = dataframe[dataframe[DATE_COLUMN].isin(test_dates)].copy()

    if train_dataframe.empty or test_dataframe.empty:
        raise ModelTrainingError("Train or test dataset is empty after split.")

    return train_dataframe, test_dataframe


def get_features_and_target(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separate features and classification target.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Model dataset.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        Feature matrix and target vector.
    """
    missing_features = set(FEATURE_COLUMNS) - set(dataframe.columns)
    if missing_features:
        raise ModelTrainingError(f"Missing feature columns: {sorted(missing_features)}")

    if CLASSIFICATION_TARGET_COLUMN not in dataframe.columns:
        raise ModelTrainingError(
            f"Missing target column: {CLASSIFICATION_TARGET_COLUMN}"
        )

    X = dataframe[FEATURE_COLUMNS]
    y = dataframe[CLASSIFICATION_TARGET_COLUMN]

    return X, y


def build_logistic_regression_pipeline() -> Pipeline:
    """
    Build Logistic Regression pipeline.

    Returns
    -------
    Pipeline
        Scikit-learn pipeline.
    """
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(**LOGISTIC_REGRESSION_PARAMS),
            ),
        ]
    )


def build_random_forest_pipeline() -> Pipeline:
    """
    Build Random Forest pipeline.

    Returns
    -------
    Pipeline
        Scikit-learn pipeline.
    """
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(**RANDOM_FOREST_PARAMS),
            ),
        ]
    )


def run_time_series_cross_validation(
    model: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> dict[str, float]:
    """
    Run time-series cross-validation.

    Parameters
    ----------
    model : Pipeline
        Scikit-learn model pipeline.
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target.

    Returns
    -------
    dict[str, float]
        Mean cross-validation metrics.
    """
    n_splits = min(N_SPLITS, max(2, len(X_train) // 30))

    time_series_split = TimeSeriesSplit(n_splits=n_splits)

    scoring = {
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    cv_results = cross_validate(
        model,
        X_train,
        y_train,
        cv=time_series_split,
        scoring=scoring,
        error_score="raise",
    )

    cv_metrics = {
        "cv_precision": cv_results["test_precision"].mean(),
        "cv_recall": cv_results["test_recall"].mean(),
        "cv_f1": cv_results["test_f1"].mean(),
        "cv_roc_auc": cv_results["test_roc_auc"].mean(),
    }

    return cv_metrics


def train_and_evaluate_classifier(
    dataframe: pd.DataFrame,
    model_name: str,
    model: Pipeline,
) -> dict[str, Any]:
    """
    Train and evaluate a classification model.

    Parameters
    ----------
    dataframe : pd.DataFrame
        Model dataset.
    model_name : str
        Model name for reporting.
    model : Pipeline
        Scikit-learn pipeline.

    Returns
    -------
    dict[str, Any]
        Trained model, metrics and artifacts.
    """
    try:
        logger.info("Training classifier: %s", model_name)

        train_dataframe, test_dataframe = split_dataset_by_time(dataframe)

        X_train, y_train = get_features_and_target(train_dataframe)
        X_test, y_test = get_features_and_target(test_dataframe)

        cv_metrics = run_time_series_cross_validation(
            model=model,
            X_train=X_train,
            y_train=y_train,
        )

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        test_metrics = evaluate_classification_model(
            y_true=y_test,
            y_pred=y_pred,
            y_pred_proba=y_pred_proba,
        )

        confusion_matrix_path = save_confusion_matrix_plot(
            y_true=y_test,
            y_pred=y_pred,
            model_name=model_name,
        )

        roc_curve_path = save_roc_curve_plot(
            y_true=y_test,
            y_pred_proba=y_pred_proba,
            model_name=model_name,
        )

        all_metrics = {
            **cv_metrics,
            **test_metrics,
        }

        logger.info("Classifier %s metrics: %s", model_name, all_metrics)

        return {
            "model_name": model_name,
            "model": model,
            "metrics": all_metrics,
            "artifacts": {
                "confusion_matrix": str(confusion_matrix_path),
                "roc_curve": str(roc_curve_path),
            },
        }

    except Exception as error:
        logger.exception("Failed to train classifier: %s", model_name)
        raise ModelTrainingError(
            f"Failed to train classifier: {model_name}"
        ) from error