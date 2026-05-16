import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))


import pandas as pd

from src.config.settings import (
    FEATURE_COLUMNS,
    LINEAR_REGRESSION_PARAMS,
    LOGISTIC_REGRESSION_PARAMS,
    LOGISTIC_REGRESSION_RUN_NAME,
    MLFLOW_TRACKING_URI,
    MODEL_DATASET_PATH,
    RANDOM_FOREST_PARAMS,
    RANDOM_FOREST_RUN_NAME,
    LINEAR_REGRESSION_RUN_NAME,
    create_required_directories,
)
from src.data.validate_data import validate_file_exists, validate_model_dataset
from src.mlops.mlflow_tracking import configure_mlflow, log_model_run
from src.models.evaluate_models import save_metrics_to_json
from src.models.train_classification_models import (
    build_logistic_regression_pipeline,
    build_random_forest_pipeline,
    train_and_evaluate_classifier,
)
from src.models.train_regression_model import train_and_evaluate_regression_model
from src.utils.logger import get_logger


logger = get_logger(__name__)


def main() -> None:
    """
    Train classification and regression models and log results into MLflow.

    Returns
    -------
    None
    """
    create_required_directories()

    validate_file_exists(MODEL_DATASET_PATH)

    dataframe = pd.read_csv(MODEL_DATASET_PATH)
    validate_model_dataset(dataframe)

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

    logger.info("Model training and MLflow tracking completed successfully")
    logger.info("Inspect MLflow UI at: %s", MLFLOW_TRACKING_URI)


if __name__ == "__main__":
    main()