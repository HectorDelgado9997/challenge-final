import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))


import pandas as pd

from src.config.settings import MODEL_DATASET_PATH, create_required_directories
from src.data.validate_data import validate_file_exists, validate_model_dataset
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
    Train classification and regression models.

    Returns
    -------
    None
    """
    create_required_directories()

    validate_file_exists(MODEL_DATASET_PATH)

    dataframe = pd.read_csv(MODEL_DATASET_PATH)
    validate_model_dataset(dataframe)

    logistic_regression_result = train_and_evaluate_classifier(
        dataframe=dataframe,
        model_name="logistic_regression",
        model=build_logistic_regression_pipeline(),
    )

    random_forest_result = train_and_evaluate_classifier(
        dataframe=dataframe,
        model_name="random_forest",
        model=build_random_forest_pipeline(),
    )

    linear_regression_result = train_and_evaluate_regression_model(
        dataframe=dataframe,
    )

    metrics = {
        "logistic_regression": logistic_regression_result["metrics"],
        "random_forest": random_forest_result["metrics"],
        "linear_regression": linear_regression_result["metrics"],
    }

    save_metrics_to_json(metrics)

    logger.info("Model training completed successfully")


if __name__ == "__main__":
    main()