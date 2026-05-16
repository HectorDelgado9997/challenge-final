from pathlib import Path
from typing import Any, Optional

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature

from src.config.settings import (
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
)
from src.utils.exceptions import ModelTrainingError
from src.utils.logger import get_logger


logger = get_logger(__name__)


def configure_mlflow() -> None:
    """
    Configure MLflow tracking URI and experiment.

    Returns
    -------
    None
    """
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

        logger.info("MLflow tracking URI configured: %s", MLFLOW_TRACKING_URI)
        logger.info("MLflow experiment configured: %s", MLFLOW_EXPERIMENT_NAME)

    except Exception as error:
        logger.exception("Failed to configure MLflow")
        raise ModelTrainingError("Failed to configure MLflow.") from error


def log_model_run(
    run_name: str,
    model: Any,
    model_type: str,
    metrics: dict[str, float],
    params: Optional[dict[str, Any]] = None,
    artifacts: Optional[dict[str, str]] = None,
    input_example: Optional[pd.DataFrame] = None,
) -> None:
    """
    Log a model training run into MLflow.

    Parameters
    ----------
    run_name : str
        MLflow run name.
    model : Any
        Trained model object.
    model_type : str
        Human-readable model type.
    metrics : dict[str, float]
        Model metrics to log.
    params : Optional[dict[str, Any]]
        Model parameters to log.
    artifacts : Optional[dict[str, str]]
        Artifact name and file path mapping.
    input_example : Optional[pd.DataFrame]
        Sample input used to infer model signature.

    Returns
    -------
    None
    """
    try:
        logger.info("Starting MLflow run: %s", run_name)

        with mlflow.start_run(run_name=run_name):
            mlflow.log_param("model_type", model_type)

            if params:
                for param_name, param_value in params.items():
                    mlflow.log_param(param_name, param_value)

            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, float(metric_value))

            if artifacts:
                for artifact_name, artifact_path in artifacts.items():
                    path = Path(artifact_path)

                    if path.exists():
                        mlflow.log_artifact(
                            local_path=str(path),
                            artifact_path=artifact_name,
                        )
                    else:
                        logger.warning(
                            "Artifact path does not exist and will not be logged: %s",
                            artifact_path,
                        )

            signature = None

            if input_example is not None and not input_example.empty:
                prediction_example = model.predict(input_example)
                signature = infer_signature(input_example, prediction_example)

            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                signature=signature,
                input_example=input_example,
            )

            logger.info("MLflow run completed successfully: %s", run_name)

    except Exception as error:
        logger.exception("Failed to log MLflow run: %s", run_name)
        raise ModelTrainingError(f"Failed to log MLflow run: {run_name}") from error