import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from src.config.settings import FIGURES_DIR, METRICS_DIR, MODEL_METRICS_PATH
from src.utils.logger import get_logger


logger = get_logger(__name__)


def evaluate_classification_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray,
) -> dict[str, float]:
    """
    Evaluate a binary classification model.

    Parameters
    ----------
    y_true : np.ndarray
        True target values.
    y_pred : np.ndarray
        Predicted class labels.
    y_pred_proba : np.ndarray
        Predicted probabilities for the positive class.

    Returns
    -------
    dict[str, float]
        Classification metrics.
    """
    metrics = {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_pred_proba),
    }

    return metrics


def evaluate_regression_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """
    Evaluate a regression model.

    Parameters
    ----------
    y_true : np.ndarray
        True target values.
    y_pred : np.ndarray
        Predicted values.

    Returns
    -------
    dict[str, float]
        Regression metrics.
    """
    rmse = mean_squared_error(y_true, y_pred) ** 0.5

    metrics = {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": rmse,
        "r2": r2_score(y_true, y_pred),
    }

    return metrics


def save_confusion_matrix_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    output_dir: Path = FIGURES_DIR,
) -> Path:
    """
    Save confusion matrix plot.

    Parameters
    ----------
    y_true : np.ndarray
        True target values.
    y_pred : np.ndarray
        Predicted class labels.
    model_name : str
        Model name.
    output_dir : Path
        Output directory.

    Returns
    -------
    Path
        Saved plot path.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    matrix = confusion_matrix(y_true, y_pred)
    display = ConfusionMatrixDisplay(confusion_matrix=matrix)

    display.plot()
    plt.title(f"Confusion Matrix - {model_name}")

    output_path = output_dir / f"confusion_matrix_{model_name}.png"
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    logger.info("Confusion matrix plot saved to %s", output_path)

    return output_path


def save_roc_curve_plot(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    model_name: str,
    output_dir: Path = FIGURES_DIR,
) -> Path:
    """
    Save ROC curve plot.

    Parameters
    ----------
    y_true : np.ndarray
        True target values.
    y_pred_proba : np.ndarray
        Predicted probabilities for positive class.
    model_name : str
        Model name.
    output_dir : Path
        Output directory.

    Returns
    -------
    Path
        Saved plot path.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    RocCurveDisplay.from_predictions(y_true, y_pred_proba)
    plt.title(f"ROC Curve - {model_name}")

    output_path = output_dir / f"roc_curve_{model_name}.png"
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    logger.info("ROC curve plot saved to %s", output_path)

    return output_path


def save_metrics_to_json(
    metrics: dict[str, Any],
    output_path: Path = MODEL_METRICS_PATH,
) -> None:
    """
    Save model metrics to JSON.

    Parameters
    ----------
    metrics : dict[str, Any]
        Metrics dictionary.
    output_path : Path
        Output path.

    Returns
    -------
    None
    """
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    logger.info("Metrics saved to %s", output_path)