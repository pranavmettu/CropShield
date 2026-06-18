"""
Evaluation metrics for CropShield.

Provides regression and classification metric computation with results
formatted for JSON serialisation and human-readable reporting.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

logger = logging.getLogger(__name__)


def regression_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    model_name: str = "model",
) -> dict[str, float]:
    """Compute standard regression metrics.

    Parameters
    ----------
    y_true : array-like
        Ground-truth yield anomaly values.
    y_pred : array-like
        Predicted yield anomaly values.
    model_name : str
        Label for logging.

    Returns
    -------
    dict
        Dictionary with keys: ``rmse``, ``mae``, ``r2``.
    """
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    results = {"rmse": rmse, "mae": mae, "r2": r2}
    logger.info(
        "%s — RMSE: %.3f  MAE: %.3f  R²: %.3f",
        model_name,
        rmse,
        mae,
        r2,
    )
    return results


def classification_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    y_prob: pd.Series | np.ndarray | None = None,
    model_name: str = "model",
) -> dict[str, float]:
    """Compute binary classification metrics for severe-risk prediction.

    Returns
    -------
    dict
        Keys: ``accuracy``, ``precision``, ``recall``, ``f1``,
        ``tn``, ``fp``, ``fn``, ``tp``.
    """
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        precision_recall_fscore_support,
    )

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0,
    )
    acc = float(accuracy_score(y_true, y_pred))
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    results = {
        "accuracy": acc,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    logger.info(
        "%s — acc: %.3f  precision: %.3f  recall: %.3f  F1: %.3f",
        model_name, acc, precision, recall, f1,
    )
    return results


def save_metrics(
    metrics_dict: dict,
    output_path: str | Path = "reports/metrics.json",
) -> None:
    """Save a metrics dictionary to a JSON file.

    Parameters
    ----------
    metrics_dict : dict
        Nested dictionary of model names → metric dictionaries.
    output_path : str or Path
        Destination JSON file path.
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics_dict, f, indent=2)
    logger.info("Metrics saved to %s", output_path)
