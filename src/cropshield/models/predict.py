"""
Prediction utilities for CropShield.

Loads trained model artifacts and generates predictions on new data,
saving results to data/processed/predictions.csv.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from cropshield.models.train_tree_model import load_model

logger = logging.getLogger(__name__)


def predict_yield_anomaly(
    panel_df: pd.DataFrame,
    feature_cols: list[str],
    model_name: str = "xgboost",
    model_dir: Path = Path("models"),
) -> pd.Series:
    """Load a trained model and generate yield anomaly predictions.

    Parameters
    ----------
    panel_df : pd.DataFrame
        Feature matrix (may include non-feature columns).
    feature_cols : list[str]
        Column names to use as model inputs.
    model_name : str
        Name of the saved model artifact (without ``.joblib`` extension).
    model_dir : Path
        Directory containing model artifacts.

    Returns
    -------
    pd.Series
        Predicted yield anomaly values (bu/acre).
    """
    # TODO: Implement prediction
    # 1. Load model via load_model()
    # 2. Subset panel_df to feature_cols
    # 3. Check for NaN features and warn
    # 4. Call model.predict()
    # 5. Return as pd.Series
    raise NotImplementedError("predict_yield_anomaly is not yet implemented.")


def save_predictions(
    panel_df: pd.DataFrame,
    predictions: pd.Series,
    output_path: str | Path = "data/processed/predictions.csv",
) -> None:
    """Attach predictions to the panel and save to disk.

    Parameters
    ----------
    panel_df : pd.DataFrame
        Original panel with identifier columns.
    predictions : pd.Series
        Predicted values aligned with panel_df index.
    output_path : str or Path
        Destination CSV path.
    """
    # TODO: Attach predictions and save
    raise NotImplementedError("save_predictions is not yet implemented.")
