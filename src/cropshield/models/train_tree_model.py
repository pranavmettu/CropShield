"""
Tree-based model training for CropShield.

Trains RandomForest, XGBoost, and LightGBM regressors (and optionally
classifiers) on the county-crop-year modeling panel.

Important validation note
--------------------------
All model evaluation uses a temporal split: train on earlier years, test on
later years. Random train/test splits are **not** used as the primary result
because they allow future-year data to leak into the training set, producing
optimistically biased validation metrics.

Model artifacts
---------------
Trained models are saved to models/ using joblib for reproducibility.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

logger = logging.getLogger(__name__)

MODEL_DIR = Path("models")


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: dict[str, Any] | None = None,
) -> RandomForestRegressor:
    """Train a RandomForestRegressor on the provided training data.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix.
    y_train : pd.Series
        Training target (yield anomaly, bu/acre).
    params : dict, optional
        Hyperparameters to pass to RandomForestRegressor.
        Defaults to sensible values from model_config.yaml.

    Returns
    -------
    RandomForestRegressor
        Fitted sklearn model.
    """
    # TODO: Implement RF training
    # 1. Merge default params with provided params
    # 2. Instantiate RandomForestRegressor(**params)
    # 3. Fit on X_train, y_train
    # 4. Log OOB score if oob_score=True
    # 5. Return fitted model
    raise NotImplementedError("train_random_forest is not yet implemented.")


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame | None = None,
    y_val: pd.Series | None = None,
    params: dict[str, Any] | None = None,
) -> Any:
    """Train an XGBoost regressor with optional early stopping.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix.
    y_train : pd.Series
        Training target.
    X_val : pd.DataFrame, optional
        Validation feature matrix for early stopping.
    y_val : pd.Series, optional
        Validation target for early stopping.
    params : dict, optional
        XGBoost hyperparameters.

    Returns
    -------
    XGBRegressor
        Fitted XGBoost model.
    """
    # TODO: Implement XGBoost training with early stopping if val provided
    raise NotImplementedError("train_xgboost is not yet implemented.")


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame | None = None,
    y_val: pd.Series | None = None,
    params: dict[str, Any] | None = None,
) -> Any:
    """Train a LightGBM regressor with optional early stopping.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix.
    y_train : pd.Series
        Training target.
    X_val : pd.DataFrame, optional
        Validation feature matrix for early stopping.
    y_val : pd.Series, optional
        Validation target for early stopping.
    params : dict, optional
        LightGBM hyperparameters.

    Returns
    -------
    LGBMRegressor
        Fitted LightGBM model.
    """
    # TODO: Implement LightGBM training
    raise NotImplementedError("train_lightgbm is not yet implemented.")


def save_model(model: Any, name: str, model_dir: Path = MODEL_DIR) -> Path:
    """Serialise a fitted model to disk using joblib.

    Parameters
    ----------
    model : Any
        Fitted sklearn-compatible model.
    name : str
        Base name for the saved file (e.g. ``"random_forest"``).
    model_dir : Path
        Directory to save model artifacts.

    Returns
    -------
    Path
        Path to the saved model file.
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / f"{name}.joblib"
    joblib.dump(model, path)
    logger.info("Saved model to %s", path)
    return path


def load_model(name: str, model_dir: Path = MODEL_DIR) -> Any:
    """Load a serialised model from disk.

    Parameters
    ----------
    name : str
        Base name used when saving (e.g. ``"random_forest"``).
    model_dir : Path
        Directory containing model artifacts.

    Returns
    -------
    Any
        Fitted model object.
    """
    path = model_dir / f"{name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found: {path}")
    return joblib.load(path)
