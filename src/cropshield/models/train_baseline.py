"""
Baseline model training for CropShield.

Implements simple baseline predictors for yield anomaly regression and
severe-risk classification. Baselines serve as sanity-check comparators:
any ML model that cannot beat these is not adding value.

Baseline strategies
-------------------
1. ZeroAnomalyBaseline
   Predicts yield_anomaly = 0 for every county-year.
   Equivalent to assuming every year will match the historical trend.
   RMSE of this baseline reflects the natural year-to-year variability.

2. CountyMeanBaseline
   Predicts the mean yield_anomaly observed for that county in the
   training set. Slightly better than zero if counties have persistent
   above/below-trend tendencies.

These baselines are intentionally simple. A well-engineered ML model
should outperform them on held-out years.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ZeroAnomalyBaseline:
    """Baseline that predicts zero yield anomaly for all inputs.

    Attributes
    ----------
    name : str
        Human-readable model name.
    """

    name = "ZeroAnomalyBaseline"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ZeroAnomalyBaseline":
        """No-op fit — this model has no learned parameters."""
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return zero for every row."""
        return np.zeros(len(X))


class CountyMeanBaseline:
    """Baseline that predicts the training-set county mean yield anomaly.

    For counties not seen in training (OOV), falls back to the global
    training-set mean.

    Parameters
    ----------
    county_col : str
        Column name for county identifier in the feature matrix.
    """

    name = "CountyMeanBaseline"

    def __init__(self, county_col: str = "county_fips") -> None:
        self.county_col = county_col
        self._county_means: dict[str, float] = {}
        self._global_mean: float = 0.0

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "CountyMeanBaseline":
        """Learn county-level mean anomalies from training data."""
        # TODO: Implement county mean calculation
        # 1. Group X[county_col] with y, compute mean per county
        # 2. Store in self._county_means
        # 3. Compute overall mean as self._global_mean
        raise NotImplementedError("CountyMeanBaseline.fit is not yet implemented.")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return county mean or global mean if county unseen."""
        # TODO: Implement prediction
        raise NotImplementedError("CountyMeanBaseline.predict is not yet implemented.")


def train_baselines(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = "yield_anomaly",
) -> dict[str, Any]:
    """Train all baseline models and return fitted instances.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training split of the modeling panel.
    feature_cols : list[str]
        Feature column names.
    target_col : str
        Target column name.

    Returns
    -------
    dict
        Dictionary of ``{model_name: fitted_model_instance}``.
    """
    # TODO: Implement baseline training loop
    raise NotImplementedError("train_baselines is not yet implemented.")
