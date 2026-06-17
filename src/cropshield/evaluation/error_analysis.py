"""
Error analysis for CropShield.

Examines where and when models fail to identify systematic biases,
geographic blind spots, and stress-condition breakdown.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def compute_residuals(
    df: pd.DataFrame,
    actual_col: str = "yield_anomaly",
    pred_col: str = "yield_anomaly_pred",
) -> pd.DataFrame:
    """Add residual columns to a predictions DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with actual and predicted yield anomaly columns.
    actual_col : str
        Column name for actual values.
    pred_col : str
        Column name for predicted values.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with added ``residual`` and ``abs_residual`` columns.
    """
    df = df.copy()
    df["residual"] = df[actual_col] - df[pred_col]
    df["abs_residual"] = df["residual"].abs()
    return df


def residuals_by_group(
    df: pd.DataFrame,
    group_col: str,
    residual_col: str = "residual",
) -> pd.DataFrame:
    """Summarise residuals grouped by a categorical column.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a residual column.
    group_col : str
        Column to group by (e.g. ``"state"`` or ``"year"``).
    residual_col : str
        Name of the residual column.

    Returns
    -------
    pd.DataFrame
        Summary table with columns: group_col, mean_residual,
        std_residual, mae, count.
    """
    # TODO: Implement grouped residual summary
    raise NotImplementedError("residuals_by_group is not yet implemented.")


def worst_predictions(
    df: pd.DataFrame,
    n: int = 20,
    abs_residual_col: str = "abs_residual",
) -> pd.DataFrame:
    """Return the N rows with the largest absolute prediction errors.

    Parameters
    ----------
    df : pd.DataFrame
        Predictions DataFrame with an absolute residual column.
    n : int
        Number of worst predictions to return.
    abs_residual_col : str
        Column name for absolute residuals.

    Returns
    -------
    pd.DataFrame
        Top-N worst predictions, sorted descending by abs_residual.
    """
    return df.nlargest(n, abs_residual_col)
