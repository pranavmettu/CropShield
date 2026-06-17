"""
Diagnostic plots for CropShield.

Generates model evaluation and data exploration figures saved to
reports/figures/.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FIGURES_DIR = Path("reports/figures")


def plot_predicted_vs_actual(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    model_name: str = "Model",
    output_path: str | Path | None = "reports/figures/predicted_vs_actual.png",
) -> plt.Figure:
    """Scatter plot of predicted vs actual yield anomaly.

    Parameters
    ----------
    y_true : array-like
        Actual yield anomaly values.
    y_pred : array-like
        Predicted yield anomaly values.
    model_name : str
        Label for the plot title.
    output_path : str or Path, optional
        Path to save the figure.

    Returns
    -------
    plt.Figure
    """
    # TODO: Implement predicted vs actual scatter
    # 1. Create scatter plot with identity line
    # 2. Label axes, add R² annotation
    # 3. Save and return figure
    raise NotImplementedError("plot_predicted_vs_actual is not yet implemented.")


def plot_residuals_by_year(
    df: pd.DataFrame,
    year_col: str = "year",
    residual_col: str = "residual",
    output_path: str | Path | None = "reports/figures/residuals_by_year.png",
) -> plt.Figure:
    """Box plot of residuals grouped by year.

    Parameters
    ----------
    df : pd.DataFrame
        Predictions DataFrame with year and residual columns.
    year_col : str
        Year column name.
    residual_col : str
        Residual column name.
    output_path : str or Path, optional
        Path to save the figure.

    Returns
    -------
    plt.Figure
    """
    # TODO: Implement residuals by year box plot
    raise NotImplementedError("plot_residuals_by_year is not yet implemented.")


def plot_residuals_by_state(
    df: pd.DataFrame,
    state_col: str = "state",
    residual_col: str = "residual",
    output_path: str | Path | None = "reports/figures/residuals_by_state.png",
) -> plt.Figure:
    """Box plot of residuals grouped by state.

    Parameters
    ----------
    df : pd.DataFrame
        Predictions DataFrame with state and residual columns.
    state_col : str
        State column name.
    residual_col : str
        Residual column name.
    output_path : str or Path, optional
        Path to save the figure.

    Returns
    -------
    plt.Figure
    """
    # TODO: Implement residuals by state box plot
    raise NotImplementedError("plot_residuals_by_state is not yet implemented.")
