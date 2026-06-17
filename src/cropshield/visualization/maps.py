"""
County-level choropleth map visualisation for CropShield.

Generates static (matplotlib/geopandas) and interactive (plotly) maps
of yield anomalies and risk classifications by county.

Requires geopandas and a county shapefile or the census TIGER/Line data.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def plot_yield_anomaly_map(
    predictions_df: pd.DataFrame,
    year: int,
    fips_col: str = "county_fips",
    anomaly_col: str = "yield_anomaly_pred",
    output_path: str | Path | None = None,
) -> None:
    """Plot a choropleth map of predicted yield anomaly for a given year.

    Parameters
    ----------
    predictions_df : pd.DataFrame
        Predictions with county FIPS and anomaly values.
    year : int
        Year to display.
    fips_col : str
        Column with 5-digit county FIPS codes.
    anomaly_col : str
        Column with yield anomaly predictions.
    output_path : str or Path, optional
        Save path for the figure. If None, displays interactively.
    """
    # TODO: Implement choropleth map using geopandas + matplotlib
    # or plotly.express.choropleth()
    raise NotImplementedError("plot_yield_anomaly_map is not yet implemented.")
