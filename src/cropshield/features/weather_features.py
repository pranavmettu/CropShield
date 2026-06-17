"""
Growing-season weather feature engineering for CropShield.

Converts raw daily NASA POWER weather records into county-year growing-season
aggregates suitable for machine learning.

Growing season definition (MVP)
--------------------------------
April 1 through August 31 of each year.

Feature catalogue
-----------------
cumulative_precip      : Total precipitation (mm) over the growing season.
mean_temp              : Mean daily temperature (°C) over the growing season.
max_temp               : Maximum single-day temperature (°C).
extreme_heat_days      : Days with Tmax ≥ 35 °C.
dry_days               : Days with precipitation ≤ 1 mm.
longest_dry_spell      : Longest consecutive dry-day run (days).
growing_degree_days    : Sum of max(0, Tavg - base_temp) over the growing season.
precip_anomaly         : Departure from county's historical mean growing-season
                         precipitation (mm), using only prior years to avoid leakage.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

GROWING_SEASON_START_MONTH = 4   # April
GROWING_SEASON_END_MONTH = 8     # August
GDD_BASE_TEMP_C = 10.0           # Growing degree day base (°C)
EXTREME_HEAT_THRESHOLD_C = 35.0  # Maximum daily temperature threshold (°C)
DRY_DAY_THRESHOLD_MM = 1.0       # Daily precipitation threshold for "dry" (mm)


def filter_growing_season(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Keep only rows falling within the growing season window.

    Parameters
    ----------
    df : pd.DataFrame
        Daily weather records with a date column.
    date_col : str
        Name of the date column.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame containing only growing-season rows.
    """
    # TODO: Parse date_col to datetime, filter month in [4, 5, 6, 7, 8]
    raise NotImplementedError("filter_growing_season is not yet implemented.")


def cumulative_precip(daily: pd.Series) -> float:
    """Return total precipitation (mm) over a growing season."""
    return float(daily.sum())


def mean_temp(daily_tavg: pd.Series) -> float:
    """Return mean daily temperature (°C) over a growing season."""
    return float(daily_tavg.mean())


def max_temp(daily_tmax: pd.Series) -> float:
    """Return the maximum single-day temperature (°C) over a growing season."""
    return float(daily_tmax.max())


def extreme_heat_days(
    daily_tmax: pd.Series,
    threshold: float = EXTREME_HEAT_THRESHOLD_C,
) -> int:
    """Count days with Tmax above the heat stress threshold."""
    return int((daily_tmax >= threshold).sum())


def dry_days(
    daily_precip: pd.Series,
    threshold: float = DRY_DAY_THRESHOLD_MM,
) -> int:
    """Count days with precipitation at or below the dry-day threshold."""
    return int((daily_precip <= threshold).sum())


def longest_dry_spell(
    daily_precip: pd.Series,
    threshold: float = DRY_DAY_THRESHOLD_MM,
) -> int:
    """Return the length (days) of the longest consecutive dry-day run.

    A dry day is defined as a day where precipitation ≤ ``threshold`` mm.

    Parameters
    ----------
    daily_precip : pd.Series
        Ordered daily precipitation values (mm).
    threshold : float
        Precipitation threshold below which a day is considered dry.

    Returns
    -------
    int
        Length in days of the longest consecutive dry spell.
        Returns 0 if there are no dry days.

    Examples
    --------
    >>> s = pd.Series([0.0, 0.5, 0.0, 0.0, 5.0, 0.0])
    >>> longest_dry_spell(s)
    2
    """
    # TODO: Implement consecutive dry-day run detection
    # Hint: Use (daily_precip <= threshold).astype(int), then detect runs via
    # cumsum trick or itertools.groupby
    raise NotImplementedError("longest_dry_spell is not yet implemented.")


def growing_degree_days(
    daily_tavg: pd.Series,
    base_temp: float = GDD_BASE_TEMP_C,
) -> float:
    """Compute accumulated growing degree days (GDD) over a growing season.

    GDD = Σ max(0, Tavg - base_temp)

    Parameters
    ----------
    daily_tavg : pd.Series
        Daily average temperature values (°C).
    base_temp : float
        Base temperature below which no growth is assumed (°C).

    Returns
    -------
    float
        Total GDD for the period.

    Examples
    --------
    >>> s = pd.Series([8.0, 12.0, 15.0, 9.0])
    >>> growing_degree_days(s, base_temp=10.0)
    7.0
    """
    return float(np.maximum(0.0, daily_tavg - base_temp).sum())


def precip_anomaly_from_history(
    county_year_df: pd.DataFrame,
    year: int,
    precip_col: str = "cumulative_precip",
    group_cols: list[str] | None = None,
) -> float:
    """Compute growing-season precipitation anomaly using only prior-year history.

    Parameters
    ----------
    county_year_df : pd.DataFrame
        DataFrame of county-level growing-season precip summaries across years.
        Must contain ``year`` and ``precip_col`` columns.
    year : int
        Target year for which the anomaly is computed.
    precip_col : str
        Column name of the cumulative precipitation feature.
    group_cols : list[str], optional
        Columns to group by (in addition to year) when computing the baseline.

    Returns
    -------
    float
        Precipitation anomaly (mm) = current_year_precip - historical_mean.
        Returns NaN if there is insufficient prior history.
    """
    # TODO: Implement leakage-safe precip anomaly
    # Steps:
    # 1. Filter to prior years only: county_year_df[county_year_df['year'] < year]
    # 2. Compute mean of precip_col over prior years
    # 3. Return current - mean (or NaN if no prior data)
    raise NotImplementedError("precip_anomaly_from_history is not yet implemented.")


def compute_weather_features(
    daily_df: pd.DataFrame,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Aggregate daily weather data into county-year growing-season features.

    Parameters
    ----------
    daily_df : pd.DataFrame
        Daily weather records with columns: ``county_fips``, ``state``,
        ``year``, ``date``, ``PRECTOTCORR``, ``T2M``, ``T2M_MIN``, ``T2M_MAX``.
    group_cols : list[str], optional
        Columns to group on. Defaults to ``["county_fips", "state", "year"]``.

    Returns
    -------
    pd.DataFrame
        One row per (county, year) with all weather features.
    """
    # TODO: Implement feature aggregation
    # Steps:
    # 1. Filter to growing season via filter_growing_season()
    # 2. Group by group_cols
    # 3. For each group, call the individual feature functions
    # 4. Assemble results into a single wide DataFrame
    # 5. Add precip_anomaly using precip_anomaly_from_history() (requires second pass)
    raise NotImplementedError("compute_weather_features is not yet implemented.")
