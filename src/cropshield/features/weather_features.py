"""
Growing-season weather feature engineering for CropShield.

Converts raw daily NASA POWER weather records into county-year growing-season
aggregates suitable for machine learning.

Growing season definition (MVP)
--------------------------------
April 1 through August 31 of each year.

Feature catalogue
-----------------
cumulative_precip   : Total precipitation (mm) over the growing season.
mean_temp           : Mean daily temperature (°C) over the growing season.
max_temp            : Maximum single-day temperature (°C).
extreme_heat_days   : Days with Tmax ≥ 35 °C.
dry_days            : Days with precipitation ≤ 1 mm.
longest_dry_spell   : Longest consecutive dry-day run (days).
growing_degree_days : Sum of max(0, Tavg - base_temp) over the growing season.
precip_anomaly      : Departure from county's historical mean growing-season
                      precipitation, computed from prior years only (leakage-safe).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

GROWING_SEASON_START_MONTH = 4   # April
GROWING_SEASON_END_MONTH   = 8   # August
GDD_BASE_TEMP_C            = 10.0
EXTREME_HEAT_THRESHOLD_C   = 35.0
DRY_DAY_THRESHOLD_MM       = 1.0

GROUP_COLS = ["county_fips", "state_fips", "year"]


# ── Season filter ─────────────────────────────────────────────────────────────

def filter_growing_season(
    df: pd.DataFrame,
    date_col: str = "date",
    cutoff_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Keep only rows that fall within the April–August growing season.

    Optionally applies an as-of-date cutoff so that features computed from
    a partially-elapsed season do not leak future weather observations.

    Parameters
    ----------
    df : pd.DataFrame
        Daily weather records with a date column.
    date_col : str
        Name of the date column.
    cutoff_date : str or Timestamp, optional
        If provided, records *after* this date are excluded even if they fall
        within the April–August window.  Useful for computing mid-season
        features and for testing that future observations cannot affect past
        feature values.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame containing only growing-season rows up to
        (and including) ``cutoff_date``.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    month = df[date_col].dt.month
    mask = (month >= GROWING_SEASON_START_MONTH) & (month <= GROWING_SEASON_END_MONTH)
    if cutoff_date is not None:
        cutoff_ts = pd.Timestamp(cutoff_date)
        mask = mask & (df[date_col] <= cutoff_ts)
    return df[mask].reset_index(drop=True)


# ── Individual feature functions ──────────────────────────────────────────────

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
    """Count days with Tmax at or above the heat stress threshold."""
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
    >>> s = pd.Series([0.0, 5.0, 0.0, 0.0, 0.0, 5.0])
    >>> longest_dry_spell(s)
    3
    """
    is_dry = daily_precip <= threshold
    if not is_dry.any():
        return 0

    # Group consecutive runs: the group id changes each time is_dry flips
    group_id = (is_dry != is_dry.shift(fill_value=~is_dry.iloc[0])).cumsum()
    # Sum is_dry within each group → 0 for wet runs, N for dry runs of length N
    run_lengths = is_dry.groupby(group_id).sum()
    return int(run_lengths.max())


def growing_degree_days(
    daily_tavg: pd.Series,
    base_temp: float = GDD_BASE_TEMP_C,
) -> float:
    """Compute accumulated growing degree days (GDD) over a growing season.

    GDD = Σ max(0, Tavg − base_temp)

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


# ── Leakage-safe precipitation anomaly ───────────────────────────────────────

def add_precip_anomaly(features_df: pd.DataFrame, precip_col: str = "cumulative_precip") -> pd.DataFrame:
    """Add a leakage-safe precipitation anomaly column to the county-year features table.

    For each (county_fips, year), the anomaly is:
        current_year_precip − mean(prior_years_precip)

    Only prior years are used in the baseline mean, preventing leakage.

    Parameters
    ----------
    features_df : pd.DataFrame
        County-year features DataFrame with ``county_fips``, ``year``, and
        a cumulative precipitation column.
    precip_col : str
        Name of the cumulative precipitation column.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with an added ``precip_anomaly`` column.
    """
    df = features_df.copy()
    df = df.sort_values(["county_fips", "year"]).reset_index(drop=True)

    # For each county, rolling mean of prior years via shift(1)
    df["precip_anomaly"] = (
        df.groupby("county_fips", group_keys=False)[precip_col]
        .transform(lambda s: s - s.shift(1).expanding(min_periods=1).mean())
    )
    return df


# ── Main aggregation pipeline ─────────────────────────────────────────────────

def compute_weather_features(
    daily_df: pd.DataFrame,
    group_cols: list[str] | None = None,
    cutoff_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Aggregate daily weather into county-year growing-season features.

    Parameters
    ----------
    daily_df : pd.DataFrame
        Daily weather records with columns: ``county_fips``, ``state_fips``,
        ``year``, ``date``, ``PRECTOTCORR``, ``T2M``, ``T2M_MIN``, ``T2M_MAX``.
    group_cols : list[str], optional
        Columns to group on. Defaults to ``["county_fips", "state_fips", "year"]``.
    cutoff_date : str or Timestamp, optional
        If set, records after this date are excluded before aggregation.
        Passed through to ``filter_growing_season``.

    Returns
    -------
    pd.DataFrame
        One row per (county, year) with columns:
        ``county_fips``, ``state_fips``, ``year``, ``cumulative_precip``,
        ``mean_temp``, ``max_temp``, ``extreme_heat_days``, ``dry_days``,
        ``longest_dry_spell``, ``growing_degree_days``, ``precip_anomaly``.
    """
    grp_cols = group_cols or GROUP_COLS

    # ── 1. Filter to growing season ──────────────────────────────────────────
    season_df = filter_growing_season(daily_df, cutoff_date=cutoff_date)
    logger.info(
        "compute_weather_features: %d daily rows after growing-season filter",
        len(season_df),
    )

    # ── 2. Aggregate per (county, year) ──────────────────────────────────────
    records = []
    for keys, grp in season_df.groupby(grp_cols, sort=True):
        key_dict = dict(zip(grp_cols, keys if isinstance(keys, tuple) else (keys,)))

        precip = grp["PRECTOTCORR"].dropna()
        tavg   = grp["T2M"].dropna()
        tmax   = grp["T2M_MAX"].dropna()

        record = {
            **key_dict,
            "cumulative_precip":  cumulative_precip(precip),
            "mean_temp":          mean_temp(tavg),
            "max_temp":           max_temp(tmax),
            "extreme_heat_days":  extreme_heat_days(tmax),
            "dry_days":           dry_days(precip),
            "longest_dry_spell":  longest_dry_spell(precip),
            "growing_degree_days": growing_degree_days(tavg),
            "obs_days":           len(grp),   # diagnostic: how many days had data
        }
        records.append(record)

    features = pd.DataFrame(records)

    # ── 3. Add leakage-safe precipitation anomaly ────────────────────────────
    features = add_precip_anomaly(features)

    logger.info(
        "compute_weather_features: %d county-year rows | %d counties | years %d–%d",
        len(features),
        features["county_fips"].nunique(),
        int(features["year"].min()),
        int(features["year"].max()),
    )
    return features


def save_weather_features(
    features: pd.DataFrame,
    output_path: str | Path = "data/processed/weather_features.csv",
) -> None:
    """Save the weather features DataFrame to disk.

    Parameters
    ----------
    features : pd.DataFrame
        Output of ``compute_weather_features()``.
    output_path : str or Path
        Destination CSV path.
    """
    from pathlib import Path as _Path
    path = _Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(path, index=False)
    logger.info("Weather features saved → %s  (%d rows)", path, len(features))
