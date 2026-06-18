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
# April(30) + May(31) + June(30) + July(31) + August(31)
FULL_GROWING_SEASON_DAYS   = 153
GDD_BASE_TEMP_C            = 10.0
EXTREME_HEAT_THRESHOLD_C   = 35.0
DRY_DAY_THRESHOLD_MM       = 1.0

GROUP_COLS = ["county_fips", "state_fips", "year"]

# Checkpoint definitions: name → "MM-DD" cutoff within the growing season.
# None means no intra-season cutoff (use all available April-August data).
CHECKPOINT_CONFIGS: dict[str, str | None] = {
    "may_31":     "05-31",
    "june_30":    "06-30",
    "july_31":    "07-31",
    "august_31":  "08-31",
    "full_season": None,
}

# Days in growing season through each checkpoint month-end
_CHECKPOINT_DAYS = {
    "may_31":     61,   # Apr(30) + May(31)
    "june_30":    91,   # + Jun(30)
    "july_31":    122,  # + Jul(31)
    "august_31":  153,  # + Aug(31)
    "full_season": 153,
}


# ── Season filter ─────────────────────────────────────────────────────────────

def filter_growing_season(
    df: pd.DataFrame,
    date_col: str = "date",
    cutoff_date: str | pd.Timestamp | None = None,
    cutoff_month_day: str | None = None,
) -> pd.DataFrame:
    """Keep only rows that fall within the April–August growing season.

    Optionally applies a cutoff to prevent future weather from leaking into
    earlier-checkpoint features.  Two cutoff modes are supported:

    ``cutoff_date``
        Absolute timestamp — filters rows after that specific date.  Useful
        for unit-testing leakage with a fixed calendar date.

    ``cutoff_month_day``
        Year-agnostic ``"MM-DD"`` string (e.g. ``"05-31"`` for May 31).
        Applied relative to each row's own calendar year so that all years
        are consistently trimmed to the same within-season window.  This is
        the correct mode for building multi-checkpoint panels.

    When both are provided, ``cutoff_month_day`` takes priority.

    Parameters
    ----------
    df : pd.DataFrame
        Daily weather records with a date column.
    date_col : str
        Name of the date column.
    cutoff_date : str or Timestamp, optional
        Absolute cutoff date.
    cutoff_month_day : str, optional
        ``"MM-DD"`` within-season cutoff applied per calendar year.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame containing only qualifying growing-season rows.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    month = df[date_col].dt.month
    day   = df[date_col].dt.day
    mask  = (month >= GROWING_SEASON_START_MONTH) & (month <= GROWING_SEASON_END_MONTH)

    if cutoff_month_day is not None:
        # Parse "MM-DD"
        parts = cutoff_month_day.split("-")
        cutoff_mm, cutoff_dd = int(parts[0]), int(parts[1])
        within_cutoff = (month < cutoff_mm) | (
            (month == cutoff_mm) & (day <= cutoff_dd)
        )
        mask = mask & within_cutoff
    elif cutoff_date is not None:
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
    cutoff_month_day: str | None = None,
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
        Absolute cutoff. Passed through to ``filter_growing_season``.
    cutoff_month_day : str, optional
        Year-agnostic ``"MM-DD"`` cutoff (e.g. ``"05-31"``).  Applied per
        calendar year so each year's features are trimmed to the same
        within-season window.  Takes priority over ``cutoff_date``.

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
    season_df = filter_growing_season(
        daily_df,
        cutoff_date=cutoff_date,
        cutoff_month_day=cutoff_month_day,
    )
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

    # ── 3. Flag incomplete growing seasons ───────────────────────────────────
    features["is_partial_year"] = features["obs_days"] < FULL_GROWING_SEASON_DAYS

    # ── 4. Add leakage-safe precipitation anomaly ────────────────────────────
    features = add_precip_anomaly(features)

    logger.info(
        "compute_weather_features: %d county-year rows | %d counties | years %d–%d",
        len(features),
        features["county_fips"].nunique(),
        int(features["year"].min()),
        int(features["year"].max()),
    )
    return features


def compute_multi_checkpoint_weather_features(
    daily_df: pd.DataFrame,
    checkpoints: list[str] | None = None,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Compute weather features for each named checkpoint.

    For every checkpoint in ``checkpoints``, the growing-season window is
    trimmed to the checkpoint's month-day boundary using
    ``CHECKPOINT_CONFIGS``.  This ensures that a ``may_31`` feature row for
    year 2020 only contains weather observations through 2020-05-31.

    The ``precip_anomaly`` for each checkpoint is computed independently
    using only prior-year data for that same seasonal window — leakage-safe.

    Parameters
    ----------
    daily_df : pd.DataFrame
        Full daily weather records (all years, all growing-season months).
    checkpoints : list[str], optional
        Subset of ``CHECKPOINT_CONFIGS`` keys to compute.
        Defaults to all five: may_31, june_30, july_31, august_31,
        full_season.
    group_cols : list[str], optional
        Columns to group on within each checkpoint.

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame with one row per
        ``(county_fips, year, checkpoint)``.  Adds a ``checkpoint`` column.
    """
    ckpts = checkpoints or list(CHECKPOINT_CONFIGS.keys())
    frames: list[pd.DataFrame] = []

    for name in ckpts:
        if name not in CHECKPOINT_CONFIGS:
            raise ValueError(
                f"Unknown checkpoint {name!r}. "
                f"Choose from: {list(CHECKPOINT_CONFIGS.keys())}"
            )
        month_day = CHECKPOINT_CONFIGS[name]
        features = compute_weather_features(
            daily_df,
            group_cols=group_cols,
            cutoff_month_day=month_day,
        )
        features.insert(features.columns.get_loc("year") + 1, "checkpoint", name)
        frames.append(features)
        logger.info("Checkpoint %s: %d county-year rows", name, len(features))

    combined = pd.concat(frames, ignore_index=True)
    logger.info(
        "compute_multi_checkpoint_weather_features: %d total rows "
        "(%d checkpoints × ~%d county-years)",
        len(combined),
        len(ckpts),
        len(combined) // len(ckpts) if ckpts else 0,
    )
    return combined


def filter_incomplete_current_year(
    features_df: pd.DataFrame,
    *,
    allow_partial_year: bool = False,
    current_year: int | None = None,
    year_col: str = "year",
    obs_col: str = "obs_days",
) -> pd.DataFrame:
    """Exclude or flag incomplete weather for the current calendar year.

    By default, rows for ``current_year`` with fewer than
    ``FULL_GROWING_SEASON_DAYS`` observations are **dropped** so they cannot
    enter model training with biased partial-season aggregates.

    Parameters
    ----------
    features_df : pd.DataFrame
        County-year weather features including ``obs_days``.
    allow_partial_year : bool
        When ``True``, keep partial rows and ensure ``is_partial_year`` is set.
    current_year : int, optional
        Calendar year treated as "current". Defaults to today's year.
    year_col, obs_col : str
        Column names for year and observation count.

    Returns
    -------
    pd.DataFrame
        Filtered (or flagged) features.
    """
    import datetime as _dt

    df = features_df.copy()
    if "is_partial_year" not in df.columns and obs_col in df.columns:
        df["is_partial_year"] = df[obs_col] < FULL_GROWING_SEASON_DAYS

    cy = current_year or _dt.date.today().year
    partial_current = (df[year_col] == cy) & df.get("is_partial_year", False)

    if allow_partial_year:
        n_partial = int(partial_current.sum())
        if n_partial:
            logger.warning(
                "allow_partial_year=True: keeping %d partial current-year (%d) rows",
                n_partial, cy,
            )
        return df

    n_drop = int(partial_current.sum())
    if n_drop:
        logger.info(
            "Excluding %d incomplete current-year (%d) weather rows "
            "(obs_days < %d). Set allow_partial_year=True to keep them.",
            n_drop, cy, FULL_GROWING_SEASON_DAYS,
        )
        df = df[~partial_current].copy()
    return df


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
