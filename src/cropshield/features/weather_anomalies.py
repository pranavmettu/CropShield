"""
County-normalised weather anomaly features for CropShield.

These features express each county-checkpoint-year's weather relative to that
**same county and checkpoint's prior-year climatology**.  A wet July in a
historically dry county is a much stronger signal than the raw precipitation
total, and normalising removes the large between-county baseline differences
that otherwise dominate raw weather features.

Leakage rules
-------------
- Climatology baselines are built per ``(county_fips, checkpoint)`` using only
  years strictly before the target year.  The mechanism is
  ``groupby(...).shift(1).expanding(min_periods=1).mean()`` — year T's baseline
  never includes year T or any future year.
- The first observed year for a county-checkpoint has no prior history, so its
  anomaly is ``NaN`` (imputed later in the sklearn pipeline).
- Future-year weather (however extreme) cannot change an earlier year's anomaly
  because expanding means only look backwards.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ANOMALY_GROUP_KEYS = ["county_fips", "checkpoint"]

# Map output feature → source column.  Anomaly = value - prior-year mean.
_ANOMALY_SOURCES = {
    "precip_anomaly_from_county_checkpoint_mean":  "cumulative_precip",
    "gdd_anomaly_from_county_checkpoint_mean":     "growing_degree_days",
    "heat_days_anomaly_from_county_checkpoint_mean": "extreme_heat_days",
    "dry_days_anomaly_from_county_checkpoint_mean": "dry_days",
    "temp_mean_anomaly_from_county_checkpoint_mean": "mean_temp",
}

# Percentage-of-baseline features (value / prior-year mean).
_PCT_SOURCES = {
    "precip_pct_of_county_checkpoint_mean": "cumulative_precip",
}


def _prior_mean(s: pd.Series) -> pd.Series:
    """Expanding mean over strictly-prior rows (shift(1) then expanding)."""
    return s.shift(1).expanding(min_periods=1).mean()


def add_weather_anomalies(
    weather_df: pd.DataFrame,
    *,
    year_col: str = "year",
) -> pd.DataFrame:
    """Add county+checkpoint-normalised weather anomaly features.

    Parameters
    ----------
    weather_df : pd.DataFrame
        Per ``(county_fips, year, checkpoint)`` weather features.  Must contain
        ``county_fips``, ``checkpoint``, ``year`` and the source columns that
        exist (``cumulative_precip``, ``growing_degree_days``,
        ``extreme_heat_days``, optionally ``dry_days`` and ``mean_temp``).

    Returns
    -------
    pd.DataFrame
        Input with anomaly / pct columns added (only for source columns that
        are present).  NaN where insufficient prior history exists.
    """
    df = weather_df.copy()
    df = df.sort_values(ANOMALY_GROUP_KEYS + [year_col]).reset_index(drop=True)
    grp = df.groupby(ANOMALY_GROUP_KEYS, group_keys=False)

    added: list[str] = []

    for out_col, src in _ANOMALY_SOURCES.items():
        if src not in df.columns:
            continue
        baseline = grp[src].transform(_prior_mean)
        df[out_col] = df[src] - baseline
        added.append(out_col)

    for out_col, src in _PCT_SOURCES.items():
        if src not in df.columns:
            continue
        baseline = grp[src].transform(_prior_mean)
        # Avoid divide-by-zero → NaN (imputed later)
        df[out_col] = np.where(
            (baseline.notna()) & (baseline != 0),
            df[src] / baseline,
            np.nan,
        )
        added.append(out_col)

    logger.info(
        "add_weather_anomalies: added %d anomaly features: %s",
        len(added), added,
    )
    return df


def weather_anomaly_columns(df: pd.DataFrame) -> list[str]:
    """Return the names of weather-anomaly columns present in ``df``."""
    candidates = list(_ANOMALY_SOURCES.keys()) + list(_PCT_SOURCES.keys())
    return [c for c in candidates if c in df.columns]
