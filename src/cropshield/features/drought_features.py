"""
Growing-season drought feature engineering for CropShield.

Converts weekly county-level U.S. Drought Monitor records into growing-season
aggregate features suitable for machine learning.

Drought categories (USDM)
--------------------------
D0 : Abnormally Dry
D1 : Moderate Drought
D2 : Severe Drought
D3 : Extreme Drought
D4 : Exceptional Drought

Feature catalogue (county-year)
--------------------------------
weeks_d0 … weeks_d4   : Count of weeks with any area in each category (> 0%).
weeks_d2_plus         : Weeks with D2 + D3 + D4 combined area > 0%.
max_drought_category  : Highest severity index (0–4) observed in the season.
mean_drought_severity : Mean weekly dominant severity index (0–4).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from cropshield.data.fips_utils import normalise_fips_series

logger = logging.getLogger(__name__)

DROUGHT_CATEGORIES = ["D0", "D1", "D2", "D3", "D4"]
GROWING_SEASON_START_MONTH = 4
GROWING_SEASON_END_MONTH = 8
GROUP_COLS = ["county_fips", "year"]

# April(30)+May(31)+June(30)+July(31)+August(31)
FULL_GROWING_SEASON_WEEKS = 22  # approximate weekly count for sanity checks


def filter_growing_season_drought(
    df: pd.DataFrame,
    date_col: str = "date",
    cutoff_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Keep weekly drought records within April–August, optionally capped at cutoff."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    month = df[date_col].dt.month
    mask = (month >= GROWING_SEASON_START_MONTH) & (month <= GROWING_SEASON_END_MONTH)
    if cutoff_date is not None:
        mask &= df[date_col] <= pd.Timestamp(cutoff_date)
    return df[mask].reset_index(drop=True)


def _dominant_severity(row: pd.Series) -> int:
    """Return the highest drought category index (0–4) with any coverage that week."""
    for idx in range(4, -1, -1):
        if row[DROUGHT_CATEGORIES[idx]] > 0:
            return idx
    return 0


def compute_drought_features(
    drought_df: pd.DataFrame,
    group_cols: list[str] | None = None,
    cutoff_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Aggregate weekly drought data into county-year growing-season features.

    Parameters
    ----------
    drought_df : pd.DataFrame
        Weekly records with ``county_fips``, ``date``, and ``D0``–``D4`` columns.
    group_cols : list[str], optional
        Grouping columns. Defaults to ``["county_fips", "year"]``.
    cutoff_date : str or Timestamp, optional
        If set, only drought weeks on or before this date are used.

    Returns
    -------
    pd.DataFrame
        One row per (county_fips, year) with drought feature columns.
    """
    grp_cols = group_cols or GROUP_COLS
    df = drought_df.copy()
    if "county_fips" in df.columns:
        df["county_fips"] = normalise_fips_series(df["county_fips"])
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = pd.to_numeric(
        df["year"] if "year" in df.columns else df["date"].dt.year,
        errors="coerce",
    ).astype("Int64")

    for cat in DROUGHT_CATEGORIES:
        if cat not in df.columns:
            df[cat] = 0.0
        df[cat] = pd.to_numeric(df[cat], errors="coerce").fillna(0.0)

    season = filter_growing_season_drought(df, cutoff_date=cutoff_date)
    if season.empty:
        return pd.DataFrame(columns=grp_cols + [
            "weeks_d0", "weeks_d1", "weeks_d2", "weeks_d3", "weeks_d4",
            "weeks_d2_plus", "max_drought_category", "mean_drought_severity",
            "checkpoint",
        ])

    season["D2_plus"] = season["D2"] + season["D3"] + season["D4"]
    season["dominant_severity"] = season.apply(_dominant_severity, axis=1)

    records = []
    for keys, grp in season.groupby(grp_cols, sort=True):
        key_dict = dict(zip(grp_cols, keys if isinstance(keys, tuple) else (keys,)))
        record = {
            **key_dict,
            "weeks_d0": int((grp["D0"] > 0).sum()),
            "weeks_d1": int((grp["D1"] > 0).sum()),
            "weeks_d2": int((grp["D2"] > 0).sum()),
            "weeks_d3": int((grp["D3"] > 0).sum()),
            "weeks_d4": int((grp["D4"] > 0).sum()),
            "weeks_d2_plus": int((grp["D2_plus"] > 0).sum()),
            "max_drought_category": int(grp["dominant_severity"].max()),
            "mean_drought_severity": float(grp["dominant_severity"].mean()),
            "checkpoint": str(cutoff_date) if cutoff_date is not None else "full_season",
        }
        records.append(record)

    features = pd.DataFrame(records)
    logger.info(
        "compute_drought_features: %d county-year rows | cutoff=%s",
        len(features), cutoff_date or "full_season",
    )
    return features
