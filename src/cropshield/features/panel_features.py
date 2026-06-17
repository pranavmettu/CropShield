"""
Panel-level feature engineering for CropShield.

Adds cross-source derived features to the merged modeling panel after
individual feature tables have been assembled.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def add_year_index(df: pd.DataFrame, base_year: int = 2015) -> pd.DataFrame:
    """Add a normalised year index column for trend-aware modelling.

    ``year_index = year - base_year``, so 2015 → 0, 2020 → 5, etc.
    Useful as a linear trend feature without scale issues.

    Parameters
    ----------
    df : pd.DataFrame
        Modeling panel with a ``year`` column.
    base_year : int
        Year corresponding to index 0.

    Returns
    -------
    pd.DataFrame
        Panel with an added ``year_index`` column.
    """
    df = df.copy()
    df["year_index"] = pd.to_numeric(df["year"], errors="coerce") - base_year
    return df


def add_heat_stress_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add compound heat-and-dryness stress features.

    Creates interaction terms that capture simultaneous heat and water stress,
    which agronomic literature associates with the largest corn yield losses.

    Features added (only when source columns are present)
    -----------------------------------------------------
    heat_dry_stress : extreme_heat_days × dry_days
        Combines heat frequency with dryness frequency. A county that is both
        very hot and very dry during pollination is at high risk.
    heat_dry_spell  : extreme_heat_days × longest_dry_spell
        Captures heat coinciding with a prolonged dry period rather than
        scattered dry days.

    Parameters
    ----------
    df : pd.DataFrame
        Merged modeling panel. Silently skips features if source columns
        are not present (e.g. weather data not yet fetched).

    Returns
    -------
    pd.DataFrame
        Panel with interaction columns added where possible.
    """
    df = df.copy()

    if "extreme_heat_days" in df.columns and "dry_days" in df.columns:
        df["heat_dry_stress"] = df["extreme_heat_days"] * df["dry_days"]

    if "extreme_heat_days" in df.columns and "longest_dry_spell" in df.columns:
        df["heat_dry_spell"] = df["extreme_heat_days"] * df["longest_dry_spell"]

    return df


def add_lagged_yield(df: pd.DataFrame, n_lags: int = 1) -> pd.DataFrame:
    """Add lagged actual yield columns as autoregressive features.

    Yield in year T-1 can be a useful predictor of year T anomaly (e.g.
    recovery after a bad year). Only prior years are used — no leakage.

    Parameters
    ----------
    df : pd.DataFrame
        Panel with ``actual_yield``, ``county_fips``, ``crop``, ``year`` columns.
    n_lags : int
        Number of lag years to add (e.g. 1 adds ``yield_lag_1``).

    Returns
    -------
    pd.DataFrame
        Panel with ``yield_lag_1`` … ``yield_lag_n`` columns added.
    """
    df = df.copy()
    df = df.sort_values(["county_fips", "crop", "year"]).reset_index(drop=True)
    group_cols = ["county_fips", "crop"]
    for lag in range(1, n_lags + 1):
        df[f"yield_lag_{lag}"] = (
            df.groupby(group_cols, group_keys=False)["actual_yield"]
            .transform(lambda s: s.shift(lag))
        )
    return df


def get_feature_columns(panel: pd.DataFrame) -> list[str]:
    """Return the list of ML feature columns in the panel.

    Excludes identifier columns, target columns, and diagnostic columns.

    Parameters
    ----------
    panel : pd.DataFrame
        The modeling panel.

    Returns
    -------
    list[str]
        Column names suitable for use as model inputs.
    """
    non_features = {
        # Identifiers
        "year", "state", "county", "county_fips", "crop",
        # Targets
        "actual_yield", "expected_yield",
        "yield_anomaly", "yield_anomaly_pct", "severe_risk",
        # Diagnostics
        "obs_days",
    }
    return [c for c in panel.columns if c not in non_features]
