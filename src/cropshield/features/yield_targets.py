"""
Leakage-safe yield target engineering for CropShield.

This module computes the yield anomaly and risk-class targets used in all
downstream models.

Why leakage prevention matters
-------------------------------
If the current year's yield is used to compute the expected yield for that
same year, the model is trained on a target that incorporates the answer.
This produces artificially low training error and models that do not
generalise to future years — the exact scenario we want to detect.

All functions here use **only prior years** when computing expected yield
for year T. No information from year T is allowed in ``expected_yield_T``.

Expected workflow
-----------------
1. Call ``clean_yield_dataframe()`` to standardise the input.
2. Call ``add_expected_yield_rolling()`` or ``add_expected_yield_trend()``
   to add an expected yield column.
3. Call ``add_yield_anomaly()`` to compute anomaly columns.
4. Call ``add_risk_class()`` to label severe-risk counties.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

GROUP_KEYS = ["state", "county_fips", "crop"]


def clean_yield_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean the raw NASS yield DataFrame for target engineering.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned NASS DataFrame with at minimum columns:
        ``year``, ``state``, ``county_fips``, ``crop``, ``value``.

    Returns
    -------
    pd.DataFrame
        Renamed and sorted DataFrame with column ``actual_yield`` in place
        of ``value``. Rows with missing yield are dropped.
    """
    # TODO: Implement cleaning
    # Steps:
    # 1. Rename 'value' -> 'actual_yield'
    # 2. Coerce 'actual_yield' to float
    # 3. Drop rows with NaN actual_yield (log how many)
    # 4. Sort by GROUP_KEYS + ['year']
    # 5. Reset index
    raise NotImplementedError("clean_yield_dataframe is not yet implemented.")


def add_expected_yield_rolling(
    df: pd.DataFrame,
    window: int = 5,
    min_periods: int = 3,
) -> pd.DataFrame:
    """Add a leakage-safe rolling-average expected yield column.

    For each (state, county, crop) group, the expected yield for year T is
    the mean of actual yields from years T-window through T-1. Year T itself
    is never included in this calculation.

    Parameters
    ----------
    df : pd.DataFrame
        Yield DataFrame sorted by group and year.
    window : int
        Number of preceding years to average over.
    min_periods : int
        Minimum number of valid prior observations required to produce a
        non-NaN expected yield. Counties with fewer years of history than
        this will have NaN expected yield (and will be dropped before modeling).

    Returns
    -------
    pd.DataFrame
        Input DataFrame with an added ``expected_yield`` column.

    Notes
    -----
    Uses ``pd.Series.shift(1)`` to ensure the rolling window never includes
    the current year — this is the critical leakage guard.
    """
    # TODO: Implement rolling expected yield
    # Steps:
    # 1. Sort df by GROUP_KEYS + ['year'] (safety sort)
    # 2. Group by GROUP_KEYS
    # 3. Use .shift(1) on actual_yield BEFORE computing .rolling(window).mean()
    #    so that the window is [T-window, T-1] not [T-window+1, T]
    # 4. Assign result to df['expected_yield']
    # 5. Log how many rows have NaN expected_yield (insufficient history)
    raise NotImplementedError("add_expected_yield_rolling is not yet implemented.")


def add_expected_yield_trend(
    df: pd.DataFrame,
    min_years: int = 5,
) -> pd.DataFrame:
    """Add a leakage-safe linear-trend expected yield column.

    For each (state, county, crop) group and year T, fits a linear regression
    on all years < T and predicts the expected yield at year T from the fitted
    trend. This accounts for secular yield improvement trends.

    Parameters
    ----------
    df : pd.DataFrame
        Yield DataFrame sorted by group and year.
    min_years : int
        Minimum number of prior years required to fit a trend. Groups with
        fewer observations receive NaN expected yield.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with an added ``expected_yield`` column.
    """
    # TODO: Implement trend-based expected yield
    # Steps:
    # 1. Sort df by GROUP_KEYS + ['year']
    # 2. For each group, iterate over years in ascending order
    # 3. For year T, filter to rows where year < T
    # 4. If len(prior_rows) < min_years, set expected_yield = NaN
    # 5. Else fit np.polyfit(prior_years, prior_yields, 1) and predict for T
    # 6. Collect results and assign to df['expected_yield']
    raise NotImplementedError("add_expected_yield_trend is not yet implemented.")


def add_yield_anomaly(df: pd.DataFrame) -> pd.DataFrame:
    """Compute yield anomaly columns from actual and expected yield.

    Requires ``actual_yield`` and ``expected_yield`` columns to already exist
    in the DataFrame (add via ``add_expected_yield_rolling`` or
    ``add_expected_yield_trend`` first).

    Adds
    ----
    yield_anomaly : float
        Absolute difference: ``actual_yield - expected_yield``.
        Negative values indicate below-trend yields.
    yield_anomaly_pct : float
        Percentage difference: ``(actual_yield - expected_yield) / expected_yield * 100``.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with ``actual_yield`` and ``expected_yield`` columns.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with added anomaly columns.
    """
    # TODO: Implement anomaly calculation
    # Steps:
    # 1. Check that both columns exist
    # 2. yield_anomaly = actual_yield - expected_yield
    # 3. yield_anomaly_pct = (yield_anomaly / expected_yield) * 100 (guard / 0)
    raise NotImplementedError("add_yield_anomaly is not yet implemented.")


def add_risk_class(
    df: pd.DataFrame,
    quantile: float = 0.20,
) -> pd.DataFrame:
    """Add a severe yield-risk binary label.

    A county-crop-year is labelled severe risk (``severe_risk = 1``) when its
    ``yield_anomaly_pct`` falls below the ``quantile``-th percentile of the
    **training distribution** for that county and crop.

    Important
    ---------
    The quantile threshold must be derived from the training set only and
    applied to the test set as a fixed threshold. This function computes
    quantiles across the entire DataFrame — callers must ensure they call
    this only on training data or pass explicit thresholds.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a ``yield_anomaly_pct`` column.
    quantile : float
        Fraction defining "severe" risk (e.g. 0.20 = bottom 20%).

    Returns
    -------
    pd.DataFrame
        Input DataFrame with an added ``severe_risk`` integer column (0 or 1).
    """
    # TODO: Implement risk class labelling
    # Steps:
    # 1. Compute threshold = df.groupby(GROUP_KEYS)['yield_anomaly_pct'].transform(
    #        lambda s: s.quantile(quantile))
    # 2. severe_risk = (yield_anomaly_pct <= threshold).astype(int)
    # 3. Log the class balance
    raise NotImplementedError("add_risk_class is not yet implemented.")
