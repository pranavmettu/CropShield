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

The critical mechanism in ``add_expected_yield_rolling`` is ``shift(1)``:
shifting the yield series down by one row before computing the rolling mean
ensures row T's window contains rows [T-window, T-1], never row T itself.

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

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

GROUP_KEYS = ["state", "county_fips", "crop"]


def clean_yield_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean the raw NASS yield DataFrame for target engineering.

    Renames ``value`` → ``actual_yield``, coerces to float, drops rows with
    missing yield, and sorts deterministically for rolling operations.

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

    Raises
    ------
    KeyError
        If any required columns are absent.
    """
    required = {"year", "state", "county_fips", "crop", "value"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"clean_yield_dataframe: missing columns {missing}")

    df = df.copy()
    df = df.rename(columns={"value": "actual_yield"})
    df["actual_yield"] = pd.to_numeric(df["actual_yield"], errors="coerce")
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

    n_before = len(df)
    df = df.dropna(subset=["actual_yield"])
    n_dropped = n_before - len(df)
    if n_dropped:
        logger.info("Dropped %d rows with missing actual_yield (%.1f%%)",
                    n_dropped, 100 * n_dropped / n_before)

    df = df.sort_values(GROUP_KEYS + ["year"]).reset_index(drop=True)

    if len(df) == 0:
        logger.warning("clean_yield_dataframe: 0 rows after cleaning — all yields were missing or suppressed")
        return df

    logger.info("clean_yield_dataframe: %d rows, %d unique county-crops, years %d–%d",
                len(df), df.groupby(GROUP_KEYS).ngroups,
                int(df["year"].min()), int(df["year"].max()))
    return df


def add_expected_yield_rolling(
    df: pd.DataFrame,
    window: int = 5,
    min_periods: int = 3,
) -> pd.DataFrame:
    """Add a leakage-safe rolling-average expected yield column.

    For each (state, county, crop) group, the expected yield for year T is
    the mean of actual yields from years T-window through T-1. Year T itself
    is never included in this calculation.

    The leakage guard is ``shift(1)``: shifting the yield series back by one
    position before computing the rolling mean moves the window entirely into
    the past. Without this shift, the rolling mean for row T would include
    row T's own value.

    Parameters
    ----------
    df : pd.DataFrame
        Yield DataFrame with ``actual_yield`` column, sorted by group and year.
    window : int
        Number of preceding rows (years) to average over.
    min_periods : int
        Minimum number of non-NaN observations required in the window to
        produce a non-NaN result. Counties with fewer prior years of history
        receive NaN expected_yield and will be dropped before modeling.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with an added ``expected_yield`` column.

    Notes
    -----
    The window operates on row position, not calendar year. If a county has
    missing years (e.g. suppressed NASS records), the window may span more
    than ``window`` calendar years. This is a known MVP limitation.
    """
    df = df.copy()
    df = df.sort_values(GROUP_KEYS + ["year"]).reset_index(drop=True)

    df["expected_yield"] = (
        df.groupby(GROUP_KEYS, group_keys=False)["actual_yield"]
        .transform(
            lambda s: s.shift(1).rolling(window=window, min_periods=min_periods).mean()
        )
    )

    n_nan = df["expected_yield"].isna().sum()
    logger.info(
        "add_expected_yield_rolling (window=%d, min_periods=%d): "
        "%d rows have expected_yield, %d rows have NaN (insufficient history)",
        window, min_periods, len(df) - n_nan, n_nan,
    )
    return df


def add_expected_yield_trend(
    df: pd.DataFrame,
    min_years: int = 5,
) -> pd.DataFrame:
    """Add a leakage-safe linear-trend expected yield column.

    For each (state, county, crop) group and year T, fits an OLS linear
    regression on all prior years (year < T) and predicts the expected yield
    at year T from the fitted slope and intercept. This removes the secular
    trend in yield improvement (approximately 2–3 bu/acre/year for corn) from
    the anomaly signal.

    Parameters
    ----------
    df : pd.DataFrame
        Yield DataFrame with ``actual_yield`` and ``year`` columns.
    min_years : int
        Minimum number of prior-year observations needed to fit a trend.
        Groups with fewer prior years receive NaN expected_yield.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with an added ``expected_yield`` column.

    Notes
    -----
    Uses ``numpy.polyfit`` (degree 1) for the linear fit. For very sparse
    county histories this may be noisy; ``add_expected_yield_rolling`` is
    a safer default for MVP.
    """
    df = df.copy()
    df = df.sort_values(GROUP_KEYS + ["year"]).reset_index(drop=True)

    expected: list[float] = [float("nan")] * len(df)

    for group_keys, group in df.groupby(GROUP_KEYS, sort=False):
        group_sorted = group.sort_values("year")
        years = group_sorted["year"].astype(float).values
        yields = group_sorted["actual_yield"].values
        indices = group_sorted.index.tolist()

        for i, (idx, yr) in enumerate(zip(indices, years)):
            prior_mask = years[:i]          # years before current position
            prior_yields = yields[:i]

            # Drop any NaN in prior yields before fitting
            valid = ~np.isnan(prior_yields)
            prior_mask_valid = prior_mask[valid]
            prior_yields_valid = prior_yields[valid]

            if len(prior_yields_valid) < min_years:
                expected[idx] = float("nan")
            else:
                coef = np.polyfit(prior_mask_valid, prior_yields_valid, 1)
                expected[idx] = float(np.polyval(coef, yr))

    df["expected_yield"] = expected
    n_nan = df["expected_yield"].isna().sum()
    logger.info(
        "add_expected_yield_trend (min_years=%d): "
        "%d rows have expected_yield, %d rows have NaN (insufficient history)",
        min_years, len(df) - n_nan, n_nan,
    )
    return df


def add_yield_anomaly(df: pd.DataFrame) -> pd.DataFrame:
    """Compute yield anomaly columns from actual and expected yield.

    Requires ``actual_yield`` and ``expected_yield`` to already exist in the
    DataFrame (add via ``add_expected_yield_rolling`` or
    ``add_expected_yield_trend`` first).

    Adds
    ----
    yield_anomaly : float
        Absolute difference: ``actual_yield - expected_yield`` (bu/acre).
        Negative values indicate below-trend yields.
    yield_anomaly_pct : float
        Percentage difference: ``(actual_yield - expected_yield) /
        expected_yield * 100``. NaN when expected_yield is zero or NaN.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with ``actual_yield`` and ``expected_yield`` columns.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with added ``yield_anomaly`` and
        ``yield_anomaly_pct`` columns.

    Raises
    ------
    KeyError
        If ``actual_yield`` or ``expected_yield`` columns are missing.
    """
    for col in ("actual_yield", "expected_yield"):
        if col not in df.columns:
            raise KeyError(
                f"add_yield_anomaly: '{col}' column not found. "
                "Run add_expected_yield_rolling() or add_expected_yield_trend() first."
            )

    df = df.copy()
    df["yield_anomaly"] = df["actual_yield"] - df["expected_yield"]

    # Guard against division by zero (expected_yield == 0 is extremely rare
    # for corn yields but we protect anyway)
    with np.errstate(divide="ignore", invalid="ignore"):
        df["yield_anomaly_pct"] = np.where(
            df["expected_yield"].notna() & (df["expected_yield"] != 0),
            (df["yield_anomaly"] / df["expected_yield"]) * 100.0,
            float("nan"),
        )

    valid = df["yield_anomaly"].notna().sum()
    logger.info(
        "add_yield_anomaly: %d valid anomaly rows | "
        "mean anomaly=%.2f bu/acre (%.2f%%)",
        valid,
        df["yield_anomaly"].mean(),
        df["yield_anomaly_pct"].mean(),
    )
    return df


def add_risk_class(
    df: pd.DataFrame,
    quantile: float = 0.20,
) -> pd.DataFrame:
    """Add a severe yield-risk binary label.

    A county-crop-year is labelled severe risk (``severe_risk = 1``) when its
    ``yield_anomaly_pct`` falls at or below the ``quantile``-th percentile of
    the anomaly distribution for that county-crop group in the provided
    DataFrame.

    Important
    ---------
    The quantile thresholds are computed from the rows present in ``df``.
    To avoid leakage into model evaluation, callers should pass only training
    data when computing thresholds, then apply those thresholds separately to
    the test set. A helper ``compute_risk_thresholds`` is provided for this.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a ``yield_anomaly_pct`` column (NaN rows are ignored
        when computing thresholds but kept in the output as NaN severe_risk).
    quantile : float
        Fraction defining "severe" risk (e.g. 0.20 = bottom 20th percentile).

    Returns
    -------
    pd.DataFrame
        Input DataFrame with an added ``severe_risk`` integer column (0 or 1).
        Rows where ``yield_anomaly_pct`` is NaN receive NaN severe_risk.
    """
    if "yield_anomaly_pct" not in df.columns:
        raise KeyError(
            "add_risk_class: 'yield_anomaly_pct' not found. "
            "Run add_yield_anomaly() first."
        )

    df = df.copy()

    # Compute per-county-crop quantile threshold using only non-NaN values
    threshold = df.groupby(GROUP_KEYS, group_keys=False)["yield_anomaly_pct"].transform(
        lambda s: s.quantile(quantile)
    )

    # Label: 1 if anomaly is at or below threshold, NaN if anomaly is NaN
    df["severe_risk"] = np.where(
        df["yield_anomaly_pct"].notna(),
        (df["yield_anomaly_pct"] <= threshold).astype(int),
        float("nan"),
    )
    df["severe_risk"] = pd.to_numeric(df["severe_risk"], errors="coerce")

    n_risk = int((df["severe_risk"] == 1).sum())
    n_total = int(df["severe_risk"].notna().sum())
    logger.info(
        "add_risk_class (quantile=%.2f): %d severe-risk rows / %d total (%.1f%%)",
        quantile, n_risk, n_total, 100 * n_risk / n_total if n_total else 0,
    )
    return df


def compute_risk_thresholds(
    train_df: pd.DataFrame,
    quantile: float = 0.20,
) -> pd.Series:
    """Compute per-county-crop risk thresholds from the training set only.

    Use this when you need to apply a fixed threshold to a held-out test set
    without re-computing quantiles on test data (which would be leakage).

    Parameters
    ----------
    train_df : pd.DataFrame
        Training split with ``yield_anomaly_pct`` column.
    quantile : float
        Risk quantile threshold.

    Returns
    -------
    pd.Series
        Series indexed by (state, county_fips, crop) MultiIndex with the
        threshold value for each group.

    Examples
    --------
    >>> thresholds = compute_risk_thresholds(train_df)
    >>> test_df = apply_risk_thresholds(test_df, thresholds)
    """
    return (
        train_df.groupby(GROUP_KEYS)["yield_anomaly_pct"]
        .quantile(quantile)
        .rename("risk_threshold")
    )


def apply_risk_thresholds(
    df: pd.DataFrame,
    thresholds: pd.Series,
) -> pd.DataFrame:
    """Apply pre-computed risk thresholds to a DataFrame (e.g. test set).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with ``yield_anomaly_pct`` column.
    thresholds : pd.Series
        Per-group thresholds from ``compute_risk_thresholds()``.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with an added ``severe_risk`` column.
        Counties not seen in training receive NaN severe_risk.
    """
    df = df.copy()
    df = df.join(thresholds, on=GROUP_KEYS, how="left")
    df["severe_risk"] = np.where(
        df["yield_anomaly_pct"].notna() & df["risk_threshold"].notna(),
        (df["yield_anomaly_pct"] <= df["risk_threshold"]).astype(int),
        float("nan"),
    )
    df = df.drop(columns=["risk_threshold"])
    return df


def build_yield_targets(
    nass_df: pd.DataFrame,
    method: str = "rolling",
    window: int = 5,
    min_periods: int = 3,
    min_years: int = 5,
    risk_quantile: float = 0.20,
    output_path: str | None = "data/interim/yield_targets.csv",
) -> pd.DataFrame:
    """Run the full yield target engineering pipeline in one call.

    Convenience wrapper that calls all target engineering functions in order.

    Parameters
    ----------
    nass_df : pd.DataFrame
        Cleaned NASS yield DataFrame (output of ``fetch_nass_yield``).
    method : str
        ``"rolling"`` for rolling-average expected yield or ``"trend"``
        for linear-trend expected yield.
    window : int
        Rolling window size (used when ``method="rolling"``).
    min_periods : int
        Minimum rolling periods (used when ``method="rolling"``).
    min_years : int
        Minimum prior years for trend fit (used when ``method="trend"``).
    risk_quantile : float
        Quantile threshold for severe-risk labelling.
    output_path : str, optional
        If provided, saves the result to this path.

    Returns
    -------
    pd.DataFrame
        DataFrame with all target columns added:
        ``actual_yield``, ``expected_yield``, ``yield_anomaly``,
        ``yield_anomaly_pct``, ``severe_risk``.
    """
    df = clean_yield_dataframe(nass_df)

    if method == "rolling":
        df = add_expected_yield_rolling(df, window=window, min_periods=min_periods)
    elif method == "trend":
        df = add_expected_yield_trend(df, min_years=min_years)
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'rolling' or 'trend'.")

    df = add_yield_anomaly(df)
    df = add_risk_class(df, quantile=risk_quantile)

    if output_path:
        from pathlib import Path
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info("Yield targets saved → %s  (%d rows)", output_path, len(df))

    return df
