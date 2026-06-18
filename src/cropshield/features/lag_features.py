"""
Leakage-safe lagged yield features for CropShield.

For each ``(county_fips, crop)`` time series, these features summarise the
county-crop's *prior-year* yield behaviour.  They are computed on the unique
county-crop-year yield table (one row per county-crop-year) and then merged
onto the checkpoint-expanded modeling panel by ``(county_fips, crop, year)``.

Leakage rules
-------------
- Every feature for year T uses only years strictly before T.
- The mechanism is ``groupby(...).shift(1)`` *before* any rolling window, so
  year T's window can never include year T (or any future year).
- Grouping is by ``[county_fips, crop]`` so corn and soybean histories never
  mix.
- Missing prior history yields ``NaN`` (e.g. a county's first observed year).
  These are imputed later inside the sklearn pipeline — never back-filled with
  future data here.

``prior_year_severe_risk_rate`` is intentionally **not** computed here: a
leakage-safe severe-risk label only exists after the temporal split via
``assign_modeling_risk_labels(train, test)``.  Computing a risk rate at
panel-build time would bake test-derived thresholds into a training feature.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

LAG_GROUP_KEYS = ["county_fips", "crop"]
ROLLING_WINDOW = 3

LAG_FEATURE_COLUMNS = [
    "prior_year_yield_anomaly",
    "prior_year_yield",
    "rolling_3yr_mean_yield_anomaly",
    "rolling_3yr_std_yield_anomaly",
    "rolling_3yr_mean_yield",
    "rolling_3yr_std_yield",
]


def add_lag_features(
    yield_df: pd.DataFrame,
    *,
    anomaly_col: str = "yield_anomaly",
    yield_col: str = "actual_yield",
    year_col: str = "year",
) -> pd.DataFrame:
    """Add leakage-safe lagged / rolling yield-history features.

    Parameters
    ----------
    yield_df : pd.DataFrame
        County-crop-year yield table containing at least
        ``county_fips``, ``crop``, ``year``, ``actual_yield``, ``yield_anomaly``.
        Should be unique by ``(county_fips, crop, year)``.
    anomaly_col, yield_col, year_col : str
        Column names.

    Returns
    -------
    pd.DataFrame
        Input with the six lag feature columns added (NaN where insufficient
        prior history exists).
    """
    df = yield_df.copy()
    df = df.sort_values(LAG_GROUP_KEYS + [year_col]).reset_index(drop=True)

    grp = df.groupby(LAG_GROUP_KEYS, group_keys=False)

    # ── 1. Prior-year (lag-1) values ─────────────────────────────────────────
    df["prior_year_yield_anomaly"] = grp[anomaly_col].shift(1)
    df["prior_year_yield"] = grp[yield_col].shift(1)

    # ── 2. Rolling windows over PRIOR years only (shift(1) first) ─────────────
    def _roll_mean(s: pd.Series) -> pd.Series:
        return s.shift(1).rolling(ROLLING_WINDOW, min_periods=1).mean()

    def _roll_std(s: pd.Series) -> pd.Series:
        # std needs >= 2 observations to be meaningful
        return s.shift(1).rolling(ROLLING_WINDOW, min_periods=2).std()

    df["rolling_3yr_mean_yield_anomaly"] = grp[anomaly_col].transform(_roll_mean)
    df["rolling_3yr_std_yield_anomaly"] = grp[anomaly_col].transform(_roll_std)
    df["rolling_3yr_mean_yield"] = grp[yield_col].transform(_roll_mean)
    df["rolling_3yr_std_yield"] = grp[yield_col].transform(_roll_std)

    n_with_history = int(df["prior_year_yield_anomaly"].notna().sum())
    logger.info(
        "add_lag_features: %d rows | %d with prior-year history | %d feature cols",
        len(df), n_with_history, len(LAG_FEATURE_COLUMNS),
    )
    return df


def build_lag_features_table(
    yield_df: pd.DataFrame,
    *,
    keys: list[str] | None = None,
) -> pd.DataFrame:
    """Return a merge-ready table of ``keys`` + lag feature columns.

    Deduplicates to one row per ``(county_fips, crop, year)`` so the table can
    be left-joined onto the checkpoint-expanded panel without fanning out rows.
    """
    keys = keys or ["county_fips", "crop", "year"]
    with_lags = add_lag_features(yield_df)
    cols = keys + LAG_FEATURE_COLUMNS
    table = with_lags[cols].drop_duplicates(subset=keys, keep="first").reset_index(drop=True)
    return table
