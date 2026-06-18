"""
Feature-group definitions for CropShield ablation studies.

Each named group is a leakage-safe subset of panel columns.  The ablation in
``scripts/05_feature_ablation.py`` adds one family at a time on top of a small
base context (``crop``, ``checkpoint``, ``state``, ``expected_yield``) so the
marginal value of each family is directly comparable.  ``baseline_features`` is
the base context alone; ``all_features`` is every allowed non-leakage column.

Leakage safety
--------------
``LEAKAGE_COLUMNS`` (imported from ml_models) — the realised-outcome / target /
risk-label columns — are never included in any group.  ``select_columns``
asserts this.
"""

from __future__ import annotations

import logging

import pandas as pd

from cropshield.models.ml_models import LEAKAGE_COLUMNS

logger = logging.getLogger(__name__)

# Base context shared by every additive group (and the baseline group itself).
BASE_CATEGORICAL = ["crop", "checkpoint", "state"]
BASE_NUMERIC = ["expected_yield"]

# Raw growing-season weather (directly aggregated + stress-timing + interactions).
WEATHER_RAW_COLUMNS = [
    "cumulative_precip", "mean_temp", "max_temp", "extreme_heat_days",
    "dry_days", "longest_dry_spell", "growing_degree_days",
    "max_consecutive_dry_days", "extreme_heat_days_after_july_1",
    "precip_last_30_days_before_checkpoint",
    "heat_days_last_30_days_before_checkpoint",
    "gdd_last_30_days_before_checkpoint",
    "heat_dry_stress", "heat_dry_spell",
]

# County+checkpoint-normalised weather anomalies.
WEATHER_ANOMALY_COLUMNS = [
    "precip_anomaly",  # legacy county-level anomaly
    "precip_anomaly_from_county_checkpoint_mean",
    "precip_pct_of_county_checkpoint_mean",
    "gdd_anomaly_from_county_checkpoint_mean",
    "heat_days_anomaly_from_county_checkpoint_mean",
    "dry_days_anomaly_from_county_checkpoint_mean",
    "temp_mean_anomaly_from_county_checkpoint_mean",
]

# Prior-year / rolling yield history.
LAGGED_YIELD_COLUMNS = [
    "prior_year_yield_anomaly", "prior_year_yield",
    "rolling_3yr_mean_yield_anomaly", "rolling_3yr_std_yield_anomaly",
    "rolling_3yr_mean_yield", "rolling_3yr_std_yield",
]

# Drought features (present only if raw drought data was integrated).
DROUGHT_COLUMNS = [
    "weeks_d0", "weeks_d1", "weeks_d2", "weeks_d3", "weeks_d4",
    "weeks_d2_plus", "max_drought_category", "mean_drought_severity",
]

# Map group name → numeric feature column list (categorical handled separately).
_GROUP_NUMERIC = {
    "weather_raw": WEATHER_RAW_COLUMNS,
    "weather_anomalies": WEATHER_ANOMALY_COLUMNS,
    "lagged_yield": LAGGED_YIELD_COLUMNS,
    "drought": DROUGHT_COLUMNS,
}

FEATURE_GROUPS = [
    "baseline_features",
    "weather_raw",
    "weather_anomalies",
    "lagged_yield",
    "drought",
    "all_features",
]


def available_groups(panel: pd.DataFrame) -> list[str]:
    """Return the feature groups that have at least one column present in panel."""
    groups = ["baseline_features"]
    for g, cols in _GROUP_NUMERIC.items():
        if any(c in panel.columns for c in cols):
            groups.append(g)
    groups.append("all_features")
    return groups


def get_feature_set(
    panel: pd.DataFrame,
    group: str,
    *,
    include_base: bool = True,
) -> tuple[list[str], list[str]]:
    """Return ``(numeric, categorical)`` columns for a feature group.

    Parameters
    ----------
    panel : pd.DataFrame
        The modeling panel (used to keep only columns that actually exist).
    group : str
        One of ``FEATURE_GROUPS``.
    include_base : bool
        Add the base context (crop/checkpoint/state + expected_yield) to
        non-baseline groups so models share crop context and are comparable.

    Returns
    -------
    (numeric, categorical) : tuple[list[str], list[str]]
    """
    if group not in FEATURE_GROUPS:
        raise ValueError(f"Unknown feature group {group!r}. Choose from {FEATURE_GROUPS}")

    categorical = [c for c in BASE_CATEGORICAL if c in panel.columns]
    numeric: list[str] = []

    if group == "baseline_features":
        numeric = [c for c in BASE_NUMERIC if c in panel.columns]
    elif group == "all_features":
        all_numeric = (
            BASE_NUMERIC + WEATHER_RAW_COLUMNS + WEATHER_ANOMALY_COLUMNS
            + LAGGED_YIELD_COLUMNS + DROUGHT_COLUMNS + ["year_index"]
        )
        numeric = [c for c in dict.fromkeys(all_numeric) if c in panel.columns]
    else:
        group_cols = [c for c in _GROUP_NUMERIC[group] if c in panel.columns]
        base = [c for c in BASE_NUMERIC if c in panel.columns] if include_base else []
        numeric = list(dict.fromkeys(base + group_cols))
        if not include_base:
            categorical = []

    # Leakage guard
    selected = set(numeric) | set(categorical)
    leaked = selected & LEAKAGE_COLUMNS
    if leaked:
        raise ValueError(f"Feature group {group!r} leaks columns: {sorted(leaked)}")

    return numeric, categorical
