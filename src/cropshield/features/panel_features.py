"""
Panel-level feature engineering for CropShield.

Adds cross-source derived features to the merged modeling panel after
individual feature tables have been assembled.

Examples of panel-level features
---------------------------------
- heat_x_drought : Interaction between extreme_heat_days and d2_plus_weeks.
- year_index     : Normalised year (for trend-aware models).
- state_encoded  : Ordinal or one-hot state encoding.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add interaction terms between weather and drought stress features.

    Parameters
    ----------
    df : pd.DataFrame
        Merged modeling panel.

    Returns
    -------
    pd.DataFrame
        Panel with additional interaction columns.
    """
    # TODO: Implement interaction features
    # Examples:
    # - heat_drought_stress = extreme_heat_days * d2_plus_weeks
    # - dry_drought_index   = dry_days * d2_plus_max_pct
    raise NotImplementedError("add_interaction_features is not yet implemented.")


def add_year_index(df: pd.DataFrame, base_year: int = 2015) -> pd.DataFrame:
    """Add a normalised year index column for trend-aware modelling.

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
    df["year_index"] = df["year"] - base_year
    return df
