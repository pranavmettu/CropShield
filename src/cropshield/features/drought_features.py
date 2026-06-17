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

Each weekly value is the percentage of county area in that category (0–100).

Feature catalogue
-----------------
d0_max_pct       : Maximum weekly D0 area percentage during the growing season.
d1_max_pct       : Maximum weekly D1 area percentage.
d2_max_pct       : Maximum weekly D2 area percentage.
d3_max_pct       : Maximum weekly D3 area percentage.
d4_max_pct       : Maximum weekly D4 area percentage.
d2_plus_weeks    : Number of weeks with D2+ (severe or worse) coverage > 0%.
d2_plus_max_pct  : Maximum weekly D2+ combined area percentage.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

DROUGHT_CATEGORIES = ["D0", "D1", "D2", "D3", "D4"]
GROWING_SEASON_START_MONTH = 4
GROWING_SEASON_END_MONTH = 8


def filter_growing_season_drought(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Keep only weekly drought records that fall within the growing season.

    Parameters
    ----------
    df : pd.DataFrame
        Weekly drought monitor records with a date column.
    date_col : str
        Name of the date column.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame for April–August weeks.
    """
    # TODO: Filter to growing season months
    raise NotImplementedError("filter_growing_season_drought is not yet implemented.")


def compute_drought_features(
    drought_df: pd.DataFrame,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Aggregate weekly drought data into county-year growing-season features.

    Parameters
    ----------
    drought_df : pd.DataFrame
        Weekly drought records with columns: ``county_fips``, ``state``,
        ``year``, ``date``, ``D0``, ``D1``, ``D2``, ``D3``, ``D4``.
    group_cols : list[str], optional
        Columns to group on. Defaults to ``["county_fips", "state", "year"]``.

    Returns
    -------
    pd.DataFrame
        One row per (county, year) with all drought features.
    """
    # TODO: Implement drought feature aggregation
    # Steps:
    # 1. Filter to growing season
    # 2. Add D2_plus = D2 + D3 + D4 column
    # 3. Group by group_cols
    # 4. For each category, compute max weekly percentage
    # 5. Count weeks where D2_plus > 0
    # 6. Compute max D2_plus percentage
    # 7. Assemble into wide DataFrame
    raise NotImplementedError("compute_drought_features is not yet implemented.")
