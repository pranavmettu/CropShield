"""
U.S. Drought Monitor data fetcher for CropShield.

Retrieves weekly county-level drought severity statistics from the
U.S. Drought Monitor and saves them to the data pipeline.

Data Source
-----------
U.S. Drought Monitor: https://droughtmonitor.unl.edu/
County-level tabular statistics are available via:
  https://usdm.climate.unl.edu/DM_Export.ashx?

Drought Categories
------------------
D0 : Abnormally Dry
D1 : Moderate Drought
D2 : Severe Drought
D3 : Extreme Drought
D4 : Exceptional Drought

Each category value represents the percentage of the county area in that
drought category for a given week.

Notes
-----
- No API key is required.
- Data is reported weekly (typically on Tuesdays).
- This module only fetches and lightly cleans the data. Growing-season
  aggregation lives in src/cropshield/features/drought_features.py.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

DROUGHT_CATEGORIES = ["D0", "D1", "D2", "D3", "D4"]
DROUGHT_BASE_URL = "https://usdm.climate.unl.edu/DM_Export.ashx"


def fetch_drought_monitor(
    state_fips_list: list[str],
    start_year: int = 2015,
    end_year: int | None = None,
    output_raw: str | Path = "data/raw/drought_monitor.csv",
    output_clean: str | Path = "data/interim/drought_features.csv",
) -> pd.DataFrame:
    """Fetch weekly county-level drought statistics from the U.S. Drought Monitor.

    Parameters
    ----------
    state_fips_list : list[str]
        Two-digit state FIPS codes (e.g. ``["19", "17"]`` for Iowa and Illinois).
    start_year : int
        First year to fetch (inclusive).
    end_year : int, optional
        Last year to fetch (inclusive). Defaults to current year.
    output_raw : str or Path
        Destination for the raw response CSV.
    output_clean : str or Path
        Destination for the cleaned drought features CSV.

    Returns
    -------
    pd.DataFrame
        Weekly drought statistics with columns: ``date``, ``county_fips``,
        ``D0``, ``D1``, ``D2``, ``D3``, ``D4``.
    """
    # TODO: Implement Drought Monitor API call
    # Steps:
    # 1. Build date range: weekly Tuesdays from start_year through end_year
    # 2. For each state FIPS, construct request URL with StatisticTypeId=1 (county area)
    # 3. Parse the response (CSV or JSON format)
    # 4. Standardise column names and FIPS codes
    # 5. Save raw data
    # 6. Call clean_drought_dataframe()
    # 7. Save clean data
    # 8. Return clean DataFrame
    raise NotImplementedError("fetch_drought_monitor is not yet implemented.")


def clean_drought_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise a raw Drought Monitor DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Raw drought monitor records.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with columns: ``date``, ``county_fips``,
        ``D0``, ``D1``, ``D2``, ``D3``, ``D4``.
        All category columns are floats representing percentage area (0–100).
    """
    # TODO: Implement cleaning
    # Steps:
    # 1. Parse date column to datetime
    # 2. Standardise FIPS codes to 5-digit zero-padded strings
    # 3. Coerce drought category columns to float
    # 4. Fill NaN drought categories with 0.0
    # 5. Add a D2_plus column = D2 + D3 + D4 (for quick filtering)
    raise NotImplementedError("clean_drought_dataframe is not yet implemented.")
