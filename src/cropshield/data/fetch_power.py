"""
NASA POWER weather data fetcher for CropShield.

Retrieves daily weather data from the NASA POWER Agroclimatology API for
county centroids and saves it to the data pipeline.

API Reference
-------------
https://power.larc.nasa.gov/docs/services/api/

Notes
-----
- No API key is required for NASA POWER (as of 2024).
- Requests are made per county centroid (lat/lon point) per year.
- The MVP fetches data for April–August (growing season) only.
- Feature engineering from the raw daily data lives in
  src/cropshield/features/weather_features.py.
- County centroids must be provided; a reference CSV is stored in
  data/external/county_fips.csv.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
POWER_BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
POWER_PARAMETERS = ["PRECTOTCORR", "T2M", "T2M_MIN", "T2M_MAX"]
POWER_COMMUNITY = "AG"
GROWING_SEASON_START = "0401"  # April 1  (MMDD)
GROWING_SEASON_END = "0831"    # August 31 (MMDD)


def fetch_power_for_county(
    lat: float,
    lon: float,
    year: int,
    parameters: list[str] = POWER_PARAMETERS,
    start_mmdd: str = GROWING_SEASON_START,
    end_mmdd: str = GROWING_SEASON_END,
    max_retries: int = 3,
    retry_delay: float = 5.0,
) -> pd.DataFrame:
    """Fetch daily weather data from NASA POWER for a single county centroid.

    Parameters
    ----------
    lat : float
        Latitude of the county centroid.
    lon : float
        Longitude of the county centroid.
    year : int
        Calendar year to fetch.
    parameters : list[str]
        NASA POWER parameter codes (e.g. ``"PRECTOTCORR"``, ``"T2M_MAX"``).
    start_mmdd : str
        Growing season start as MMDD string (e.g. ``"0401"`` for April 1).
    end_mmdd : str
        Growing season end as MMDD string (e.g. ``"0831"`` for August 31).
    max_retries : int
        Number of retry attempts on API failure.
    retry_delay : float
        Seconds to wait between retries.

    Returns
    -------
    pd.DataFrame
        Daily weather records with columns: ``date``, ``lat``, ``lon``, and
        one column per requested parameter.

    Raises
    ------
    requests.HTTPError
        If the API returns a non-200 status after all retries.
    """
    # TODO: Implement NASA POWER API call
    # Steps:
    # 1. Build URL parameters: latitude, longitude, start, end, parameters, community, format=JSON
    # 2. Send GET request with retry loop
    # 3. Parse JSON response: response["properties"]["parameter"]
    # 4. Reshape to long DataFrame with date, parameter, value columns
    # 5. Pivot to wide format (one row per date)
    # 6. Add lat/lon columns
    # 7. Return DataFrame
    raise NotImplementedError("fetch_power_for_county is not yet implemented. See Prompt 4.")


def fetch_power_all_counties(
    county_centroids: pd.DataFrame,
    years: list[int],
    output_raw: str | Path = "data/raw/weather_daily_raw.csv",
    checkpoint_every: int = 10,
) -> pd.DataFrame:
    """Fetch NASA POWER data for all county centroids across multiple years.

    Loops over each (county, year) combination and saves a checkpoint CSV
    every ``checkpoint_every`` counties to allow resuming interrupted runs.

    Parameters
    ----------
    county_centroids : pd.DataFrame
        DataFrame with columns: ``county_fips``, ``county``, ``state``,
        ``lat``, ``lon``.
    years : list[int]
        Calendar years to fetch.
    output_raw : str or Path
        Output path for the concatenated raw weather CSV.
    checkpoint_every : int
        Save a checkpoint to disk after this many counties are processed.

    Returns
    -------
    pd.DataFrame
        Concatenated daily weather records for all counties and years.
    """
    # TODO: Implement batch fetching loop with checkpointing
    # Steps:
    # 1. Check if a checkpoint file exists; if so, load it and skip completed (fips, year) pairs
    # 2. Loop over each row in county_centroids × years
    # 3. Call fetch_power_for_county() for each pair
    # 4. Append county_fips, state, county columns to the result
    # 5. Every checkpoint_every counties, save progress to disk
    # 6. Concatenate all results and save final CSV
    raise NotImplementedError("fetch_power_all_counties is not yet implemented.")
