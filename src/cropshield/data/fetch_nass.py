"""
USDA NASS Quick Stats data fetcher for CropShield.

Retrieves county-level crop yield data from the USDA National Agricultural
Statistics Service (NASS) Quick Stats API and saves it to the data pipeline.

API Reference
-------------
https://quickstats.nass.usda.gov/api

Notes
-----
- A free API key is required. Register at https://quickstats.nass.usda.gov/api
- Set the key as NASS_API_KEY in your .env file.
- County-level records may be suppressed by NASS for confidentiality when
  fewer than 3 operations exist in a county. Suppressed records are dropped.
- This module only downloads and lightly cleans data. Target engineering
  (yield anomaly calculation) lives in src/cropshield/features/yield_targets.py.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
NASS_API_URL = "https://quickstats.nass.usda.gov/api/api_GET/"
REQUIRED_COLUMNS = ["year", "state_name", "county_name", "county_ansi", "value"]


def fetch_nass_yield(
    api_key: Optional[str] = None,
    crop: str = "CORN",
    states: list[str] | None = None,
    start_year: int = 2015,
    end_year: Optional[int] = None,
    output_raw: str | Path = "data/raw/nass_yield.csv",
    output_clean: str | Path = "data/interim/nass_yield_clean.csv",
) -> pd.DataFrame:
    """Fetch county-level crop yield from USDA NASS Quick Stats and save to disk.

    Parameters
    ----------
    api_key : str, optional
        NASS API key. Falls back to the ``NASS_API_KEY`` environment variable.
    crop : str
        NASS commodity description (e.g. ``"CORN"`` or ``"SOYBEANS"``).
    states : list[str], optional
        List of NASS state names in ALL CAPS (e.g. ``["IOWA", "ILLINOIS"]``).
        Defaults to the MVP states when ``None``.
    start_year : int
        First year to include (inclusive).
    end_year : int, optional
        Last year to include (inclusive). Defaults to the most recent year
        available when ``None``.
    output_raw : str or Path
        Destination for the unmodified API response CSV.
    output_clean : str or Path
        Destination for the cleaned, standardised CSV.

    Returns
    -------
    pd.DataFrame
        Cleaned yield dataframe with standardised columns.

    Raises
    ------
    EnvironmentError
        If no API key is found in arguments or environment.
    requests.HTTPError
        If the NASS API returns a non-200 status.

    Examples
    --------
    >>> df = fetch_nass_yield(states=["IOWA", "ILLINOIS"], start_year=2015)
    >>> df.columns.tolist()
    ['year', 'state', 'county', 'county_fips', 'crop', 'value', 'unit']
    """
    # TODO: Implement API call logic
    # Steps:
    # 1. Resolve API key from argument or environment
    # 2. Build request parameters dict
    # 3. Loop over each state (API supports one state per call)
    # 4. Retry failed requests up to 3 times
    # 5. Concatenate state responses
    # 6. Save raw CSV
    # 7. Call clean_nass_response() to standardise columns
    # 8. Save clean CSV
    # 9. Return cleaned DataFrame
    raise NotImplementedError("fetch_nass_yield is not yet implemented. See Prompt 2.")


def clean_nass_response(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise a raw NASS API response DataFrame.

    Handles:
    - Renaming verbose NASS column names to short canonical names.
    - Converting the ``Value`` column to numeric (removes commas, coerces
      suppressed entries like ``" (D)"`` to NaN).
    - Dropping rows with suppressed or missing yield values.
    - Creating a ``county_fips`` column from state + county ANSI codes.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe as returned by the NASS API.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe with columns:
        ``year``, ``state``, ``county``, ``county_fips``, ``crop``,
        ``value``, ``unit``.
    """
    # TODO: Implement cleaning logic
    # Steps:
    # 1. Rename columns (year, state_name -> state, county_name -> county, etc.)
    # 2. Strip whitespace from string columns
    # 3. Convert Value to numeric: pd.to_numeric(df["Value"].str.replace(",", ""), errors="coerce")
    # 4. Drop rows where value is NaN (suppressed or missing)
    # 5. Build county_fips = state_fips_code.zfill(2) + county_ansi.zfill(3)
    # 6. Drop unnecessary NASS metadata columns
    # 7. Return clean DataFrame
    raise NotImplementedError("clean_nass_response is not yet implemented.")


def validate_nass_dataframe(df: pd.DataFrame) -> None:
    """Assert that the cleaned NASS DataFrame has the expected schema.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned NASS DataFrame.

    Raises
    ------
    ValueError
        If required columns are missing or if the value column is not numeric.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"NASS DataFrame is missing required columns: {missing}")
    if not pd.api.types.is_numeric_dtype(df["value"]):
        raise ValueError("Column 'value' must be numeric after cleaning.")
    logger.info(
        "NASS validation passed: %d rows, %d unique counties, years %d–%d",
        len(df),
        df["county_fips"].nunique() if "county_fips" in df.columns else -1,
        int(df["year"].min()),
        int(df["year"].max()),
    )
