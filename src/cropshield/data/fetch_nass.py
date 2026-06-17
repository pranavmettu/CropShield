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
- Set the key as NASS_API_KEY in your .env file (see .env.example).
- County-level records may be suppressed by NASS for confidentiality when
  fewer than 3 operations exist in a county. Suppressed records (marked "(D)")
  are coerced to NaN and dropped.
- This module only downloads and lightly cleans data. Target engineering
  (yield anomaly calculation) lives in src/cropshield/features/yield_targets.py.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from dotenv import load_dotenv
import os

load_dotenv()

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
NASS_API_URL = "https://quickstats.nass.usda.gov/api/api_GET/"
MVP_STATES = ["IOWA", "ILLINOIS"]

# Columns required in the final cleaned DataFrame
CLEAN_COLUMNS = ["year", "state", "county", "county_fips", "crop", "value", "unit"]

# NASS API always returns these two columns alongside Value
_RAW_RENAME = {
    "year":             "year",
    "state_name":       "state",
    "county_name":      "county",
    "commodity_desc":   "crop",
    "unit_desc":        "unit",
    "Value":            "value",
    "state_fips_code":  "state_fips",
    "county_ansi":      "county_ansi",
}


def fetch_nass_yield(
    api_key: Optional[str] = None,
    crop: str = "CORN",
    states: list[str] | None = None,
    start_year: int = 2015,
    end_year: Optional[int] = None,
    output_raw: str | Path = "data/raw/nass_yield.csv",
    output_clean: str | Path = "data/interim/nass_yield_clean.csv",
    max_retries: int = 3,
    retry_delay: float = 5.0,
) -> pd.DataFrame:
    """Fetch county-level crop yield from USDA NASS Quick Stats and save to disk.

    Loops over each requested state (the NASS API works best per-state),
    concatenates responses, saves the raw dump, then cleans and saves the
    interim file.

    Parameters
    ----------
    api_key : str, optional
        NASS API key. Falls back to the ``NASS_API_KEY`` environment variable.
    crop : str
        NASS commodity description in ALL CAPS (e.g. ``"CORN"`` or ``"SOYBEANS"``).
    states : list[str], optional
        State names in ALL CAPS (e.g. ``["IOWA", "ILLINOIS"]``).
        Defaults to the MVP states when ``None``.
    start_year : int
        First year to include (inclusive).
    end_year : int, optional
        Last year to include (inclusive). Fetches all available years when ``None``.
    output_raw : str or Path
        Destination for the unmodified API response CSV.
    output_clean : str or Path
        Destination for the cleaned, standardised CSV.
    max_retries : int
        Number of retry attempts on transient API failures.
    retry_delay : float
        Seconds to wait between retries.

    Returns
    -------
    pd.DataFrame
        Cleaned yield DataFrame with columns:
        ``year``, ``state``, ``county``, ``county_fips``, ``crop``,
        ``value``, ``unit``.

    Raises
    ------
    EnvironmentError
        If no API key is found in arguments or the ``NASS_API_KEY`` env var.
    requests.HTTPError
        If the NASS API returns a non-200 status after all retries.
    ValueError
        If the API returns an error payload (e.g. too many records).

    Examples
    --------
    >>> df = fetch_nass_yield(states=["IOWA", "ILLINOIS"], start_year=2015)
    >>> df.columns.tolist()
    ['year', 'state', 'county', 'county_fips', 'crop', 'value', 'unit']
    """
    # ── 1. Resolve API key ───────────────────────────────────────────────────
    key = api_key or os.getenv("NASS_API_KEY", "").strip()
    if not key:
        raise EnvironmentError(
            "No NASS API key found. Set NASS_API_KEY in your .env file or pass "
            "api_key= directly. Register for a free key at "
            "https://quickstats.nass.usda.gov/api"
        )

    states = states or MVP_STATES
    logger.info(
        "Fetching NASS %s yield for states=%s, years=%d–%s",
        crop, states, start_year, end_year or "latest",
    )

    # ── 2. Fetch one state at a time ─────────────────────────────────────────
    all_frames: list[pd.DataFrame] = []
    for state in states:
        logger.info("  Requesting state: %s", state)
        params: dict = {
            "key":              key,
            "source_desc":      "SURVEY",
            "sector_desc":      "CROPS",
            "group_desc":       "FIELD CROPS",
            "commodity_desc":   crop,
            "statisticcat_desc": "YIELD",
            "unit_desc":        "BU / ACRE",
            "agg_level_desc":   "COUNTY",
            "domain_desc":      "TOTAL",
            "state_name":       state,
            "year__GE":         str(start_year),
            "format":           "JSON",
        }
        if end_year is not None:
            params["year__LE"] = str(end_year)

        frame = _fetch_with_retry(
            url=NASS_API_URL,
            params=params,
            max_retries=max_retries,
            retry_delay=retry_delay,
            label=state,
        )
        if frame is not None and len(frame) > 0:
            all_frames.append(frame)
            logger.info("    → %d rows received", len(frame))
        else:
            logger.warning("    → No data returned for %s", state)

    if not all_frames:
        raise ValueError(
            "No data returned from NASS API for any requested state. "
            "Check your API key and parameters."
        )

    raw_df = pd.concat(all_frames, ignore_index=True)
    logger.info("Total raw rows fetched: %d", len(raw_df))

    # ── 3. Save raw CSV ──────────────────────────────────────────────────────
    output_raw = Path(output_raw)
    output_raw.parent.mkdir(parents=True, exist_ok=True)
    raw_df.to_csv(output_raw, index=False)
    logger.info("Raw data saved → %s", output_raw)

    # ── 4. Clean and save ────────────────────────────────────────────────────
    clean_df = clean_nass_response(raw_df)
    output_clean = Path(output_clean)
    output_clean.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(output_clean, index=False)
    logger.info("Clean data saved → %s  (%d rows)", output_clean, len(clean_df))

    # ── 5. Validate ──────────────────────────────────────────────────────────
    validate_nass_dataframe(clean_df)

    return clean_df


def _fetch_with_retry(
    url: str,
    params: dict,
    max_retries: int,
    retry_delay: float,
    label: str = "",
) -> pd.DataFrame | None:
    """Send a GET request to the NASS API with exponential-backoff retries.

    Returns a raw DataFrame of the ``data`` key from the JSON response,
    or ``None`` if the response contains zero records.

    Raises
    ------
    requests.HTTPError
        After all retries are exhausted on a non-200 response.
    ValueError
        If the API returns an error message (e.g. record count limit exceeded).
    """
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=60)
            resp.raise_for_status()
            payload = resp.json()

            # NASS wraps errors in {"error": [...]} instead of HTTP error codes
            if "error" in payload:
                msg = "; ".join(payload["error"])
                raise ValueError(f"NASS API error for {label!r}: {msg}")

            records = payload.get("data", [])
            if not records:
                return None
            return pd.DataFrame(records)

        except (requests.RequestException, ValueError) as exc:
            last_exc = exc
            if attempt < max_retries:
                wait = retry_delay * attempt
                logger.warning(
                    "NASS request failed for %s (attempt %d/%d): %s — retrying in %.0fs",
                    label, attempt, max_retries, exc, wait,
                )
                time.sleep(wait)
            else:
                logger.error(
                    "NASS request failed for %s after %d attempts: %s",
                    label, max_retries, exc,
                )

    if last_exc is not None:
        raise last_exc
    return None


def clean_nass_response(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise a raw NASS API response DataFrame.

    Handles
    -------
    - Renaming verbose NASS column names to short canonical names.
    - Stripping whitespace from all string columns.
    - Converting the ``Value`` column to numeric (removes commas; coerces
      suppressed entries like ``" (D)"`` and ``" (Z)"`` to NaN).
    - Dropping rows with suppressed or missing yield values.
    - Building a 5-digit ``county_fips`` from state + county ANSI codes.
    - Returning only the columns needed downstream.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame as returned by the NASS API (one row per record).

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with columns:
        ``year``, ``state``, ``county``, ``county_fips``, ``crop``,
        ``value``, ``unit``.
    """
    df = df.copy()

    # ── Strip whitespace on all object columns ────────────────────────────────
    str_cols = df.select_dtypes(include="object").columns
    df[str_cols] = df[str_cols].apply(lambda col: col.str.strip())

    # ── Rename to canonical names ─────────────────────────────────────────────
    df = df.rename(columns=_RAW_RENAME)

    # ── Convert year to int ───────────────────────────────────────────────────
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")

    # ── Convert value to numeric (handles commas and NASS suppression codes) ──
    # Suppressed values come in as "(D)", "(Z)", "(NA)", etc.
    df["value"] = pd.to_numeric(
        df["value"].astype(str).str.replace(",", "", regex=False),
        errors="coerce",
    )

    # ── Drop rows with missing yield ──────────────────────────────────────────
    n_before = len(df)
    df = df.dropna(subset=["value"])
    n_dropped = n_before - len(df)
    if n_dropped:
        logger.info(
            "Dropped %d rows with suppressed or missing yield values (%.1f%%)",
            n_dropped, 100 * n_dropped / n_before,
        )

    # ── Build 5-digit county FIPS ─────────────────────────────────────────────
    # state_fips is 2 digits; county_ansi is 3 digits
    if "state_fips" in df.columns and "county_ansi" in df.columns:
        df["county_fips"] = (
            df["state_fips"].astype(str).str.zfill(2)
            + df["county_ansi"].astype(str).str.zfill(3)
        )
        # Mark rows where either component is missing/non-numeric
        df.loc[
            df["state_fips"].astype(str).str.strip().isin(["", "nan"])
            | df["county_ansi"].astype(str).str.strip().isin(["", "nan"]),
            "county_fips",
        ] = pd.NA
    else:
        logger.warning("state_fips_code / county_ansi not found; county_fips will be missing.")
        df["county_fips"] = pd.NA

    # ── Keep only the clean output columns ───────────────────────────────────
    for col in CLEAN_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    df = df[CLEAN_COLUMNS].copy()
    df = df.sort_values(["state", "county", "year"]).reset_index(drop=True)

    return df


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
    missing = [c for c in CLEAN_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"NASS DataFrame is missing required columns: {missing}")
    if not pd.api.types.is_numeric_dtype(df["value"]):
        raise ValueError("Column 'value' must be numeric after cleaning.")
    if df["value"].isna().all():
        raise ValueError("All values in 'value' column are NaN — cleaning may have failed.")

    logger.info(
        "NASS validation passed: %d rows | %d unique counties | years %d–%d | "
        "mean yield %.1f bu/acre",
        len(df),
        df["county_fips"].nunique(),
        int(df["year"].min()),
        int(df["year"].max()),
        df["value"].mean(),
    )
