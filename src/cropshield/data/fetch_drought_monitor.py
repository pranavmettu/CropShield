"""
U.S. Drought Monitor data fetcher for CropShield.

Live API fetching is optional; ``clean_drought_dataframe`` and feature
engineering in ``drought_features.py`` work on any raw CSV matching the
expected schema.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from cropshield.data.fips_utils import normalise_fips_series, nass_csv_dtypes

logger = logging.getLogger(__name__)

DROUGHT_CATEGORIES = ["D0", "D1", "D2", "D3", "D4"]
DROUGHT_BASE_URL = "https://usdm.climate.unl.edu/DM_Export.ashx"


def fetch_drought_monitor(
    state_fips_list: list[str],
    start_year: int = 2015,
    end_year: int | None = None,
    output_raw: str | Path = "data/raw/drought_monitor.csv",
    output_clean: str | Path = "data/interim/drought_monitor_clean.csv",
) -> pd.DataFrame:
    """Fetch weekly county-level drought statistics from the U.S. Drought Monitor.

    Live API integration is not yet implemented.  If ``output_raw`` already
    exists on disk, it is cleaned and returned instead.

    Raises
    ------
    NotImplementedError
        When no raw file exists and a live fetch would be required.
    """
    output_raw = Path(output_raw)
    if output_raw.exists():
        logger.info("Loading existing drought raw file → %s", output_raw)
        header = pd.read_csv(output_raw, nrows=0)
        raw = pd.read_csv(output_raw, dtype=nass_csv_dtypes(header.columns))
        clean = clean_drought_dataframe(raw)
        output_clean = Path(output_clean)
        output_clean.parent.mkdir(parents=True, exist_ok=True)
        clean.to_csv(output_clean, index=False)
        return clean

    raise NotImplementedError(
        "Live Drought Monitor API fetch is not yet implemented. "
        f"Place raw weekly data at {output_raw} or implement the API client first."
    )


def clean_drought_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise a raw Drought Monitor DataFrame.

    Returns
    -------
    pd.DataFrame
        Columns: ``date``, ``county_fips``, ``year``, ``D0``–``D4``.
        Category columns are floats (0–100).  ``county_fips`` is a 5-char string.
    """
    df = df.copy()

    # Normalise common column name variants
    rename_map = {
        "MapDate": "date",
        "FIPS": "county_fips",
        "CountyFIPS": "county_fips",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    if "date" not in df.columns:
        raise KeyError("clean_drought_dataframe: 'date' column not found.")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    if "county_fips" in df.columns:
        df["county_fips"] = normalise_fips_series(df["county_fips"])

    df["year"] = df["date"].dt.year.astype("Int64")

    for cat in DROUGHT_CATEGORIES:
        if cat not in df.columns:
            df[cat] = 0.0
        df[cat] = pd.to_numeric(df[cat], errors="coerce").fillna(0.0).clip(0.0, 100.0)

    keep = ["date", "county_fips", "year"] + DROUGHT_CATEGORIES
    df = df[keep].sort_values(["county_fips", "date"]).reset_index(drop=True)
    logger.info(
        "clean_drought_dataframe: %d weekly rows | %d counties",
        len(df), df["county_fips"].nunique(),
    )
    return df
