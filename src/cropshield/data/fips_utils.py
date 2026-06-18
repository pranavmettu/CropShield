"""
FIPS code normalization utilities for CropShield.

Centralises county and state FIPS handling so CSV round-trips (where pandas
may infer ``17001.0`` as float) never produce invalid join keys.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Columns that must be read as strings when loading NASS / yield CSVs from disk.
NASS_FIPS_COLUMNS = ("county_fips", "state_fips", "county_ansi", "state_fips_code")
YIELD_FIPS_COLUMNS = ("county_fips",)


def normalise_fips_series(series: pd.Series) -> pd.Series:
    """Convert a FIPS Series to zero-padded 5-character strings.

    Handles floats (17001.0), ints (17001), and strings ('17001', '01001').
    NaN / None / empty strings become ``pd.NA`` (never ``"00nan"``).
    """
    s = series.astype(str).str.strip()
    missing_mask = s.isin({"nan", "None", "NaN", ""})
    result = (
        s.str.split(".").str[0]
        .str.strip()
        .str.zfill(5)
    )
    return result.where(~missing_mask, other=pd.NA)


def normalise_state_fips_series(series: pd.Series) -> pd.Series:
    """Convert a 2-digit state FIPS Series to zero-padded strings."""
    s = series.astype(str).str.strip()
    missing_mask = s.isin({"nan", "None", "NaN", ""})
    result = s.str.split(".").str[0].str.strip().str.zfill(2)
    return result.where(~missing_mask, other=pd.NA)


def normalise_county_ansi_series(series: pd.Series) -> pd.Series:
    """Convert a 3-digit county ANSI Series to zero-padded strings."""
    s = series.astype(str).str.strip()
    missing_mask = s.isin({"nan", "None", "NaN", ""})
    result = s.str.split(".").str[0].str.strip().str.zfill(3)
    return result.where(~missing_mask, other=pd.NA)


def build_county_fips_from_components(
    state_fips: pd.Series,
    county_ansi: pd.Series,
) -> pd.Series:
    """Build 5-digit county FIPS from state (2-digit) + county ANSI (3-digit)."""
    st = normalise_state_fips_series(state_fips)
    co = normalise_county_ansi_series(county_ansi)
    combined = st.astype(str) + co.astype(str)
    invalid = st.isna() | co.isna()
    return combined.where(~invalid, other=pd.NA)


def nass_csv_dtypes(columns: pd.Index) -> dict[str, str]:
    """Return dtype map for FIPS-related columns present in *columns*."""
    return {col: "string" for col in NASS_FIPS_COLUMNS if col in columns}


def load_nass_yield_csv(path: str | Path) -> pd.DataFrame:
    """Load a NASS or yield CSV with FIPS columns coerced to strings immediately."""
    path = Path(path)
    header = pd.read_csv(path, nrows=0)
    df = pd.read_csv(path, dtype=nass_csv_dtypes(header.columns))
    return normalise_nass_fips_columns(df)


def normalise_nass_fips_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise all FIPS-related columns in a NASS/yield DataFrame."""
    df = df.copy()
    if "county_fips" in df.columns:
        df["county_fips"] = normalise_fips_series(df["county_fips"])
    if "state_fips" in df.columns:
        df["state_fips"] = normalise_state_fips_series(df["state_fips"])
    if "state_fips_code" in df.columns:
        df["state_fips_code"] = normalise_state_fips_series(df["state_fips_code"])
    if "county_ansi" in df.columns:
        df["county_ansi"] = normalise_county_ansi_series(df["county_ansi"])
    # Rebuild county_fips from components when missing but components exist
    st_col = "state_fips" if "state_fips" in df.columns else (
        "state_fips_code" if "state_fips_code" in df.columns else None
    )
    if st_col and "county_ansi" in df.columns:
        rebuilt = build_county_fips_from_components(df[st_col], df["county_ansi"])
        if "county_fips" not in df.columns:
            df["county_fips"] = rebuilt
        else:
            df["county_fips"] = df["county_fips"].fillna(rebuilt)
    return df
