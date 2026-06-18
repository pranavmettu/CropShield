"""
NASS FIPS CSV round-trip tests.

Proves that saving and re-loading NASS / yield CSVs with float-like FIPS
values still produces valid 5-character county_fips strings.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from cropshield.data.fetch_nass import clean_nass_response
from cropshield.data.fips_utils import (
    build_county_fips_from_components,
    load_nass_yield_csv,
    normalise_fips_series,
)


def _raw_row(state_fips, county_ansi, value="185.0") -> dict:
    return {
        "year": "2020",
        "state_name": "IOWA",
        "county_name": "STORY",
        "commodity_desc": "CORN",
        "unit_desc": "BU / ACRE",
        "Value": value,
        "state_fips_code": state_fips,
        "county_ansi": county_ansi,
    }


class TestFipsComponentsFromFloatCsv:
    def test_float_components_produce_valid_fips(self):
        """CSV round-trip: state_fips=19.0, county_ansi=169.0 → '19169'."""
        raw = pd.DataFrame([_raw_row(19.0, 169.0)])
        cleaned = clean_nass_response(raw)
        assert cleaned["county_fips"].iloc[0] == "19169"

    def test_small_county_fips_zero_padded(self):
        """Alabama Autauga: state=1.0, county=1.0 → '01001'."""
        raw = pd.DataFrame([_raw_row(1.0, 1.0)])
        cleaned = clean_nass_response(raw)
        assert cleaned["county_fips"].iloc[0] == "01001"

    def test_build_from_float_series(self):
        st = pd.Series([19.0, 17.0, 1.0])
        co = pd.Series([169.0, 1.0, 1.0])
        result = build_county_fips_from_components(st, co)
        assert list(result) == ["19169", "17001", "01001"]


class TestLoadNassYieldCsvRoundTrip:
    def test_reload_float_fips_csv(self):
        """Save cleaned data, reload via load_nass_yield_csv — FIPS must stay valid."""
        raw = pd.DataFrame([
            _raw_row("19", "169"),
            _raw_row("17", "1"),
        ])
        cleaned = clean_nass_response(raw)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nass_clean.csv"
            # Simulate pandas writing floats for numeric-looking FIPS
            sim = cleaned.copy()
            sim["county_fips"] = sim["county_fips"].astype(float)
            sim.to_csv(path, index=False)

            reloaded = load_nass_yield_csv(path)
            assert set(reloaded["county_fips"]) == {"19169", "17001"}
            for fips in reloaded["county_fips"]:
                assert len(fips) == 5

    def test_reload_with_component_columns(self):
        """When county_fips is float-corrupted, rebuild from state_fips + county_ansi."""
        df = pd.DataFrame({
            "year": [2020, 2020],
            "state": ["IOWA", "ILLINOIS"],
            "county": ["STORY", "ADAMS"],
            "county_fips": [19169.0, 17001.0],
            "state_fips": [19.0, 17.0],
            "county_ansi": [169.0, 1.0],
            "crop": ["CORN", "CORN"],
            "value": [185.0, 190.0],
            "unit": ["BU / ACRE", "BU / ACRE"],
        })
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "yield.csv"
            df.to_csv(path, index=False)
            reloaded = load_nass_yield_csv(path)
            assert list(reloaded["county_fips"]) == ["19169", "17001"]

    def test_nan_fips_not_00nan(self):
        s = pd.Series([19169.0, float("nan"), "17001"])
        result = normalise_fips_series(s)
        assert result.iloc[0] == "19169"
        assert pd.isna(result.iloc[1])
        assert result.iloc[2] == "17001"
