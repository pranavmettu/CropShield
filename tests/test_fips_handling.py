"""
FIPS normalization tests for CropShield.

County FIPS codes flow through three separate code paths, each with its own
normalization logic:

  1. build_county_panel._normalise_fips   – cleans FIPS before panel merge
  2. fetch_nass.clean_nass_response       – constructs FIPS from state+county ANSI
  3. fetch_power.load_county_centroids    – normalises filter list

A single bad FIPS representation causes silent join failures (all weather
features become NaN for that county) without raising any error.

Test philosophy
---------------
Every test has an explicit expected value — "assert df is not None" is not
enough.  Tests cover the full range of real-world FIPS representations seen
in NASS CSV round-trips, pandas CSV ingest, and Census centroid files.
"""

from __future__ import annotations

import pandas as pd
import pytest

from cropshield.data.build_county_panel import _normalise_fips
from cropshield.data.fetch_nass import clean_nass_response


# ── _normalise_fips ───────────────────────────────────────────────────────────

class TestNormaliseFips:
    """Unit tests for build_county_panel._normalise_fips."""

    def test_float_large_state(self):
        """17001.0 (Illinois Adams) → '17001'."""
        result = _normalise_fips(pd.Series(["17001.0"]))
        assert result.iloc[0] == "17001"

    def test_float_small_state(self):
        """1001.0 (Alabama Autauga) → '01001' (must zero-pad to 5 chars)."""
        result = _normalise_fips(pd.Series(["1001.0"]))
        assert result.iloc[0] == "01001"

    def test_string_already_padded(self):
        """'17001' stays '17001'."""
        result = _normalise_fips(pd.Series(["17001"]))
        assert result.iloc[0] == "17001"

    def test_string_short_needs_padding(self):
        """'1001' → '01001'."""
        result = _normalise_fips(pd.Series(["1001"]))
        assert result.iloc[0] == "01001"

    def test_integer_large(self):
        """17001 (int stored as string by pandas) → '17001'."""
        result = _normalise_fips(pd.Series([17001]))
        assert result.iloc[0] == "17001"

    def test_integer_small(self):
        """1001 → '01001'."""
        result = _normalise_fips(pd.Series([1001]))
        assert result.iloc[0] == "01001"

    def test_nan_does_not_produce_00nan(self):
        """NaN must NOT produce the invalid string '00nan'."""
        result = _normalise_fips(pd.Series([float("nan")]))
        assert pd.isna(result.iloc[0]), (
            f"Expected pd.NA for NaN input, got {result.iloc[0]!r}"
        )

    def test_none_string_becomes_na(self):
        """'None' (string) must become pd.NA, not '0None'."""
        result = _normalise_fips(pd.Series(["None"]))
        assert pd.isna(result.iloc[0])

    def test_empty_string_becomes_na(self):
        result = _normalise_fips(pd.Series([""]))
        assert pd.isna(result.iloc[0])

    def test_mixed_series(self):
        """Mixed float, string, and NaN in one Series."""
        s = pd.Series(["17001.0", "01001", float("nan"), "19169"])
        result = _normalise_fips(s)
        assert result.iloc[0] == "17001"
        assert result.iloc[1] == "01001"
        assert pd.isna(result.iloc[2])
        assert result.iloc[3] == "19169"

    def test_all_five_chars(self):
        """Every non-null result must be exactly 5 characters."""
        s = pd.Series(["17001.0", "1001", "19169", "01001"])
        result = _normalise_fips(s)
        for val in result:
            assert len(val) == 5, f"Expected 5-char FIPS, got {val!r}"

    def test_join_float_matches_string(self):
        """A float FIPS on the yield side must join to a string FIPS on weather side."""
        yields = pd.DataFrame({
            "county_fips": ["17001.0", "19169.0"],
            "year":        [2020, 2020],
            "yield_anomaly": [5.0, -3.0],
        })
        weather = pd.DataFrame({
            "county_fips": ["17001", "19169"],
            "year":        [2020, 2020],
            "cumulative_precip": [400.0, 350.0],
        })
        yields["county_fips"]  = _normalise_fips(yields["county_fips"])
        weather["county_fips"] = _normalise_fips(weather["county_fips"])
        merged = yields.merge(weather, on=["county_fips", "year"], how="left")
        assert merged["cumulative_precip"].notna().all(), (
            "Float FIPS on yield side failed to join to string FIPS on weather side"
        )


# ── clean_nass_response FIPS construction ────────────────────────────────────

def _make_raw_nass_row(state_fips: str, county_ansi: str, value: str = "185.0") -> dict:
    """Build a minimal raw NASS API row dictionary."""
    return {
        "year":             "2020",
        "state_name":       "IOWA",
        "county_name":      "STORY",
        "commodity_desc":   "CORN",
        "unit_desc":        "BU / ACRE",
        "Value":            value,
        "state_fips_code":  state_fips,
        "county_ansi":      county_ansi,
    }


class TestNassFipsConstruction:
    """Tests for FIPS built inside clean_nass_response."""

    def test_iowa_story_county(self):
        """Iowa (19) + Story county (169) → '19169'."""
        raw = pd.DataFrame([_make_raw_nass_row("19", "169")])
        cleaned = clean_nass_response(raw)
        assert cleaned["county_fips"].iloc[0] == "19169"

    def test_alabama_autauga_county(self):
        """Alabama (1) + Autauga county (1) → '01001'."""
        raw = pd.DataFrame([_make_raw_nass_row("1", "1")])
        cleaned = clean_nass_response(raw)
        assert cleaned["county_fips"].iloc[0] == "01001"

    def test_illinois_adams_county(self):
        """Illinois (17) + Adams county (1) → '17001'."""
        raw = pd.DataFrame([_make_raw_nass_row("17", "1")])
        cleaned = clean_nass_response(raw)
        assert cleaned["county_fips"].iloc[0] == "17001"

    def test_five_char_fips_always(self):
        """All constructed FIPS codes must be exactly 5 characters."""
        rows = [
            _make_raw_nass_row("1",  "1"),
            _make_raw_nass_row("17", "1"),
            _make_raw_nass_row("19", "169"),
        ]
        cleaned = clean_nass_response(pd.DataFrame(rows))
        for fips in cleaned["county_fips"]:
            assert isinstance(fips, str) and len(fips) == 5, (
                f"Expected 5-char string, got {fips!r}"
            )

    def test_suppressed_value_row_is_dropped(self):
        """Rows with NASS suppression code '(D)' must be dropped (not kept as FIPS '00nan')."""
        rows = [
            _make_raw_nass_row("19", "169", "(D)"),
            _make_raw_nass_row("19", "169", "185.0"),
        ]
        cleaned = clean_nass_response(pd.DataFrame(rows))
        assert len(cleaned) == 1
        assert cleaned["value"].iloc[0] == pytest.approx(185.0)

    def test_z_suppression_code_dropped(self):
        """'(Z)' (rounds to zero) must also become NaN and be dropped."""
        raw = pd.DataFrame([_make_raw_nass_row("19", "169", "(Z)")])
        cleaned = clean_nass_response(raw)
        assert len(cleaned) == 0

    def test_value_with_comma_parsed(self):
        """'1,234.5' (NASS comma-formatted large numbers) must be parsed to 1234.5."""
        raw = pd.DataFrame([_make_raw_nass_row("19", "169", "1,234.5")])
        cleaned = clean_nass_response(raw)
        assert cleaned["value"].iloc[0] == pytest.approx(1234.5)

    def test_county_name_not_used_as_join_key(self):
        """Two counties with the same name but different FIPS must remain distinct."""
        rows = [
            # "WASHINGTON" county exists in many states
            {
                "year": "2020", "state_name": "IOWA",
                "county_name": "WASHINGTON",
                "commodity_desc": "CORN", "unit_desc": "BU / ACRE",
                "Value": "185.0", "state_fips_code": "19", "county_ansi": "183",
            },
            {
                "year": "2020", "state_name": "ILLINOIS",
                "county_name": "WASHINGTON",
                "commodity_desc": "CORN", "unit_desc": "BU / ACRE",
                "Value": "175.0", "state_fips_code": "17", "county_ansi": "189",
            },
        ]
        cleaned = clean_nass_response(pd.DataFrame(rows))
        fips_set = set(cleaned["county_fips"])
        assert "19183" in fips_set
        assert "17189" in fips_set
        assert len(fips_set) == 2, "County name collision collapsed two distinct counties"
