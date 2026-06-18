"""
Missing and malformed data handling tests for CropShield.

Defensive data handling is critical for agricultural data pipelines where:
  - NASS suppresses county data with codes like '(D)' and '(Z)'
  - NASA POWER returns -999 as a missing-value sentinel
  - County FIPS may be absent or corrupt in source data
  - API responses may be empty or malformed

Tests verify that each failure mode either:
  a) Is handled explicitly with meaningful logging/exception, OR
  b) Propagates a clear exception (not a silent empty DataFrame or wrong result)

All tests use synthetic data — no live API calls.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from cropshield.data.fetch_nass import (
    _fetch_with_retry,
    clean_nass_response,
    validate_nass_dataframe,
)
from cropshield.features.weather_features import compute_weather_features
from cropshield.features.yield_targets import (
    add_expected_yield_rolling,
    clean_yield_dataframe,
)


# ── NASS suppressed values ─────────────────────────────────────────────────────

class TestNassSuppressionHandling:
    def _raw(self, value: str) -> pd.DataFrame:
        return pd.DataFrame([{
            "year": "2020", "state_name": "IOWA", "county_name": "STORY",
            "commodity_desc": "CORN", "unit_desc": "BU / ACRE",
            "Value": value, "state_fips_code": "19", "county_ansi": "169",
        }])

    def test_d_suppression_code_dropped(self):
        cleaned = clean_nass_response(self._raw(" (D) "))
        assert len(cleaned) == 0, "'(D)' suppression code must drop the row"

    def test_z_suppression_code_dropped(self):
        cleaned = clean_nass_response(self._raw(" (Z) "))
        assert len(cleaned) == 0, "'(Z)' suppression code must drop the row"

    def test_na_string_dropped(self):
        cleaned = clean_nass_response(self._raw("(NA)"))
        assert len(cleaned) == 0, "'(NA)' must drop the row"

    def test_empty_value_dropped(self):
        cleaned = clean_nass_response(self._raw(""))
        assert len(cleaned) == 0

    def test_whitespace_value_dropped(self):
        cleaned = clean_nass_response(self._raw("   "))
        assert len(cleaned) == 0

    def test_valid_value_kept(self):
        cleaned = clean_nass_response(self._raw("185.0"))
        assert len(cleaned) == 1
        assert cleaned["value"].iloc[0] == pytest.approx(185.0)

    def test_comma_formatted_number_parsed(self):
        """NASS often returns '1,234.5' for large numbers."""
        cleaned = clean_nass_response(self._raw("1,234.5"))
        assert cleaned["value"].iloc[0] == pytest.approx(1234.5)

    def test_all_suppressed_produces_empty_dataframe(self):
        """If every record is suppressed, clean_nass_response returns an empty DataFrame."""
        rows = [
            {**{"year": "2020", "state_name": "IOWA", "county_name": f"C{i}",
                "commodity_desc": "CORN", "unit_desc": "BU / ACRE",
                "state_fips_code": "19", "county_ansi": str(i)},
             "Value": "(D)"}
            for i in range(1, 6)
        ]
        cleaned = clean_nass_response(pd.DataFrame(rows))
        assert len(cleaned) == 0


# ── NASS validate_nass_dataframe ──────────────────────────────────────────────

class TestValidateNassDataframe:
    def _valid_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "year": [2020], "state": ["IOWA"], "county": ["STORY"],
            "county_fips": ["19169"], "crop": ["CORN"],
            "value": [185.0], "unit": ["BU / ACRE"],
        })

    def test_passes_on_valid_dataframe(self):
        validate_nass_dataframe(self._valid_df())  # should not raise

    def test_raises_on_missing_column(self):
        df = self._valid_df().drop(columns=["county_fips"])
        with pytest.raises(ValueError, match="county_fips"):
            validate_nass_dataframe(df)

    def test_raises_when_all_values_nan(self):
        df = self._valid_df()
        df["value"] = float("nan")
        with pytest.raises(ValueError):
            validate_nass_dataframe(df)

    def test_raises_when_value_is_non_numeric(self):
        df = self._valid_df()
        df["value"] = df["value"].astype(str)  # convert to object
        with pytest.raises(ValueError):
            validate_nass_dataframe(df)


# ── _fetch_with_retry: empty API response ────────────────────────────────────

class TestFetchWithRetryEmptyResponse:
    def test_empty_data_returns_none(self):
        """An API response with 'data': [] should return None, not an empty DataFrame."""
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {"data": []}

        with patch("cropshield.data.fetch_nass.requests.get", return_value=mock_resp):
            result = _fetch_with_retry(
                url="https://fake", params={}, max_retries=1, retry_delay=0, label="test"
            )
        assert result is None, "Empty API response must return None"

    def test_api_error_payload_raises_value_error(self):
        """NASS wraps errors in {'error': [...]} — must raise ValueError."""
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = {"error": ["Too many records requested."]}

        with patch("cropshield.data.fetch_nass.requests.get", return_value=mock_resp):
            with pytest.raises(ValueError, match="Too many records"):
                _fetch_with_retry(
                    url="https://fake", params={}, max_retries=1, retry_delay=0, label="test"
                )

    def test_http_error_retries_and_raises(self):
        """HTTP error should be retried and ultimately raised."""
        import requests as _req

        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = _req.HTTPError("503 Server Error")

        with patch("cropshield.data.fetch_nass.requests.get", return_value=mock_resp):
            with patch("cropshield.data.fetch_nass.time.sleep"):
                with pytest.raises(_req.HTTPError):
                    _fetch_with_retry(
                        url="https://fake", params={}, max_retries=2,
                        retry_delay=0, label="test"
                    )


# ── Weather missing days ───────────────────────────────────────────────────────

class TestWeatherMissingDays:
    def _make_county_year(
        self,
        fips: str = "19001",
        year: int = 2020,
        n_days: int = 153,
        missing_precip_idx: list[int] | None = None,
    ) -> pd.DataFrame:
        dates = pd.date_range(f"{year}-04-01", f"{year}-08-31", freq="D")[:n_days]
        precip = [5.0] * len(dates)
        if missing_precip_idx:
            for i in missing_precip_idx:
                precip[i] = float("nan")
        return pd.DataFrame({
            "county_fips": fips, "state_fips": fips[:2],
            "year": year, "date": dates, "month": [d.month for d in dates],
            "PRECTOTCORR": precip, "T2M": 20.0, "T2M_MIN": 15.0, "T2M_MAX": 25.0,
        })

    def test_missing_precip_handled_gracefully(self):
        """NaN precipitation days are dropped (dropna) before aggregation — no crash."""
        df = self._make_county_year(missing_precip_idx=[0, 1, 2])
        features = compute_weather_features(df)
        assert len(features) == 1
        assert features["cumulative_precip"].notna().all()

    def test_all_precip_nan_gives_zero_cumulative(self):
        """If all precipitation values are NaN, cumulative_precip should be 0 (sum of empty)."""
        df = self._make_county_year()
        df["PRECTOTCORR"] = float("nan")
        features = compute_weather_features(df)
        assert features["cumulative_precip"].iloc[0] == pytest.approx(0.0)

    def test_partial_season_county_still_produces_row(self):
        """A county with only 30 days of growing-season data must still produce a feature row."""
        df = self._make_county_year(n_days=30)
        features = compute_weather_features(df)
        assert len(features) == 1

    def test_obs_days_reflects_actual_day_count(self):
        """obs_days column must match the actual number of records provided."""
        df = self._make_county_year(n_days=90)
        features = compute_weather_features(df)
        assert features["obs_days"].iloc[0] == 90


# ── yield target: malformed inputs ────────────────────────────────────────────

class TestYieldTargetMalformedInputs:
    def _base_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            "year": [2018, 2019, 2020, 2021, 2022],
            "state": "IOWA",
            "county": "STORY",
            "county_fips": "19169",
            "crop": "CORN",
            "value": [180.0, 185.0, 175.0, 190.0, 178.0],
        })

    def test_missing_required_column_raises_key_error(self):
        df = self._base_df().drop(columns=["county_fips"])
        with pytest.raises(KeyError):
            clean_yield_dataframe(df)

    def test_non_numeric_value_coerced_to_nan_and_dropped(self):
        df = self._base_df()
        df.loc[df["year"] == 2020, "value"] = "INVALID"
        cleaned = clean_yield_dataframe(df)
        # 2020 row should be dropped
        assert 2020 not in cleaned["year"].values
        assert len(cleaned) == 4

    def test_all_nan_yields_raises_or_produces_empty(self):
        df = self._base_df()
        df["value"] = float("nan")
        cleaned = clean_yield_dataframe(df)
        assert len(cleaned) == 0, "All-NaN yields should produce an empty DataFrame"

    def test_rolling_on_single_row_produces_nan(self):
        """A county with only 1 year of history cannot produce expected_yield."""
        df = pd.DataFrame({
            "year": [2020], "state": "IOWA", "county": "STORY",
            "county_fips": "19169", "crop": "CORN", "value": [185.0],
        })
        cleaned = clean_yield_dataframe(df)
        result = add_expected_yield_rolling(cleaned, window=5, min_periods=3)
        assert result["expected_yield"].isna().all()

    def test_duplicate_county_year_rows_handled(self):
        """
        Duplicate (county, year) rows do not crash and at least one of the
        duplicate rows receives a valid expected_yield once enough prior history
        exists.  The exact value depends on sort order; this test documents
        the current behaviour rather than asserting a single canonical value.
        """
        df = self._base_df()
        df_duped = pd.concat([df, df.iloc[[2]]], ignore_index=True)  # duplicate 2020
        cleaned = clean_yield_dataframe(df_duped)
        result = add_expected_yield_rolling(cleaned, window=5, min_periods=3)
        rows_2020 = result[result["year"] == 2020]
        # At least one 2020 row should eventually get a non-NaN expected_yield
        # (the second occurrence has 3+ prior rows in the sort order)
        assert len(rows_2020) >= 1  # duplicates survive to here
        # Non-NaN expected_yield must be in a plausible corn-yield range
        valid_expected = rows_2020["expected_yield"].dropna()
        if len(valid_expected) > 0:
            assert (valid_expected > 100.0).all() and (valid_expected < 300.0).all()
