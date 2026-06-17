"""
Tests for weather feature engineering (src/cropshield/features/weather_features.py).

Tests use small, hand-crafted DataFrames to verify correctness of each
individual feature function and the full aggregation pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cropshield.features.weather_features import (
    compute_weather_features,
    cumulative_precip,
    dry_days,
    extreme_heat_days,
    filter_growing_season,
    growing_degree_days,
    longest_dry_spell,
    mean_temp,
    max_temp,
    add_precip_anomaly,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_daily_df(
    county_fips: str = "19001",
    state_fips: str = "19",
    year: int = 2020,
    n_days: int = 153,  # April 1 – August 31 = 153 days
) -> pd.DataFrame:
    """Synthetic daily weather DataFrame for one county-year growing season."""
    dates = pd.date_range(f"{year}-04-01", f"{year}-08-31", freq="D")
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "county_fips":  county_fips,
        "state_fips":   state_fips,
        "year":         year,
        "date":         dates,
        "month":        dates.month,
        "PRECTOTCORR":  rng.exponential(scale=3.0, size=len(dates)),
        "T2M":          rng.normal(loc=20.0, scale=5.0, size=len(dates)),
        "T2M_MIN":      rng.normal(loc=14.0, scale=4.0, size=len(dates)),
        "T2M_MAX":      rng.normal(loc=26.0, scale=6.0, size=len(dates)),
    })


# ── filter_growing_season ─────────────────────────────────────────────────────

class TestFilterGrowingSeason:
    def test_removes_non_growing_months(self):
        dates = pd.date_range("2020-01-01", "2020-12-31", freq="D")
        df = pd.DataFrame({"date": dates, "val": 1.0})
        result = filter_growing_season(df)
        assert result["date"].dt.month.min() == 4
        assert result["date"].dt.month.max() == 8

    def test_keeps_correct_day_count(self):
        dates = pd.date_range("2020-01-01", "2020-12-31", freq="D")
        df = pd.DataFrame({"date": dates, "val": 1.0})
        result = filter_growing_season(df)
        # April(30) + May(31) + June(30) + July(31) + Aug(31) = 153
        assert len(result) == 153

    def test_handles_string_dates(self):
        df = pd.DataFrame({"date": ["2020-04-01", "2020-09-01"], "val": [1, 2]})
        result = filter_growing_season(df)
        assert len(result) == 1


# ── longest_dry_spell ─────────────────────────────────────────────────────────

class TestLongestDrySpell:
    def test_basic_spell_after_wet_start(self):
        # 5.0 is wet, then 3 dry, then 5.0 wet, then 1 dry
        s = pd.Series([5.0, 0.0, 0.0, 0.0, 5.0, 0.0])
        assert longest_dry_spell(s) == 3

    def test_no_dry_days(self):
        s = pd.Series([2.0, 5.0, 3.0, 10.0])
        assert longest_dry_spell(s) == 0

    def test_all_dry_days(self):
        s = pd.Series([0.0, 0.0, 0.5, 0.3, 0.0])
        assert longest_dry_spell(s) == 5

    def test_single_dry_day(self):
        assert longest_dry_spell(pd.Series([0.0])) == 1
        assert longest_dry_spell(pd.Series([5.0])) == 0

    def test_threshold_boundary(self):
        # Exactly at threshold counts as dry
        s = pd.Series([1.0, 1.0, 2.0])
        assert longest_dry_spell(s, threshold=1.0) == 2

    def test_multiple_spells_returns_longest(self):
        # Spell of 2, then wet, then spell of 3
        s = pd.Series([0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 5.0])
        assert longest_dry_spell(s) == 3

    def test_alternating_days(self):
        s = pd.Series([0.0, 5.0, 0.0, 5.0, 0.0])
        assert longest_dry_spell(s) == 1

    def test_returns_int(self):
        s = pd.Series([0.0, 0.0, 5.0])
        result = longest_dry_spell(s)
        assert isinstance(result, int)


# ── growing_degree_days ───────────────────────────────────────────────────────

class TestGrowingDegreeDays:
    def test_basic_gdd(self):
        s = pd.Series([8.0, 12.0, 15.0, 9.0])
        result = growing_degree_days(s, base_temp=10.0)
        # max(0,8-10)=0, max(0,12-10)=2, max(0,15-10)=5, max(0,9-10)=0 → 7
        assert result == pytest.approx(7.0)

    def test_all_below_base(self):
        s = pd.Series([5.0, 6.0, 7.0])
        assert growing_degree_days(s, base_temp=10.0) == 0.0

    def test_zero_base(self):
        s = pd.Series([10.0, 15.0, 20.0])
        assert growing_degree_days(s, base_temp=0.0) == pytest.approx(45.0)

    def test_empty_series(self):
        assert growing_degree_days(pd.Series([], dtype=float)) == 0.0

    def test_negative_temperatures(self):
        s = pd.Series([-5.0, -2.0, 0.0])
        assert growing_degree_days(s, base_temp=10.0) == 0.0

    def test_returns_float(self):
        assert isinstance(growing_degree_days(pd.Series([15.0])), float)


# ── extreme_heat_days ─────────────────────────────────────────────────────────

class TestExtremeHeatDays:
    def test_counts_correctly(self):
        s = pd.Series([30.0, 35.0, 36.0, 34.9, 35.1])
        assert extreme_heat_days(s, threshold=35.0) == 3

    def test_none_above_threshold(self):
        s = pd.Series([20.0, 25.0, 30.0])
        assert extreme_heat_days(s, threshold=35.0) == 0

    def test_all_above(self):
        s = pd.Series([36.0, 37.0, 38.0])
        assert extreme_heat_days(s, threshold=35.0) == 3


# ── dry_days ──────────────────────────────────────────────────────────────────

class TestDryDays:
    def test_counts_correctly(self):
        s = pd.Series([0.0, 0.5, 2.0, 1.0, 5.0])
        assert dry_days(s, threshold=1.0) == 3

    def test_all_wet(self):
        s = pd.Series([5.0, 10.0, 3.0])
        assert dry_days(s, threshold=1.0) == 0

    def test_all_dry(self):
        s = pd.Series([0.0, 0.5, 1.0])
        assert dry_days(s, threshold=1.0) == 3


# ── cumulative_precip / mean_temp / max_temp ──────────────────────────────────

class TestScalarFeatures:
    def test_cumulative_precip(self):
        assert cumulative_precip(pd.Series([1.0, 2.0, 3.0])) == pytest.approx(6.0)

    def test_mean_temp(self):
        assert mean_temp(pd.Series([10.0, 20.0, 30.0])) == pytest.approx(20.0)

    def test_max_temp(self):
        assert max_temp(pd.Series([10.0, 35.0, 20.0])) == pytest.approx(35.0)


# ── add_precip_anomaly ────────────────────────────────────────────────────────

class TestPrecipAnomaly:
    def test_first_year_is_nan(self):
        df = pd.DataFrame({
            "county_fips": ["19001"] * 3,
            "year": [2015, 2016, 2017],
            "cumulative_precip": [300.0, 350.0, 280.0],
        })
        result = add_precip_anomaly(df)
        assert pd.isna(result.loc[result["year"] == 2015, "precip_anomaly"].iloc[0])

    def test_second_year_anomaly(self):
        df = pd.DataFrame({
            "county_fips": ["19001"] * 3,
            "year": [2015, 2016, 2017],
            "cumulative_precip": [300.0, 350.0, 280.0],
        })
        result = add_precip_anomaly(df)
        # 2016 anomaly = 350 - mean([300]) = 50
        anomaly_2016 = result.loc[result["year"] == 2016, "precip_anomaly"].iloc[0]
        assert anomaly_2016 == pytest.approx(50.0)

    def test_no_future_leakage(self):
        """The anomaly for year T must not use year T's own precipitation."""
        df = pd.DataFrame({
            "county_fips": ["19001"] * 4,
            "year": [2015, 2016, 2017, 2018],
            "cumulative_precip": [300.0, 350.0, 280.0, 400.0],
        })
        result = add_precip_anomaly(df)
        # 2018 anomaly should use mean([300, 350, 280]) = 310
        # So anomaly = 400 - 310 = 90
        anomaly_2018 = result.loc[result["year"] == 2018, "precip_anomaly"].iloc[0]
        expected = 400.0 - np.mean([300.0, 350.0, 280.0])
        assert anomaly_2018 == pytest.approx(expected, abs=0.01)


# ── compute_weather_features (integration) ────────────────────────────────────

class TestComputeWeatherFeatures:
    def test_output_shape(self):
        """One row per county-year in the output."""
        df1 = _make_daily_df("19001", "19", 2019)
        df2 = _make_daily_df("19001", "19", 2020)
        df3 = _make_daily_df("17001", "17", 2020)
        daily = pd.concat([df1, df2, df3], ignore_index=True)
        features = compute_weather_features(daily)
        assert len(features) == 3

    def test_expected_columns_present(self):
        daily = _make_daily_df()
        features = compute_weather_features(daily)
        expected = {
            "county_fips", "state_fips", "year",
            "cumulative_precip", "mean_temp", "max_temp",
            "extreme_heat_days", "dry_days", "longest_dry_spell",
            "growing_degree_days", "precip_anomaly",
        }
        assert expected.issubset(set(features.columns))

    def test_cumulative_precip_positive(self):
        daily = _make_daily_df()
        features = compute_weather_features(daily)
        assert (features["cumulative_precip"] >= 0).all()

    def test_growing_degree_days_nonnegative(self):
        daily = _make_daily_df()
        features = compute_weather_features(daily)
        assert (features["growing_degree_days"] >= 0).all()

    def test_filters_non_growing_months(self):
        """Features should only aggregate April–August data."""
        dates_full = pd.date_range("2020-01-01", "2020-12-31", freq="D")
        df_full = pd.DataFrame({
            "county_fips": "19001", "state_fips": "19",
            "year": 2020, "date": dates_full,
            "month": dates_full.month,
            "PRECTOTCORR": 5.0, "T2M": 20.0,
            "T2M_MIN": 15.0, "T2M_MAX": 25.0,
        })
        dates_gs = pd.date_range("2020-04-01", "2020-08-31", freq="D")
        df_gs = pd.DataFrame({
            "county_fips": "19001", "state_fips": "19",
            "year": 2020, "date": dates_gs,
            "month": dates_gs.month,
            "PRECTOTCORR": 5.0, "T2M": 20.0,
            "T2M_MIN": 15.0, "T2M_MAX": 25.0,
        })
        feat_full = compute_weather_features(df_full)
        feat_gs   = compute_weather_features(df_gs)
        assert feat_full["cumulative_precip"].iloc[0] == pytest.approx(
            feat_gs["cumulative_precip"].iloc[0]
        )
