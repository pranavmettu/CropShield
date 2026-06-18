"""
Incomplete current-year weather guard tests.
"""

from __future__ import annotations

import pandas as pd
import pytest

from cropshield.features.weather_features import (
    FULL_GROWING_SEASON_DAYS,
    compute_weather_features,
    filter_incomplete_current_year,
)


def _daily(fips="19001", year=2020, n_days=153):
    dates = pd.date_range(f"{year}-04-01", periods=n_days, freq="D")
    return pd.DataFrame({
        "county_fips": fips,
        "state_fips": fips[:2],
        "year": year,
        "date": dates,
        "month": dates.month,
        "PRECTOTCORR": 5.0,
        "T2M": 20.0,
        "T2M_MIN": 15.0,
        "T2M_MAX": 25.0,
    })


class TestPartialYearGuard:
    def test_full_season_not_partial(self):
        features = compute_weather_features(_daily(n_days=FULL_GROWING_SEASON_DAYS))
        assert not features["is_partial_year"].iloc[0]

    def test_short_season_is_partial(self):
        features = compute_weather_features(_daily(n_days=60))
        assert features["is_partial_year"].iloc[0]

    def test_default_excludes_partial_current_year(self):
        current = 2026
        features = pd.DataFrame([
            {"county_fips": "19001", "year": current, "obs_days": 60, "is_partial_year": True},
            {"county_fips": "19001", "year": 2020, "obs_days": 60, "is_partial_year": True},
        ])
        filtered = filter_incomplete_current_year(
            features, allow_partial_year=False, current_year=current,
        )
        assert len(filtered) == 1
        assert filtered["year"].iloc[0] == 2020

    def test_allow_partial_year_keeps_rows(self):
        current = 2026
        features = pd.DataFrame([
            {"county_fips": "19001", "year": current, "obs_days": 60, "is_partial_year": True},
        ])
        kept = filter_incomplete_current_year(
            features, allow_partial_year=True, current_year=current,
        )
        assert len(kept) == 1
        assert kept["is_partial_year"].iloc[0]

    def test_complete_current_year_kept(self):
        current = 2026
        features = pd.DataFrame([
            {"county_fips": "19001", "year": current,
             "obs_days": FULL_GROWING_SEASON_DAYS, "is_partial_year": False},
        ])
        filtered = filter_incomplete_current_year(
            features, allow_partial_year=False, current_year=current,
        )
        assert len(filtered) == 1
