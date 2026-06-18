"""
Weather feature leakage tests for CropShield.

Scientific validity requirement: features computed for a given date cutoff
must not incorporate observations that fall after that cutoff.  This is
important for two scenarios:

  1. Mid-season predictions: features as of June 30 must only use April–June
     weather, even if July–August data is present in the input.

  2. Retrospective reproducibility: re-computing features with a different
     end date must give a different (not the same) result unless the cutoff
     is beyond the full growing season.

Tests use synthetic daily weather with a known "extreme values after cutoff"
pattern: the future dates have precipitation = 9999 mm/day.  If the cutoff
is respected, cumulative_precip must not include those extreme values.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cropshield.features.weather_features import (
    compute_weather_features,
    filter_growing_season,
)

EXTREME_PRECIP = 9999.0   # clearly impossible value to detect leakage


def _make_full_season(
    fips: str = "19001",
    year: int = 2020,
    precip_normal: float = 5.0,
    precip_extreme: float = EXTREME_PRECIP,
    extreme_from_month: int = 7,   # July onwards has extreme values
) -> pd.DataFrame:
    """Synthetic full growing season (Apr–Aug) with extreme values from a given month."""
    dates = pd.date_range(f"{year}-04-01", f"{year}-08-31", freq="D")
    precip = [
        precip_extreme if d.month >= extreme_from_month else precip_normal
        for d in dates
    ]
    return pd.DataFrame({
        "county_fips":  fips,
        "state_fips":   fips[:2],
        "year":         year,
        "date":         dates,
        "month":        [d.month for d in dates],
        "PRECTOTCORR":  precip,
        "T2M":          20.0,
        "T2M_MIN":      15.0,
        "T2M_MAX":      25.0,
        "lat":          42.0,
        "lon":          -93.0,
    })


# ── filter_growing_season cutoff_date ─────────────────────────────────────────

class TestFilterGrowingSeasonCutoff:
    def test_no_cutoff_keeps_full_season(self):
        df = _make_full_season()
        result = filter_growing_season(df)
        # April (30) + May (31) + June (30) + July (31) + August (31) = 153
        assert len(result) == 153

    def test_may31_cutoff_excludes_june_onwards(self):
        df = _make_full_season()
        result = filter_growing_season(df, cutoff_date="2020-05-31")
        # April (30) + May (31) = 61
        assert len(result) == 61
        assert result["date"].dt.month.max() == 5

    def test_june30_cutoff(self):
        df = _make_full_season()
        result = filter_growing_season(df, cutoff_date="2020-06-30")
        # April (30) + May (31) + June (30) = 91
        assert len(result) == 91

    def test_july31_cutoff(self):
        df = _make_full_season()
        result = filter_growing_season(df, cutoff_date="2020-07-31")
        # April + May + June + July = 122
        assert len(result) == 122

    def test_cutoff_on_boundary_is_inclusive(self):
        """The cutoff date itself must be included (≤, not <)."""
        df = _make_full_season()
        result = filter_growing_season(df, cutoff_date="2020-04-01")
        assert len(result) == 1
        assert result["date"].dt.date.iloc[0].isoformat() == "2020-04-01"

    def test_cutoff_before_growing_season_returns_empty(self):
        df = _make_full_season()
        result = filter_growing_season(df, cutoff_date="2020-03-31")
        assert len(result) == 0


# ── compute_weather_features with cutoff_date ─────────────────────────────────

class TestWeatherFeaturesCutoffLeakage:
    """
    Verify that extreme values on future dates do not affect features
    computed with a cutoff date.
    """

    def test_may31_cutoff_excludes_extreme_july_precip(self):
        """
        Daily data has normal precip in April–June and 9999 mm/day in July–August.
        Features computed with a May 31 cutoff must NOT include July–August values.
        """
        df = _make_full_season(extreme_from_month=7)  # July onwards = extreme
        # Compute with May 31 cutoff
        features_may = compute_weather_features(df, cutoff_date="2020-05-31")
        assert len(features_may) == 1
        assert features_may["cumulative_precip"].iloc[0] < EXTREME_PRECIP, (
            "Extreme July-August precipitation leaked into May 31 cumulative precip"
        )

    def test_june30_cutoff_excludes_extreme_july_precip(self):
        df = _make_full_season(extreme_from_month=7)
        features_june = compute_weather_features(df, cutoff_date="2020-06-30")
        assert features_june["cumulative_precip"].iloc[0] < EXTREME_PRECIP

    def test_altering_future_dates_does_not_change_cutoff_features(self):
        """
        Changing precipitation values on dates after the cutoff must NOT affect
        features computed with that cutoff.

        This is the core leakage test: if this fails, future data is leaking
        into the feature set.
        """
        base_df = _make_full_season(precip_normal=5.0, extreme_from_month=7)

        # Compute features through June 30 on original data
        features_normal = compute_weather_features(base_df, cutoff_date="2020-06-30")

        # Make July–August precipiation absurdly extreme
        extreme_df = base_df.copy()
        extreme_df.loc[extreme_df["date"].dt.month >= 7, "PRECTOTCORR"] = EXTREME_PRECIP

        # Features through June 30 must be identical regardless of future values
        features_extreme = compute_weather_features(extreme_df, cutoff_date="2020-06-30")

        pd.testing.assert_frame_equal(
            features_normal.reset_index(drop=True),
            features_extreme.reset_index(drop=True),
            check_exact=False,
            atol=1e-6,
            obj="Weather features with cutoff_date='2020-06-30'",
        )

    def test_no_cutoff_includes_full_season(self):
        """Without cutoff, full season including extreme July values is aggregated."""
        df = _make_full_season(precip_normal=5.0, extreme_from_month=7)
        features_full = compute_weather_features(df)
        # Some days in July-August have EXTREME_PRECIP, so cumulative > anything possible
        assert features_full["cumulative_precip"].iloc[0] > 1000, (
            "Full-season features should include July-August extreme values"
        )

    def test_earlier_cutoff_gives_lower_cumulative_precip(self):
        """Features through May 31 must have lower cumulative precip than through August."""
        df = _make_full_season(precip_normal=5.0, extreme_from_month=100)  # no extreme
        f_may  = compute_weather_features(df, cutoff_date="2020-05-31")
        f_full = compute_weather_features(df)
        assert f_may["cumulative_precip"].iloc[0] < f_full["cumulative_precip"].iloc[0]

    def test_extreme_heat_days_in_future_excluded_by_cutoff(self):
        """
        Extreme heat in July–August must not appear in June 30 cutoff features.
        """
        dates = pd.date_range("2020-04-01", "2020-08-31", freq="D")
        tmax = [20.0 if d.month < 7 else 45.0 for d in dates]  # 45°C in July-Aug
        df = pd.DataFrame({
            "county_fips": "19001", "state_fips": "19",
            "year": 2020, "date": dates, "month": [d.month for d in dates],
            "PRECTOTCORR": 5.0, "T2M": 20.0, "T2M_MIN": 15.0, "T2M_MAX": tmax,
        })
        features = compute_weather_features(df, cutoff_date="2020-06-30")
        assert features["extreme_heat_days"].iloc[0] == 0, (
            "Extreme heat in July should be invisible through June 30 cutoff"
        )
