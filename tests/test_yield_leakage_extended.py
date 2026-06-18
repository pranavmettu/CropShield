"""
Extended yield target leakage tests for CropShield.

These tests focus on two scenarios not covered in test_yield_targets.py:

  1. Crop-grouping independence: CORN and SOYBEAN baselines for the same
     county must be computed entirely independently.  If the groups mixed,
     the wrong baseline would corrupt both anomalies.

  2. Future-year isolation: adding an extreme future year (e.g. 9999 bu/acre
     in 2025) must not change the expected_yield or yield_anomaly values for
     earlier years.

Both rolling and trend methods are tested.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cropshield.features.yield_targets import (
    add_expected_yield_rolling,
    add_expected_yield_trend,
    add_yield_anomaly,
    clean_yield_dataframe,
)

EXTREME_YIELD = 9999.0  # clearly impossible — detects leakage instantly


def _make_yield_df(
    fips: str = "19001",
    state: str = "IOWA",
    county: str = "STORY",
    crop: str = "CORN",
    years: list[int] | None = None,
    yields: list[float] | None = None,
) -> pd.DataFrame:
    """Minimal yield DataFrame for a single county-crop sequence."""
    if years is None:
        years = [2015, 2016, 2017, 2018, 2019, 2020]
    if yields is None:
        yields = [180.0, 185.0, 175.0, 190.0, 178.0, 188.0]
    return pd.DataFrame({
        "year":        years,
        "state":       state,
        "county":      county,
        "county_fips": fips,
        "crop":        crop,
        "value":       yields,
    })


# ── Crop-grouping independence ─────────────────────────────────────────────────

class TestCropGroupingIndependence:
    """CORN and SOYBEAN baselines for the same county must be computed separately."""

    def _make_mixed_crop_df(self) -> pd.DataFrame:
        corn = _make_yield_df(crop="CORN",     yields=[180.0, 185.0, 175.0, 190.0, 178.0, 188.0])
        soy  = _make_yield_df(crop="SOYBEANS", yields=[45.0,  48.0,  42.0,  50.0,  46.0,  49.0])
        raw  = pd.concat([corn, soy], ignore_index=True)
        return clean_yield_dataframe(raw)

    def test_rolling_expected_yield_corn_uses_only_corn_history(self):
        df = self._make_mixed_crop_df()
        result = add_expected_yield_rolling(df, window=5, min_periods=3)

        corn_rows = result[result["crop"] == "CORN"]
        soy_rows  = result[result["crop"] == "SOYBEANS"]

        # Year 2020 rolling mean for CORN should be near corn yield range (150–200)
        corn_2020_expected = corn_rows.loc[corn_rows["year"] == 2020, "expected_yield"].iloc[0]
        assert 150.0 <= corn_2020_expected <= 220.0, (
            f"CORN expected_yield {corn_2020_expected:.1f} looks contaminated by soybean values"
        )

        # Year 2020 rolling mean for SOYBEANS should be near soybean yield range (35–55)
        soy_2020_expected = soy_rows.loc[soy_rows["year"] == 2020, "expected_yield"].iloc[0]
        assert 35.0 <= soy_2020_expected <= 60.0, (
            f"SOYBEAN expected_yield {soy_2020_expected:.1f} looks contaminated by corn values"
        )

    def test_rolling_corn_and_soy_expected_yields_are_different(self):
        """Sanity check: corn and soybean baselines must not be identical."""
        df = self._make_mixed_crop_df()
        result = add_expected_yield_rolling(df, window=5, min_periods=3)
        corn_exp = result.loc[result["crop"] == "CORN", "expected_yield"].dropna()
        soy_exp  = result.loc[result["crop"] == "SOYBEANS", "expected_yield"].dropna()
        assert not corn_exp.equals(soy_exp), "CORN and SOYBEAN expected yields must differ"

    def test_trend_expected_yield_corn_independent_of_soybeans(self):
        df = self._make_mixed_crop_df()
        result = add_expected_yield_trend(df, min_years=3)

        corn_2020 = result.loc[
            (result["crop"] == "CORN") & (result["year"] == 2020), "expected_yield"
        ].iloc[0]
        # Should be a reasonable extrapolation of corn yields (~170–200 range)
        assert 150.0 <= corn_2020 <= 220.0, (
            f"CORN trend expected_yield {corn_2020:.1f} appears contaminated"
        )

    def test_anomaly_sign_preserved_per_crop(self):
        """A bad corn year must not become a bad soybean year via group contamination."""
        # Make 2019 a very bad corn year but a good soybean year
        corn = _make_yield_df(crop="CORN",     yields=[180.0, 185.0, 175.0, 190.0, 100.0, 188.0])
        soy  = _make_yield_df(crop="SOYBEANS", yields=[45.0,  48.0,  42.0,  50.0,   60.0,  49.0])
        raw  = pd.concat([corn, soy], ignore_index=True)
        df   = clean_yield_dataframe(raw)
        df   = add_expected_yield_rolling(df, window=5, min_periods=3)
        df   = add_yield_anomaly(df)

        corn_2019_anomaly = df.loc[
            (df["crop"] == "CORN") & (df["year"] == 2019), "yield_anomaly"
        ].iloc[0]
        soy_2019_anomaly = df.loc[
            (df["crop"] == "SOYBEANS") & (df["year"] == 2019), "yield_anomaly"
        ].iloc[0]

        assert corn_2019_anomaly < 0, "CORN 2019 should have a negative anomaly (bad year)"
        assert soy_2019_anomaly > 0, "SOYBEAN 2019 should have a positive anomaly (good year)"


# ── Future-year isolation ──────────────────────────────────────────────────────

class TestFutureYearIsolation:
    """Adding an extreme future year must not alter earlier years' baselines."""

    def _base_yields(self) -> list[float]:
        return [180.0, 185.0, 175.0, 190.0, 178.0, 188.0]

    def _base_years(self) -> list[int]:
        return [2015, 2016, 2017, 2018, 2019, 2020]

    def test_rolling_earlier_expected_unchanged_by_future_extreme(self):
        """
        Rolling expected_yield for 2018 must be the same whether or not a
        9999 bu/acre year 2025 exists.
        """
        base_df = clean_yield_dataframe(
            _make_yield_df(years=self._base_years(), yields=self._base_yields())
        )
        extreme_df = clean_yield_dataframe(
            _make_yield_df(
                years=self._base_years() + [2025],
                yields=self._base_yields() + [EXTREME_YIELD],
            )
        )

        base_result    = add_expected_yield_rolling(base_df,    window=5, min_periods=3)
        extreme_result = add_expected_yield_rolling(extreme_df, window=5, min_periods=3)

        for yr in [2018, 2019, 2020]:
            base_val    = base_result.loc[base_result["year"] == yr, "expected_yield"].iloc[0]
            extreme_val = extreme_result.loc[extreme_result["year"] == yr, "expected_yield"].iloc[0]
            assert base_val == pytest.approx(extreme_val, abs=1e-6), (
                f"Year {yr} expected_yield changed from {base_val:.2f} to {extreme_val:.2f} "
                f"when extreme year 2025 was added — leakage detected!"
            )

    def test_trend_earlier_expected_unchanged_by_future_extreme(self):
        base_df = clean_yield_dataframe(
            _make_yield_df(years=self._base_years(), yields=self._base_yields())
        )
        extreme_df = clean_yield_dataframe(
            _make_yield_df(
                years=self._base_years() + [2025],
                yields=self._base_yields() + [EXTREME_YIELD],
            )
        )
        base_result    = add_expected_yield_trend(base_df,    min_years=3)
        extreme_result = add_expected_yield_trend(extreme_df, min_years=3)

        for yr in [2018, 2019, 2020]:
            base_val    = base_result.loc[base_result["year"] == yr, "expected_yield"].iloc[0]
            extreme_val = extreme_result.loc[extreme_result["year"] == yr, "expected_yield"].iloc[0]
            assert base_val == pytest.approx(exact=False, rel=1e-6)(extreme_val) if False else \
                   base_val == pytest.approx(extreme_val, abs=1e-6), (
                f"Trend year {yr} changed when extreme future year added — leakage!"
            )

    def test_rolling_first_row_is_nan_always(self):
        """First chronological year for a county must always have NaN expected_yield."""
        df = clean_yield_dataframe(_make_yield_df())
        result = add_expected_yield_rolling(df, window=5, min_periods=3)
        first_year_row = result[result["year"] == result["year"].min()]
        assert first_year_row["expected_yield"].isna().all(), (
            "First year cannot have an expected_yield — no prior history exists"
        )

    def test_yield_anomaly_propagates_nan_expected(self):
        """Rows with NaN expected_yield must produce NaN yield_anomaly."""
        df = clean_yield_dataframe(_make_yield_df())
        df = add_expected_yield_rolling(df, window=5, min_periods=3)
        df = add_yield_anomaly(df)
        nan_expected = df[df["expected_yield"].isna()]
        assert nan_expected["yield_anomaly"].isna().all(), (
            "yield_anomaly must be NaN when expected_yield is NaN"
        )

    def test_rolling_current_year_not_in_window(self):
        """
        The rolling window for year T must be purely [T-window .. T-1].
        Verify by checking that modifying year T's actual_yield does not
        change its own expected_yield.
        """
        df_low = clean_yield_dataframe(
            _make_yield_df(yields=[180.0, 185.0, 175.0, 190.0, 178.0, 50.0])  # 2020 low
        )
        df_high = clean_yield_dataframe(
            _make_yield_df(yields=[180.0, 185.0, 175.0, 190.0, 178.0, 999.0]) # 2020 extreme
        )
        r_low  = add_expected_yield_rolling(df_low,  window=5, min_periods=3)
        r_high = add_expected_yield_rolling(df_high, window=5, min_periods=3)

        exp_low  = r_low.loc[r_low["year"] == 2020,   "expected_yield"].iloc[0]
        exp_high = r_high.loc[r_high["year"] == 2020, "expected_yield"].iloc[0]

        assert exp_low == pytest.approx(exp_high, abs=1e-6), (
            f"expected_yield for 2020 changed ({exp_low:.2f} vs {exp_high:.2f}) "
            "when only 2020's actual_yield was modified — current year leaked into its own baseline!"
        )
