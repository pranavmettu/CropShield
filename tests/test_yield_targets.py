"""
Tests for yield target engineering (src/cropshield/features/yield_targets.py).

Critical invariants tested
---------------------------
1. No current-year data leaks into expected_yield computation.
2. Counties with insufficient history receive NaN expected_yield.
3. yield_anomaly = actual_yield - expected_yield (to floating-point precision).
4. severe_risk is binary (0 or 1) with no NaN values for rows that have anomaly.
5. Cleaned DataFrame has the correct schema and no NaN actual_yield rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cropshield.features.yield_targets import (
    add_expected_yield_rolling,
    add_expected_yield_trend,
    add_risk_class,
    add_yield_anomaly,
    clean_yield_dataframe,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def simple_yield_df() -> pd.DataFrame:
    """Minimal single-county, single-crop yield series for testing."""
    return pd.DataFrame({
        "year":        [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022],
        "state":       ["IOWA"] * 8,
        "county_fips": ["19001"] * 8,
        "crop":        ["CORN"] * 8,
        "value":       [170.0, 175.0, 168.0, 180.0, 172.0, 177.0, 165.0, 183.0],
    })


@pytest.fixture
def two_county_df() -> pd.DataFrame:
    """Two counties with independent yield histories."""
    iowa = pd.DataFrame({
        "year":        list(range(2015, 2023)),
        "state":       ["IOWA"] * 8,
        "county_fips": ["19001"] * 8,
        "crop":        ["CORN"] * 8,
        "value":       [170.0, 175.0, 168.0, 180.0, 172.0, 177.0, 165.0, 183.0],
    })
    illinois = pd.DataFrame({
        "year":        list(range(2015, 2023)),
        "state":       ["ILLINOIS"] * 8,
        "county_fips": ["17001"] * 8,
        "crop":        ["CORN"] * 8,
        "value":       [165.0, 160.0, 170.0, 155.0, 168.0, 162.0, 173.0, 158.0],
    })
    return pd.concat([iowa, illinois], ignore_index=True)


# ── clean_yield_dataframe ─────────────────────────────────────────────────────

class TestCleanYieldDataframe:
    def test_renames_value_to_actual_yield(self, simple_yield_df):
        result = clean_yield_dataframe(simple_yield_df)
        assert "actual_yield" in result.columns
        assert "value" not in result.columns

    def test_drops_nan_yield_rows(self):
        df = pd.DataFrame({
            "year": [2015, 2016, 2017],
            "state": ["IOWA"] * 3,
            "county_fips": ["19001"] * 3,
            "crop": ["CORN"] * 3,
            "value": [170.0, np.nan, 168.0],
        })
        result = clean_yield_dataframe(df)
        assert result["actual_yield"].isna().sum() == 0
        assert len(result) == 2

    def test_sorted_by_group_and_year(self, two_county_df):
        # Shuffle the input
        shuffled = two_county_df.sample(frac=1, random_state=42)
        result = clean_yield_dataframe(shuffled)
        # Within each county, years should be ascending
        for _, grp in result.groupby(["state", "county_fips", "crop"]):
            assert grp["year"].is_monotonic_increasing

    def test_raises_on_missing_columns(self):
        with pytest.raises(KeyError):
            clean_yield_dataframe(pd.DataFrame({"year": [2015], "value": [100.0]}))

    def test_actual_yield_is_float(self, simple_yield_df):
        result = clean_yield_dataframe(simple_yield_df)
        assert result["actual_yield"].dtype == float


# ── Leakage prevention ────────────────────────────────────────────────────────

class TestNoLeakage:
    """Critical tests: expected yield must never include the current year."""

    def test_rolling_expected_yield_excludes_current_year(self, simple_yield_df):
        df = clean_yield_dataframe(simple_yield_df)
        df = add_expected_yield_rolling(df, window=5)

        # For each non-NaN row, manually verify expected_yield equals the
        # rolling mean of prior years only
        for _, row in df.dropna(subset=["expected_yield"]).iterrows():
            year = int(row["year"])
            prior = df[
                (df["county_fips"] == row["county_fips"]) & (df["year"] < year)
            ]
            prior_mean = prior["actual_yield"].tail(5).mean()
            assert abs(row["expected_yield"] - prior_mean) < 1e-6, (
                f"Leakage detected at year {year}: "
                f"expected_yield={row['expected_yield']:.4f} != prior_mean={prior_mean:.4f}"
            )

    def test_trend_expected_yield_excludes_current_year(self, simple_yield_df):
        df = clean_yield_dataframe(simple_yield_df)
        df = add_expected_yield_trend(df, min_years=3)
        # The first min_years rows must all be NaN (no enough prior history)
        first_years = df.nsmallest(3, "year")
        assert first_years["expected_yield"].isna().all(), (
            "First 3 years should have NaN expected_yield (fewer than min_years=3 prior rows)"
        )

    def test_insufficient_history_produces_nan(self, simple_yield_df):
        df = clean_yield_dataframe(simple_yield_df)
        # With window=5 and min_periods=5, need 5 prior rows → first 5 years are NaN
        df = add_expected_yield_rolling(df, window=5, min_periods=5)
        early_rows = df[df["year"] <= 2019]
        assert early_rows["expected_yield"].isna().all(), (
            "Rows with fewer than 5 prior years should have NaN expected_yield"
        )

    def test_rolling_first_row_always_nan(self, simple_yield_df):
        """The very first year of any county must always be NaN (no prior data)."""
        df = clean_yield_dataframe(simple_yield_df)
        df = add_expected_yield_rolling(df, window=5, min_periods=1)
        first_year = df["year"].min()
        assert df[df["year"] == first_year]["expected_yield"].isna().all()


# ── yield anomaly ─────────────────────────────────────────────────────────────

class TestYieldAnomaly:
    def test_anomaly_equals_actual_minus_expected(self, simple_yield_df):
        df = clean_yield_dataframe(simple_yield_df)
        df = add_expected_yield_rolling(df, window=5)
        df = add_yield_anomaly(df)
        valid = df.dropna(subset=["yield_anomaly"])
        np.testing.assert_allclose(
            valid["yield_anomaly"].values,
            (valid["actual_yield"] - valid["expected_yield"]).values,
            rtol=1e-6,
        )

    def test_pct_anomaly_formula(self, simple_yield_df):
        df = clean_yield_dataframe(simple_yield_df)
        df = add_expected_yield_rolling(df, window=5)
        df = add_yield_anomaly(df)
        valid = df.dropna(subset=["yield_anomaly_pct"])
        expected_pct = (valid["yield_anomaly"] / valid["expected_yield"]) * 100.0
        np.testing.assert_allclose(valid["yield_anomaly_pct"].values,
                                   expected_pct.values, rtol=1e-6)

    def test_raises_on_missing_column(self, simple_yield_df):
        df = clean_yield_dataframe(simple_yield_df)
        # expected_yield column not added yet
        with pytest.raises(KeyError):
            add_yield_anomaly(df)

    def test_nan_anomaly_when_expected_is_nan(self, simple_yield_df):
        df = clean_yield_dataframe(simple_yield_df)
        df = add_expected_yield_rolling(df, window=5, min_periods=5)
        df = add_yield_anomaly(df)
        # Rows where expected_yield is NaN should have NaN anomaly too
        nan_mask = df["expected_yield"].isna()
        assert df.loc[nan_mask, "yield_anomaly"].isna().all()


# ── risk class ────────────────────────────────────────────────────────────────

class TestRiskClass:
    def test_risk_class_is_binary(self, two_county_df):
        df = clean_yield_dataframe(two_county_df)
        df = add_expected_yield_rolling(df, window=5)
        df = add_yield_anomaly(df)
        df = df.dropna(subset=["yield_anomaly"])
        df = add_risk_class(df, quantile=0.20)
        valid = df.dropna(subset=["severe_risk_descriptive"])
        assert set(valid["severe_risk_descriptive"].unique()).issubset({0, 1})

    def test_risk_class_no_nan_for_valid_rows(self, two_county_df):
        df = clean_yield_dataframe(two_county_df)
        df = add_expected_yield_rolling(df, window=5)
        df = add_yield_anomaly(df)
        df = df.dropna(subset=["yield_anomaly"])
        df = add_risk_class(df, quantile=0.20)
        assert df["severe_risk_descriptive"].isna().sum() == 0

    def test_risk_class_respects_quantile(self, two_county_df):
        df = clean_yield_dataframe(two_county_df)
        df = add_expected_yield_rolling(df, window=5)
        df = add_yield_anomaly(df)
        df = df.dropna(subset=["yield_anomaly"])
        quantile = 0.20
        df = add_risk_class(df, quantile=quantile)
        actual_rate = df["severe_risk_descriptive"].mean()
        # Allow ±10% tolerance due to discrete data and per-county computation
        assert abs(actual_rate - quantile) < 0.10, (
            f"Expected ~{quantile:.0%} severe risk rate, got {actual_rate:.2%}"
        )

    def test_risk_class_raises_on_missing_column(self, simple_yield_df):
        df = clean_yield_dataframe(simple_yield_df)
        with pytest.raises(KeyError):
            add_risk_class(df)
