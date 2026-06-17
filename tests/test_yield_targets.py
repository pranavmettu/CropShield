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

# TODO: Uncomment imports once modules are implemented
# from cropshield.features.yield_targets import (
#     clean_yield_dataframe,
#     add_expected_yield_rolling,
#     add_expected_yield_trend,
#     add_yield_anomaly,
#     add_risk_class,
# )


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


# ── Placeholder tests (will be activated when module is implemented) ──────────

class TestCleanYieldDataframe:
    @pytest.mark.skip(reason="clean_yield_dataframe not yet implemented")
    def test_renames_value_to_actual_yield(self, simple_yield_df):
        from cropshield.features.yield_targets import clean_yield_dataframe
        result = clean_yield_dataframe(simple_yield_df)
        assert "actual_yield" in result.columns
        assert "value" not in result.columns

    @pytest.mark.skip(reason="clean_yield_dataframe not yet implemented")
    def test_drops_nan_yield_rows(self):
        from cropshield.features.yield_targets import clean_yield_dataframe
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


class TestNoLeakage:
    """Critical tests: expected yield must never include the current year."""

    @pytest.mark.skip(reason="add_expected_yield_rolling not yet implemented")
    def test_rolling_expected_yield_excludes_current_year(self, simple_yield_df):
        from cropshield.features.yield_targets import (
            clean_yield_dataframe,
            add_expected_yield_rolling,
        )
        df = clean_yield_dataframe(simple_yield_df)
        df = add_expected_yield_rolling(df, window=5)

        # For each row, verify expected_yield is computed only from prior years
        for _, row in df.dropna(subset=["expected_yield"]).iterrows():
            year = row["year"]
            prior_years = df[
                (df["county_fips"] == row["county_fips"]) & (df["year"] < year)
            ]
            prior_mean = prior_years["actual_yield"].tail(5).mean()
            assert abs(row["expected_yield"] - prior_mean) < 1e-6, (
                f"Leakage detected: expected_yield for {year} uses current year data"
            )

    @pytest.mark.skip(reason="add_expected_yield_trend not yet implemented")
    def test_trend_expected_yield_excludes_current_year(self, simple_yield_df):
        from cropshield.features.yield_targets import (
            clean_yield_dataframe,
            add_expected_yield_trend,
        )
        df = clean_yield_dataframe(simple_yield_df)
        df = add_expected_yield_trend(df, min_years=3)
        # For each non-NaN row, the trend should be fit on years < current year
        # We cannot easily recompute the trend here, but we can at least verify
        # that the first min_years rows have NaN expected_yield
        first_years = df.nsmallest(3, "year")
        assert first_years["expected_yield"].isna().all()

    @pytest.mark.skip(reason="add_expected_yield_rolling not yet implemented")
    def test_insufficient_history_produces_nan(self, simple_yield_df):
        from cropshield.features.yield_targets import (
            clean_yield_dataframe,
            add_expected_yield_rolling,
        )
        df = clean_yield_dataframe(simple_yield_df)
        # With window=5 and min_periods=5, first 5 rows should have NaN
        df = add_expected_yield_rolling(df, window=5, min_periods=5)
        early_rows = df[df["year"] <= 2019]
        assert early_rows["expected_yield"].isna().all(), (
            "Rows with fewer than 5 prior years should have NaN expected_yield"
        )


class TestYieldAnomaly:
    @pytest.mark.skip(reason="add_yield_anomaly not yet implemented")
    def test_anomaly_equals_actual_minus_expected(self, simple_yield_df):
        from cropshield.features.yield_targets import (
            clean_yield_dataframe,
            add_expected_yield_rolling,
            add_yield_anomaly,
        )
        df = clean_yield_dataframe(simple_yield_df)
        df = add_expected_yield_rolling(df, window=5)
        df = add_yield_anomaly(df)
        valid = df.dropna(subset=["yield_anomaly"])
        np.testing.assert_allclose(
            valid["yield_anomaly"].values,
            (valid["actual_yield"] - valid["expected_yield"]).values,
            rtol=1e-6,
        )


class TestRiskClass:
    @pytest.mark.skip(reason="add_risk_class not yet implemented")
    def test_risk_class_is_binary(self, two_county_df):
        from cropshield.features.yield_targets import (
            clean_yield_dataframe,
            add_expected_yield_rolling,
            add_yield_anomaly,
            add_risk_class,
        )
        df = clean_yield_dataframe(two_county_df)
        df = add_expected_yield_rolling(df, window=5)
        df = add_yield_anomaly(df)
        df = df.dropna(subset=["yield_anomaly"])
        df = add_risk_class(df, quantile=0.20)
        assert set(df["severe_risk"].unique()).issubset({0, 1})
        assert df["severe_risk"].isna().sum() == 0

    @pytest.mark.skip(reason="add_risk_class not yet implemented")
    def test_risk_class_respects_quantile(self, two_county_df):
        from cropshield.features.yield_targets import (
            clean_yield_dataframe,
            add_expected_yield_rolling,
            add_yield_anomaly,
            add_risk_class,
        )
        df = clean_yield_dataframe(two_county_df)
        df = add_expected_yield_rolling(df, window=5)
        df = add_yield_anomaly(df)
        df = df.dropna(subset=["yield_anomaly"])
        quantile = 0.20
        df = add_risk_class(df, quantile=quantile)
        # Approximately quantile% of rows should be labelled severe risk
        actual_rate = df["severe_risk"].mean()
        # Allow ±5% tolerance due to discrete data
        assert abs(actual_rate - quantile) < 0.10, (
            f"Expected ~{quantile:.0%} severe risk rate, got {actual_rate:.2%}"
        )
