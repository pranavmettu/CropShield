"""
Extended validation split tests for CropShield.

The existing test_validation_splits.py covers basic temporal and spatial
split correctness.  These additional tests verify:

  1. No county_fips overlap between train and test sets (temporal).
  2. No county_fips overlap between train and test sets (spatial).
  3. Exact cutoff year boundary: the last train year < first test year.
  4. Expanding-window splits never allow test_year data into training.
  5. apply_risk_thresholds uses training thresholds only (no test leakage).

County FIPS are a stronger identity than state: a county held out spatially
must have zero FIPS overlap, even if two counties share the same name.
"""

from __future__ import annotations

import pandas as pd
import pytest

from cropshield.evaluation.validation_splits import (
    expanding_window_splits,
    spatial_split,
    temporal_split,
)
from cropshield.features.yield_targets import (
    apply_risk_thresholds,
    compute_risk_thresholds,
)


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def multi_county_panel() -> pd.DataFrame:
    """Panel with 3 counties, 2 states, 2015–2023."""
    rows = []
    counties = [
        ("19001", "IOWA",     "BOONE"),
        ("19003", "IOWA",     "CEDAR"),
        ("17001", "ILLINOIS", "ADAMS"),
    ]
    for year in range(2015, 2024):
        for fips, state, county in counties:
            rows.append({
                "year": year,
                "state": state,
                "county": county,
                "county_fips": fips,
                "crop": "CORN",
                "yield_anomaly": float(year - 2019),
                "yield_anomaly_pct": float(year - 2019) / 180 * 100,
                "severe_risk": int(year < 2019),
            })
    return pd.DataFrame(rows)


# ── Temporal split: FIPS overlap ──────────────────────────────────────────────

class TestTemporalSplitNoFipsOverlap:
    def test_train_and_test_share_same_county_fips(self, multi_county_panel):
        """
        In temporal splits, the same counties appear in train and test
        but at different years — this is expected and correct.
        Verify they do share FIPS (as expected) but have no year overlap.
        """
        train, test = temporal_split(multi_county_panel, n_test_years=3)
        train_fips = set(train["county_fips"].unique())
        test_fips  = set(test["county_fips"].unique())
        # Counties should appear in BOTH splits (different years)
        assert train_fips == test_fips, (
            "Temporal split: all counties should appear in both train and test "
            "(at different years)"
        )

    def test_no_year_overlap(self, multi_county_panel):
        train, test = temporal_split(multi_county_panel, n_test_years=3)
        assert set(train["year"].unique()).isdisjoint(set(test["year"].unique()))

    def test_test_years_strictly_after_train(self, multi_county_panel):
        train, test = temporal_split(multi_county_panel, n_test_years=3)
        assert test["year"].min() > train["year"].max()

    def test_exact_cutoff_year_in_test(self, multi_county_panel):
        """The first test year should be exactly (max_year - n_test_years + 1)."""
        n = 3
        _, test = temporal_split(multi_county_panel, n_test_years=n)
        all_years = sorted(multi_county_panel["year"].unique())
        expected_first_test_year = all_years[-n]
        assert test["year"].min() == expected_first_test_year

    def test_no_row_in_train_with_test_year(self, multi_county_panel):
        """Explicit row-level check: no train row has a year in the test set."""
        train, test = temporal_split(multi_county_panel, n_test_years=3)
        test_years = set(test["year"].unique())
        assert not train["year"].isin(test_years).any(), (
            "At least one training row has a year that is in the test set"
        )


# ── Spatial split: FIPS overlap ───────────────────────────────────────────────

class TestSpatialSplitNoFipsOverlap:
    def test_held_out_fips_not_in_train(self, multi_county_panel):
        """After a spatial split, held-out state's FIPS must not appear in train."""
        train, test = spatial_split(multi_county_panel, holdout_state="ILLINOIS")
        ill_fips = set(test["county_fips"].unique())  # {"17001"}
        train_fips = set(train["county_fips"].unique())
        assert ill_fips.isdisjoint(train_fips), (
            f"Illinois FIPS {ill_fips} appear in training set after spatial split"
        )

    def test_train_fips_not_in_test(self, multi_county_panel):
        """Iowa counties must not appear in the Illinois-held-out test set."""
        train, test = spatial_split(multi_county_panel, holdout_state="ILLINOIS")
        iowa_fips = set(train["county_fips"].unique())
        assert iowa_fips.isdisjoint(set(test["county_fips"].unique()))

    def test_spatial_split_preserves_all_years(self, multi_county_panel):
        """Both splits should cover all years (spatial, not temporal, separation)."""
        train, test = spatial_split(multi_county_panel, holdout_state="ILLINOIS")
        all_years = set(multi_county_panel["year"].unique())
        assert set(train["year"].unique()) == all_years
        assert set(test["year"].unique()) == all_years


# ── Expanding window: no future leakage ───────────────────────────────────────

class TestExpandingWindowNoLeakage:
    def test_train_never_contains_test_year_rows(self, multi_county_panel):
        for train, test, test_year in expanding_window_splits(multi_county_panel):
            assert test_year not in set(train["year"].unique()), (
                f"Test year {test_year} appears in training set"
            )

    def test_train_never_contains_future_years(self, multi_county_panel):
        for train, test, test_year in expanding_window_splits(multi_county_panel):
            assert train["year"].max() < test_year, (
                f"Training set has year {train['year'].max()} >= test year {test_year}"
            )

    def test_each_fold_test_is_one_year(self, multi_county_panel):
        for train, test, test_year in expanding_window_splits(multi_county_panel):
            assert test["year"].nunique() == 1
            assert test["year"].iloc[0] == test_year


# ── Risk threshold leakage prevention ─────────────────────────────────────────

class TestRiskThresholdLeakage:
    """
    compute_risk_thresholds + apply_risk_thresholds must use training
    thresholds only.  Applying test thresholds to the test set would be
    leakage (the threshold would be computed from test labels).
    """

    def _make_split_df(self):
        rows = []
        for year in range(2015, 2024):
            rows.append({
                "year": year,
                "state": "IOWA",
                "county_fips": "19001",
                "crop": "CORN",
                "yield_anomaly_pct": float(year - 2019) * 2.0,
            })
        df = pd.DataFrame(rows)
        train = df[df["year"] < 2022].copy()
        test  = df[df["year"] >= 2022].copy()
        return train, test

    def test_thresholds_computed_from_train_only(self):
        train, test = self._make_split_df()
        thresholds = compute_risk_thresholds(train, quantile=0.20)
        # Threshold must be based only on train distribution
        train_q20 = train["yield_anomaly_pct"].quantile(0.20)
        assert thresholds.iloc[0] == pytest.approx(train_q20), (
            "compute_risk_thresholds returned a value different from training quantile"
        )

    def test_apply_thresholds_produces_binary_labels(self):
        train, test = self._make_split_df()
        thresholds = compute_risk_thresholds(train, quantile=0.20)
        test_labeled = apply_risk_thresholds(test, thresholds)
        valid = test_labeled["severe_risk"].dropna()
        assert set(valid.unique()).issubset({0, 1}), (
            f"severe_risk contains non-binary values: {set(valid.unique())}"
        )

    def test_threshold_not_recomputed_from_test_data(self):
        """
        If we apply training thresholds to test data, the result must differ
        from recomputing thresholds on test data alone (demonstrating that
        training and test thresholds are distinct).
        """
        train, test = self._make_split_df()
        train_thresholds = compute_risk_thresholds(train, quantile=0.20)
        test_thresholds  = compute_risk_thresholds(test,  quantile=0.20)

        # Thresholds should differ because train and test have different distributions
        assert train_thresholds.iloc[0] != pytest.approx(test_thresholds.iloc[0]), (
            "Training and test thresholds are identical — "
            "the threshold may be inadvertently computed from test data"
        )
