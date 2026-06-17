"""
Tests for temporal and spatial validation splits
(src/cropshield/evaluation/validation_splits.py).

Validation split correctness is essential: a bug here could silently allow
future data into training, invalidating all reported metrics.
"""

from __future__ import annotations

import pandas as pd
import pytest

from cropshield.evaluation.validation_splits import (
    expanding_window_splits,
    spatial_split,
    temporal_split,
)


@pytest.fixture
def panel_df() -> pd.DataFrame:
    """Minimal panel with two counties across 2015–2023."""
    rows = []
    for year in range(2015, 2024):
        for fips, state in [("19001", "IOWA"), ("17001", "ILLINOIS")]:
            rows.append({
                "year": year,
                "state": state,
                "county_fips": fips,
                "crop": "CORN",
                "yield_anomaly": float(year - 2019),
            })
    return pd.DataFrame(rows)


class TestTemporalSplit:
    def test_train_and_test_are_disjoint_by_year(self, panel_df):
        train, test = temporal_split(panel_df, n_test_years=3)
        train_years = set(train["year"].unique())
        test_years = set(test["year"].unique())
        assert train_years.isdisjoint(test_years), "Train and test years must not overlap"

    def test_test_has_correct_number_of_years(self, panel_df):
        n = 3
        _, test = temporal_split(panel_df, n_test_years=n)
        assert test["year"].nunique() == n

    def test_test_years_are_the_latest(self, panel_df):
        train, test = temporal_split(panel_df, n_test_years=3)
        assert test["year"].min() > train["year"].max()

    def test_no_rows_lost(self, panel_df):
        train, test = temporal_split(panel_df, n_test_years=3)
        assert len(train) + len(test) == len(panel_df)

    def test_raises_when_not_enough_years(self, panel_df):
        with pytest.raises(ValueError):
            temporal_split(panel_df, n_test_years=100)


class TestSpatialSplit:
    def test_holdout_state_in_test_only(self, panel_df):
        train, test = spatial_split(panel_df, holdout_state="ILLINOIS")
        assert "ILLINOIS" not in train["state"].values
        assert set(test["state"].unique()) == {"ILLINOIS"}

    def test_no_rows_lost(self, panel_df):
        train, test = spatial_split(panel_df, holdout_state="IOWA")
        assert len(train) + len(test) == len(panel_df)

    def test_raises_for_unknown_state(self, panel_df):
        with pytest.raises(ValueError):
            spatial_split(panel_df, holdout_state="CALIFORNIA")


class TestExpandingWindowSplits:
    def test_generates_correct_number_of_folds(self, panel_df):
        years = sorted(panel_df["year"].unique())
        min_train = 5
        expected_folds = len(years) - min_train
        folds = list(expanding_window_splits(panel_df, min_train_years=min_train))
        assert len(folds) == expected_folds

    def test_train_never_includes_test_year(self, panel_df):
        for train, test, test_year in expanding_window_splits(panel_df):
            assert test_year not in train["year"].values

    def test_training_set_grows_monotonically(self, panel_df):
        sizes = [len(train) for train, _, _ in expanding_window_splits(panel_df)]
        assert all(sizes[i] < sizes[i + 1] for i in range(len(sizes) - 1))
