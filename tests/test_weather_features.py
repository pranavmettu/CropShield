"""
Tests for weather feature engineering (src/cropshield/features/weather_features.py).

Tests use small, hand-crafted DataFrames to verify correctness of each
individual feature function before they are wired into the full pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# These tests are written ahead of implementation and will be unskipped as
# each function is built.


class TestLongestDrySpell:
    """Tests for the longest_dry_spell() function."""

    @pytest.mark.skip(reason="longest_dry_spell not yet implemented")
    def test_basic_dry_spell(self):
        from cropshield.features.weather_features import longest_dry_spell
        s = pd.Series([0.0, 0.5, 0.0, 0.0, 5.0, 0.0])
        assert longest_dry_spell(s) == 2

    @pytest.mark.skip(reason="longest_dry_spell not yet implemented")
    def test_no_dry_days(self):
        from cropshield.features.weather_features import longest_dry_spell
        s = pd.Series([2.0, 5.0, 3.0, 10.0])
        assert longest_dry_spell(s) == 0

    @pytest.mark.skip(reason="longest_dry_spell not yet implemented")
    def test_all_dry_days(self):
        from cropshield.features.weather_features import longest_dry_spell
        s = pd.Series([0.0, 0.0, 0.5, 0.3, 0.0])
        assert longest_dry_spell(s) == 5

    @pytest.mark.skip(reason="longest_dry_spell not yet implemented")
    def test_single_day_series(self):
        from cropshield.features.weather_features import longest_dry_spell
        assert longest_dry_spell(pd.Series([0.0])) == 1
        assert longest_dry_spell(pd.Series([5.0])) == 0

    @pytest.mark.skip(reason="longest_dry_spell not yet implemented")
    def test_threshold_boundary(self):
        from cropshield.features.weather_features import longest_dry_spell
        # Exactly at threshold should count as a dry day
        s = pd.Series([1.0, 1.0, 2.0])
        assert longest_dry_spell(s, threshold=1.0) == 2

    @pytest.mark.skip(reason="longest_dry_spell not yet implemented")
    def test_multiple_spells_returns_longest(self):
        from cropshield.features.weather_features import longest_dry_spell
        # Two spells: length 2 and length 3
        s = pd.Series([0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 5.0])
        assert longest_dry_spell(s) == 3


class TestGrowingDegreeDays:
    """Tests for the growing_degree_days() function."""

    @pytest.mark.skip(reason="growing_degree_days not yet implemented")
    def test_basic_gdd(self):
        from cropshield.features.weather_features import growing_degree_days
        s = pd.Series([8.0, 12.0, 15.0, 9.0])
        result = growing_degree_days(s, base_temp=10.0)
        # max(0, 8-10)=0, max(0, 12-10)=2, max(0, 15-10)=5, max(0, 9-10)=0 → 7
        assert result == pytest.approx(7.0)

    @pytest.mark.skip(reason="growing_degree_days not yet implemented")
    def test_all_below_base(self):
        from cropshield.features.weather_features import growing_degree_days
        s = pd.Series([5.0, 6.0, 7.0])
        result = growing_degree_days(s, base_temp=10.0)
        assert result == 0.0

    @pytest.mark.skip(reason="growing_degree_days not yet implemented")
    def test_zero_base(self):
        from cropshield.features.weather_features import growing_degree_days
        s = pd.Series([10.0, 15.0, 20.0])
        result = growing_degree_days(s, base_temp=0.0)
        assert result == pytest.approx(45.0)

    @pytest.mark.skip(reason="growing_degree_days not yet implemented")
    def test_empty_series(self):
        from cropshield.features.weather_features import growing_degree_days
        result = growing_degree_days(pd.Series([], dtype=float))
        assert result == 0.0

    @pytest.mark.skip(reason="growing_degree_days not yet implemented")
    def test_negative_temperatures(self):
        from cropshield.features.weather_features import growing_degree_days
        s = pd.Series([-5.0, -2.0, 0.0])
        result = growing_degree_days(s, base_temp=10.0)
        assert result == 0.0


class TestExtremeHeatDays:
    @pytest.mark.skip(reason="extreme_heat_days not yet implemented")
    def test_counts_correctly(self):
        from cropshield.features.weather_features import extreme_heat_days
        s = pd.Series([30.0, 35.0, 36.0, 34.9, 35.1])
        assert extreme_heat_days(s, threshold=35.0) == 3

    @pytest.mark.skip(reason="extreme_heat_days not yet implemented")
    def test_none_above_threshold(self):
        from cropshield.features.weather_features import extreme_heat_days
        s = pd.Series([20.0, 25.0, 30.0])
        assert extreme_heat_days(s, threshold=35.0) == 0


class TestDryDays:
    @pytest.mark.skip(reason="dry_days not yet implemented")
    def test_counts_correctly(self):
        from cropshield.features.weather_features import dry_days
        s = pd.Series([0.0, 0.5, 2.0, 1.0, 5.0])
        assert dry_days(s, threshold=1.0) == 3

    @pytest.mark.skip(reason="dry_days not yet implemented")
    def test_all_wet(self):
        from cropshield.features.weather_features import dry_days
        s = pd.Series([5.0, 10.0, 3.0])
        assert dry_days(s, threshold=1.0) == 0
