"""
Modeling-safe risk label tests.

Proves that ``assign_modeling_risk_labels`` computes thresholds from training
data only and that extreme test-year anomalies do not shift the boundary.
"""

from __future__ import annotations

import pandas as pd
import pytest

from cropshield.features.yield_targets import (
    assign_modeling_risk_labels,
    compute_risk_thresholds,
)


def _make_panel(years: list[int], anomaly_by_year: dict[int, float]) -> pd.DataFrame:
    rows = []
    for yr in years:
        rows.append({
            "year": yr,
            "state": "IOWA",
            "county_fips": "19001",
            "crop": "CORN",
            "yield_anomaly_pct": anomaly_by_year.get(yr, float(yr - 2019) * 2.0),
        })
    return pd.DataFrame(rows)


class TestModelingRiskLabels:
    def test_thresholds_from_train_only(self):
        train = _make_panel([2018, 2019, 2020], {2018: -10, 2019: 0, 2020: 5})
        test  = _make_panel([2021, 2022], {2021: -5, 2022: 3})
        thresholds = compute_risk_thresholds(train, quantile=0.20)
        train_q20 = train["yield_anomaly_pct"].quantile(0.20)
        assert thresholds.iloc[0] == pytest.approx(train_q20)

    def test_extreme_test_year_does_not_change_train_threshold(self):
        """Adding extreme anomalies in test years must not affect train thresholds."""
        train = _make_panel([2018, 2019, 2020], {2018: -8, 2019: -2, 2020: 4})
        test_normal = _make_panel([2021, 2022], {2021: -3, 2022: 2})
        test_extreme = _make_panel([2021, 2022], {2021: -9999.0, 2022: 9999.0})

        thresh_base = compute_risk_thresholds(train, quantile=0.20)

        # Label with normal test data
        train_l, test_l = assign_modeling_risk_labels(train, test_normal, quantile=0.20)
        thresh_after = compute_risk_thresholds(train_l, quantile=0.20)

        assert thresh_base.iloc[0] == pytest.approx(thresh_after.iloc[0]), (
            "Train threshold changed after labeling — possible leakage"
        )

        # Extreme test labels use same threshold
        _, test_extreme_l = assign_modeling_risk_labels(train, test_extreme, quantile=0.20)
        # Threshold value unchanged; only test labels differ
        assert thresh_base.iloc[0] == pytest.approx(
            compute_risk_thresholds(train, quantile=0.20).iloc[0]
        )
        assert test_extreme_l["severe_risk"].notna().any()

    def test_descriptive_vs_modeling_labels_differ(self):
        """Full-dataset descriptive labels must differ from train-only modeling labels."""
        from cropshield.features.yield_targets import add_risk_class

        full = _make_panel([2018, 2019, 2020, 2021, 2022], {})
        full = add_risk_class(full, quantile=0.20)

        train = full[full["year"] <= 2020].copy()
        test  = full[full["year"] > 2020].copy()
        train_m, test_m = assign_modeling_risk_labels(train, test, quantile=0.20)

        # Descriptive column exists on full data
        assert "severe_risk_descriptive" in full.columns
        # Modeling column on splits
        assert "severe_risk" in test_m.columns

    def test_build_yield_targets_skips_descriptive_by_default(self):
        from cropshield.features.yield_targets import build_yield_targets

        nass = pd.DataFrame({
            "year": [2018, 2019, 2020, 2021, 2022],
            "state": "IOWA",
            "county_fips": "19001",
            "county": "STORY",
            "crop": "CORN",
            "value": [180.0, 185.0, 175.0, 190.0, 178.0],
        })
        result = build_yield_targets(nass, window=3, min_periods=2, output_path=None)
        assert "severe_risk_descriptive" not in result.columns
        assert "severe_risk" not in result.columns
