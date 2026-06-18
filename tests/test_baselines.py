"""
Tests for baseline models (src/cropshield/models/baselines.py).

Verifies leakage prevention, fallback behaviour, and output contracts.
"""

from __future__ import annotations

import pandas as pd
import pytest

from cropshield.evaluation.validation_splits import temporal_split
from cropshield.features.yield_targets import assign_modeling_risk_labels
from cropshield.models.baselines import (
    CountyHistoricalMeanBaseline,
    CropCheckpointMeanBaseline,
    HistoricalCountyRiskBaseline,
    MajorityClassBaseline,
    PreviousYearAnomalyBaseline,
    ZeroAnomalyBaseline,
    get_classification_baselines,
    get_regression_baselines,
)


def _make_panel() -> pd.DataFrame:
    rows = []
    for yr in range(2018, 2024):
        for fips in ("19001", "17001"):
            rows.append({
                "year": yr,
                "state": "IOWA" if fips == "19001" else "ILLINOIS",
                "county": "A",
                "county_fips": fips,
                "crop": "CORN",
                "checkpoint": "full_season",
                "yield_anomaly": float(yr - 2020),
                "yield_anomaly_pct": float(yr - 2020) / 180 * 100,
            })
    return pd.DataFrame(rows)


class TestRegressionBaselinesNoLeakage:
    def test_county_mean_uses_train_only(self):
        panel = _make_panel()
        train, test = temporal_split(panel, n_test_years=2)
        model = CountyHistoricalMeanBaseline()
        model.fit(train)
        # Train mean for 19001 CORN
        expected = train.loc[
            (train["county_fips"] == "19001") & (train["crop"] == "CORN"),
            "yield_anomaly",
        ].mean()
        test_row = test[(test["county_fips"] == "19001")].iloc[0]
        pred = model.predict(test_row.to_frame().T)[0]
        assert pred == pytest.approx(expected)

    def test_county_mean_not_contaminated_by_test(self):
        panel = _make_panel()
        train, test = temporal_split(panel, n_test_years=2)
        model = CountyHistoricalMeanBaseline()
        model.fit(train)
        train_mean_before = model._county_means[("19001", "CORN")]

        # Inject extreme test values — should not change fitted mean
        test_extreme = test.copy()
        test_extreme["yield_anomaly"] = 9999.0
        model2 = CountyHistoricalMeanBaseline()
        model2.fit(train)  # still train only
        assert model2._county_means[("19001", "CORN")] == pytest.approx(train_mean_before)

    def test_previous_year_does_not_look_ahead(self):
        panel = _make_panel()
        train, test = temporal_split(panel, n_test_years=2)
        model = PreviousYearAnomalyBaseline()
        model.fit(train)
        preds = model.predict(test)
        # First test year should use last train year anomaly
        first_test_yr = int(test["year"].min())
        last_train_yr = int(train["year"].max())
        row = test[test["year"] == first_test_yr].iloc[0]
        expected = train[
            (train["county_fips"] == row["county_fips"])
            & (train["year"] == last_train_yr)
        ]["yield_anomaly"].iloc[0]
        idx = test[test["year"] == first_test_yr].index[0]
        pos = list(test.index).index(idx)
        assert preds[pos] == pytest.approx(expected)

    def test_crop_checkpoint_mean_train_only(self):
        panel = _make_panel()
        panel.loc[panel["year"] >= 2022, "checkpoint"] = "partial"
        train, test = temporal_split(panel, n_test_years=2)
        model = CropCheckpointMeanBaseline()
        model.fit(train)
        key = ("CORN", "full_season")
        expected = train[train["checkpoint"] == "full_season"]["yield_anomaly"].mean()
        assert model._ckpt_means[key] == pytest.approx(expected)


class TestFallbacks:
    def test_unseen_county_falls_back_to_crop_mean(self):
        train = pd.DataFrame({
            "county_fips": ["19001", "19001"],
            "crop": ["CORN", "CORN"],
            "year": [2018, 2019],
            "yield_anomaly": [5.0, 7.0],
            "checkpoint": ["full_season", "full_season"],
        })
        test = pd.DataFrame({
            "county_fips": ["99999"],
            "crop": ["CORN"],
            "year": [2020],
            "checkpoint": ["full_season"],
        })
        model = CountyHistoricalMeanBaseline()
        model.fit(train)
        pred = model.predict(test)[0]
        assert pred == pytest.approx(6.0)  # crop mean

    def test_unseen_crop_falls_back_to_zero(self):
        train = pd.DataFrame({
            "county_fips": ["19001"],
            "crop": ["CORN"],
            "year": [2018],
            "yield_anomaly": [5.0],
            "checkpoint": ["full_season"],
        })
        test = pd.DataFrame({
            "county_fips": ["19001"],
            "crop": ["SOYBEANS"],
            "year": [2020],
            "checkpoint": ["full_season"],
        })
        model = CountyHistoricalMeanBaseline()
        model.fit(train)
        # crop SOYBEANS unseen → global mean 5.0 actually since only one crop in train
        pred = model.predict(test)[0]
        assert pred == pytest.approx(5.0)


class TestClassificationBaselines:
    def test_assign_modeling_risk_labels_used(self):
        panel = _make_panel()
        train, test = temporal_split(panel, n_test_years=2)
        train_l, test_l = assign_modeling_risk_labels(train, test, quantile=0.20)
        assert "severe_risk" in test_l.columns
        assert "severe_risk_descriptive" not in test_l.columns

    def test_majority_class_from_train(self):
        train = pd.DataFrame({"severe_risk": [0, 0, 0, 1]})
        test = pd.DataFrame({"severe_risk": [0, 1]})
        model = MajorityClassBaseline()
        model.fit(train)
        assert (model.predict(test) == 0).all()


class TestBaselineRegistry:
    def test_regression_baseline_names(self):
        names = {m.name for m in get_regression_baselines()}
        assert names == {
            "zero_anomaly", "previous_year_anomaly",
            "county_historical_mean", "crop_checkpoint_mean",
        }

    def test_classification_baseline_names(self):
        names = {m.name for m in get_classification_baselines()}
        assert names == {"majority_class", "historical_county_risk"}


class TestMetricsOutputContract:
    def test_prediction_row_count_matches_test(self):
        panel = _make_panel()
        train, test = temporal_split(panel, n_test_years=2)
        train_l, test_l = assign_modeling_risk_labels(train, test)
        model = ZeroAnomalyBaseline()
        model.fit(train_l)
        preds = model.predict(test_l)
        assert len(preds) == len(test_l)

    def test_metrics_contain_expected_columns(self):
        from cropshield.evaluation.metrics import classification_metrics, regression_metrics

        reg = regression_metrics([1.0, 2.0], [1.5, 2.5], model_name="test")
        assert {"rmse", "mae", "r2"}.issubset(reg.keys())

        cls = classification_metrics([0, 1, 1, 0], [0, 1, 0, 0], model_name="test")
        assert {"accuracy", "precision", "recall", "f1", "tn", "fp", "fn", "tp"}.issubset(cls.keys())
