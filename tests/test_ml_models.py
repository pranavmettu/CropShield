"""
Tests for the first-generation ML models and training pipeline.

Verifies leakage safety, feature selection contracts, preprocessing fit
behaviour, and prediction/metric output shape.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cropshield.features.yield_targets import assign_modeling_risk_labels
from cropshield.evaluation.validation_splits import temporal_split
from cropshield.models.ml_models import (
    LEAKAGE_COLUMNS,
    build_classification_models,
    build_regression_models,
    build_preprocessor,
    select_feature_columns,
    assert_no_leakage_features,
)


# ── Synthetic panel fixture ─────────────────────────────────────────────────────

def _make_panel(seed: int = 0) -> pd.DataFrame:
    """Build a synthetic multi-crop, multi-checkpoint panel."""
    rng = np.random.default_rng(seed)
    checkpoints = ["may_31", "june_30", "july_31", "august_31", "full_season"]
    counties = ["17001", "17003", "19001", "19003"]
    years = list(range(2016, 2026))
    records = []
    for fips in counties:
        state = "ILLINOIS" if fips.startswith("17") else "IOWA"
        for crop in ["CORN", "SOYBEANS"]:
            for yr in years:
                expected = 180.0 if crop == "CORN" else 55.0
                for ckpt in checkpoints:
                    precip = rng.normal(400, 80)
                    gdd = rng.normal(1500, 200)
                    heat = rng.integers(0, 25)
                    anomaly = rng.normal(0, 12)
                    records.append({
                        "year": yr, "state": state, "county": "TEST",
                        "county_fips": fips, "crop": crop, "checkpoint": ckpt,
                        "actual_yield": expected + anomaly,
                        "expected_yield": expected,
                        "yield_anomaly": anomaly,
                        "yield_anomaly_pct": anomaly / expected,
                        "cumulative_precip": precip,
                        "mean_temp": rng.normal(20, 2),
                        "max_temp": rng.normal(33, 3),
                        "extreme_heat_days": heat,
                        "dry_days": rng.integers(0, 40),
                        "longest_dry_spell": rng.integers(0, 15),
                        "growing_degree_days": gdd,
                        "obs_days": 153,
                        "is_partial_year": False,
                        "precip_anomaly": rng.normal(0, 50),
                        "year_index": yr - 2015,
                        "heat_dry_stress": heat * rng.integers(0, 40),
                        "heat_dry_spell": heat * rng.integers(0, 15),
                    })
    return pd.DataFrame(records)


# ── Feature selection contracts ─────────────────────────────────────────────────

class TestFeatureSelection:

    def setup_method(self):
        self.panel = _make_panel()

    def test_no_leakage_columns_selected(self):
        numeric, categorical = select_feature_columns(self.panel)
        selected = set(numeric) | set(categorical)
        assert selected.isdisjoint(LEAKAGE_COLUMNS)

    def test_target_columns_excluded(self):
        numeric, categorical = select_feature_columns(self.panel)
        selected = set(numeric) | set(categorical)
        for target in ("yield_anomaly", "yield_anomaly_pct", "actual_yield"):
            assert target not in selected

    def test_expected_yield_is_allowed_feature(self):
        numeric, _ = select_feature_columns(self.panel)
        assert "expected_yield" in numeric

    def test_categorical_features_present(self):
        _, categorical = select_feature_columns(self.panel)
        assert "crop" in categorical
        assert "checkpoint" in categorical
        assert "state" in categorical

    def test_county_excluded_by_default(self):
        numeric, categorical = select_feature_columns(self.panel)
        assert "county_fips" not in numeric
        assert "county_fips" not in categorical

    def test_county_included_when_requested(self):
        _, categorical = select_feature_columns(self.panel, include_county=True)
        assert "county_fips" in categorical

    def test_year_index_excluded_by_default(self):
        numeric, _ = select_feature_columns(self.panel)
        assert "year_index" not in numeric

    def test_year_index_included_when_requested(self):
        numeric, _ = select_feature_columns(self.panel, include_year_index=True)
        assert "year_index" in numeric

    def test_raw_year_never_a_feature(self):
        numeric, categorical = select_feature_columns(
            self.panel, include_year_index=True
        )
        assert "year" not in numeric
        assert "year" not in categorical

    def test_assert_no_leakage_raises_on_target(self):
        with pytest.raises(ValueError, match="Leakage columns"):
            assert_no_leakage_features(["yield_anomaly", "cumulative_precip"], ["crop"])


# ── Model construction ──────────────────────────────────────────────────────────

class TestModelConstruction:

    def setup_method(self):
        self.panel = _make_panel()
        self.numeric, self.categorical = select_feature_columns(self.panel)

    def test_three_regression_models(self):
        models = build_regression_models(self.numeric, self.categorical)
        assert set(models.keys()) == {
            "ridge", "random_forest_reg", "hist_gradient_boosting_reg"
        }

    def test_two_classification_models(self):
        models = build_classification_models(self.numeric, self.categorical)
        assert set(models.keys()) == {"logistic_regression", "random_forest_clf"}

    def test_regression_models_are_pipelines(self):
        from sklearn.pipeline import Pipeline
        models = build_regression_models(self.numeric, self.categorical)
        for m in models.values():
            assert isinstance(m, Pipeline)
            assert m.steps[0][0] == "preprocessor"

    def test_classifiers_use_balanced_class_weight(self):
        models = build_classification_models(self.numeric, self.categorical)
        for name, pipe in models.items():
            model = pipe.named_steps["model"]
            assert model.class_weight == "balanced", name

    def test_build_raises_if_leakage_feature_passed(self):
        with pytest.raises(ValueError, match="Leakage columns"):
            build_regression_models(["yield_anomaly"], ["crop"])


# ── Preprocessing fit-on-train-only ───────────────────────────────────────────

class TestPreprocessingLeakage:

    def test_preprocessor_fit_uses_train_statistics_only(self):
        """
        Median imputation must use TRAIN medians.  If the test set has a wildly
        different distribution, the imputed value for a train-NaN must still be
        the train median, proving the preprocessor never saw test data.
        """
        numeric = ["cumulative_precip"]
        categorical = ["crop"]
        prep = build_preprocessor(numeric, categorical)

        train = pd.DataFrame({
            "cumulative_precip": [10.0, 20.0, 30.0, np.nan],
            "crop": ["CORN", "CORN", "SOYBEANS", "CORN"],
        })
        prep.fit(train)

        num_steps = prep.named_transformers_["num"].named_steps
        # Train median of [10,20,30] = 20.0 (imputer fit on TRAIN only)
        assert num_steps["imputer"].statistics_[0] == 20.0
        # Scaler mean is the TRAIN mean (after imputation): [10,20,30,20] → 20.0
        assert num_steps["scaler"].mean_[0] == 20.0

        # Transforming a test row with NaN should fill with the TRAIN median (20),
        # then scale by TRAIN stats → (20 - 20) / train_std = 0.0.  This proves the
        # preprocessor never saw test data.
        test = pd.DataFrame({
            "cumulative_precip": [np.nan, 9999.0],
            "crop": ["SOYBEANS", "CORN"],
        })
        transformed = prep.transform(test)
        assert np.isclose(transformed[0, 0], 0.0)

    def test_onehot_handles_unseen_category(self):
        numeric = ["cumulative_precip"]
        categorical = ["crop"]
        prep = build_preprocessor(numeric, categorical)
        train = pd.DataFrame({
            "cumulative_precip": [10.0, 20.0],
            "crop": ["CORN", "SOYBEANS"],
        })
        prep.fit(train)
        # Unseen category "WHEAT" must not raise (handle_unknown="ignore")
        test = pd.DataFrame({"cumulative_precip": [15.0], "crop": ["WHEAT"]})
        result = prep.transform(test)
        assert result.shape[0] == 1


# ── End-to-end training + prediction shape ────────────────────────────────────

class TestEndToEndTraining:

    def setup_method(self):
        self.panel = _make_panel()
        self.train_raw, self.test_raw = temporal_split(self.panel, n_test_years=3)
        self.train_df, self.test_df = assign_modeling_risk_labels(
            self.train_raw, self.test_raw, quantile=0.20
        )
        self.numeric, self.categorical = select_feature_columns(self.panel)
        self.feature_cols = self.numeric + self.categorical

    def test_temporal_split_no_year_overlap(self):
        train_years = set(int(y) for y in self.train_df["year"].unique())
        test_years = set(int(y) for y in self.test_df["year"].unique())
        assert train_years.isdisjoint(test_years)

    def test_risk_labels_assigned_after_split(self):
        assert "severe_risk" in self.train_df.columns
        assert "severe_risk" in self.test_df.columns

    def test_descriptive_risk_never_used(self):
        assert "severe_risk_descriptive" not in self.train_df.columns
        assert "severe_risk_descriptive" not in self.feature_cols

    def test_regression_prediction_row_count(self):
        models = build_regression_models(self.numeric, self.categorical)
        X_train = self.train_df[self.feature_cols]
        y_train = self.train_df["yield_anomaly"].values
        X_test = self.test_df[self.feature_cols]
        for name, pipe in models.items():
            pipe.fit(X_train, y_train)
            preds = pipe.predict(X_test)
            assert len(preds) == len(self.test_df), name

    def test_long_format_prediction_count_equals_test_rows_times_models(self):
        """Long-format regression predictions = test rows × number of ML models."""
        reg_models = build_regression_models(self.numeric, self.categorical)
        X_train = self.train_df[self.feature_cols]
        y_train = self.train_df["yield_anomaly"].values
        X_test = self.test_df[self.feature_cols]

        frames = []
        for name, pipe in reg_models.items():
            pipe.fit(X_train, y_train)
            frames.append(pd.DataFrame({
                "model_name": name,
                "y_pred": pipe.predict(X_test),
            }))
        preds = pd.concat(frames, ignore_index=True)
        assert len(preds) == len(self.test_df) * len(reg_models)

    def test_classification_proba_available(self):
        models = build_classification_models(self.numeric, self.categorical)
        train_cls = self.train_df.dropna(subset=["severe_risk"])
        test_cls = self.test_df.dropna(subset=["severe_risk"])
        X_train = train_cls[self.feature_cols]
        y_train = train_cls["severe_risk"].astype(int).values
        X_test = test_cls[self.feature_cols]
        for name, pipe in models.items():
            pipe.fit(X_train, y_train)
            assert hasattr(pipe, "predict_proba"), name
            proba = pipe.predict_proba(X_test)
            assert proba.shape == (len(test_cls), 2)

    def test_prediction_output_has_required_columns(self):
        """model output includes model_name, crop, checkpoint, year, county_fips, y_true, y_pred."""
        models = build_regression_models(self.numeric, self.categorical)
        X_train = self.train_df[self.feature_cols]
        y_train = self.train_df["yield_anomaly"].values
        X_test = self.test_df[self.feature_cols]
        meta_cols = ["year", "state", "county_fips", "crop", "checkpoint"]
        frames = []
        for name, pipe in models.items():
            pipe.fit(X_train, y_train)
            frames.append(self.test_df[meta_cols].assign(
                model_name=name, task="regression",
                y_true=self.test_df["yield_anomaly"].values,
                y_pred=pipe.predict(X_test), y_proba=np.nan,
            ))
        out = pd.concat(frames, ignore_index=True)
        required = {"model_name", "crop", "checkpoint", "year", "county_fips", "y_true", "y_pred"}
        assert required.issubset(out.columns)


# ── Metrics include all expected models ────────────────────────────────────────

class TestMetricsCoverage:

    def test_all_regression_models_appear_in_metrics(self):
        panel = _make_panel()
        train_raw, test_raw = temporal_split(panel, n_test_years=3)
        train_df, test_df = assign_modeling_risk_labels(train_raw, test_raw)
        numeric, categorical = select_feature_columns(panel)
        feature_cols = numeric + categorical

        from cropshield.evaluation.metrics import regression_metrics
        models = build_regression_models(numeric, categorical)
        rows = []
        for name, pipe in models.items():
            pipe.fit(train_df[feature_cols], train_df["yield_anomaly"].values)
            preds = pipe.predict(test_df[feature_cols])
            rows.append({"model": name, **regression_metrics(test_df["yield_anomaly"].values, preds)})
        df = pd.DataFrame(rows)
        assert set(df["model"]) == set(models.keys())
