"""
Tests for multi-checkpoint panel building and baseline prediction format.

Validates:
- compute_multi_checkpoint_weather_features produces the correct checkpoint names
- Earlier checkpoints do NOT use future weather (leakage guard)
- All five checkpoints appear in the final panel
- CORN and SOYBEANS both appear in the panel when built from multi-crop data
- Uniqueness holds by (county_fips, crop, year, checkpoint)
- Baseline predictions have correct long-format row count
- Baseline metrics are reported by crop and checkpoint
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cropshield.features.weather_features import (
    CHECKPOINT_CONFIGS,
    compute_multi_checkpoint_weather_features,
    compute_weather_features,
)
from cropshield.data.build_county_panel import build_modeling_panel
from cropshield.models.baselines import get_regression_baselines, get_classification_baselines


# ── Synthetic daily weather fixture ────────────────────────────────────────────

def _make_daily_df(
    n_counties: int = 2,
    years: list[int] | None = None,
    extreme_month: int = 7,   # July values will be extreme
    extreme_precip: float = 999.0,
    normal_precip: float = 5.0,
) -> pd.DataFrame:
    """Synthetic daily weather: April–August for multiple counties and years."""
    years = years or [2018, 2019, 2020]
    records = []
    for fips_int in range(n_counties):
        fips = str(fips_int + 1).zfill(5)
        for year in years:
            for month in range(4, 9):  # April–August
                import calendar
                n_days = calendar.monthrange(year, month)[1]
                for day in range(1, n_days + 1):
                    precip = extreme_precip if month == extreme_month else normal_precip
                    records.append({
                        "county_fips": fips,
                        "state_fips": "99",
                        "year": year,
                        "date": pd.Timestamp(year=year, month=month, day=day),
                        "PRECTOTCORR": precip,
                        "T2M": 20.0,
                        "T2M_MIN": 15.0,
                        "T2M_MAX": 25.0,
                    })
    return pd.DataFrame(records)


# ── CHECKPOINT_CONFIGS contract ─────────────────────────────────────────────────

class TestCheckpointConfigs:

    def test_all_five_checkpoint_names_present(self):
        expected = {"may_31", "june_30", "july_31", "august_31", "full_season"}
        assert set(CHECKPOINT_CONFIGS.keys()) == expected

    def test_checkpoint_month_day_values(self):
        assert CHECKPOINT_CONFIGS["may_31"] == "05-31"
        assert CHECKPOINT_CONFIGS["june_30"] == "06-30"
        assert CHECKPOINT_CONFIGS["july_31"] == "07-31"
        assert CHECKPOINT_CONFIGS["august_31"] == "08-31"
        assert CHECKPOINT_CONFIGS["full_season"] is None

    def test_unknown_checkpoint_raises(self):
        daily = _make_daily_df()
        with pytest.raises(ValueError, match="Unknown checkpoint"):
            compute_multi_checkpoint_weather_features(daily, checkpoints=["bad_checkpoint"])


# ── compute_multi_checkpoint_weather_features ───────────────────────────────────

class TestComputeMultiCheckpointWeatherFeatures:

    def setup_method(self):
        self.daily = _make_daily_df(n_counties=3, years=[2018, 2019, 2020])

    def test_checkpoint_column_present_in_output(self):
        out = compute_multi_checkpoint_weather_features(self.daily)
        assert "checkpoint" in out.columns

    def test_all_five_checkpoints_in_output(self):
        out = compute_multi_checkpoint_weather_features(self.daily)
        assert set(out["checkpoint"].unique()) == set(CHECKPOINT_CONFIGS.keys())

    def test_row_count_is_checkpoints_times_county_years(self):
        out = compute_multi_checkpoint_weather_features(self.daily)
        n_county_years = self.daily[["county_fips", "year"]].drop_duplicates().shape[0]
        assert len(out) == 5 * n_county_years

    def test_subset_of_checkpoints(self):
        out = compute_multi_checkpoint_weather_features(
            self.daily, checkpoints=["may_31", "july_31"]
        )
        assert set(out["checkpoint"].unique()) == {"may_31", "july_31"}

    def test_may_31_has_lower_precip_than_full_season(self):
        """May checkpoint uses fewer months → less cumulative precip."""
        out = compute_multi_checkpoint_weather_features(self.daily)
        may_precip  = out[out["checkpoint"] == "may_31"]["cumulative_precip"].mean()
        full_precip = out[out["checkpoint"] == "full_season"]["cumulative_precip"].mean()
        assert may_precip < full_precip, (
            f"may_31 precip ({may_precip:.1f}) should be less than full_season ({full_precip:.1f})"
        )

    def test_june_30_obs_days_less_than_full_season(self):
        out = compute_multi_checkpoint_weather_features(self.daily)
        june_obs = out[out["checkpoint"] == "june_30"]["obs_days"].mean()
        full_obs  = out[out["checkpoint"] == "full_season"]["obs_days"].mean()
        assert june_obs < full_obs


# ── Leakage: earlier checkpoints must not include future weather ────────────────

class TestCheckpointLeakage:

    def _extreme_after_cutoff(self, cutoff_name: str, cutoff_mm: int) -> tuple[float, float]:
        """Return (cutoff_cumprec, full_cumprec) with extreme precip after cutoff month."""
        daily = _make_daily_df(
            n_counties=1,
            years=[2019, 2020],
            extreme_month=cutoff_mm + 1,   # first month AFTER the cutoff
            extreme_precip=9999.0,
            normal_precip=1.0,
        )
        out = compute_multi_checkpoint_weather_features(daily)
        cutoff_val = float(
            out[out["checkpoint"] == cutoff_name]["cumulative_precip"].mean()
        )
        full_val = float(
            out[out["checkpoint"] == "full_season"]["cumulative_precip"].mean()
        )
        return cutoff_val, full_val

    def test_may_31_excludes_june_extreme(self):
        may_val, full_val = self._extreme_after_cutoff("may_31", cutoff_mm=5)
        assert full_val > may_val * 10, (
            "full_season precip should be >> may_31 when June+ is extreme"
        )

    def test_june_30_excludes_july_extreme(self):
        june_val, full_val = self._extreme_after_cutoff("june_30", cutoff_mm=6)
        assert full_val > june_val * 10, (
            "full_season precip should be >> june_30 when July+ is extreme"
        )

    def test_july_31_excludes_august_extreme(self):
        july_val, full_val = self._extreme_after_cutoff("july_31", cutoff_mm=7)
        assert full_val > july_val * 10, (
            "full_season precip should be >> july_31 when August is extreme"
        )

    def test_earlier_checkpoints_do_not_change_when_later_weather_changes(self):
        """
        Regression: modifying July+ weather should have zero effect on may_31
        features (same county, same year, same April–May records).
        """
        base = _make_daily_df(n_counties=1, years=[2020], normal_precip=5.0, extreme_month=7)
        alt = base.copy()
        # Double the precipitation for all July+August rows
        late_mask = alt["date"].dt.month >= 7
        alt.loc[late_mask, "PRECTOTCORR"] = 999.0

        base_out = compute_multi_checkpoint_weather_features(base, checkpoints=["may_31"])
        alt_out  = compute_multi_checkpoint_weather_features(alt,  checkpoints=["may_31"])

        pd.testing.assert_series_equal(
            base_out["cumulative_precip"].reset_index(drop=True),
            alt_out["cumulative_precip"].reset_index(drop=True),
            check_names=False,
        )


# ── Panel uniqueness ────────────────────────────────────────────────────────────

class TestPanelUniqueness:

    def _build_minimal_panel(self, tmp_path) -> pd.DataFrame:
        """Build a tiny two-crop, two-county, multi-checkpoint panel in tmp_path."""
        fips_list = [("17001", "ILLINOIS", "ADAMS"), ("19001", "IOWA", "ADAIR")]
        years = [2017, 2018, 2019, 2020, 2021]

        # Yield targets (two crops, two counties)
        records = []
        for fips, state, county in fips_list:
            for crop in ["CORN", "SOYBEANS"]:
                for yr in years:
                    records.append({
                        "year": yr, "state": state, "county": county,
                        "county_fips": fips, "crop": crop,
                        "value": 180.0 + yr * 0.1,
                        "unit": "BU / ACRE",
                    })
        yield_raw = pd.DataFrame(records)
        from cropshield.features.yield_targets import build_yield_targets
        yield_targets_path = tmp_path / "yield_targets.csv"
        build_yield_targets(yield_raw, output_path=str(yield_targets_path))

        # Weather features: build daily data with the SAME FIPS as yield targets
        import calendar
        daily_records = []
        for fips, _, _ in fips_list:
            for yr in years:
                for month in range(4, 9):
                    n_days = calendar.monthrange(yr, month)[1]
                    for day in range(1, n_days + 1):
                        daily_records.append({
                            "county_fips": fips,
                            "state_fips": fips[:2],
                            "year": yr,
                            "date": pd.Timestamp(year=yr, month=month, day=day),
                            "PRECTOTCORR": 5.0,
                            "T2M": 20.0,
                            "T2M_MIN": 15.0,
                            "T2M_MAX": 25.0,
                        })
        daily = pd.DataFrame(daily_records)
        weather = compute_multi_checkpoint_weather_features(daily)
        weather_path = tmp_path / "weather_features.csv"
        weather.to_csv(weather_path, index=False)

        panel = build_modeling_panel(
            yield_path=str(yield_targets_path),
            weather_path=str(weather_path),
            output_path=str(tmp_path / "panel.csv"),
            drop_missing_target=True,
        )
        return panel

    def test_panel_unique_by_county_crop_year_checkpoint(self, tmp_path):
        panel = self._build_minimal_panel(tmp_path)
        dupe_count = panel.duplicated(
            subset=["county_fips", "crop", "year", "checkpoint"]
        ).sum()
        assert dupe_count == 0, f"Found {dupe_count} duplicate (county_fips, crop, year, checkpoint) rows"

    def test_panel_has_all_five_checkpoints(self, tmp_path):
        panel = self._build_minimal_panel(tmp_path)
        assert set(panel["checkpoint"].dropna().unique()) == set(CHECKPOINT_CONFIGS.keys())

    def test_panel_has_both_crops(self, tmp_path):
        panel = self._build_minimal_panel(tmp_path)
        assert "CORN" in panel["crop"].values
        assert "SOYBEANS" in panel["crop"].values


# ── Baseline prediction row count ──────────────────────────────────────────────

class TestBaselinePredictionFormat:

    def _make_train_test(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Minimal multi-crop, multi-checkpoint train/test split."""
        rng = np.random.default_rng(42)
        years = list(range(2016, 2023))
        test_years = [2023, 2024]
        records = []
        for fips in ["17001", "17003", "19001"]:
            for crop in ["CORN", "SOYBEANS"]:
                for ckpt in ["may_31", "full_season"]:
                    for yr in years + test_years:
                        anomaly = float(rng.normal(0, 10))
                        records.append({
                            "county_fips": fips, "crop": crop,
                            "checkpoint": ckpt, "year": yr,
                            "state": "ILLINOIS", "county": "TEST",
                            "actual_yield": 180.0,
                            "expected_yield": 180.0,
                            "yield_anomaly": anomaly,
                            "yield_anomaly_pct": anomaly / 180.0,
                        })
        df = pd.DataFrame(records)
        train = df[df["year"].isin(years)].copy()
        test  = df[df["year"].isin(test_years)].copy()

        from cropshield.features.yield_targets import assign_modeling_risk_labels
        train, test = assign_modeling_risk_labels(train, test, quantile=0.20)
        return train, test

    def test_regression_predictions_length_matches_test_rows(self):
        train, test = self._make_train_test()
        n_test = len(test)
        all_models = get_regression_baselines()
        for model in all_models:
            model.fit(train)
            preds = model.predict(test)
            assert len(preds) == n_test, (
                f"{model.name}: expected {n_test} predictions, got {len(preds)}"
            )

    def test_classification_predictions_length_matches_test_rows(self):
        train, test = self._make_train_test()
        test_cls = test.dropna(subset=["severe_risk"])
        n_test = len(test_cls)
        for model in get_classification_baselines():
            model.fit(train)
            preds = model.predict(test_cls)
            assert len(preds) == n_test, (
                f"{model.name}: expected {n_test} predictions, got {len(preds)}"
            )

    def test_long_format_predictions_row_count(self):
        """
        Long-format predictions = test rows × number of baseline models.
        """
        train, test = self._make_train_test()
        reg_models  = get_regression_baselines()
        cls_models  = get_classification_baselines()
        test_cls = test.dropna(subset=["severe_risk"])

        pred_frames = []
        for m in reg_models:
            m.fit(train)
            pred_frames.append(pd.DataFrame({
                "model": m.name, "y_true": test["yield_anomaly"].values,
                "y_pred": m.predict(test),
            }))
        for m in cls_models:
            m.fit(train)
            pred_frames.append(pd.DataFrame({
                "model": m.name, "y_true": test_cls["severe_risk"].values,
                "y_pred": m.predict(test_cls),
            }))

        preds = pd.concat(pred_frames, ignore_index=True)
        expected_reg_rows = len(test) * len(reg_models)
        expected_cls_rows = len(test_cls) * len(cls_models)
        assert len(preds) == expected_reg_rows + expected_cls_rows


# ── Metrics reported by crop and checkpoint ────────────────────────────────────

class TestBaselineMetricsByGroup:

    def _run_eval_regression(self, model_name, y_true, y_pred, meta, group_cols):
        """Minimal copy of _eval_regression from 03_train_baselines.py."""
        from cropshield.evaluation.metrics import regression_metrics
        rows = [{"model": model_name, "group": "overall", "group_value": "all",
                 **regression_metrics(y_true, y_pred, model_name=model_name)}]
        for col in group_cols:
            for val in meta[col].unique():
                mask = (meta[col] == val).values
                if mask.sum() >= 2:
                    rows.append({
                        "model": model_name, "group": col, "group_value": str(val),
                        **regression_metrics(y_true[mask], y_pred[mask]),
                    })
        return rows

    def test_metrics_reported_by_crop(self):
        rng = np.random.default_rng(42)
        n = 200
        meta = pd.DataFrame({
            "crop": rng.choice(["CORN", "SOYBEANS"], size=n),
            "checkpoint": rng.choice(["may_31", "full_season"], size=n),
        })
        y_true = rng.normal(0, 10, n)
        y_pred = np.zeros(n)
        rows = self._run_eval_regression("zero", y_true, y_pred, meta, ["crop", "checkpoint"])
        df = pd.DataFrame(rows)
        assert "CORN" in df["group_value"].values
        assert "SOYBEANS" in df["group_value"].values

    def test_metrics_reported_by_checkpoint(self):
        rng = np.random.default_rng(42)
        n = 200
        meta = pd.DataFrame({
            "crop": rng.choice(["CORN", "SOYBEANS"], size=n),
            "checkpoint": rng.choice(["may_31", "june_30", "full_season"], size=n),
        })
        y_true = rng.normal(0, 10, n)
        y_pred = np.zeros(n)
        rows = self._run_eval_regression("zero", y_true, y_pred, meta, ["crop", "checkpoint"])
        df = pd.DataFrame(rows)
        ckpts_reported = set(df[df["group"] == "checkpoint"]["group_value"].unique())
        assert {"may_31", "june_30", "full_season"}.issubset(ckpts_reported)
