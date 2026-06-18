"""
Tests for Prompt 9 feature engineering: lagged yield, county-normalised weather
anomalies, drought merge safety, and feature-group ablation contracts.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cropshield.features.lag_features import add_lag_features, build_lag_features_table
from cropshield.features.weather_anomalies import add_weather_anomalies
from cropshield.models.feature_sets import (
    FEATURE_GROUPS, available_groups, get_feature_set,
)
from cropshield.models.ml_models import LEAKAGE_COLUMNS


# ── Lagged yield features ───────────────────────────────────────────────────────

def _yield_df(extreme_future: bool = False) -> pd.DataFrame:
    """Two crops, two counties, years 2016–2021."""
    rows = []
    for fips in ["17001", "19001"]:
        for crop in ["CORN", "SOYBEANS"]:
            base = 180.0 if crop == "CORN" else 55.0
            for i, yr in enumerate(range(2016, 2022)):
                anomaly = float(i)  # 0,1,2,3,4,5
                yld = base + anomaly
                if extreme_future and yr == 2021:
                    anomaly = 9999.0
                    yld = base + 9999.0
                rows.append({
                    "county_fips": fips, "crop": crop, "year": yr,
                    "actual_yield": yld, "yield_anomaly": anomaly,
                })
    return pd.DataFrame(rows)


class TestLagFeatures:

    def test_prior_year_uses_only_prior(self):
        df = add_lag_features(_yield_df())
        # For 2018 CORN at 17001, prior_year_yield_anomaly should be the 2017 value (=1)
        row = df[(df.county_fips == "17001") & (df.crop == "CORN") & (df.year == 2018)].iloc[0]
        assert row["prior_year_yield_anomaly"] == 1.0

    def test_first_year_has_nan_lag(self):
        df = add_lag_features(_yield_df())
        row = df[(df.county_fips == "17001") & (df.crop == "CORN") & (df.year == 2016)].iloc[0]
        assert pd.isna(row["prior_year_yield_anomaly"])

    def test_corn_and_soybean_histories_do_not_mix(self):
        df = add_lag_features(_yield_df())
        # SOYBEANS prior_year_yield for 2018 must come from soybean history (~56),
        # never corn (~181).
        row = df[(df.county_fips == "17001") & (df.crop == "SOYBEANS") & (df.year == 2018)].iloc[0]
        assert 55.0 <= row["prior_year_yield"] <= 58.0

    def test_future_extreme_yield_does_not_change_earlier_lag(self):
        base = add_lag_features(_yield_df(extreme_future=False))
        alt = add_lag_features(_yield_df(extreme_future=True))
        # 2020's lag features depend on years < 2020, so the extreme 2021 value
        # must not change them.
        key = (base.county_fips == "17001") & (base.crop == "CORN") & (base.year == 2020)
        kalt = (alt.county_fips == "17001") & (alt.crop == "CORN") & (alt.year == 2020)
        for col in ["prior_year_yield_anomaly", "rolling_3yr_mean_yield_anomaly",
                    "rolling_3yr_mean_yield"]:
            assert base[key].iloc[0][col] == alt[kalt].iloc[0][col]

    def test_rolling_3yr_mean_correct(self):
        df = add_lag_features(_yield_df())
        # 2019 CORN: prior anomalies are 2016,2017,2018 = 0,1,2 → mean 1.0
        row = df[(df.county_fips == "17001") & (df.crop == "CORN") & (df.year == 2019)].iloc[0]
        assert row["rolling_3yr_mean_yield_anomaly"] == pytest.approx(1.0)

    def test_lag_table_unique_by_keys(self):
        table = build_lag_features_table(_yield_df())
        assert table.duplicated(subset=["county_fips", "crop", "year"]).sum() == 0


# ── Weather anomalies ───────────────────────────────────────────────────────────

def _weather_df(extreme_future: bool = False) -> pd.DataFrame:
    rows = []
    for fips in ["17001", "19001"]:
        for ckpt in ["may_31", "full_season"]:
            for i, yr in enumerate(range(2016, 2022)):
                precip = 100.0 + i * 10  # 100,110,...
                gdd = 1000.0 + i * 50
                heat = i
                if extreme_future and yr == 2021:
                    precip = 99999.0
                    gdd = 99999.0
                    heat = 9999
                rows.append({
                    "county_fips": fips, "checkpoint": ckpt, "year": yr,
                    "cumulative_precip": precip, "growing_degree_days": gdd,
                    "extreme_heat_days": heat, "dry_days": 10, "mean_temp": 20.0,
                })
    return pd.DataFrame(rows)


class TestWeatherAnomalies:

    def test_anomaly_uses_only_prior_years(self):
        df = add_weather_anomalies(_weather_df())
        # 2018 may_31 17001: prior precip 100,110 → mean 105; value 120 → anomaly 15
        row = df[(df.county_fips == "17001") & (df.checkpoint == "may_31") & (df.year == 2018)].iloc[0]
        assert row["precip_anomaly_from_county_checkpoint_mean"] == pytest.approx(15.0)

    def test_first_year_anomaly_is_nan(self):
        df = add_weather_anomalies(_weather_df())
        row = df[(df.county_fips == "17001") & (df.checkpoint == "may_31") & (df.year == 2016)].iloc[0]
        assert pd.isna(row["precip_anomaly_from_county_checkpoint_mean"])

    def test_future_extreme_weather_does_not_change_earlier_anomaly(self):
        base = add_weather_anomalies(_weather_df(extreme_future=False))
        alt = add_weather_anomalies(_weather_df(extreme_future=True))
        k = (base.county_fips == "17001") & (base.checkpoint == "may_31") & (base.year == 2019)
        ka = (alt.county_fips == "17001") & (alt.checkpoint == "may_31") & (alt.year == 2019)
        for col in ["precip_anomaly_from_county_checkpoint_mean",
                    "gdd_anomaly_from_county_checkpoint_mean",
                    "heat_days_anomaly_from_county_checkpoint_mean"]:
            assert base[k].iloc[0][col] == alt[ka].iloc[0][col]

    def test_checkpoints_normalised_independently(self):
        """may_31 and full_season climatologies must not mix."""
        df = _weather_df()
        # Make full_season precip much larger so a shared baseline would differ
        df.loc[df.checkpoint == "full_season", "cumulative_precip"] *= 4
        out = add_weather_anomalies(df)
        may = out[(out.county_fips == "17001") & (out.checkpoint == "may_31") & (out.year == 2018)].iloc[0]
        # may_31 anomaly should still be ~15 (unaffected by full_season scaling)
        assert may["precip_anomaly_from_county_checkpoint_mean"] == pytest.approx(15.0)


# ── Feature-set / ablation contracts ───────────────────────────────────────────

def _panel() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for fips in ["17001", "19001"]:
        for crop in ["CORN", "SOYBEANS"]:
            for ckpt in ["may_31", "full_season"]:
                for yr in range(2018, 2026):
                    rows.append({
                        "year": yr, "state": "ILLINOIS", "county": "T",
                        "county_fips": fips, "crop": crop, "checkpoint": ckpt,
                        "actual_yield": 180.0, "expected_yield": 180.0,
                        "yield_anomaly": float(rng.normal(0, 10)),
                        "yield_anomaly_pct": 0.0,
                        "cumulative_precip": rng.normal(400, 50),
                        "growing_degree_days": rng.normal(1500, 100),
                        "extreme_heat_days": rng.integers(0, 20),
                        "prior_year_yield_anomaly": rng.normal(0, 10),
                        "precip_anomaly_from_county_checkpoint_mean": rng.normal(0, 30),
                    })
    return pd.DataFrame(rows)


class TestFeatureSets:

    def test_available_groups_includes_expected(self):
        groups = available_groups(_panel())
        assert "baseline_features" in groups
        assert "weather_raw" in groups
        assert "weather_anomalies" in groups
        assert "lagged_yield" in groups
        assert "all_features" in groups

    def test_drought_group_absent_when_no_drought_columns(self):
        groups = available_groups(_panel())
        assert "drought" not in groups  # synthetic panel has no drought cols

    def test_no_feature_group_includes_leakage_columns(self):
        panel = _panel()
        for group in available_groups(panel):
            numeric, categorical = get_feature_set(panel, group)
            selected = set(numeric) | set(categorical)
            assert selected.isdisjoint(LEAKAGE_COLUMNS), group

    def test_descriptive_risk_never_in_feature_set(self):
        panel = _panel()
        panel["severe_risk_descriptive"] = 0
        for group in available_groups(panel):
            numeric, categorical = get_feature_set(panel, group)
            assert "severe_risk_descriptive" not in (numeric + categorical)

    def test_baseline_features_content(self):
        numeric, categorical = get_feature_set(_panel(), "baseline_features")
        assert "expected_yield" in numeric
        assert set(categorical) == {"crop", "checkpoint", "state"}

    def test_unknown_group_raises(self):
        with pytest.raises(ValueError, match="Unknown feature group"):
            get_feature_set(_panel(), "nonexistent")


# ── Drought merge fan-out safety ────────────────────────────────────────────────

class TestDroughtMergeSafety:

    def test_drought_merge_does_not_fan_out(self, tmp_path):
        """If drought data is present, merging must not multiply panel rows."""
        from cropshield.features.yield_targets import build_yield_targets
        from cropshield.features.weather_features import compute_multi_checkpoint_weather_features
        from cropshield.data.build_county_panel import build_modeling_panel
        import calendar

        fips_list = [("17001", "ILLINOIS"), ("19001", "IOWA")]
        years = [2017, 2018, 2019, 2020, 2021]

        # Yield targets
        yrows = []
        for fips, state in fips_list:
            for crop in ["CORN", "SOYBEANS"]:
                for yr in years:
                    yrows.append({"year": yr, "state": state, "county": "T",
                                  "county_fips": fips, "crop": crop,
                                  "value": 180.0, "unit": "BU / ACRE"})
        ypath = tmp_path / "yt.csv"
        build_yield_targets(pd.DataFrame(yrows), output_path=str(ypath))

        # Weather (multi-checkpoint)
        drows = []
        for fips, _ in fips_list:
            for yr in years:
                for month in range(4, 9):
                    for day in range(1, calendar.monthrange(yr, month)[1] + 1):
                        drows.append({"county_fips": fips, "state_fips": fips[:2],
                                      "year": yr, "date": pd.Timestamp(yr, month, day),
                                      "PRECTOTCORR": 5.0, "T2M": 20.0,
                                      "T2M_MIN": 15.0, "T2M_MAX": 25.0})
        weather = compute_multi_checkpoint_weather_features(pd.DataFrame(drows))
        wpath = tmp_path / "wf.csv"
        weather.to_csv(wpath, index=False)

        # Drought features (one row per county-year) WITH duplicate to test dedup
        drought = pd.DataFrame([
            {"county_fips": f, "year": yr, "weeks_d0": 1, "weeks_d1": 0, "weeks_d2": 0,
             "weeks_d3": 0, "weeks_d4": 0, "weeks_d2_plus": 0,
             "max_drought_category": 0, "mean_drought_severity": 0.0}
            for f, _ in fips_list for yr in years
        ])
        # add a duplicate row that would fan out a naive merge
        drought = pd.concat([drought, drought.iloc[[0]]], ignore_index=True)
        dpath = tmp_path / "drought.csv"
        drought.to_csv(dpath, index=False)

        panel = build_modeling_panel(
            yield_path=str(ypath), weather_path=str(wpath), drought_path=str(dpath),
            output_path=str(tmp_path / "panel.csv"), drop_missing_target=True,
        )
        # Unique by full key
        assert panel.duplicated(subset=["county_fips", "crop", "year", "checkpoint"]).sum() == 0
        # Drought columns present
        assert "mean_drought_severity" in panel.columns
