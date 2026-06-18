"""
County panel integrity tests for CropShield.

The modeling panel is the final artifact fed to the ML model.  Silent
integrity failures here (duplicate rows, missing columns, inflated row
counts from fanout joins) produce wrong training data and wrong metrics
without any exception being raised.

Tests verify:
  - No duplicate (county_fips, crop, year) combinations.
  - All required columns are present after panel construction.
  - Row counts match the yield side (no fanout from a many-side join).
  - Weather duplicates are either de-duplicated or raise a clear error.
  - FIPS normalization works end-to-end inside build_modeling_panel.
  - Missing yield_anomaly rows are dropped (not silently kept).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from cropshield.data.build_county_panel import (
    _normalise_fips,
    build_modeling_panel,
    report_missingness,
    validate_merge_keys,
)

# Minimum required columns in the final modeling panel
REQUIRED_PANEL_COLUMNS = {
    "year", "state", "county", "county_fips", "crop",
    "actual_yield", "expected_yield", "yield_anomaly", "yield_anomaly_pct",
}


# ── Fixtures ───────────────────────────────────────────────────────────────────

def _make_yield_targets(n_counties: int = 3, n_years: int = 4) -> pd.DataFrame:
    rows = []
    for i in range(n_counties):
        fips = f"1900{i + 1}"
        for yr in range(2018, 2018 + n_years):
            rows.append({
                "year": yr,
                "state": "IOWA",
                "county": f"COUNTY_{i}",
                "county_fips": fips,
                "crop": "CORN",
                "actual_yield": 185.0 + i,
                "expected_yield": 180.0 + i if yr > 2018 else None,
                "yield_anomaly": 5.0 if yr > 2018 else None,
                "yield_anomaly_pct": 2.5 if yr > 2018 else None,
                "severe_risk_descriptive": 0 if yr > 2018 else None,
            })
    return pd.DataFrame(rows)


def _make_weather_features(n_counties: int = 3, n_years: int = 4) -> pd.DataFrame:
    rows = []
    for i in range(n_counties):
        fips = f"1900{i + 1}"
        for yr in range(2018, 2018 + n_years):
            rows.append({
                "county_fips": fips,
                "year": yr,
                "cumulative_precip": 400.0,
                "mean_temp": 20.0,
                "max_temp": 32.0,
                "extreme_heat_days": 5,
                "dry_days": 80,
                "longest_dry_spell": 12,
                "growing_degree_days": 1750.0,
                "precip_anomaly": 10.0,
                "obs_days": 153,
            })
    return pd.DataFrame(rows)


def _write_csvs(
    yields: pd.DataFrame,
    weather: pd.DataFrame,
    tmpdir: str,
) -> tuple[Path, Path, Path]:
    yield_path   = Path(tmpdir) / "yield_targets.csv"
    weather_path = Path(tmpdir) / "weather_features.csv"
    output_path  = Path(tmpdir) / "modeling_panel.csv"
    yields.to_csv(yield_path,   index=False)
    weather.to_csv(weather_path, index=False)
    return yield_path, weather_path, output_path


# ── Required columns ──────────────────────────────────────────────────────────

class TestPanelRequiredColumns:
    def test_required_columns_present(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yields  = _make_yield_targets()
            weather = _make_weather_features()
            yp, wp, op = _write_csvs(yields, weather, tmpdir)
            panel = build_modeling_panel(
                yield_path=yp, weather_path=wp, output_path=op,
                missingness_path=Path(tmpdir) / "miss.csv",
                drop_missing_target=True,
            )
            missing = REQUIRED_PANEL_COLUMNS - set(panel.columns)
            assert not missing, f"Panel is missing required columns: {missing}"

    def test_feature_columns_present(self):
        """Weather-derived feature columns must exist after the merge."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yields  = _make_yield_targets()
            weather = _make_weather_features()
            yp, wp, op = _write_csvs(yields, weather, tmpdir)
            panel = build_modeling_panel(
                yield_path=yp, weather_path=wp, output_path=op,
                missingness_path=Path(tmpdir) / "miss.csv",
            )
            for col in ("cumulative_precip", "extreme_heat_days", "growing_degree_days"):
                assert col in panel.columns, f"Expected feature column '{col}' missing from panel"


# ── Row count integrity ────────────────────────────────────────────────────────

class TestPanelRowCount:
    def test_row_count_equals_yield_rows_with_anomaly(self):
        """
        After the left join, panel row count must equal the number of
        yield_target rows that have a non-null yield_anomaly.
        No fanout: one yield row → at most one panel row.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            yields  = _make_yield_targets()
            weather = _make_weather_features()
            yp, wp, op = _write_csvs(yields, weather, tmpdir)
            panel = build_modeling_panel(
                yield_path=yp, weather_path=wp, output_path=op,
                missingness_path=Path(tmpdir) / "miss.csv",
                drop_missing_target=True,
            )
            expected_rows = yields["yield_anomaly"].notna().sum()
            assert len(panel) == expected_rows, (
                f"Panel has {len(panel)} rows but expected {expected_rows}. "
                "A many-to-one join may have multiplied rows."
            )

    def test_weather_duplicates_do_not_multiply_panel_rows(self):
        """
        Duplicate (county_fips, year) rows in weather must NOT multiply
        the panel — if weather has 2 rows for the same key, the left join
        would produce 2 panel rows per yield row.
        This test documents the current behavior; ideally it should warn or deduplicate.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            yields  = _make_yield_targets(n_counties=1, n_years=2)
            weather = _make_weather_features(n_counties=1, n_years=2)
            # Duplicate every weather row
            weather_duped = pd.concat([weather, weather], ignore_index=True)
            yp, wp, op = _write_csvs(yields, weather_duped, tmpdir)

            panel = build_modeling_panel(
                yield_path=yp, weather_path=wp, output_path=op,
                missingness_path=Path(tmpdir) / "miss.csv",
                drop_missing_target=True,
            )
            # Expected: row count must NOT be doubled by the duplication
            # (deduplication should happen before or during merge)
            non_null = yields["yield_anomaly"].notna().sum()
            assert len(panel) == non_null, (
                f"Duplicate weather rows multiplied the panel: {len(panel)} rows "
                f"instead of {non_null}. build_modeling_panel should deduplicate weather."
            )


# ── Uniqueness constraint ─────────────────────────────────────────────────────

class TestPanelUniqueness:
    def test_no_duplicate_fips_crop_year_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yields  = _make_yield_targets()
            weather = _make_weather_features()
            yp, wp, op = _write_csvs(yields, weather, tmpdir)
            panel = build_modeling_panel(
                yield_path=yp, weather_path=wp, output_path=op,
                missingness_path=Path(tmpdir) / "miss.csv",
            )
            dupes = panel.duplicated(subset=["county_fips", "crop", "year"])
            assert not dupes.any(), (
                f"Panel has {dupes.sum()} duplicate (county_fips, crop, year) rows"
            )


# ── FIPS normalization in build_modeling_panel ────────────────────────────────

class TestPanelFipsNormalization:
    def test_float_fips_in_yields_matches_string_fips_in_weather(self):
        """
        FIPS stored as floats in yield CSV must join correctly to
        string FIPS in weather CSV.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            yields = pd.DataFrame({
                "year": [2020, 2020],
                "state": ["IOWA", "ILLINOIS"],
                "county": ["STORY", "ADAMS"],
                "county_fips": [19169.0, 17001.0],   # floats from CSV read
                "crop": ["CORN", "CORN"],
                "actual_yield":    [185.0, 190.0],
                "expected_yield":  [180.0, 187.0],
                "yield_anomaly":   [5.0, 3.0],
                "yield_anomaly_pct": [2.5, 1.6],
                "severe_risk_descriptive": [0, 0],
            })
            weather = pd.DataFrame({
                "county_fips": ["19169", "17001"],    # strings
                "year": [2020, 2020],
                "cumulative_precip": [400.0, 380.0],
                "mean_temp": [20.0, 21.0],
                "max_temp": [30.0, 31.0],
                "extreme_heat_days": [5, 8],
                "dry_days": [80, 85],
                "longest_dry_spell": [12, 14],
                "growing_degree_days": [1750.0, 1800.0],
                "precip_anomaly": [10.0, -5.0],
                "obs_days": [153, 153],
            })
            yp = Path(tmpdir) / "yields.csv"
            wp = Path(tmpdir) / "weather.csv"
            op = Path(tmpdir) / "panel.csv"
            yields.to_csv(yp, index=False)
            weather.to_csv(wp, index=False)
            panel = build_modeling_panel(
                yield_path=yp, weather_path=wp, output_path=op,
                missingness_path=Path(tmpdir) / "miss.csv",
            )
            assert panel["cumulative_precip"].notna().all(), (
                "Float FIPS in yields failed to join to string FIPS in weather — "
                "all weather features are NaN"
            )


# ── Missing target handling ───────────────────────────────────────────────────

class TestMissingTargetDropping:
    def test_rows_with_null_anomaly_dropped_by_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yields  = _make_yield_targets()
            weather = _make_weather_features()
            yp, wp, op = _write_csvs(yields, weather, tmpdir)
            panel = build_modeling_panel(
                yield_path=yp, weather_path=wp, output_path=op,
                missingness_path=Path(tmpdir) / "miss.csv",
                drop_missing_target=True,
            )
            assert panel["yield_anomaly"].notna().all(), (
                "Panel contains rows with NaN yield_anomaly after drop_missing_target=True"
            )

    def test_rows_with_null_anomaly_kept_when_flag_false(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yields  = _make_yield_targets()
            weather = _make_weather_features()
            yp, wp, op = _write_csvs(yields, weather, tmpdir)
            panel_keep = build_modeling_panel(
                yield_path=yp, weather_path=wp, output_path=op,
                missingness_path=Path(tmpdir) / "miss.csv",
                drop_missing_target=False,
            )
            panel_drop = build_modeling_panel(
                yield_path=yp, weather_path=wp, output_path=op,
                missingness_path=Path(tmpdir) / "miss.csv",
                drop_missing_target=True,
            )
            assert len(panel_keep) >= len(panel_drop), (
                "Keeping NaN rows should produce >= rows than dropping them"
            )


# ── validate_merge_keys ───────────────────────────────────────────────────────

class TestValidateMergeKeys:
    def test_raises_on_missing_key(self):
        df = pd.DataFrame({"county_fips": ["19001"], "other": [1]})
        with pytest.raises(KeyError, match="year"):
            validate_merge_keys(df, "test df")

    def test_passes_when_all_keys_present(self):
        df = pd.DataFrame({"county_fips": ["19001"], "year": [2020]})
        validate_merge_keys(df, "test df")  # should not raise


# ── report_missingness ────────────────────────────────────────────────────────

class TestReportMissingness:
    def test_returns_one_row_per_column(self):
        df = pd.DataFrame({"a": [1, None, 3], "b": [None, None, None], "c": [1, 2, 3]})
        miss = report_missingness(df)
        assert len(miss) == 3

    def test_missing_pct_accurate(self):
        df = pd.DataFrame({"a": [1.0, None, None, None]})
        miss = report_missingness(df)
        row = miss[miss["column"] == "a"].iloc[0]
        assert row["missing_pct"] == pytest.approx(75.0)

    def test_no_missing_all_zeros(self):
        df = pd.DataFrame({"x": [1, 2, 3]})
        miss = report_missingness(df)
        assert (miss["missing_count"] == 0).all()
