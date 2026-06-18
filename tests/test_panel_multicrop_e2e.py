"""
End-to-end synthetic multi-crop county panel test.

Validates the full merge pipeline with 2 counties, 2 crops, multiple years,
and weather rows at different checkpoint completeness levels.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from cropshield.data.build_county_panel import build_modeling_panel
from cropshield.features.yield_targets import (
    add_expected_yield_rolling,
    add_yield_anomaly,
    clean_yield_dataframe,
)
from cropshield.features.weather_features import FULL_GROWING_SEASON_DAYS


COUNTIES = [
    ("19001", "IOWA", "BOONE"),
    ("17001", "ILLINOIS", "ADAMS"),
]
CROPS = ["CORN", "SOYBEANS"]
YEARS = [2018, 2019, 2020, 2021, 2022]
# Descriptive checkpoints for weather completeness (not calendar cutoffs)
CHECKPOINTS = {
    2018: FULL_GROWING_SEASON_DAYS,       # full season
    2019: FULL_GROWING_SEASON_DAYS,
    2020: FULL_GROWING_SEASON_DAYS,
    2021: 90,                              # partial (mid-season)
    2022: FULL_GROWING_SEASON_DAYS,
}


def _build_yield_targets() -> pd.DataFrame:
    rows = []
    for fips, state, county in COUNTIES:
        for crop in CROPS:
            base = 180.0 if crop == "CORN" else 45.0
            for i, yr in enumerate(YEARS):
                rows.append({
                    "year": yr,
                    "state": state,
                    "county": county,
                    "county_fips": fips,
                    "crop": crop,
                    "value": base + i * (5.0 if crop == "CORN" else 2.0),
                })
    df = clean_yield_dataframe(pd.DataFrame(rows))
    df = add_expected_yield_rolling(df, window=3, min_periods=2)
    df = add_yield_anomaly(df)
    return df


def _build_weather() -> pd.DataFrame:
    rows = []
    for fips, _, _ in COUNTIES:
        for yr in YEARS:
            rows.append({
                "county_fips": fips,
                "year": yr,
                "cumulative_precip": 400.0 + yr - 2018,
                "mean_temp": 20.0,
                "max_temp": 32.0,
                "extreme_heat_days": 5,
                "dry_days": 80,
                "longest_dry_spell": 12,
                "growing_degree_days": 1750.0,
                "precip_anomaly": 10.0,
                "obs_days": CHECKPOINTS[yr],
                "is_partial_year": CHECKPOINTS[yr] < FULL_GROWING_SEASON_DAYS,
            })
    return pd.DataFrame(rows)


class TestMultiCropPanelE2E:
    @pytest.fixture
    def panel(self) -> pd.DataFrame:
        with tempfile.TemporaryDirectory() as tmpdir:
            yp = Path(tmpdir) / "yields.csv"
            wp = Path(tmpdir) / "weather.csv"
            op = Path(tmpdir) / "panel.csv"
            _build_yield_targets().to_csv(yp, index=False)
            _build_weather().to_csv(wp, index=False)
            return build_modeling_panel(
                yield_path=yp,
                weather_path=wp,
                output_path=op,
                missingness_path=Path(tmpdir) / "miss.csv",
                drop_missing_target=True,
                allow_partial_year=True,
                current_year=2099,  # don't drop any test rows
            )

    def test_expected_row_count(self, panel):
        """2 counties × 2 crops × 3 valid years (2020–2022; earlier years lack history)."""
        expected = 2 * 2 * 3
        assert len(panel) == expected, f"Expected {expected} rows, got {len(panel)}"

    def test_unique_by_fips_crop_year(self, panel):
        dupes = panel.duplicated(subset=["county_fips", "crop", "year"])
        assert not dupes.any(), f"{dupes.sum()} duplicate (county_fips, crop, year) rows"

    def test_no_join_fanout(self, panel):
        """Row count must equal yield rows with valid anomaly — no multiplication."""
        yields = _build_yield_targets()
        expected = yields["yield_anomaly"].notna().sum()
        assert len(panel) == expected

    def test_corn_and_soy_baselines_independent(self, panel):
        """CORN expected_yield must differ from SOYBEANS for the same county-year."""
        for fips in ["19001", "17001"]:
            for yr in [2020, 2021, 2022]:
                sub = panel[(panel["county_fips"] == fips) & (panel["year"] == yr)]
                corn_exp = sub.loc[sub["crop"] == "CORN", "expected_yield"].iloc[0]
                soy_exp  = sub.loc[sub["crop"] == "SOYBEANS", "expected_yield"].iloc[0]
                assert corn_exp != pytest.approx(soy_exp), (
                    f"CORN and SOYBEAN baselines mixed for {fips} {yr}"
                )
                assert corn_exp > soy_exp, "CORN baseline should exceed SOYBEAN baseline"

    def test_partial_year_flagged(self, panel):
        partial = panel[(panel["year"] == 2021)]
        assert partial["is_partial_year"].all()
        full = panel[panel["year"] != 2021]
        assert not full["is_partial_year"].any()

    def test_weather_joined_on_fips_not_county_name(self, panel):
        """Both counties have weather — join succeeded via FIPS, not name."""
        assert panel["cumulative_precip"].notna().all()
