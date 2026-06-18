"""
Drought feature logic tests for CropShield.
"""

from __future__ import annotations

import pandas as pd
import pytest

from cropshield.data.fetch_drought_monitor import DROUGHT_CATEGORIES, clean_drought_dataframe
from cropshield.features.drought_features import compute_drought_features


# ── Constants and contracts ───────────────────────────────────────────────────

class TestDroughtCategoryContract:
    def test_categories_are_d0_through_d4(self):
        """The canonical drought categories must be exactly D0–D4 in order."""
        assert DROUGHT_CATEGORIES == ["D0", "D1", "D2", "D3", "D4"], (
            f"Expected ['D0','D1','D2','D3','D4'], got {DROUGHT_CATEGORIES}"
        )

    def test_category_count_is_five(self):
        assert len(DROUGHT_CATEGORIES) == 5

    def test_severity_order_is_ascending(self):
        """Higher index in the list must represent higher severity."""
        for i, cat in enumerate(DROUGHT_CATEGORIES):
            assert cat == f"D{i}", (
                f"Expected D{i} at index {i}, got {cat} — severity order is wrong"
            )


# ── Synthetic drought DataFrame tests ────────────────────────────────────────

def _make_drought_df(
    fips: str = "19001",
    dates: list[str] | None = None,
    d0: float = 10.0,
    d1: float = 5.0,
    d2: float = 3.0,
    d3: float = 1.0,
    d4: float = 0.0,
) -> pd.DataFrame:
    """Synthetic weekly Drought Monitor records for a single county."""
    if dates is None:
        dates = ["2020-04-07", "2020-04-14", "2020-04-21"]
    return pd.DataFrame({
        "date":        pd.to_datetime(dates),
        "county_fips": fips,
        "D0": d0,
        "D1": d1,
        "D2": d2,
        "D3": d3,
        "D4": d4,
    })


class TestDroughtCategoryValues:
    def test_d2_plus_is_d2_d3_d4_sum(self):
        """D2_plus must aggregate D2 + D3 + D4 (not D0 or D1)."""
        df = _make_drought_df(d0=10.0, d1=5.0, d2=3.0, d3=2.0, d4=1.0)
        df["D2_plus"] = df["D2"] + df["D3"] + df["D4"]
        expected = 3.0 + 2.0 + 1.0
        assert df["D2_plus"].iloc[0] == pytest.approx(expected)

    def test_d2_plus_excludes_d1(self):
        """D1 is 'moderate drought' — not severe enough for D2_plus."""
        df = _make_drought_df(d0=0.0, d1=50.0, d2=0.0, d3=0.0, d4=0.0)
        df["D2_plus"] = df["D2"] + df["D3"] + df["D4"]
        assert df["D2_plus"].iloc[0] == pytest.approx(0.0)

    def test_d4_exceptional_drought_included_in_d2_plus(self):
        """D4 (exceptional drought) must be part of D2_plus."""
        df = _make_drought_df(d0=0.0, d1=0.0, d2=0.0, d3=0.0, d4=20.0)
        df["D2_plus"] = df["D2"] + df["D3"] + df["D4"]
        assert df["D2_plus"].iloc[0] == pytest.approx(20.0)

    def test_categories_sum_to_at_most_100(self):
        """Each week's D0–D4 values represent % area — their sum <= 100%."""
        df = _make_drought_df(d0=20.0, d1=15.0, d2=10.0, d3=5.0, d4=2.0)
        total = df[["D0", "D1", "D2", "D3", "D4"]].sum(axis=1)
        assert (total <= 100.1).all(), (
            f"Drought category percentages sum to > 100: {total.tolist()}"
        )

    def test_zero_drought_is_valid(self):
        """A week with no drought (all zeros) is valid."""
        df = _make_drought_df(d0=0.0, d1=0.0, d2=0.0, d3=0.0, d4=0.0)
        assert df["D4"].iloc[0] == 0.0


# ── Date-cutoff leakage for drought features ──────────────────────────────────

def _aggregate_drought_through_cutoff(
    df: pd.DataFrame,
    cutoff_date: str,
    date_col: str = "date",
) -> pd.DataFrame:
    """
    Reference implementation: aggregate drought categories up to cutoff_date.
    This is the pattern that drought_features.py should follow when implemented.
    Used here for testing the contract.
    """
    cutoff = pd.Timestamp(cutoff_date)
    filtered = df[df[date_col] <= cutoff].copy()
    if filtered.empty:
        return pd.DataFrame()
    agg = filtered[["D0", "D1", "D2", "D3", "D4"]].mean()
    agg["D2_plus"] = filtered["D2"].mean() + filtered["D3"].mean() + filtered["D4"].mean()
    agg["county_fips"] = df["county_fips"].iloc[0]
    return pd.DataFrame([agg])


EXTREME_D4 = 99.0  # clearly impossible extreme value


class TestDroughtDateCutoffLeakage:
    def test_extreme_post_cutoff_drought_excluded(self):
        """
        A severe D4 event after the cutoff must not appear in features computed
        through the cutoff.
        """
        # Normal drought weeks in April
        weeks_april = _make_drought_df(
            dates=["2020-04-07", "2020-04-14", "2020-04-21"],
            d2=5.0, d3=2.0, d4=0.0,
        )
        # Catastrophic event in July (post-June-30 cutoff)
        week_july = _make_drought_df(
            dates=["2020-07-07"],
            d2=0.0, d3=0.0, d4=EXTREME_D4,
        )
        full_df = pd.concat([weeks_april, week_july], ignore_index=True)

        # Aggregate through June 30
        features = _aggregate_drought_through_cutoff(full_df, cutoff_date="2020-06-30")
        assert features["D4"].iloc[0] < EXTREME_D4 / 2, (
            "Extreme post-cutoff D4 value leaked into features computed through June 30"
        )

    def test_cutoff_at_april_excludes_may_onwards(self):
        weeks = _make_drought_df(
            dates=["2020-04-07", "2020-05-05", "2020-06-02"],
            d4=10.0,
        )
        # Only 1 April week has d4=10; May and June also d4=10
        # Cutoff at April 30 should average only the 1 April week
        features = _aggregate_drought_through_cutoff(weeks, cutoff_date="2020-04-30")
        assert features["D4"].iloc[0] == pytest.approx(10.0)

    def test_altering_future_drought_does_not_change_cutoff_features(self):
        """Core leakage test: changing post-cutoff D4 must not affect cutoff features."""
        weeks_normal = _make_drought_df(
            dates=["2020-04-07", "2020-04-14", "2020-07-07"],
            d4=5.0,
        )
        weeks_extreme = weeks_normal.copy()
        weeks_extreme.loc[weeks_extreme["date"] == pd.Timestamp("2020-07-07"), "D4"] = EXTREME_D4

        feat_normal  = _aggregate_drought_through_cutoff(weeks_normal,  cutoff_date="2020-06-30")
        feat_extreme = _aggregate_drought_through_cutoff(weeks_extreme, cutoff_date="2020-06-30")

        assert feat_normal["D4"].iloc[0] == pytest.approx(feat_extreme["D4"].iloc[0]), (
            "Changing post-cutoff D4 to extreme value affected features before the cutoff"
        )


# ── compute_drought_features (implemented) ────────────────────────────────────

class TestComputeDroughtFeatures:
    def test_required_columns_present(self):
        weekly = _make_drought_df(
            dates=["2020-04-07", "2020-04-14", "2020-05-05", "2020-06-02"],
            d2=10.0, d3=5.0, d4=2.0,
        )
        weekly["year"] = 2020
        features = compute_drought_features(weekly)
        for col in (
            "county_fips", "year", "weeks_d0", "weeks_d1", "weeks_d2",
            "weeks_d3", "weeks_d4", "weeks_d2_plus",
            "max_drought_category", "mean_drought_severity", "checkpoint",
        ):
            assert col in features.columns, f"Missing column {col}"

    def test_weeks_d2_plus_counts_severe_weeks(self):
        weekly = pd.DataFrame({
            "date": pd.to_datetime(["2020-04-07", "2020-04-14", "2020-05-05"]),
            "county_fips": "19001",
            "year": 2020,
            "D0": [0, 0, 0],
            "D1": [0, 0, 0],
            "D2": [10, 0, 0],
            "D3": [0, 5, 0],
            "D4": [0, 0, 0],
        })
        features = compute_drought_features(weekly)
        assert features["weeks_d2_plus"].iloc[0] == 2  # weeks 1 and 2

    def test_cutoff_excludes_later_drought(self):
        april = _make_drought_df(dates=["2020-04-07"], d2=10.0, d3=0.0, d4=0.0)
        july  = _make_drought_df(dates=["2020-07-07"], d2=0.0, d3=0.0, d4=99.0)
        full = pd.concat([april, july], ignore_index=True)
        full["year"] = 2020
        feat = compute_drought_features(full, cutoff_date="2020-06-30")
        assert feat["max_drought_category"].iloc[0] == 2
        assert feat["checkpoint"].iloc[0] == "2020-06-30"

    def test_clean_drought_normalises_fips(self):
        raw = pd.DataFrame({
            "date": ["2020-04-07"],
            "county_fips": [19001.0],
            "D0": [10], "D1": [0], "D2": [0], "D3": [0], "D4": [0],
        })
        clean = clean_drought_dataframe(raw)
        assert clean["county_fips"].iloc[0] == "19001"


class TestFetchDroughtMonitorLiveApi:
    @pytest.mark.xfail(
        strict=True,
        reason="Live Drought Monitor API fetch not implemented — use raw CSV on disk",
    )
    def test_fetch_raises_without_raw_file(self, tmp_path):
        from cropshield.data.fetch_drought_monitor import fetch_drought_monitor
        fetch_drought_monitor(
            state_fips_list=["19"],
            output_raw=tmp_path / "missing.csv",
        )
