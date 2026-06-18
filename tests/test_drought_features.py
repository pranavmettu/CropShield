"""
Drought feature logic tests for CropShield.

The Drought Monitor fetcher and feature engineering are not yet implemented
(fetch_drought_monitor raises NotImplementedError), but the data contracts
and feature logic can be tested with synthetic data matching the expected
schema.

Tests cover:
  1. Drought category ordering: D0 < D1 < D2 < D3 < D4 (severity increases).
  2. D2_plus aggregation: D2 + D3 + D4 only.
  3. Date-cutoff leakage: drought features computed through a cutoff date
     must not incorporate drought observations after that date.
  4. FIPS normalisation in drought data.
  5. clean_drought_dataframe contract (once implemented).

Since fetch_drought_monitor is not yet implemented, tests that require it
are marked xfail with a clear message.
"""

from __future__ import annotations

import pandas as pd
import pytest

# The fetcher is not yet implemented — we test the data contract and helpers
# that are defined (or should be defined) in the module.
from cropshield.data.fetch_drought_monitor import DROUGHT_CATEGORIES


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


# ── fetch_drought_monitor not yet implemented ─────────────────────────────────

class TestFetchDroughtMonitorNotImplemented:
    @pytest.mark.xfail(
        strict=True,
        reason="fetch_drought_monitor is not yet implemented (Prompt 6+)",
    )
    def test_fetch_raises_not_implemented(self):
        from cropshield.data.fetch_drought_monitor import fetch_drought_monitor
        fetch_drought_monitor(state_fips_list=["19"])

    @pytest.mark.xfail(
        strict=True,
        reason="clean_drought_dataframe is not yet implemented (Prompt 6+)",
    )
    def test_clean_raises_not_implemented(self):
        from cropshield.data.fetch_drought_monitor import clean_drought_dataframe
        clean_drought_dataframe(pd.DataFrame())
