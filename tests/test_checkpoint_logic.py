"""
NASA POWER checkpoint and resume tests for CropShield.

The checkpoint mechanism must satisfy two correctness properties:

  1. Cumulative persistence: every county fetched so far is saved to disk,
     not just the latest batch.
  2. Year-range awareness: a county is only considered "complete" if the
     checkpoint contains data for EVERY year in the requested range.
     Checking only county_fips would cause a silent data gap if the year
     range is extended between runs (e.g. 2015-2024 → 2015-2025).

Tests here use unittest.mock to avoid live API calls.  They verify the
checkpoint contract directly without network access.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from cropshield.data.fetch_power import _save_checkpoint, fetch_power_all_counties


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_centroids(fips_list: list[str]) -> pd.DataFrame:
    """Minimal county_centroids DataFrame for a list of FIPS codes."""
    return pd.DataFrame({
        "county_fips": fips_list,
        "county":      [f"COUNTY_{f}" for f in fips_list],
        "state_fips":  [f[:2] for f in fips_list],
        "lat":         [42.0] * len(fips_list),
        "lon":         [-93.0] * len(fips_list),
    })


def _make_county_weather(
    fips: str,
    years: list[int],
    precip: float = 5.0,
) -> pd.DataFrame:
    """Synthetic daily weather DataFrame for one county across multiple years."""
    frames = []
    for yr in years:
        dates = pd.date_range(f"{yr}-04-01", f"{yr}-08-31", freq="D")
        frames.append(pd.DataFrame({
            "county_fips":  fips,
            "state_fips":   fips[:2],
            "county":       f"COUNTY_{fips}",
            "date":         dates,
            "year":         yr,
            "month":        dates.month,
            "PRECTOTCORR":  precip,
            "T2M":          20.0,
            "T2M_MIN":      15.0,
            "T2M_MAX":      25.0,
            "lat":          42.0,
            "lon":          -93.0,
        }))
    return pd.concat(frames, ignore_index=True)


# ── _save_checkpoint ──────────────────────────────────────────────────────────

class TestSaveCheckpoint:
    def test_writes_all_accumulated_data(self):
        """Checkpoint must contain ALL counties accumulated so far, not just latest."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ckpt.csv"

            batch1 = [_make_county_weather("17001", [2020, 2021])]
            _save_checkpoint(batch1, [], path)
            # Read with dtype=str to avoid int64 inference of numeric FIPS
            ckpt1 = pd.read_csv(path, dtype={"county_fips": str})
            assert set(ckpt1["county_fips"].unique()) == {"17001"}

            # Append second county — checkpoint must have BOTH
            batch2 = batch1 + [_make_county_weather("19001", [2020, 2021])]
            _save_checkpoint(batch2, [], path)
            ckpt2 = pd.read_csv(path, dtype={"county_fips": str})
            assert set(ckpt2["county_fips"].unique()) == {"17001", "19001"}, (
                "Second checkpoint overwrote first county's data"
            )

    def test_row_count_grows_monotonically(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ckpt.csv"
            data: list[pd.DataFrame] = []
            for fips in ["17001", "17003", "17005"]:
                data.append(_make_county_weather(fips, [2020, 2021]))
                _save_checkpoint(data, [], path)
                saved = pd.read_csv(path)
                n_counties = saved["county_fips"].nunique()
                assert n_counties == len(data), (
                    f"Expected {len(data)} counties after save, got {n_counties}"
                )


# ── fetch_power_all_counties resume logic ────────────────────────────────────

class TestCheckpointResume:
    """Tests that fetch_power_all_counties correctly skips completed counties."""

    def _make_fetch_side_effect(self, county_frames: dict[str, pd.DataFrame]):
        """Return a mock for fetch_power_for_county keyed by (lat, lon)."""
        # Map (lat, lon) → DataFrame (strip county/fips columns added by caller)
        lat_lon_map = {}
        for fips, df in county_frames.items():
            lat = df["lat"].iloc[0]
            lon = df["lon"].iloc[0]
            # Return only the weather columns (without county/fips identifiers)
            lat_lon_map[(lat, lon)] = df.drop(
                columns=[c for c in ("county_fips", "state_fips", "county") if c in df.columns]
            )

        def _fetch(lat, lon, start_year, end_year=None, **kw):
            return lat_lon_map.get((lat, lon), pd.DataFrame())

        return _fetch

    def test_completed_county_not_re_fetched(self):
        """A county already in the checkpoint must not trigger a new API call."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "weather.csv"
            checkpoint = output.with_suffix(".checkpoint.csv")

            # Pre-populate checkpoint with county 17001 for years 2015-2016
            existing = _make_county_weather("17001", [2015, 2016])
            existing.to_csv(checkpoint, index=False)

            centroids = _make_centroids(["17001", "17003"])
            call_count = {"n": 0}

            def mock_fetch(lat, lon, start_year, end_year=None, **kw):
                call_count["n"] += 1
                fips = "17003"  # only 17003 should be fetched
                years = list(range(start_year, (end_year or 2016) + 1))
                return _make_county_weather(fips, years).drop(
                    columns=["county_fips", "state_fips", "county"]
                )

            with patch(
                "cropshield.data.fetch_power.fetch_power_for_county",
                side_effect=mock_fetch,
            ):
                fetch_power_all_counties(
                    county_centroids=centroids,
                    start_year=2015,
                    end_year=2016,
                    output_raw=str(output),
                    checkpoint_every=100,
                    request_delay=0.0,
                )

            assert call_count["n"] == 1, (
                f"Expected 1 API call (only 17003), but got {call_count['n']}. "
                "County 17001 should have been skipped from checkpoint."
            )

    def test_checkpoint_data_preserved_in_output(self):
        """Data from checkpoint must appear in the final output file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "weather.csv"
            checkpoint = output.with_suffix(".checkpoint.csv")

            checkpoint_data = _make_county_weather("17001", [2015, 2016])
            checkpoint_data.to_csv(checkpoint, index=False)

            centroids = _make_centroids(["17001", "17003"])

            def mock_fetch(lat, lon, start_year, end_year=None, **kw):
                years = list(range(start_year, (end_year or 2016) + 1))
                return _make_county_weather("17003", years).drop(
                    columns=["county_fips", "state_fips", "county"]
                )

            with patch(
                "cropshield.data.fetch_power.fetch_power_for_county",
                side_effect=mock_fetch,
            ):
                result = fetch_power_all_counties(
                    county_centroids=centroids,
                    start_year=2015,
                    end_year=2016,
                    output_raw=str(output),
                    checkpoint_every=100,
                    request_delay=0.0,
                )

            fips_in_output = set(result["county_fips"].unique())
            assert "17001" in fips_in_output, "Checkpoint county missing from output"
            assert "17003" in fips_in_output, "Newly fetched county missing from output"

    def test_checkpoint_deleted_after_full_success(self):
        """Checkpoint file must be removed once the final output is written."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "weather.csv"
            checkpoint = output.with_suffix(".checkpoint.csv")

            centroids = _make_centroids(["17001"])

            def mock_fetch(lat, lon, start_year, end_year=None, **kw):
                years = list(range(start_year, (end_year or 2016) + 1))
                return _make_county_weather("17001", years).drop(
                    columns=["county_fips", "state_fips", "county"]
                )

            with patch(
                "cropshield.data.fetch_power.fetch_power_for_county",
                side_effect=mock_fetch,
            ):
                fetch_power_all_counties(
                    county_centroids=centroids,
                    start_year=2015,
                    end_year=2016,
                    output_raw=str(output),
                    checkpoint_every=100,
                    request_delay=0.0,
                )

            assert not checkpoint.exists(), (
                "Checkpoint file was not deleted after successful completion"
            )


# ── REGRESSION: county-level checkpoint skipping vs. year-range awareness ─────

class TestCheckpointYearRangeRegression:
    """
    Regression tests for the year-range awareness bug.

    BUG (pre-fix): completed_fips was built from county_fips alone.
    If a county was fetched for 2015-2024, then the next run requested
    2015-2025 (new year added), the county would be incorrectly skipped
    because its FIPS was in the checkpoint — silently leaving year 2025
    missing for that county.

    FIX: A county is only considered 'complete' if the checkpoint contains
    data for every year in [start_year, end_year].
    """

    def test_county_with_partial_years_is_refetched(self):
        """A county in checkpoint covering 2015-2016 must be re-fetched if end_year=2017."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "weather.csv"
            checkpoint = output.with_suffix(".checkpoint.csv")

            # Checkpoint only has years 2015-2016 for county 17001
            partial = _make_county_weather("17001", [2015, 2016])
            partial.to_csv(checkpoint, index=False)

            centroids = _make_centroids(["17001"])
            fetched_fips = []

            def mock_fetch(lat, lon, start_year, end_year=None, **kw):
                fetched_fips.append("17001")
                years = list(range(start_year, (end_year or 2017) + 1))
                return _make_county_weather("17001", years).drop(
                    columns=["county_fips", "state_fips", "county"]
                )

            with patch(
                "cropshield.data.fetch_power.fetch_power_for_county",
                side_effect=mock_fetch,
            ):
                fetch_power_all_counties(
                    county_centroids=centroids,
                    start_year=2015,
                    end_year=2017,          # year 2017 not in checkpoint!
                    output_raw=str(output),
                    checkpoint_every=100,
                    request_delay=0.0,
                )

            assert len(fetched_fips) == 1, (
                "County 17001 should have been re-fetched because year 2017 "
                "was missing from the checkpoint. "
                "This would silently FAIL with the old county-only check. "
                f"fetch calls: {fetched_fips}"
            )

    def test_county_with_all_years_is_skipped(self):
        """A county with all requested years in checkpoint must NOT be re-fetched."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "weather.csv"
            checkpoint = output.with_suffix(".checkpoint.csv")

            full = _make_county_weather("17001", [2015, 2016, 2017])
            full.to_csv(checkpoint, index=False)

            centroids = _make_centroids(["17001"])
            call_count = {"n": 0}

            def mock_fetch(*a, **kw):
                call_count["n"] += 1
                return pd.DataFrame()

            with patch(
                "cropshield.data.fetch_power.fetch_power_for_county",
                side_effect=mock_fetch,
            ):
                fetch_power_all_counties(
                    county_centroids=centroids,
                    start_year=2015,
                    end_year=2017,
                    output_raw=str(output),
                    checkpoint_every=100,
                    request_delay=0.0,
                )

            assert call_count["n"] == 0, (
                "County with all required years in checkpoint should be skipped"
            )
