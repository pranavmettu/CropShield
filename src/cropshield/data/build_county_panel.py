"""
County panel builder for CropShield.

Merges yield targets, weather features, and (optionally) drought features
into a single county-crop-year modeling table ready for ML training.

Input files
-----------
- data/interim/yield_targets.csv    (from build_yield_targets())
- data/processed/weather_features.csv  (from compute_weather_features())
- data/interim/drought_features.csv    (optional, from Drought Monitor)

Output
------
- data/processed/modeling_panel.csv

Merge strategy
--------------
Yield targets drive the panel (left join). Weather and drought features
are joined on (county_fips, year). Rows with missing yield_anomaly
(counties with insufficient rolling history) are dropped before saving
the model-ready panel, but reported first.

Notes
-----
- county_fips is normalised to a zero-padded 5-character string before
  merging to handle floats (17001.0) stored in CSVs.
- A missingness report is always printed and saved to reports/.
- Never drop rows silently — every removal is counted and logged.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import numpy as np

from cropshield.features.panel_features import add_year_index, add_heat_stress_features

logger = logging.getLogger(__name__)

# Yield targets use (county_fips, year) as the weather join key;
# state and crop come from the yield side.
WEATHER_MERGE_KEYS = ["county_fips", "year"]
TARGET_COLUMN = "yield_anomaly"

TARGET_COLUMNS = [
    "actual_yield", "expected_yield",
    "yield_anomaly", "yield_anomaly_pct", "severe_risk",
]


def _normalise_fips(series: pd.Series) -> pd.Series:
    """Convert a county_fips Series to zero-padded 5-character strings.

    Handles floats (17001.0), ints (17001), and existing strings ('17001').
    NaN / None / empty strings are preserved as ``pd.NA`` rather than
    being silently converted to the invalid string ``"00nan"``.
    """
    s = series.astype(str).str.strip()
    missing_mask = s.isin({"nan", "None", "NaN", ""})
    result = (
        s.str.split(".").str[0]  # strip ".0" from float repr
        .str.strip()
        .str.zfill(5)
    )
    result = result.where(~missing_mask, other=pd.NA)
    return result


def build_modeling_panel(
    yield_path: str | Path = "data/interim/yield_targets.csv",
    weather_path: str | Path = "data/processed/weather_features.csv",
    drought_path: str | Path | None = None,
    output_path: str | Path = "data/processed/modeling_panel.csv",
    missingness_path: str | Path = "reports/missingness_report.csv",
    drop_missing_target: bool = True,
) -> pd.DataFrame:
    """Merge all feature sources into the county-crop-year modeling panel.

    Parameters
    ----------
    yield_path : str or Path
        Path to the yield targets CSV (output of ``build_yield_targets()``).
    weather_path : str or Path
        Path to aggregated weather features CSV.
    drought_path : str or Path, optional
        Path to aggregated drought features CSV. Skipped if ``None``.
    output_path : str or Path
        Destination for the final modeling panel CSV.
    missingness_path : str or Path
        Destination for the column-level missingness report CSV.
    drop_missing_target : bool
        Drop rows where ``yield_anomaly`` is NaN (insufficient rolling history).
        Set to ``False`` only for auditing — these rows cannot be used for modeling.

    Returns
    -------
    pd.DataFrame
        Merged modeling panel ready for ML training.

    Raises
    ------
    FileNotFoundError
        If yield_path or weather_path do not exist.
    KeyError
        If required merge key columns are absent from any input file.
    """
    # ── 1. Load yield targets ────────────────────────────────────────────────
    yield_path = Path(yield_path)
    if not yield_path.exists():
        raise FileNotFoundError(
            f"Yield targets not found at {yield_path}. "
            "Run scripts/02_build_features.py --targets-only first."
        )
    yields = pd.read_csv(yield_path)
    yields["county_fips"] = _normalise_fips(yields["county_fips"])
    yields["year"] = pd.to_numeric(yields["year"], errors="coerce").astype("Int64")
    logger.info("Loaded yield targets: %d rows", len(yields))

    # ── 2. Load weather features ─────────────────────────────────────────────
    weather_path = Path(weather_path)
    if not weather_path.exists():
        raise FileNotFoundError(
            f"Weather features not found at {weather_path}. "
            "Run scripts/01_fetch_data.py then scripts/02_build_features.py first."
        )
    weather = pd.read_csv(weather_path)
    weather["county_fips"] = _normalise_fips(weather["county_fips"])
    weather["year"] = pd.to_numeric(weather["year"], errors="coerce").astype("Int64")

    # Drop county/state columns that came from weather centroids (avoid collision
    # with the authoritative columns from yield targets)
    weather = weather.drop(
        columns=[c for c in ("county", "state", "state_fips") if c in weather.columns]
    )
    # Deduplicate weather on merge keys before joining; duplicates would fan out
    # the panel (1 yield row → N panel rows) without any exception being raised.
    n_weather_before = len(weather)
    weather = weather.drop_duplicates(subset=WEATHER_MERGE_KEYS, keep="first")
    if len(weather) < n_weather_before:
        logger.warning(
            "Dropped %d duplicate (county_fips, year) rows from weather features "
            "before merge to prevent row multiplication.",
            n_weather_before - len(weather),
        )
    logger.info("Loaded weather features: %d county-year rows", len(weather))
    validate_merge_keys(weather, "weather features", keys=WEATHER_MERGE_KEYS)

    # ── 3. Left-join weather onto yield targets ───────────────────────────────
    panel = yields.merge(weather, on=WEATHER_MERGE_KEYS, how="left")
    n_unmatched = panel[weather.columns.difference(WEATHER_MERGE_KEYS)].isnull().all(axis=1).sum()
    if n_unmatched:
        logger.warning(
            "%d yield rows had no matching weather record (%.1f%% of panel). "
            "Possible causes: county fetch failed, FIPS mismatch, or weather "
            "fetch not yet complete.",
            n_unmatched, 100 * n_unmatched / len(panel),
        )
    logger.info("After weather merge: %d rows", len(panel))

    # ── 4. Optionally merge drought features ─────────────────────────────────
    if drought_path is not None:
        drought_path = Path(drought_path)
        if drought_path.exists():
            drought = pd.read_csv(drought_path)
            drought["county_fips"] = _normalise_fips(drought["county_fips"])
            drought["year"] = pd.to_numeric(drought["year"], errors="coerce").astype("Int64")
            validate_merge_keys(drought, "drought features", keys=WEATHER_MERGE_KEYS)
            panel = panel.merge(drought, on=WEATHER_MERGE_KEYS, how="left")
            logger.info("After drought merge: %d rows", len(panel))
        else:
            logger.warning("Drought path provided but file not found: %s", drought_path)

    # ── 5. Drop stray metadata columns from source files ─────────────────────
    drop_cols = [c for c in ("unit",) if c in panel.columns]
    if drop_cols:
        panel = panel.drop(columns=drop_cols)
        logger.info("Dropped non-feature metadata columns: %s", drop_cols)

    # ── 6. Add panel-level derived features ──────────────────────────────────
    panel = add_year_index(panel)
    panel = add_heat_stress_features(panel)

    # ── 6. Report missingness (before dropping) ───────────────────────────────
    miss = report_missingness(panel, save_path=missingness_path)
    _print_missingness_summary(miss)

    # ── 7. Drop rows with missing target ─────────────────────────────────────
    n_before = len(panel)
    if drop_missing_target:
        panel = panel.dropna(subset=[TARGET_COLUMN])
        n_dropped = n_before - len(panel)
        if n_dropped:
            logger.info(
                "Dropped %d rows with missing %s (%.1f%%) — "
                "these are counties with insufficient rolling history.",
                n_dropped, TARGET_COLUMN, 100 * n_dropped / n_before,
            )

    # ── 8. Final sort and save ────────────────────────────────────────────────
    panel = panel.sort_values(["state", "county", "year", "crop"]).reset_index(drop=True)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_csv(output_path, index=False)
    logger.info(
        "Modeling panel saved → %s  (%d rows | %d counties | %d features)",
        output_path,
        len(panel),
        panel["county_fips"].nunique(),
        len([c for c in panel.columns if c not in TARGET_COLUMNS + ["year", "state", "county", "county_fips", "crop"]]),
    )
    return panel


def report_missingness(df: pd.DataFrame, save_path: str | Path | None = None) -> pd.DataFrame:
    """Compute and optionally save a column-level missingness summary.

    Parameters
    ----------
    df : pd.DataFrame
        The panel to audit.
    save_path : str or Path, optional
        If provided, saves the missingness table as a CSV.

    Returns
    -------
    pd.DataFrame
        Summary with columns: ``column``, ``missing_count``,
        ``missing_pct``, ``dtype``.
    """
    summary = pd.DataFrame({
        "column":        df.columns,
        "missing_count": df.isnull().sum().values,
        "missing_pct":   (df.isnull().mean() * 100).round(2).values,
        "dtype":         [str(d) for d in df.dtypes.values],
    }).sort_values("missing_pct", ascending=False).reset_index(drop=True)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(save_path, index=False)
        logger.info("Missingness report saved → %s", save_path)
    return summary


def validate_merge_keys(
    df: pd.DataFrame,
    name: str,
    keys: list[str] = WEATHER_MERGE_KEYS,
) -> None:
    """Raise an informative error if any merge key columns are missing.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to check.
    name : str
        Human-readable label for error messages.
    keys : list[str]
        Required column names.

    Raises
    ------
    KeyError
        If one or more merge key columns are absent.
    """
    missing = [k for k in keys if k not in df.columns]
    if missing:
        raise KeyError(
            f"{name} is missing merge key columns: {missing}. "
            f"Found columns: {df.columns.tolist()}"
        )


def _print_missingness_summary(miss: pd.DataFrame) -> None:
    """Print a concise missingness table to stdout."""
    problematic = miss[miss["missing_pct"] > 0]
    if problematic.empty:
        logger.info("No missing values in panel.")
        return
    print("\n── Missingness Report ────────────────────────────────────────")
    print(f"{'Column':<35} {'Missing':>8} {'Missing %':>10}")
    print("─" * 57)
    for _, row in problematic.iterrows():
        print(f"  {row['column']:<33} {int(row['missing_count']):>8,} {row['missing_pct']:>9.1f}%")
    print()
