"""
County panel builder for CropShield.

Merges yield targets, weather features, and drought features into a single
county-crop-year modeling table ready for ML training.

Input files (expected in data/processed/ or data/interim/)
-----------------------------------------------------------
- data/interim/nass_yield_clean.csv
- data/processed/weather_features.csv
- data/interim/drought_features.csv  (optional for MVP)

Output
------
- data/processed/modeling_panel.csv

Notes
-----
- Merge key: (year, state, county_fips, crop)
- Rows with missing targets (yield_anomaly) are dropped with a warning.
- Missingness across all feature columns is reported before the model-ready
  panel is saved. Do not silently discard data.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

MERGE_KEYS = ["year", "state", "county_fips", "crop"]
TARGET_COLUMNS = ["actual_yield", "expected_yield", "yield_anomaly", "yield_anomaly_pct", "severe_risk"]


def build_modeling_panel(
    yield_path: str | Path = "data/interim/nass_yield_clean.csv",
    weather_path: str | Path = "data/processed/weather_features.csv",
    drought_path: str | Path | None = None,
    output_path: str | Path = "data/processed/modeling_panel.csv",
    drop_missing_target: bool = True,
) -> pd.DataFrame:
    """Merge all feature sources into the county-crop-year modeling panel.

    Parameters
    ----------
    yield_path : str or Path
        Path to cleaned NASS yield file with target columns.
    weather_path : str or Path
        Path to aggregated weather features.
    drought_path : str or Path, optional
        Path to aggregated drought features. If ``None``, drought features
        are omitted (valid for early MVP).
    output_path : str or Path
        Destination for the final modeling panel CSV.
    drop_missing_target : bool
        Whether to drop rows where ``yield_anomaly`` is NaN. Always ``True``
        for modeling; set to ``False`` for data auditing.

    Returns
    -------
    pd.DataFrame
        Merged modeling panel.
    """
    # TODO: Implement panel assembly
    # Steps:
    # 1. Load each input file
    # 2. Validate that each file has the expected merge key columns
    # 3. Merge yield + weather on MERGE_KEYS (left join from yield side)
    # 4. Optionally merge drought features
    # 5. Call report_missingness() to log/save a missingness summary
    # 6. Drop rows with missing target if drop_missing_target=True
    # 7. Log how many rows were dropped and why
    # 8. Save to output_path
    # 9. Return the final DataFrame
    raise NotImplementedError("build_modeling_panel is not yet implemented. See Prompt 5.")


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
        Summary DataFrame with columns: ``column``, ``missing_count``,
        ``missing_pct``, ``dtype``.
    """
    # TODO: Implement missingness report
    summary = pd.DataFrame({
        "column": df.columns,
        "missing_count": df.isnull().sum().values,
        "missing_pct": (df.isnull().mean() * 100).round(2).values,
        "dtype": df.dtypes.values,
    })
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(save_path, index=False)
        logger.info("Missingness report saved to %s", save_path)
    return summary


def validate_merge_keys(df: pd.DataFrame, name: str, keys: list[str] = MERGE_KEYS) -> None:
    """Raise an informative error if any expected merge keys are missing from a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to check.
    name : str
        Human-readable name for error messages (e.g. ``"weather features"``).
    keys : list[str]
        Required column names.

    Raises
    ------
    KeyError
        If one or more merge key columns are absent.
    """
    missing = [k for k in keys if k not in df.columns]
    if missing:
        raise KeyError(f"{name} is missing merge key columns: {missing}. Found: {df.columns.tolist()}")
