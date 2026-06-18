"""
Panel audit utilities for CropShield baseline modeling.

Validates schema, uniqueness, and produces a concise data-quality report
before model training.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from cropshield.features.weather_features import FULL_GROWING_SEASON_DAYS

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = {
    "county_fips", "state", "county", "crop", "year",
    "actual_yield", "expected_yield", "yield_anomaly",
}


def add_checkpoint_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add a ``checkpoint`` column describing growing-season completeness.

    ``full_season`` — complete April–August weather (``obs_days >= 153``).
    ``partial``     — incomplete growing-season observations.
    """
    df = df.copy()
    if "checkpoint" in df.columns:
        return df
    if "is_partial_year" in df.columns:
        df["checkpoint"] = df["is_partial_year"].map({True: "partial", False: "full_season"})
    elif "obs_days" in df.columns:
        df["checkpoint"] = df["obs_days"].apply(
            lambda d: "full_season" if pd.notna(d) and d >= FULL_GROWING_SEASON_DAYS else "partial"
        )
    else:
        df["checkpoint"] = "full_season"
    return df


def audit_panel(df: pd.DataFrame) -> dict:
    """Run schema and quality checks; return an audit summary dict."""
    df = add_checkpoint_column(df)
    missing_required = REQUIRED_COLUMNS - set(df.columns)
    if missing_required:
        raise KeyError(f"Panel missing required columns: {sorted(missing_required)}")

    dupes = df.duplicated(subset=["county_fips", "crop", "year", "checkpoint"]).sum()
    miss = df.isnull().mean().mul(100).round(2).to_dict()

    target_by_crop = (
        df.groupby("crop")["yield_anomaly"]
        .agg(["count", "mean", "std"])
        .round(3)
        .to_dict("index")
    )
    target_by_checkpoint = (
        df.dropna(subset=["checkpoint"]).groupby("checkpoint")["yield_anomaly"]
        .agg(["count", "mean", "std"])
        .round(3)
        .to_dict("index")
    )

    summary = {
        "row_count": len(df),
        "n_counties": int(df["county_fips"].nunique()),
        "n_states": int(df["state"].nunique()),
        "states": sorted(df["state"].unique()),
        "crops": sorted(df["crop"].unique()),
        "years": sorted(int(y) for y in df["year"].unique()),
        "checkpoints": sorted(df["checkpoint"].dropna().unique()),
        "duplicate_rows": int(dupes),
        "missingness_pct": miss,
        "target_by_crop": target_by_crop,
        "target_by_checkpoint": target_by_checkpoint,
    }
    return summary


def format_audit_markdown(summary: dict) -> str:
    """Render audit summary as markdown."""
    lines = [
        "# Modeling Panel Audit",
        "",
        f"- **Rows:** {summary['row_count']:,}",
        f"- **Counties:** {summary['n_counties']}",
        f"- **States:** {summary['n_states']} ({', '.join(summary['states'])})",
        f"- **Crops:** {', '.join(summary['crops'])}",
        f"- **Years:** {summary['years'][0]}–{summary['years'][-1]}",
        f"- **Checkpoints:** {', '.join(summary['checkpoints'])}",
        f"- **Duplicate (county_fips, crop, year, checkpoint) rows:** {summary['duplicate_rows']}",
        "",
        "## Missingness (%)",
        "",
    ]
    for col, pct in sorted(summary["missingness_pct"].items(), key=lambda x: -x[1]):
        if pct > 0:
            lines.append(f"- `{col}`: {pct}%")
    if not any(v > 0 for v in summary["missingness_pct"].values()):
        lines.append("- No missing values.")
    lines += ["", "## Target distribution by crop", ""]
    for crop, stats in summary["target_by_crop"].items():
        lines.append(f"- **{crop}**: n={stats['count']}, mean={stats['mean']}, std={stats['std']}")
    lines += ["", "## Target distribution by checkpoint", ""]
    for ckpt, stats in summary["target_by_checkpoint"].items():
        lines.append(f"- **{ckpt}**: n={stats['count']}, mean={stats['mean']}, std={stats['std']}")
    lines.append("")
    return "\n".join(lines)


def save_panel_audit(
    df: pd.DataFrame,
    output_path: str | Path = "reports/panel_audit.md",
) -> dict:
    """Audit the panel and save a markdown report."""
    summary = audit_panel(df)
    md = format_audit_markdown(summary)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(md)
    logger.info("Panel audit saved → %s", output_path)
    return summary
