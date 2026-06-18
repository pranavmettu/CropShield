"""
Script 02: Build the feature engineering pipeline.

Orchestrates:
  1. Yield targets (expected yield, anomaly, risk class)
  2. Weather growing-season features (if raw weather data is available)
  3. Drought growing-season features (placeholder)
  4. Final merged modeling panel

Usage
-----
    python scripts/02_build_features.py
    python scripts/02_build_features.py --method trend --window 5
    python scripts/02_build_features.py --targets-only   # skip weather merge

Run from the project root directory.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd

from cropshield.features.yield_targets import build_yield_targets
from cropshield.features.weather_features import (
    compute_weather_features,
    compute_multi_checkpoint_weather_features,
    compute_checkpoint_stress_features,
    save_weather_features,
    filter_incomplete_current_year,
    CHECKPOINT_CONFIGS,
    STRESS_FEATURE_COLUMNS,
)
from cropshield.features.weather_anomalies import add_weather_anomalies, weather_anomaly_columns
from cropshield.data.build_county_panel import build_modeling_panel
from cropshield.data.fips_utils import load_nass_yield_csv
from cropshield.features.panel_features import get_feature_columns
from cropshield.evaluation.panel_audit import save_panel_audit

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build CropShield feature engineering pipeline."
    )
    parser.add_argument(
        "--method", choices=["rolling", "trend"], default="rolling",
        help="Expected yield method (default: rolling)",
    )
    parser.add_argument("--window",      type=int,   default=5,    help="Rolling window (years)")
    parser.add_argument("--min-periods", type=int,   default=3,    help="Min periods for rolling")
    parser.add_argument("--min-years",   type=int,   default=5,    help="Min years for trend fit")
    parser.add_argument("--quantile",    type=float, default=0.20, help="Severe-risk quantile")
    parser.add_argument(
        "--nass-file", default="data/interim/nass_yield_clean.csv",
        help="Path to cleaned NASS yield CSV (may contain multiple crops)",
    )
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        default=list(CHECKPOINT_CONFIGS.keys()),
        help="Checkpoint names to include (default: all 5)",
    )
    parser.add_argument(
        "--single-checkpoint",
        action="store_true",
        help="Use only full_season (legacy behaviour, no multi-checkpoint fanout)",
    )
    parser.add_argument(
        "--targets-only", action="store_true",
        help="Only build yield targets; skip weather merge and panel assembly",
    )
    parser.add_argument(
        "--descriptive-risk", action="store_true",
        help="Add severe_risk_descriptive labels (EDA only — not for modeling)",
    )
    parser.add_argument(
        "--allow-partial-year", action="store_true",
        help="Keep incomplete current-year weather rows in the modeling panel",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger.info("=== CropShield: Step 2 — Build Features ===")

    # ── 1. Yield targets ─────────────────────────────────────────────────────
    nass_path = Path(args.nass_file)
    if not nass_path.exists():
        logger.error("NASS file not found at %s. Run scripts/01_fetch_data.py first.", nass_path)
        sys.exit(1)

    logger.info("--- Building yield targets (method=%s) ---", args.method)
    nass_df = load_nass_yield_csv(nass_path)
    targets_df = build_yield_targets(
        nass_df,
        method=args.method,
        window=args.window,
        min_periods=args.min_periods,
        min_years=args.min_years,
        risk_quantile=args.quantile,
        add_descriptive_risk=args.descriptive_risk,
        output_path="data/interim/yield_targets.csv",
    )
    valid_targets = targets_df.dropna(subset=["yield_anomaly"])
    logger.info(
        "Yield targets: %d total rows | %d with targets | mean anomaly=%.2f bu/acre",
        len(targets_df), len(valid_targets), valid_targets["yield_anomaly"].mean(),
    )

    if args.targets_only:
        logger.info("--targets-only flag set; skipping weather and panel steps.")
        logger.info("=== Step 2 complete (targets only) ===")
        return

    # ── 2. Weather features ───────────────────────────────────────────────────
    weather_raw_path = Path("data/raw/weather_daily_raw.csv")
    weather_features_path = Path("data/processed/weather_features.csv")

    if weather_raw_path.exists():
        logger.info("--- Computing weather features ---")
        daily_df = pd.read_csv(weather_raw_path, parse_dates=["date"])

        if args.single_checkpoint:
            logger.info("--single-checkpoint: computing full_season only")
            weather_features = compute_weather_features(daily_df)
            weather_features["checkpoint"] = "full_season"
        else:
            checkpoints = args.checkpoints
            logger.info("Computing %d checkpoints: %s", len(checkpoints), checkpoints)
            weather_features = compute_multi_checkpoint_weather_features(
                daily_df, checkpoints=checkpoints
            )

            # ── Stress-timing features from daily weather (leakage-safe) ──────
            logger.info("Computing checkpoint stress-timing features from daily weather")
            stress = compute_checkpoint_stress_features(daily_df, checkpoints=checkpoints)
            merge_keys = ["county_fips", "state_fips", "year", "checkpoint"]
            merge_keys = [k for k in merge_keys if k in weather_features.columns and k in stress.columns]
            before = len(weather_features)
            weather_features = weather_features.merge(stress, on=merge_keys, how="left")
            assert len(weather_features) == before, "stress merge fanned out weather rows"
            logger.info("Merged %d stress-timing features", len(STRESS_FEATURE_COLUMNS))

            # ── County+checkpoint-normalised weather anomalies (leakage-safe) ─
            weather_features = add_weather_anomalies(weather_features)
            logger.info(
                "Added %d weather-anomaly features", len(weather_anomaly_columns(weather_features))
            )

        weather_features = filter_incomplete_current_year(
            weather_features,
            allow_partial_year=args.allow_partial_year,
        )
        save_weather_features(weather_features, weather_features_path)
        n_ckpts = weather_features["checkpoint"].nunique() if "checkpoint" in weather_features.columns else 1
        logger.info(
            "Weather features: %d rows | %d counties | %d checkpoints | %d feature cols",
            len(weather_features),
            weather_features["county_fips"].nunique(),
            n_ckpts,
            len(weather_features.columns) - 4,
        )
    else:
        logger.warning(
            "Weather raw file not found at %s. "
            "Run: python scripts/01_fetch_data.py --skip-drought",
            weather_raw_path,
        )
        logger.warning("Skipping weather features and panel assembly.")
        logger.info("=== Step 2 complete (targets only — no weather data) ===")
        return

    # ── 3. Drought features (only if raw data is present) ─────────────────────
    drought_raw_path = Path("data/raw/drought_monitor.csv")
    drought_features_path = None
    if drought_raw_path.exists():
        logger.info("--- Computing drought features from %s ---", drought_raw_path)
        from cropshield.data.fetch_drought_monitor import clean_drought_dataframe
        from cropshield.features.drought_features import compute_drought_features

        drought_raw = pd.read_csv(drought_raw_path)
        drought_clean = clean_drought_dataframe(drought_raw)
        # Full-season drought features merged by (county_fips, year); broadcast
        # across checkpoints. (Per-checkpoint drought cutoffs are a future step
        # once raw weekly data and per-year cutoff support are wired in.)
        drought_feats = compute_drought_features(drought_clean)
        drought_feats = drought_feats.drop(columns=[c for c in ("checkpoint",) if c in drought_feats.columns])
        drought_features_path = Path("data/interim/drought_features.csv")
        drought_feats.to_csv(drought_features_path, index=False)
        logger.info("Drought features saved → %s (%d rows)", drought_features_path, len(drought_feats))
    else:
        logger.warning(
            "No raw drought data at %s — drought feature code exists "
            "(fetch_drought_monitor.clean_drought_dataframe + "
            "drought_features.compute_drought_features) but is SKIPPED. "
            "Drought columns will be absent from the panel.",
            drought_raw_path,
        )

    # ── 4. Build modeling panel ───────────────────────────────────────────────
    logger.info("--- Assembling modeling panel ---")
    panel = build_modeling_panel(
        yield_path="data/interim/yield_targets.csv",
        weather_path=str(weather_features_path),
        drought_path=str(drought_features_path) if drought_features_path else None,
        output_path="data/processed/modeling_panel.csv",
        missingness_path="reports/missingness_report.csv",
        drop_missing_target=True,
        allow_partial_year=args.allow_partial_year,
    )

    # Save panel audit
    save_panel_audit(panel, "reports/panel_audit.md")

    feature_cols = get_feature_columns(panel)
    print("\n── Modeling Panel Summary ────────────────────────────────────")
    print(f"  Rows:              {len(panel):,}")
    print(f"  Unique counties:   {panel['county_fips'].nunique()}")
    print(f"  States:            {sorted(panel['state'].unique())}")
    print(f"  Years:             {int(panel['year'].min())}–{int(panel['year'].max())}")
    if "crop" in panel.columns:
        print(f"  Crops:             {sorted(panel['crop'].unique())}")
    if "checkpoint" in panel.columns:
        ckpts = sorted(panel["checkpoint"].dropna().unique())
        print(f"  Checkpoints:       {ckpts}")
    print(f"  Feature columns:   {len(feature_cols)}")
    print(f"  Feature names:     {feature_cols}")
    print()
    print("Sample rows:")
    cols_to_show = ["year", "state", "county", "crop", "checkpoint",
                    "yield_anomaly", "cumulative_precip", "extreme_heat_days",
                    "growing_degree_days"]
    cols_present = [c for c in cols_to_show if c in panel.columns]
    print(panel[cols_present].head(8).to_string(index=False))
    print()

    logger.info("=== Step 2 complete ===")


if __name__ == "__main__":
    main()
