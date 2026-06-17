"""
Script 02: Build the feature engineering pipeline.

Reads cleaned interim data and produces:
  1. Yield targets (expected yield, anomaly, risk class)
  2. Weather growing-season features     [Prompt 4]
  3. Drought growing-season features     [later]
  4. Final merged modeling panel         [Prompt 5]

Usage
-----
    python scripts/02_build_features.py
    python scripts/02_build_features.py --method rolling --window 5
    python scripts/02_build_features.py --method trend --min-years 5

Run from the project root directory.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cropshield.features.yield_targets import build_yield_targets
from cropshield.features.weather_features import compute_weather_features, save_weather_features

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
    parser.add_argument("--window",    type=int, default=5, help="Rolling window (years)")
    parser.add_argument("--min-periods", type=int, default=3,
                        help="Min periods for rolling window")
    parser.add_argument("--min-years", type=int, default=5,
                        help="Min years for trend fit")
    parser.add_argument("--quantile",  type=float, default=0.20,
                        help="Severe-risk quantile threshold (default: 0.20)")
    parser.add_argument(
        "--nass-file", default="data/interim/nass_yield_clean.csv",
        help="Path to cleaned NASS yield CSV",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger.info("=== CropShield: Step 2 — Build Features ===")
    logger.info("Expected yield method: %s | window=%d | quantile=%.2f",
                args.method, args.window, args.quantile)

    # ── 1. Yield targets ─────────────────────────────────────────────────────
    nass_path = Path(args.nass_file)
    if not nass_path.exists():
        logger.error(
            "NASS clean file not found at %s. Run scripts/01_fetch_data.py first.",
            nass_path,
        )
        sys.exit(1)

    import pandas as pd
    nass_df = pd.read_csv(nass_path)
    logger.info("Loaded NASS data: %d rows from %s", len(nass_df), nass_path)

    targets_df = build_yield_targets(
        nass_df,
        method=args.method,
        window=args.window,
        min_periods=args.min_periods,
        min_years=args.min_years,
        risk_quantile=args.quantile,
        output_path="data/interim/yield_targets.csv",
    )

    # Print summary stats
    valid = targets_df.dropna(subset=["yield_anomaly"])
    print("\n── Yield Target Summary ──────────────────────────────")
    print(f"  Total rows:           {len(targets_df)}")
    print(f"  Rows with targets:    {len(valid)}")
    print(f"  Rows without targets: {len(targets_df) - len(valid)}  (insufficient history)")
    print(f"  Years:                {int(targets_df['year'].min())}–{int(targets_df['year'].max())}")
    print(f"  Mean yield anomaly:   {valid['yield_anomaly'].mean():.2f} bu/acre")
    print(f"  Std yield anomaly:    {valid['yield_anomaly'].std():.2f} bu/acre")
    print(f"  Severe-risk rows:     {int((valid['severe_risk'] == 1).sum())}  "
          f"({100 * (valid['severe_risk'] == 1).mean():.1f}%)")
    print()
    print("Sample rows with targets:")
    print(valid[["year","state","county","yield_anomaly","yield_anomaly_pct","severe_risk"]]
          .head(10).to_string(index=False))
    print()

    # ── 2. Weather features ──────────────────────────────────────────────────
    weather_raw_path = Path("data/raw/weather_daily_raw.csv")
    if weather_raw_path.exists():
        logger.info("--- Computing weather features ---")
        import pandas as _pd
        daily_df = _pd.read_csv(weather_raw_path, parse_dates=["date"])
        weather_features = compute_weather_features(daily_df)
        save_weather_features(weather_features, "data/processed/weather_features.csv")
        print(f"\nWeather features: {weather_features.shape}")
        print(weather_features.describe().to_string())
    else:
        logger.info("--- Weather raw file not found; run 01_fetch_data.py first ---")

    # ── 3. Drought features ──────────────────────────────────────────────────
    logger.info("--- Drought Monitor features not yet implemented ---")

    # ── 4. Build modeling panel ──────────────────────────────────────────────
    logger.info("--- Panel assembly not yet implemented (Prompt 5) ---")

    logger.info("=== Step 2 complete ===")


if __name__ == "__main__":
    main()
