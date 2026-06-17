"""
Script 02: Build the feature engineering pipeline.

Reads cleaned interim data and produces:
  1. Yield targets (expected yield, anomaly, risk class)
  2. Weather growing-season features
  3. Drought growing-season features (optional)
  4. Final merged modeling panel

Usage
-----
    python scripts/02_build_features.py
    python scripts/02_build_features.py --method rolling --window 5

Run from the project root directory.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# TODO: Import feature modules when implemented
# from cropshield.features.yield_targets import (
#     clean_yield_dataframe, add_expected_yield_rolling, add_yield_anomaly, add_risk_class
# )
# from cropshield.features.weather_features import compute_weather_features
# from cropshield.features.drought_features import compute_drought_features
# from cropshield.data.build_county_panel import build_modeling_panel

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
        "--method",
        choices=["rolling", "trend"],
        default="rolling",
        help="Expected yield method (default: rolling)",
    )
    parser.add_argument("--window", type=int, default=5, help="Rolling window size in years")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger.info("=== CropShield: Step 2 — Build Features ===")
    logger.info("Expected yield method: %s (window=%d)", args.method, args.window)

    # ── 1. Yield targets ─────────────────────────────────────────────────────
    logger.info("--- Engineering yield targets ---")
    # TODO: Implement when yield_targets.py is ready (Prompt 3)
    logger.warning("Yield target engineering not yet implemented. Skipping.")

    # ── 2. Weather features ──────────────────────────────────────────────────
    logger.info("--- Computing weather features ---")
    # TODO: Implement when weather_features.py is ready (Prompt 4)
    logger.warning("Weather feature engineering not yet implemented. Skipping.")

    # ── 3. Drought features ──────────────────────────────────────────────────
    logger.info("--- Computing drought features ---")
    # TODO: Implement when drought_features.py is ready
    logger.warning("Drought feature engineering not yet implemented. Skipping.")

    # ── 4. Build modeling panel ──────────────────────────────────────────────
    logger.info("--- Assembling modeling panel ---")
    # TODO: Implement when build_county_panel.py is ready (Prompt 5)
    logger.warning("Panel assembly not yet implemented. Skipping.")

    logger.info("=== Step 2 complete ===")


if __name__ == "__main__":
    main()
