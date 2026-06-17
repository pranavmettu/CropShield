"""
Script 01: Fetch raw data from all sources.

Runs the data ingestion pipeline for:
  1. USDA NASS yield data (county-level, Iowa + Illinois, 2015 onward)
  2. NASA POWER daily weather data (county centroids, growing season)  [Prompt 4]
  3. U.S. Drought Monitor weekly statistics                            [later]

Usage
-----
    python scripts/01_fetch_data.py
    python scripts/01_fetch_data.py --crop CORN --states IOWA ILLINOIS
    python scripts/01_fetch_data.py --start-year 2015 --skip-weather

Run from the project root directory so that relative data/ paths resolve
correctly.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running without `pip install -e .`
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cropshield.data.fetch_nass import fetch_nass_yield
# from cropshield.data.fetch_power import fetch_power_all_counties   # Prompt 4
# from cropshield.data.fetch_drought_monitor import fetch_drought_monitor  # later

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch CropShield raw data from USDA NASS, NASA POWER, and Drought Monitor."
    )
    parser.add_argument("--crop", default="CORN", help="NASS commodity (default: CORN)")
    parser.add_argument(
        "--states",
        nargs="+",
        default=["IOWA", "ILLINOIS"],
        help="States in ALL CAPS (default: IOWA ILLINOIS)",
    )
    parser.add_argument("--start-year", type=int, default=2015, help="First year to fetch")
    parser.add_argument("--end-year",   type=int, default=None,  help="Last year to fetch (default: latest)")
    parser.add_argument("--skip-weather", action="store_true", help="Skip NASA POWER fetch")
    parser.add_argument("--skip-drought", action="store_true", help="Skip Drought Monitor fetch")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logger.info("=== CropShield: Step 1 — Fetch Data ===")
    logger.info(
        "Crop: %s | States: %s | Years: %d–%s",
        args.crop, args.states, args.start_year, args.end_year or "latest",
    )

    # ── 1. USDA NASS yield ───────────────────────────────────────────────────
    logger.info("--- Fetching USDA NASS yield data ---")
    try:
        df_nass = fetch_nass_yield(
            crop=args.crop,
            states=args.states,
            start_year=args.start_year,
            end_year=args.end_year,
        )
        logger.info(
            "NASS fetch complete: %d rows | %d counties | years %d–%d",
            len(df_nass),
            df_nass["county_fips"].nunique(),
            int(df_nass["year"].min()),
            int(df_nass["year"].max()),
        )
        # Quick sanity print
        print("\nSample NASS records:")
        print(df_nass.head(10).to_string(index=False))
        print(f"\nShape: {df_nass.shape}")
        print(f"States: {sorted(df_nass['state'].unique())}")
        print(f"Years:  {sorted(df_nass['year'].unique())}")
        print(f"Mean yield: {df_nass['value'].mean():.1f} bu/acre\n")
    except EnvironmentError as exc:
        logger.error("NASS API key missing: %s", exc)
        sys.exit(1)
    except Exception as exc:
        logger.error("NASS fetch failed: %s", exc, exc_info=True)
        sys.exit(1)

    # ── 2. NASA POWER weather ────────────────────────────────────────────────
    if not args.skip_weather:
        logger.info("--- NASA POWER weather fetcher not yet implemented (Prompt 4) ---")

    # ── 3. U.S. Drought Monitor ──────────────────────────────────────────────
    if not args.skip_drought:
        logger.info("--- Drought Monitor fetcher not yet implemented ---")

    logger.info("=== Step 1 complete ===")


if __name__ == "__main__":
    main()
