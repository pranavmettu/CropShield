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
from cropshield.data.fetch_power import fetch_power_all_counties, load_county_centroids
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
    parser.add_argument(
        "--crops",
        nargs="+",
        default=["CORN", "SOYBEANS"],
        help="NASS commodities to fetch (default: CORN SOYBEANS)",
    )
    # Legacy single-crop flag kept for backward compatibility; overridden by --crops
    parser.add_argument("--crop", default=None, help="NASS commodity (deprecated; use --crops)")
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

    # Resolve crops list (--crop is deprecated; --crops takes priority)
    crops = [c.upper() for c in (args.crops or ([args.crop] if args.crop else ["CORN", "SOYBEANS"]))]

    logger.info("=== CropShield: Step 1 — Fetch Data ===")
    logger.info(
        "Crops: %s | States: %s | Years: %d–%s",
        crops, args.states, args.start_year, args.end_year or "latest",
    )

    # ── 1. USDA NASS yield ───────────────────────────────────────────────────
    logger.info("--- Fetching USDA NASS yield data ---")
    import pandas as _pd
    from pathlib import Path as _Path
    nass_frames = []
    try:
        for crop in crops:
            crop_slug = crop.lower().replace(" ", "_")
            logger.info("  Fetching %s …", crop)
            df_crop = fetch_nass_yield(
                crop=crop,
                states=args.states,
                start_year=args.start_year,
                end_year=args.end_year,
                output_raw=f"data/raw/nass_yield_{crop_slug}.csv",
                output_clean=f"data/interim/nass_yield_{crop_slug}_clean.csv",
            )
            logger.info(
                "  %s: %d rows | %d counties | years %d–%d",
                crop,
                len(df_crop),
                df_crop["county_fips"].nunique(),
                int(df_crop["year"].min()),
                int(df_crop["year"].max()),
            )
            nass_frames.append(df_crop)
    except EnvironmentError as exc:
        logger.error("NASS API key missing: %s", exc)
        sys.exit(1)
    except Exception as exc:
        logger.error("NASS fetch failed: %s", exc, exc_info=True)
        sys.exit(1)

    df_nass = _pd.concat(nass_frames, ignore_index=True) if nass_frames else _pd.DataFrame()

    # Save combined multi-crop clean file (used by 02_build_features.py)
    combined_clean = _Path("data/interim/nass_yield_clean.csv")
    combined_clean.parent.mkdir(parents=True, exist_ok=True)
    df_nass.to_csv(combined_clean, index=False)
    logger.info("Combined NASS file saved → %s  (%d rows)", combined_clean, len(df_nass))

    logger.info(
        "NASS fetch complete: %d rows | %d counties | crops: %s",
        len(df_nass),
        df_nass["county_fips"].nunique(),
        sorted(df_nass["crop"].unique()) if "crop" in df_nass.columns else crops,
    )
    print("\nSample NASS records:")
    print(df_nass.head(10).to_string(index=False))
    print(f"\nShape: {df_nass.shape}")
    if "crop" in df_nass.columns:
        for c in sorted(df_nass["crop"].unique()):
            sub = df_nass[df_nass["crop"] == c]
            print(f"  {c}: {len(sub)} rows | mean yield {sub['value'].mean():.1f} bu/acre")
    print()

    # ── 2. NASA POWER weather ────────────────────────────────────────────────
    if not args.skip_weather:
        logger.info("--- Fetching NASA POWER weather data ---")
        try:
            # Union of county FIPS across all crops (weather is crop-independent)
            county_fips_list = df_nass["county_fips"].dropna().unique().tolist()
            logger.info("Loading centroids for %d counties…", len(county_fips_list))
            centroids = load_county_centroids(county_fips_list=county_fips_list)

            df_weather = fetch_power_all_counties(
                county_centroids=centroids,
                start_year=args.start_year,
                end_year=args.end_year,
                output_raw="data/raw/weather_daily_raw.csv",
            )
            logger.info(
                "NASA POWER fetch complete: %d daily rows | %d counties | years %d–%d",
                len(df_weather),
                df_weather["county_fips"].nunique(),
                int(df_weather["year"].min()),
                int(df_weather["year"].max()),
            )
        except Exception as exc:
            logger.error("NASA POWER fetch failed: %s", exc, exc_info=True)
            sys.exit(1)

    # ── 3. U.S. Drought Monitor ──────────────────────────────────────────────
    if not args.skip_drought:
        logger.info("--- Drought Monitor fetcher not yet implemented ---")

    logger.info("=== Step 1 complete ===")


if __name__ == "__main__":
    main()
