"""
NASA POWER weather data fetcher for CropShield.

Retrieves daily weather data from the NASA POWER Agroclimatology API for
county centroids and saves it to the data pipeline.

API Reference
-------------
https://power.larc.nasa.gov/docs/services/api/

Strategy
--------
Rather than one API call per (county, year), this module makes one call per
county fetching the entire multi-year date range at once. For 201 counties
this is ~201 requests instead of ~2,200. A checkpoint CSV is saved every N
counties so the fetch can be resumed if interrupted.

Notes
-----
- No API key is required for NASA POWER (as of 2024).
- The NASA POWER API returns -999 as a sentinel for missing values; these
  are replaced with NaN.
- County centroids are downloaded from the U.S. Census Bureau county
  population centres file and cached in data/external/.
- Feature engineering from the raw daily data lives in
  src/cropshield/features/weather_features.py.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
POWER_BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
POWER_PARAMETERS = ["PRECTOTCORR", "T2M", "T2M_MIN", "T2M_MAX"]
POWER_COMMUNITY = "AG"
POWER_MISSING_VALUE = -999.0

# Census Bureau county population centres (2020)
CENSUS_CENTROIDS_URL = (
    "https://www2.census.gov/geo/docs/reference/cenpop2020/county/"
    "CenPop2020_Mean_CO.txt"
)
CENTROIDS_CACHE = Path("data/external/county_centroids.csv")

# MVP state FIPS codes
MVP_STATE_FIPS = {"19", "17"}  # Iowa, Illinois


def load_county_centroids(
    county_fips_list: list[str] | None = None,
    cache_path: Path = CENTROIDS_CACHE,
    force_download: bool = False,
) -> pd.DataFrame:
    """Return a DataFrame of county centroids for the requested FIPS codes.

    Downloads the Census Bureau county population centres file on first call
    and caches it to ``cache_path``. Subsequent calls use the cache.

    Parameters
    ----------
    county_fips_list : list[str], optional
        5-digit FIPS codes to include. Returns all counties in the cache
        when ``None``.
    cache_path : Path
        Local cache destination.
    force_download : bool
        Re-download even if the cache already exists.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ``county_fips``, ``county``, ``state_fips``,
        ``lat``, ``lon``.

    Raises
    ------
    requests.HTTPError
        If the Census URL is unreachable and no cache exists.
    """
    if not cache_path.exists() or force_download:
        logger.info("Downloading county centroids from Census Bureau…")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            resp = requests.get(CENSUS_CENTROIDS_URL, timeout=30)
            resp.raise_for_status()
            cache_path.write_bytes(resp.content)
            logger.info("Centroids cached → %s", cache_path)
        except requests.RequestException as exc:
            raise requests.HTTPError(
                f"Failed to download county centroids: {exc}"
            ) from exc

    raw = pd.read_csv(cache_path, dtype=str)
    # Columns: STATEFP, COUNTYFP, COUNAME, STNAME, POPULATION, LATITUDE, LONGITUDE
    raw.columns = [c.strip() for c in raw.columns]

    centroids = pd.DataFrame({
        "county_fips": raw["STATEFP"].str.zfill(2) + raw["COUNTYFP"].str.zfill(3),
        "county":      raw["COUNAME"].str.strip().str.upper(),
        "state_fips":  raw["STATEFP"].str.zfill(2),
        "lat":         pd.to_numeric(raw["LATITUDE"],  errors="coerce"),
        "lon":         pd.to_numeric(raw["LONGITUDE"], errors="coerce"),
    })

    if county_fips_list is not None:
        # Normalise to zero-padded 5-character strings so floats like 17001.0 match "17001"
        normalised = {
            str(int(float(f))).zfill(5) for f in county_fips_list
            if str(f) not in ("nan", "None", "")
        }
        centroids = centroids[centroids["county_fips"].isin(normalised)]

    centroids = centroids.dropna(subset=["lat", "lon"]).reset_index(drop=True)
    logger.info(
        "Loaded %d county centroids (requested %s)",
        len(centroids),
        len(county_fips_list) if county_fips_list else "all",
    )
    return centroids


def fetch_power_for_county(
    lat: float,
    lon: float,
    start_year: int = 2015,
    end_year: int | None = None,
    parameters: list[str] = POWER_PARAMETERS,
    max_retries: int = 3,
    retry_delay: float = 5.0,
) -> pd.DataFrame:
    """Fetch daily weather from NASA POWER for a single county centroid.

    Fetches all years from ``start_year`` through ``end_year`` in one request
    (full Jan-Dec date range per year), which is far more efficient than one
    request per year.

    Parameters
    ----------
    lat : float
        Latitude of the county centroid.
    lon : float
        Longitude of the county centroid.
    start_year : int
        First calendar year to include.
    end_year : int, optional
        Last calendar year to include. Defaults to the current year.
    parameters : list[str]
        NASA POWER parameter codes.
    max_retries : int
        Number of retry attempts on transient failures.
    retry_delay : float
        Base seconds to wait between retries (multiplied by attempt number).

    Returns
    -------
    pd.DataFrame
        Daily weather records with columns: ``date``, ``year``, ``month``,
        ``lat``, ``lon``, plus one column per requested parameter.
        Missing values (-999) are replaced with NaN.

    Raises
    ------
    requests.HTTPError
        After all retries are exhausted.
    """
    import datetime
    end_yr = end_year or datetime.datetime.now().year
    start_str = f"{start_year}0101"
    end_str   = f"{end_yr}1231"

    params = {
        "parameters": ",".join(parameters),
        "community":  POWER_COMMUNITY,
        "longitude":  str(round(lon, 4)),
        "latitude":   str(round(lat, 4)),
        "start":      start_str,
        "end":        end_str,
        "format":     "JSON",
    }

    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(POWER_BASE_URL, params=params, timeout=120)
            resp.raise_for_status()
            payload = resp.json()
            param_data = payload["properties"]["parameter"]
            break
        except (requests.RequestException, KeyError) as exc:
            last_exc = exc
            if attempt < max_retries:
                wait = retry_delay * attempt
                logger.warning(
                    "NASA POWER request failed (attempt %d/%d): %s — retrying in %.0fs",
                    attempt, max_retries, exc, wait,
                )
                time.sleep(wait)
    else:
        raise requests.HTTPError(
            f"NASA POWER request failed after {max_retries} attempts: {last_exc}"
        )

    # ── Reshape: one row per date ──────────────────────────────────────────────
    # param_data = {"PRECTOTCORR": {"20150101": 0.5, ...}, "T2M": {...}, ...}
    dates = sorted(next(iter(param_data.values())).keys())  # YYYYMMDD strings
    df = pd.DataFrame({"date_str": dates})
    for param, values in param_data.items():
        df[param] = [values.get(d, POWER_MISSING_VALUE) for d in dates]

    # Replace -999 sentinel with NaN
    numeric_cols = [c for c in df.columns if c != "date_str"]
    df[numeric_cols] = df[numeric_cols].replace(POWER_MISSING_VALUE, float("nan"))

    # Parse dates and add convenience columns
    df["date"]  = pd.to_datetime(df["date_str"], format="%Y%m%d")
    df["year"]  = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["lat"]   = lat
    df["lon"]   = lon
    df = df.drop(columns=["date_str"]).sort_values("date").reset_index(drop=True)

    return df


def fetch_power_all_counties(
    county_centroids: pd.DataFrame,
    start_year: int = 2015,
    end_year: int | None = None,
    output_raw: str | Path = "data/raw/weather_daily_raw.csv",
    checkpoint_every: int = 10,
    request_delay: float = 0.5,
) -> pd.DataFrame:
    """Fetch NASA POWER data for all county centroids.

    Makes one API call per county (fetching all years at once) and saves
    a checkpoint CSV every ``checkpoint_every`` counties so the fetch can
    be resumed if interrupted.

    Parameters
    ----------
    county_centroids : pd.DataFrame
        DataFrame with columns: ``county_fips``, ``county``, ``state_fips``,
        ``lat``, ``lon``. Use ``load_county_centroids()`` to build this.
    start_year : int
        First calendar year to include.
    end_year : int, optional
        Last calendar year to include.
    output_raw : str or Path
        Final output path for the concatenated raw weather CSV.
    checkpoint_every : int
        Save a checkpoint after this many counties are processed.
    request_delay : float
        Polite delay (seconds) between API requests to avoid rate-limiting.

    Returns
    -------
    pd.DataFrame
        Concatenated daily weather records for all counties and years, with
        columns: ``county_fips``, ``state_fips``, ``county``, ``date``,
        ``year``, ``month``, ``lat``, ``lon``, plus weather parameter columns.
    """
    output_raw = Path(output_raw)
    checkpoint_path = output_raw.with_suffix(".checkpoint.csv")

    # ── Resume from checkpoint if it exists ──────────────────────────────────
    import datetime as _dt
    end_yr = end_year or _dt.datetime.now().year
    required_years = set(range(start_year, end_yr + 1))

    completed_fips: set[str] = set()
    # all_data accumulates ALL fetched frames in memory for correct checkpointing
    all_data: list[pd.DataFrame] = []
    if checkpoint_path.exists():
        # Explicitly read county_fips as str to avoid int64 infer after CSV round-trip.
        # Without this, "17001" is read as int64(17001) and set-membership checks against
        # the string-typed centroids DataFrame silently fail, causing re-fetches or skips.
        ckpt = pd.read_csv(
            checkpoint_path, low_memory=False,
            dtype={"county_fips": str, "state_fips": str},
        )
        # A county is "complete" only if every year in the requested range
        # is present.  Checking only county_fips would skip a county whose
        # checkpoint was written with an older (shorter) year range.
        ckpt_years_by_fips = (
            ckpt.groupby("county_fips")["year"]
            .apply(lambda s: set(s.astype(int)))
        )
        completed_fips = {
            fips
            for fips, years in ckpt_years_by_fips.items()
            if required_years <= years
        }
        all_data.append(ckpt)
        logger.info(
            "Resuming from checkpoint: %d counties fully complete "
            "(covering years %d–%d)",
            len(completed_fips), start_year, end_yr,
        )

    remaining = county_centroids[~county_centroids["county_fips"].isin(completed_fips)]
    total = len(county_centroids)
    logger.info(
        "Fetching weather for %d counties (%d remaining, %d already done)",
        total, len(remaining), len(completed_fips),
    )

    fetched_this_run = 0

    for i, row in enumerate(remaining.itertuples(index=False), start=1):
        fips    = row.county_fips
        county  = row.county
        st_fips = row.state_fips
        lat     = row.lat
        lon     = row.lon

        logger.info(
            "  [%d/%d] %s %s (%.4f, %.4f)",
            len(completed_fips) + i, total, st_fips, county, lat, lon,
        )

        try:
            df_county = fetch_power_for_county(
                lat=lat, lon=lon,
                start_year=start_year, end_year=end_year,
            )
            df_county.insert(0, "county_fips", fips)
            df_county.insert(1, "state_fips",  st_fips)
            df_county.insert(2, "county",      county)
            all_data.append(df_county)
            fetched_this_run += 1
        except Exception as exc:
            logger.error("Failed to fetch %s (%s): %s — skipping.", fips, county, exc)

        # ── Checkpoint: write cumulative data every N counties ─────────────────
        if fetched_this_run % checkpoint_every == 0 and fetched_this_run > 0:
            _save_checkpoint(all_data, [], checkpoint_path)

        if request_delay > 0:
            time.sleep(request_delay)

    # ── Final save ────────────────────────────────────────────────────────────
    if not all_data:
        raise ValueError("No weather data was fetched successfully.")

    result = pd.concat(all_data, ignore_index=True)
    output_raw.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_raw, index=False)
    logger.info(
        "Weather data saved → %s  (%d rows, %d counties, %d years)",
        output_raw, len(result),
        result["county_fips"].nunique(),
        result["year"].nunique(),
    )

    # Remove checkpoint file after successful completion
    if checkpoint_path.exists():
        checkpoint_path.unlink()

    return result


def _save_checkpoint(
    all_data: list[pd.DataFrame],
    _unused: list,
    path: Path,
) -> None:
    """Write all accumulated data to the checkpoint CSV."""
    combined = pd.concat(all_data, ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(path, index=False)
    n_counties = combined["county_fips"].nunique()
    logger.info("Checkpoint saved: %d counties so far → %s", n_counties, path)
