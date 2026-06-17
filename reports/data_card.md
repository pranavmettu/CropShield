# CropShield Data Card

## Dataset Summary

CropShield integrates public agricultural and climate datasets into a county-crop-year
modeling panel. All data sources are publicly available and free to access. No
proprietary, licensed, or personally identifiable data is used.

---

## Sources

| Source | Provider | Variables | Granularity | Access |
|---|---|---|---|---|
| USDA NASS Quick Stats | USDA National Agricultural Statistics Service | Crop yield (bu/acre) | County × year | API (free key) |
| NASA POWER | NASA Langley Research Center | Daily weather (precip, temp) | Point (county centroid) × day | API (no key required) |
| U.S. Drought Monitor | NDMC / USDA / NOAA | Weekly D0–D4 drought area | County × week | Web download |
| County FIPS & centroids | U.S. Census Bureau | FIPS codes, lat/lon | County | Static reference |

---

## Unit of Analysis

County × crop × year (and optionally × growing-season checkpoint)

---

## MVP Scope

| Attribute | Value |
|---|---|
| Crops | Corn only |
| States | Iowa (FIPS 19), Illinois (FIPS 17) |
| Years | 2015 through latest available NASS year |
| Weather aggregation | Growing season: April 1 – August 31 |

---

## Key Variables

### Target Variables
| Variable | Type | Description |
|---|---|---|
| `actual_yield` | float | NASS-reported county corn yield (bu/acre) |
| `expected_yield` | float | Rolling 5-year prior-year mean (leakage-safe) |
| `yield_anomaly` | float | `actual_yield - expected_yield` (bu/acre) |
| `yield_anomaly_pct` | float | Anomaly as percentage of expected yield |
| `severe_risk` | binary | 1 if yield_anomaly_pct ≤ 20th percentile |

### Weather Features
| Variable | Units | Source |
|---|---|---|
| `cumulative_precip` | mm | NASA POWER (PRECTOTCORR) |
| `mean_temp` | °C | NASA POWER (T2M) |
| `max_temp` | °C | NASA POWER (T2M_MAX) |
| `extreme_heat_days` | days | Derived (Tmax ≥ 35°C) |
| `dry_days` | days | Derived (precip ≤ 1 mm) |
| `longest_dry_spell` | days | Derived (consecutive dry days) |
| `growing_degree_days` | GDD | Derived (Σ max(0, Tavg − 10)) |
| `precip_anomaly` | mm | Derived (vs. county prior-year mean) |

### Drought Features
| Variable | Units | Source |
|---|---|---|
| `d0_max_pct` through `d4_max_pct` | % county area | USDM |
| `d2_plus_weeks` | weeks | Derived |
| `d2_plus_max_pct` | % | Derived |

---

## Data Cleaning Choices

### NASS yield data
- Records with `(D)` (data suppressed for confidentiality) are dropped.
- Records marked `(Z)` (near-zero) are dropped.
- Yield values are stripped of commas before numeric conversion.
- County FIPS codes are constructed from state ANSI + county ANSI (zero-padded to 5 digits).
- Only the `TOTAL` domain is used (no organic/irrigated sub-programs).

### NASA POWER weather
- Data fetched at county centroid (not averaged over county area).
- Missing values (`-999` sentinel) are replaced with NaN.
- Days outside April–August are excluded before feature aggregation.

### Drought Monitor
- Weekly records are filtered to April–August for growing-season aggregation.
- County FIPS codes are standardised to 5-digit strings.
- Missing category percentages are filled with 0.0 (assumed no drought data = no drought).

---

## Known Limitations

| Limitation | Description |
|---|---|
| County-level aggregation | Farm-level heterogeneity in management and soil is not captured |
| Suppressed NASS records | Data-sparse counties are systematically underrepresented |
| Weather at centroids | Does not capture within-county spatial variation |
| NASS reporting lag | Final county yields are often not available until early winter |
| Drought Monitor resolution | Weekly snapshots miss sub-weekly stress events |
| Missing management data | Planting date, irrigation, and fertiliser are not in the dataset |
| Potential reporting differences | NASS survey methodology may vary across state-years |

---

## Data Access Instructions

### USDA NASS Quick Stats
1. Register for a free API key at https://quickstats.nass.usda.gov/api
2. Set `NASS_API_KEY` in your `.env` file
3. Run `python scripts/01_fetch_data.py`

### NASA POWER
- No registration required
- Run `python scripts/01_fetch_data.py --skip-drought`

### U.S. Drought Monitor
- No registration required
- Data accessed via https://usdm.climate.unl.edu/

---

## Reproducibility

All raw data is fetched programmatically via scripts. No manual download steps
are required for the core pipeline. See `configs/data_sources.yaml` for all
API endpoints and parameters.

Raw data files are gitignored (`data/raw/`, `data/interim/`, `data/processed/`).
Re-run `make fetch-data && make build-features` to reproduce the dataset.
