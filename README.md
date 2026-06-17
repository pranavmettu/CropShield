# CropShield: Drought-Aware Yield Risk Forecasting for U.S. Corn and Soybean Counties

> **Portfolio project** — an interpretable geospatial ML pipeline for county-level crop yield-risk forecasting using public agricultural and climate data.

---

## One-Sentence Summary

CropShield integrates USDA yield records, NASA weather data, and U.S. Drought Monitor severity into a reproducible machine-learning pipeline that predicts county-level corn (and later soybean) yield anomalies during the growing season.

---

## Problem and Motivation

Farmers, crop advisors, insurers, agtech companies, and agricultural researchers need earlier signals of county-level crop stress **before** final harvest yields are available. Drought, heat, rainfall deficits, and historical yield trends all contribute to outcomes — but these data sources are scattered across incompatible APIs and formats.

CropShield assembles a county-crop-year panel dataset and trains models to surface:

1. **Yield anomaly** — how much yield differs from a county's expected historical trend.
2. **Severe yield-risk classification** — whether a county is likely to fall into a low-yield anomaly group.

---

## Why This Matters

- ~90 million acres of U.S. corn are planted annually; county-level stress signals have real economic relevance.
- Public data (USDA NASS, NASA POWER, U.S. Drought Monitor) is underutilized in integrated, reproducible ML pipelines.
- Interpretable models built on public data can complement — not replace — expert agronomic judgment.

---

## Target Users

- Agricultural data scientists and ML engineers
- Agtech researchers and students
- Crop insurance analysts exploring open-data baselines
- Portfolio reviewers evaluating geospatial / time-series ML skills

---

## Data Sources

| Source | Variables | Granularity |
|---|---|---|
| USDA NASS Quick Stats | Corn yield (bu/acre) | County × Year |
| NASA POWER API | Precip, Tmin, Tmax, Tavg | Daily × County centroid |
| U.S. Drought Monitor | D0–D4 drought category area | Weekly × County |

> MVP scope: **corn only**, **Iowa and Illinois**, **2015 onward**.

---

## Methods

1. **Data ingestion** — modular fetchers for each source with raw → interim → processed pipeline.
2. **Leakage-safe target engineering** — expected yield calculated from prior years only (rolling window or trend).
3. **Weather feature engineering** — growing-season aggregates: cumulative precip, GDD, heat stress days, dry spells.
4. **Drought features** — weekly D0–D4 area statistics aggregated over the growing season.
5. **Modeling** — baseline (historical mean), RandomForest, XGBoost / LightGBM.
6. **Validation** — temporal split (train on earlier years, test on later years); spatial split when ≥ 3 states available.
7. **Interpretability** — feature importance plots, SHAP summary, error analysis by state and year.

---

## Validation Strategy

- **Primary:** temporal split — avoid data leakage across years.
- **Secondary:** spatial hold-out — one state withheld to test geographic generalization.
- Random train/test split is **not** used as the main evaluation.

---

## Results

> Results will be added after model evaluation is complete.

---

## Visual Examples

> Figures will be added after pipeline runs successfully.

---

## How to Run

### 1. Install dependencies

```bash
make install
# or
pip install -r requirements.txt
```

### 2. Configure API keys (if needed)

```bash
cp .env.example .env
# Add your USDA NASS API key if required
```

### 3. Fetch raw data

```bash
make fetch-data
```

### 4. Build feature panel

```bash
make build-features
```

### 5. Train models

```bash
make train
```

### 6. Evaluate

```bash
make evaluate
```

### 7. Launch dashboard

```bash
make app
```

### 8. Run tests

```bash
make test
```

---

## Repository Structure

```
cropshield/
  configs/          ← YAML configuration for states, sources, model hyperparams
  data/
    raw/            ← Original downloaded files (not committed)
    interim/        ← Cleaned but not yet merged
    processed/      ← Modeling-ready files
    external/       ← Reference files (FIPS codes, shapefiles)
  notebooks/        ← EDA and results notebooks
  src/cropshield/   ← Main Python package
    data/           ← Fetchers: NASS, NASA POWER, Drought Monitor
    features/       ← Target engineering, weather features, panel assembly
    models/         ← Baseline and tree-based model training
    evaluation/     ← Metrics, validation splits, error analysis
    interpretability/ ← Feature importance, SHAP
    visualization/  ← Plots and maps
  scripts/          ← End-to-end CLI entry points
  app/              ← Streamlit dashboard
  tests/            ← Unit tests for feature engineering and validation
  reports/          ← Model card, data card, final report, figures
```

---

## Key Findings

> To be updated after model evaluation is complete.

---

## Limitations

- County-level aggregation hides farm-level variation.
- Weather approximated at county centroids in MVP.
- Missing management data (planting date, irrigation, fertilizer) can confound predictions.
- Public datasets may contain missing or suppressed values in sparse counties.
- Feature importance reflects model correlations, not causal relationships.
- Results may not generalize to states or climate years outside the training distribution.

---

## Future Work

- Expand to corn + soybeans across 6 states (Iowa, Illinois, Indiana, Nebraska, Kansas, North Carolina).
- Add USDA Cropland Data Layer for crop-area weighting.
- Integrate SSURGO soil features.
- Add Crop-CASMA soil moisture / vegetation condition.
- Temporal cross-validation across multiple held-out year windows.
- FastAPI endpoint for batch county queries.

---

## Current Status

- [x] Repository scaffolded
- [ ] USDA NASS fetcher implemented
- [ ] Yield target engineering implemented
- [ ] NASA POWER weather features implemented
- [ ] Drought Monitor features implemented
- [ ] Modeling panel assembled
- [ ] Baseline and tree models trained
- [ ] Interpretability analysis complete
- [ ] Streamlit dashboard complete

---

## Resume Bullet

*(To be filled in after results exist)*

> Developed CropShield, an interpretable geospatial ML pipeline for county-level corn and soybean yield-risk forecasting, integrating USDA yield records, NASA weather data, drought severity, and optional soil/crop-condition features with temporal/spatial validation, SHAP-based interpretability, and a Streamlit dashboard.

---

## Author Note

This is a student portfolio and research-style project demonstrating how public agricultural datasets can be integrated into an interpretable early-warning system for crop stress and yield risk. It is **not** a production agronomy tool. Do not use it for farm-level recommendations, insurance pricing, or financial decisions.
