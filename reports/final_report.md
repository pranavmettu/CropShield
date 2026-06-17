# CropShield: Final Report

> **Status:** Work in progress. Sections will be updated as each pipeline milestone is completed.

---

## Abstract

CropShield is an interpretable geospatial machine learning pipeline for
county-level crop yield risk forecasting. We integrate USDA NASS historical
yields, NASA POWER daily weather data, and U.S. Drought Monitor severity
indices into a county-crop-year panel dataset. We train baseline and
tree-based models to predict yield anomaly (deviation from expected trend)
and classify counties at severe yield-risk.

*[Results summary to be added after model evaluation.]*

---

## 1. Introduction

*[To be written after initial results.]*

---

## 2. Data

### 2.1 USDA NASS Yield Data

*[Summary statistics, year/state coverage, missingness rates to be added after ingestion.]*

### 2.2 NASA POWER Weather Data

*[Coverage statistics and feature distributions to be added after ingestion.]*

### 2.3 U.S. Drought Monitor

*[Drought severity statistics and growing-season coverage to be added.]*

---

## 3. Methods

### 3.1 Target Engineering

Yield anomaly is defined as:

```
yield_anomaly = actual_yield - expected_yield
```

Expected yield is computed as the 5-year rolling mean of **prior years only**,
grouped by county and crop. This prevents target leakage: no current-year
information is used to compute the expected yield for that same year.

Severe yield risk is classified as counties falling in the bottom 20th
percentile of yield anomaly for their county-crop historical distribution.

### 3.2 Weather Feature Engineering

Growing-season features are aggregated from daily NASA POWER data
(April 1 – August 31). Features include cumulative precipitation, mean
and maximum temperature, extreme heat days, dry-day counts, longest dry spell,
growing degree days, and precipitation anomaly vs. county history.

### 3.3 Validation Strategy

**Primary validation:** Temporal split — training on years T through T−n,
testing on the most recent n years. This correctly simulates the forecasting
scenario.

**Secondary validation (standout version):** Spatial hold-out — one state
withheld to test geographic generalisation.

Random train/test splits are **not** used as the primary validation metric
because they violate temporal ordering and produce optimistically biased results.

### 3.4 Models

1. **ZeroAnomalyBaseline** — predicts zero anomaly for all inputs
2. **CountyMeanBaseline** — predicts training-set mean anomaly per county
3. **RandomForestRegressor** — sklearn ensemble model
4. **XGBoost / LightGBM** — gradient-boosted trees

---

## 4. Results

*[To be added after model training and evaluation are complete.]*

### 4.1 Regression Performance

| Model | RMSE (bu/acre) | MAE (bu/acre) | R² |
|---|---|---|---|
| ZeroAnomalyBaseline | — | — | — |
| CountyMeanBaseline | — | — | — |
| RandomForest | — | — | — |
| XGBoost | — | — | — |

### 4.2 Classification Performance (Severe Risk)

| Model | Precision | Recall | F1 | AUROC |
|---|---|---|---|---|
| Baseline | — | — | — | — |
| RandomForest | — | — | — | — |
| XGBoost | — | — | — | — |

### 4.3 Feature Importance

*[Figure and discussion to be added.]*

### 4.4 Error Analysis

*[Residuals by year, state, and drought severity to be added.]*

---

## 5. Discussion

*[To be written after results are available.]*

---

## 6. Limitations

- County-level aggregation hides farm-level variation.
- Weather approximated at county centroids.
- No management data (planting date, irrigation, fertiliser).
- MVP covers only two states; results may not generalise broadly.
- Feature importance reflects model correlations, not causal mechanisms.
- The 2015-onward training window may not capture rare extreme events.

---

## 7. Future Work

- Expand to corn + soybeans and six states
- Add USDA Cropland Data Layer for crop-area weighting
- Integrate SSURGO soil features
- Temporal cross-validation across multiple held-out year windows
- FastAPI endpoint for batch county queries
- Scenario tool for "what if drought worsens?" exploration

---

## 8. Conclusion

*[To be written after results are available.]*

---

## References

- USDA NASS. (2024). *Quick Stats Agricultural Database*. https://quickstats.nass.usda.gov/
- Sparks, A.H., et al. (2018). *nasapower: A NASA POWER Global Meteorology, Surface Solar Energy and Climatology Data Client for R*. Journal of Open Source Software.
- Svoboda, M., et al. (2002). *The Drought Monitor*. Bulletin of the American Meteorological Society.
- Mitchell, M., et al. (2019). *Model Cards for Model Reporting*. FAT* 2019.
