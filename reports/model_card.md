# CropShield Model Card

## Model Purpose

This model predicts county-level crop yield anomaly and severe yield-risk status
using public agricultural, weather, and drought data. It is designed as an
interpretable early-warning signal for in-season crop stress, not as an
official yield forecast.

---

## Intended Use

- Student portfolio demonstration
- Research-style exploration of public agricultural data integration
- Agricultural risk signal visualisation
- Hypothesis generation for crop monitoring research

---

## Not Intended For

- Farm-level agronomic recommendations
- Insurance pricing or underwriting
- Official yield forecasting or USDA projections
- Financial or commodity trading decisions
- Replacing USDA, extension service, or certified agronomist expertise

---

## Model Outputs

| Output | Type | Description |
|---|---|---|
| `yield_anomaly_pred` | Regression (float) | Predicted deviation from expected yield (bu/acre) |
| `severe_risk_pred` | Classification (binary) | 1 if county is predicted in the bottom yield-anomaly quantile |
| `severe_risk_prob` | Probability (float, optional) | Predicted probability of severe risk |

---

## Training Data

| Attribute | Value |
|---|---|
| Crops | Corn (MVP); soybeans planned |
| States | Iowa, Illinois (MVP); Indiana, Nebraska, Kansas, North Carolina planned |
| Years | 2015 through latest available NASS year |
| Unit of analysis | County × crop × year |
| Yield source | USDA NASS Quick Stats county-level yields |
| Weather source | NASA POWER daily API at county centroids |
| Drought source | U.S. Drought Monitor weekly county statistics |

---

## Features

### Weather (growing season: April 1 – August 31)
- `cumulative_precip` — Total precipitation (mm)
- `mean_temp` — Mean daily temperature (°C)
- `max_temp` — Maximum daily temperature (°C)
- `extreme_heat_days` — Days with Tmax ≥ 35°C
- `dry_days` — Days with precip ≤ 1 mm
- `longest_dry_spell` — Longest consecutive dry-day run (days)
- `growing_degree_days` — Accumulated GDD with 10°C base
- `precip_anomaly` — Departure from county historical growing-season precipitation

### Drought
- `d0_max_pct` through `d4_max_pct` — Maximum weekly area in each drought category (%)
- `d2_plus_weeks` — Weeks with severe or worse drought (D2+) coverage > 0%
- `d2_plus_max_pct` — Maximum weekly D2+ area (%)

### Historical yield
- `expected_yield` — Rolling 5-year mean from prior years only (leakage-safe)
- `year_index` — Normalised year for trend-aware features

---

## Validation Strategy

**Primary: Temporal split**
- Training set: earlier years
- Test set: most recent 3 years held out
- Rationale: simulates realistic forecasting scenario; prevents future-year leakage

**Secondary: Spatial split (standout version)**
- One state withheld to test geographic generalisation

**Not used as primary metric:** random train/test splits (inappropriate for time series)

---

## Evaluation Metrics

*(To be filled in after model training is complete)*

| Model | RMSE (bu/acre) | MAE (bu/acre) | R² | F1 (severe risk) |
|---|---|---|---|---|
| ZeroAnomalyBaseline | — | — | — | — |
| CountyMeanBaseline | — | — | — | — |
| RandomForest | — | — | — | — |
| XGBoost | — | — | — | — |

---

## Limitations

- **County-level data** hides farm-level variation in management, soil, and microclimate.
- **Missing management data** (planting date, irrigation, fertiliser) can confound predictions.
- **Suppressed NASS records** in counties with fewer than 3 reporting operations create gaps.
- **Weather at centroids** misses spatial heterogeneity within large counties.
- **Two-state MVP** scope limits generalisation to other geographies.
- **Feature importance ≠ causation.** High importance scores describe model reliance, not causal mechanisms.
- **Historical training window:** performance in extreme or novel climate conditions (outside the 2015+ training distribution) is uncertain.
- **Temporal autocorrelation:** county yield patterns may persist across years in ways the model has not captured.

---

## Ethical Considerations

- The dashboard must clearly communicate uncertainty and the portfolio/research-only nature of the project.
- Do not present predictions as official USDA forecasts or as inputs for insurance or financial decisions.
- Predictions for individual counties should always be accompanied by confidence ranges or disclaimers when possible.
- Consider potential harms if predictions were misapplied in high-stakes agricultural decisions.

---

## Caveats on Interpretation

Results published from this model should be framed as:
- "risk signals" or "yield anomaly estimates"
- "decision-support indicators" for further investigation
- "research-grade outputs" not "production forecasts"

---

*This model card template follows the framework introduced by Mitchell et al. (2019). "Model Cards for Model Reporting."*
