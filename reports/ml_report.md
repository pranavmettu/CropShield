# CropShield ML Model Report

Generated: 2026-06-17 22:49

## Setup

- Train rows: 8,545 (years 2018–2022)
- Test rows: 4,450 (years 2023–2025)
- Crops: CORN, SOYBEANS
- Checkpoints: august_31, full_season, july_31, june_30, may_31
- Numeric features (29): expected_yield, prior_year_yield_anomaly, prior_year_yield, rolling_3yr_mean_yield_anomaly, rolling_3yr_std_yield_anomaly, rolling_3yr_mean_yield, rolling_3yr_std_yield, cumulative_precip, mean_temp, max_temp, extreme_heat_days, dry_days, longest_dry_spell, growing_degree_days, obs_days, precip_anomaly, max_consecutive_dry_days, extreme_heat_days_after_july_1, precip_last_30_days_before_checkpoint, heat_days_last_30_days_before_checkpoint, gdd_last_30_days_before_checkpoint, precip_anomaly_from_county_checkpoint_mean, gdd_anomaly_from_county_checkpoint_mean, heat_days_anomaly_from_county_checkpoint_mean, dry_days_anomaly_from_county_checkpoint_mean, temp_mean_anomaly_from_county_checkpoint_mean, precip_pct_of_county_checkpoint_mean, heat_dry_stress, heat_dry_spell
- Categorical features (3): crop, checkpoint, state
- include_county=False, include_year_index=False

## Regression — overall (sorted by RMSE)

| Model | RMSE | MAE | R² |
|-------|------|-----|-----|
| random_forest_reg | 14.635 | 10.272 | -0.399 |
| ridge | 15.495 | 11.812 | -0.569 |
| hist_gradient_boosting_reg | 15.966 | 11.369 | -0.666 |

## Regression — by checkpoint (best model)

| Checkpoint | RMSE | MAE | R² |
|-----------|------|-----|-----|
| august_31 | 14.710 | 10.314 | -0.414 |
| full_season | 14.710 | 10.313 | -0.414 |
| july_31 | 15.035 | 10.555 | -0.477 |
| june_30 | 13.649 | 9.549 | -0.217 |
| may_31 | 15.024 | 10.628 | -0.475 |

## Regression — by crop (best model)

| Crop | RMSE | MAE | R² |
|------|------|-----|-----|
| CORN | 19.843 | 16.247 | -0.613 |
| SOYBEANS | 5.052 | 3.994 | -0.062 |

## Classification — overall (sorted by F1)

| Model | F1 | Accuracy | Precision | Recall |
|-------|-----|----------|-----------|--------|
| logistic_regression | 0.377 | 0.649 | 0.295 | 0.524 |
| random_forest_clf | 0.100 | 0.798 | 0.510 | 0.056 |

## Baseline comparison

- **Best ML regression**: `random_forest_reg` RMSE=14.635, R²=-0.399
- Best baseline (`crop_checkpoint_mean`) RMSE=12.988 → ML does NOT beat the baseline (14.635 vs 12.988)
- **Best ML classifier**: `logistic_regression` F1=0.377, recall=0.524
- Baseline (`historical_county_risk`) F1=0.124, recall=0.072 → ML **beats** F1 and **beats** recall

## Checkpoint trend (does later = better?)

- For `random_forest_reg`, RMSE improves from may_31 (15.024) to full_season (14.710).

## Leakage check

- Target/realised-outcome columns excluded from features: ['actual_yield', 'severe_risk', 'severe_risk_descriptive', 'yield_anomaly', 'yield_anomaly_pct']
- Preprocessing fit on training split only (sklearn Pipeline).
- severe_risk assigned after temporal split via assign_modeling_risk_labels.
