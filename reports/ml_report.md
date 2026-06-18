# CropShield ML Model Report

Generated: 2026-06-17 22:34

## Setup

- Train rows: 8,545 (years 2018–2022)
- Test rows: 4,450 (years 2023–2025)
- Crops: CORN, SOYBEANS
- Checkpoints: august_31, full_season, july_31, june_30, may_31
- Numeric features (12): expected_yield, cumulative_precip, mean_temp, max_temp, extreme_heat_days, dry_days, longest_dry_spell, growing_degree_days, obs_days, precip_anomaly, heat_dry_stress, heat_dry_spell
- Categorical features (3): crop, checkpoint, state
- include_county=False, include_year_index=False

## Regression — overall (sorted by RMSE)

| Model | RMSE | MAE | R² |
|-------|------|-----|-----|
| ridge | 15.230 | 11.725 | -0.516 |
| hist_gradient_boosting_reg | 15.438 | 10.969 | -0.557 |
| random_forest_reg | 15.699 | 10.991 | -0.610 |

## Regression — by checkpoint (best model)

| Checkpoint | RMSE | MAE | R² |
|-----------|------|-----|-----|
| august_31 | 15.400 | 12.013 | -0.550 |
| full_season | 15.400 | 12.013 | -0.550 |
| july_31 | 15.956 | 12.506 | -0.664 |
| june_30 | 14.741 | 11.231 | -0.420 |
| may_31 | 14.613 | 10.862 | -0.395 |

## Regression — by crop (best model)

| Crop | RMSE | MAE | R² |
|------|------|-----|-----|
| CORN | 19.695 | 16.235 | -0.589 |
| SOYBEANS | 8.252 | 6.986 | -1.833 |

## Classification — overall (sorted by F1)

| Model | F1 | Accuracy | Precision | Recall |
|-------|-----|----------|-----------|--------|
| logistic_regression | 0.266 | 0.601 | 0.212 | 0.357 |
| random_forest_clf | 0.266 | 0.796 | 0.494 | 0.182 |

## Baseline comparison

- **Best ML regression**: `ridge` RMSE=15.230, R²=-0.516
- Best baseline (`crop_checkpoint_mean`) RMSE=12.988 → ML does NOT beat the baseline (15.230 vs 12.988)
- **Best ML classifier**: `logistic_regression` F1=0.266, recall=0.357
- Baseline (`historical_county_risk`) F1=0.124, recall=0.072 → ML **beats** F1 and **beats** recall

## Checkpoint trend (does later = better?)

- For `ridge`, RMSE does NOT clearly improve from may_31 (14.613) to full_season (15.400).

## Leakage check

- Target/realised-outcome columns excluded from features: ['actual_yield', 'severe_risk', 'severe_risk_descriptive', 'yield_anomaly', 'yield_anomaly_pct']
- Preprocessing fit on training split only (sklearn Pipeline).
- severe_risk assigned after temporal split via assign_modeling_risk_labels.
