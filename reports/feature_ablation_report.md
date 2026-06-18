# CropShield Feature Ablation Report

Generated: 2026-06-17 22:51

Each non-baseline group adds one feature family on top of the base context (crop, checkpoint, state, expected_yield). Temporal split: train 2018–2022, test 2023–2025.

- Drought features: SKIPPED (no raw drought CSV present)

## Best regression (lowest RMSE) per mode × feature group

| Mode | Feature group | Model | RMSE | MAE | R² |
|------|---------------|-------|------|-----|-----|
| corn_only | all_features | hist_gradient_boosting_reg | 18.399 | 14.835 | -0.387 |
| corn_only | weather_anomalies | ridge | 18.824 | 15.401 | -0.451 |
| corn_only | weather_raw | hist_gradient_boosting_reg | 19.105 | 15.540 | -0.495 |
| corn_only | baseline_features | ridge | 19.405 | 16.032 | -0.542 |
| corn_only | lagged_yield | ridge | 21.914 | 18.120 | -0.967 |
| pooled | all_features | hist_gradient_boosting_reg | 13.713 | 9.431 | -0.229 |
| pooled | weather_raw | hist_gradient_boosting_reg | 13.832 | 9.687 | -0.250 |
| pooled | weather_anomalies | ridge | 13.945 | 10.441 | -0.271 |
| pooled | baseline_features | ridge | 14.276 | 10.538 | -0.332 |
| pooled | lagged_yield | ridge | 15.726 | 11.618 | -0.616 |
| soybeans_only | weather_raw | hist_gradient_boosting_reg | 5.029 | 3.938 | -0.052 |
| soybeans_only | all_features | hist_gradient_boosting_reg | 5.182 | 4.000 | -0.117 |
| soybeans_only | baseline_features | ridge | 5.435 | 4.417 | -0.229 |
| soybeans_only | lagged_yield | hist_gradient_boosting_reg | 5.625 | 4.504 | -0.316 |
| soybeans_only | weather_anomalies | ridge | 5.661 | 4.527 | -0.333 |

## Best classification (highest F1) per mode × feature group

| Mode | Feature group | Model | F1 | Recall | Precision | Acc |
|------|---------------|-------|-----|--------|-----------|-----|
| corn_only | baseline_features | logistic_regression | 0.427 | 0.759 | 0.297 | 0.627 |
| corn_only | lagged_yield | logistic_regression | 0.351 | 0.771 | 0.227 | 0.477 |
| corn_only | all_features | logistic_regression | 0.309 | 0.395 | 0.254 | 0.677 |
| corn_only | weather_raw | logistic_regression | 0.295 | 0.477 | 0.214 | 0.582 |
| corn_only | weather_anomalies | random_forest_clf | 0.267 | 0.214 | 0.353 | 0.784 |
| pooled | baseline_features | logistic_regression | 0.399 | 0.633 | 0.291 | 0.612 |
| pooled | lagged_yield | logistic_regression | 0.383 | 0.800 | 0.252 | 0.478 |
| pooled | weather_anomalies | random_forest_clf | 0.327 | 0.260 | 0.442 | 0.783 |
| pooled | weather_raw | logistic_regression | 0.327 | 0.480 | 0.248 | 0.599 |
| pooled | all_features | logistic_regression | 0.313 | 0.268 | 0.375 | 0.761 |
| soybeans_only | lagged_yield | logistic_regression | 0.384 | 0.763 | 0.257 | 0.454 |
| soybeans_only | weather_raw | logistic_regression | 0.363 | 0.489 | 0.289 | 0.617 |
| soybeans_only | baseline_features | logistic_regression | 0.352 | 0.546 | 0.260 | 0.551 |
| soybeans_only | weather_anomalies | random_forest_clf | 0.348 | 0.254 | 0.557 | 0.788 |
| soybeans_only | all_features | logistic_regression | 0.155 | 0.093 | 0.464 | 0.773 |

## Headlines

- **Lowest RMSE overall**: `hist_gradient_boosting_reg` on `weather_raw` / `soybeans_only` — RMSE 5.029, R² -0.052
- **Best POOLED regression** (apples-to-apples vs the pooled crop_checkpoint_mean baseline RMSE 12.99): `hist_gradient_boosting_reg` on `all_features` — RMSE 13.713 → does NOT beat the baseline.
- NOTE: per-crop RMSE (e.g. soybeans_only ≈ 5) is *not* comparable to the pooled baseline — soybean anomalies live on a much smaller scale (~±5 bu/acre) than corn (~±18). Compare within the same crop subset.
- **Best classification overall**: `logistic_regression` on `baseline_features` / `corn_only` — F1 0.427, recall 0.759 (prior best from script 04: F1 0.266, recall 0.357)