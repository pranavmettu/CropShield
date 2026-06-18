# CropShield Baseline Report

Generated: 2026-06-17 21:37

## Data

- Panel rows: 12,995
- Train rows: 8,545 (years 2018–2022)
- Test rows: 4,450 (years 2023–2025)
- Counties: 199
- Crops: CORN, SOYBEANS

## Best regression baseline (lowest RMSE)

- **crop_checkpoint_mean**: RMSE=12.988, MAE=9.217, R²=-0.102

## Best classification baseline (highest F1)

- **historical_county_risk**: F1=0.124, accuracy=0.794, recall=0.072

## All regression baselines (overall)

| Model | RMSE | MAE | R² |
|-------|------|-----|-----|
| crop_checkpoint_mean | 12.988 | 9.217 | -0.102 |
| zero_anomaly | 13.832 | 10.068 | -0.250 |
| previous_year_anomaly | 16.216 | 11.233 | -0.718 |
| county_historical_mean | 16.429 | 11.581 | -0.764 |

## All classification baselines (overall)

| Model | F1 | Accuracy | Precision | Recall |
|-------|-----|----------|-----------|--------|
| historical_county_risk | 0.124 | 0.794 | 0.448 | 0.072 |
| majority_class | 0.000 | 0.797 | 0.000 | 0.000 |
