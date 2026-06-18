"""
Baseline models for CropShield.

Simple, leakage-safe predictors for yield-anomaly regression and
severe-risk classification.  All baselines that learn from data use
**training rows only** during ``fit()``; prediction-time lookups for
lag features use only prior years (never future or same-year labels).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

GROUP_KEYS = ["county_fips", "crop"]
CHECKPOINT_KEYS = ["crop", "checkpoint"]


# ── Regression baselines ───────────────────────────────────────────────────────

class ZeroAnomalyBaseline:
    """Predict yield_anomaly = 0 (normal relative to expectation)."""

    name = "zero_anomaly"

    def fit(self, train_df: pd.DataFrame, target_col: str = "yield_anomaly") -> ZeroAnomalyBaseline:
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(df))


class PreviousYearAnomalyBaseline:
    """Predict the previous calendar year's yield_anomaly per county-crop."""

    name = "previous_year_anomaly"

    def fit(self, train_df: pd.DataFrame, target_col: str = "yield_anomaly") -> PreviousYearAnomalyBaseline:
        self._history = (
            train_df[["county_fips", "crop", "year", target_col]]
            .dropna(subset=[target_col])
            .copy()
        )
        return self

    def predict(self, df: pd.DataFrame, target_col: str = "yield_anomaly") -> np.ndarray:
        """Use anomalies from years strictly before each row's year."""
        history = self._history.copy()
        pred_map: dict[int, float] = {}

        for idx, row in df.sort_values(["county_fips", "crop", "year"]).iterrows():
            prior = history[
                (history["county_fips"] == row["county_fips"])
                & (history["crop"] == row["crop"])
                & (history["year"] < row["year"])
            ]
            if prior.empty:
                pred_map[idx] = 0.0
            else:
                pred_map[idx] = float(prior.sort_values("year").iloc[-1][target_col])
            history = pd.concat([
                history,
                pd.DataFrame([{
                    "county_fips": row["county_fips"],
                    "crop": row["crop"],
                    "year": row["year"],
                    target_col: row[target_col],
                }]),
            ], ignore_index=True)

        return df.index.map(pred_map).astype(float).values


class CountyHistoricalMeanBaseline:
    """County-crop mean anomaly from training data; crop-mean then 0 fallback."""

    name = "county_historical_mean"

    def fit(self, train_df: pd.DataFrame, target_col: str = "yield_anomaly") -> CountyHistoricalMeanBaseline:
        self._county_means = (
            train_df.groupby(GROUP_KEYS)[target_col].mean().to_dict()
        )
        self._crop_means = train_df.groupby("crop")[target_col].mean().to_dict()
        self._global_mean = float(train_df[target_col].mean()) if len(train_df) else 0.0
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        preds = []
        for row in df.itertuples(index=False):
            key = (row.county_fips, row.crop)
            if key in self._county_means:
                preds.append(self._county_means[key])
            elif row.crop in self._crop_means:
                preds.append(self._crop_means[row.crop])
            else:
                preds.append(self._global_mean if self._global_mean else 0.0)
        return np.array(preds)


class CropCheckpointMeanBaseline:
    """Mean anomaly by crop + checkpoint from training data only."""

    name = "crop_checkpoint_mean"

    def fit(self, train_df: pd.DataFrame, target_col: str = "yield_anomaly") -> CropCheckpointMeanBaseline:
        if "checkpoint" not in train_df.columns:
            raise KeyError("crop_checkpoint_mean requires a 'checkpoint' column")
        self._ckpt_means = (
            train_df.groupby(CHECKPOINT_KEYS)[target_col].mean().to_dict()
        )
        self._crop_means = train_df.groupby("crop")[target_col].mean().to_dict()
        self._global_mean = float(train_df[target_col].mean()) if len(train_df) else 0.0
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        preds = []
        for row in df.itertuples(index=False):
            key = (row.crop, row.checkpoint)
            if key in self._ckpt_means:
                preds.append(self._ckpt_means[key])
            elif row.crop in self._crop_means:
                preds.append(self._crop_means[row.crop])
            else:
                preds.append(self._global_mean if self._global_mean else 0.0)
        return np.array(preds)


# ── Classification baselines ───────────────────────────────────────────────────

class MajorityClassBaseline:
    """Always predict the most frequent severe_risk label in training data."""

    name = "majority_class"

    def fit(self, train_df: pd.DataFrame, label_col: str = "severe_risk") -> MajorityClassBaseline:
        valid = train_df[label_col].dropna()
        self._majority = int(valid.mode().iloc[0]) if len(valid) else 0
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.full(len(df), self._majority)


class HistoricalCountyRiskBaseline:
    """Predict severe risk if county-crop training rate exceeds threshold."""

    name = "historical_county_risk"

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold

    def fit(self, train_df: pd.DataFrame, label_col: str = "severe_risk") -> HistoricalCountyRiskBaseline:
        valid = train_df.dropna(subset=[label_col])
        self._county_rates = (
            valid.groupby(GROUP_KEYS)[label_col].mean().to_dict()
        )
        self._crop_rates = valid.groupby("crop")[label_col].mean().to_dict()
        self._global_rate = float(valid[label_col].mean()) if len(valid) else 0.0
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        preds = []
        for row in df.itertuples(index=False):
            key = (row.county_fips, row.crop)
            if key in self._county_rates:
                rate = self._county_rates[key]
            elif row.crop in self._crop_rates:
                rate = self._crop_rates[row.crop]
            else:
                rate = self._global_rate
            preds.append(1 if rate >= self.threshold else 0)
        return np.array(preds)


# ── Registry ───────────────────────────────────────────────────────────────────

def get_regression_baselines() -> list[Any]:
    return [
        ZeroAnomalyBaseline(),
        PreviousYearAnomalyBaseline(),
        CountyHistoricalMeanBaseline(),
        CropCheckpointMeanBaseline(),
    ]


def get_classification_baselines(threshold: float = 0.5) -> list[Any]:
    return [
        MajorityClassBaseline(),
        HistoricalCountyRiskBaseline(threshold=threshold),
    ]
