"""
Script 05: Feature-group ablation and per-crop modeling for CropShield.

Trains a fixed set of models across:
  - feature groups: baseline_features, weather_raw, weather_anomalies,
    lagged_yield, drought (if present), all_features
  - modeling modes: pooled (CORN+SOYBEANS), corn_only, soybeans_only

Each non-baseline group adds one feature family on top of the base context
(crop, checkpoint, state, expected_yield), so the ablation measures the marginal
value of each family.  Temporal split only (train 2018–2022, test 2023–2025);
severe_risk is assigned after the split via assign_modeling_risk_labels.

Outputs
-------
- reports/metrics/feature_ablation_metrics.csv
- reports/feature_ablation_report.md
- data/processed/feature_ablation_predictions.csv

Run from the project root.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline

from cropshield.data.fips_utils import normalise_fips_series
from cropshield.evaluation.metrics import classification_metrics, regression_metrics
from cropshield.evaluation.panel_audit import add_checkpoint_column
from cropshield.evaluation.validation_splits import temporal_split
from cropshield.features.yield_targets import assign_modeling_risk_labels
from cropshield.models.feature_sets import available_groups, get_feature_set
from cropshield.models.ml_models import build_preprocessor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

REGRESSION_TARGET = "yield_anomaly"
CLASSIFICATION_TARGET = "severe_risk"
GROUP_COLS = ["crop", "checkpoint", "year"]
RANDOM_STATE = 42
MODES = ["pooled", "corn_only", "soybeans_only"]
META_COLS = ["year", "state", "county_fips", "crop", "checkpoint"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CropShield feature-group ablation.")
    p.add_argument("--panel", default="data/processed/modeling_panel.csv")
    p.add_argument("--test-years", type=int, default=3)
    p.add_argument("--risk-quantile", type=float, default=0.20)
    p.add_argument("--metrics-out", default="reports/metrics/feature_ablation_metrics.csv")
    p.add_argument("--predictions-out", default="data/processed/feature_ablation_predictions.csv")
    p.add_argument("--report-out", default="reports/feature_ablation_report.md")
    return p.parse_args()


def load_panel(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"county_fips": "string"})
    df["county_fips"] = normalise_fips_series(df["county_fips"])
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df = add_checkpoint_column(df)
    for col in ("severe_risk", "severe_risk_descriptive"):
        if col in df.columns:
            df = df.drop(columns=[col])
    return df.dropna(subset=[REGRESSION_TARGET, "checkpoint"]).copy()


def _reg_models(numeric, categorical) -> dict[str, Pipeline]:
    def pipe(est):
        return Pipeline([("preprocessor", build_preprocessor(numeric, categorical)), ("model", est)])
    return {
        "ridge": pipe(Ridge(alpha=1.0, random_state=RANDOM_STATE)),
        "hist_gradient_boosting_reg": pipe(HistGradientBoostingRegressor(random_state=RANDOM_STATE)),
    }


def _clf_models(numeric, categorical) -> dict[str, Pipeline]:
    def pipe(est):
        return Pipeline([("preprocessor", build_preprocessor(numeric, categorical)), ("model", est)])
    return {
        "logistic_regression": pipe(LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)),
        "random_forest_clf": pipe(RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)),
    }


def _grouped_reg(tag, name, y_true, y_pred, meta) -> list[dict]:
    rows = [{**tag, "model": name, "task": "regression", "group": "overall",
             "group_value": "all", **regression_metrics(y_true, y_pred)}]
    for col in GROUP_COLS:
        for val in sorted(meta[col].dropna().unique()):
            m = (meta[col] == val).values
            if m.sum() >= 2:
                rows.append({**tag, "model": name, "task": "regression", "group": col,
                             "group_value": str(val), **regression_metrics(y_true[m], y_pred[m])})
    return rows


def _grouped_clf(tag, name, y_true, y_pred, meta) -> list[dict]:
    rows = [{**tag, "model": name, "task": "classification", "group": "overall",
             "group_value": "all", **classification_metrics(y_true, y_pred)}]
    for col in GROUP_COLS:
        for val in sorted(meta[col].dropna().unique()):
            m = (meta[col] == val).values
            if m.sum() >= 2 and len(np.unique(y_true[m])) >= 2:
                rows.append({**tag, "model": name, "task": "classification", "group": col,
                             "group_value": str(val), **classification_metrics(y_true[m], y_pred[m])})
    return rows


def run_mode(panel: pd.DataFrame, mode: str, args) -> tuple[list[dict], list[pd.DataFrame]]:
    if mode == "corn_only":
        sub = panel[panel["crop"] == "CORN"].copy()
    elif mode == "soybeans_only":
        sub = panel[panel["crop"] == "SOYBEANS"].copy()
    else:
        sub = panel.copy()

    # Guard: per-crop modes must not include the other crop
    if mode == "corn_only":
        assert set(sub["crop"].unique()) == {"CORN"}, "corn_only leaked soybeans"
    elif mode == "soybeans_only":
        assert set(sub["crop"].unique()) == {"SOYBEANS"}, "soybeans_only leaked corn"

    train_raw, test_raw = temporal_split(sub, n_test_years=args.test_years)
    train_df, test_df = assign_modeling_risk_labels(train_raw, test_raw, quantile=args.risk_quantile)

    metric_rows: list[dict] = []
    pred_frames: list[pd.DataFrame] = []

    for group in available_groups(panel):
        numeric, categorical = get_feature_set(panel, group)
        feats = numeric + categorical
        tag = {"mode": mode, "feature_group": group}

        # Regression
        y_tr, y_te = train_df[REGRESSION_TARGET].values, test_df[REGRESSION_TARGET].values
        for name, p in _reg_models(numeric, categorical).items():
            p.fit(train_df[feats], y_tr)
            y_pred = p.predict(test_df[feats])
            metric_rows.extend(_grouped_reg(tag, name, y_te, y_pred, test_df))
            pred_frames.append(test_df[META_COLS].assign(
                mode=mode, feature_group=group, model_name=name, task="regression",
                y_true=y_te, y_pred=y_pred, y_proba=np.nan))

        # Classification
        tr_c = train_df.dropna(subset=[CLASSIFICATION_TARGET])
        te_c = test_df.dropna(subset=[CLASSIFICATION_TARGET])
        if tr_c[CLASSIFICATION_TARGET].nunique() < 2 or len(te_c) == 0:
            logger.warning("mode=%s group=%s: insufficient class variety, skipping clf", mode, group)
            continue
        yc_tr = tr_c[CLASSIFICATION_TARGET].astype(int).values
        yc_te = te_c[CLASSIFICATION_TARGET].astype(int).values
        for name, p in _clf_models(numeric, categorical).items():
            p.fit(tr_c[feats], yc_tr)
            y_pred = p.predict(te_c[feats]).astype(int)
            y_proba = p.predict_proba(te_c[feats])[:, 1] if hasattr(p, "predict_proba") else np.nan
            metric_rows.extend(_grouped_clf(tag, name, yc_te, y_pred, te_c))
            pred_frames.append(te_c[META_COLS].assign(
                mode=mode, feature_group=group, model_name=name, task="classification",
                y_true=yc_te, y_pred=y_pred, y_proba=y_proba))

    return metric_rows, pred_frames


def write_report(metrics: pd.DataFrame, path: Path, drought_present: bool) -> None:
    reg = metrics[(metrics.task == "regression") & (metrics.group == "overall")]
    cls = metrics[(metrics.task == "classification") & (metrics.group == "overall")]

    lines = [
        "# CropShield Feature Ablation Report",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "Each non-baseline group adds one feature family on top of the base "
        "context (crop, checkpoint, state, expected_yield). Temporal split: "
        "train 2018–2022, test 2023–2025.",
        "",
        f"- Drought features: {'integrated' if drought_present else 'SKIPPED (no raw drought CSV present)'}",
        "",
        "## Best regression (lowest RMSE) per mode × feature group",
        "",
        "| Mode | Feature group | Model | RMSE | MAE | R² |",
        "|------|---------------|-------|------|-----|-----|",
    ]
    best_reg = (reg.sort_values("rmse").groupby(["mode", "feature_group"], as_index=False).first())
    for _, r in best_reg.sort_values(["mode", "rmse"]).iterrows():
        lines.append(f"| {r['mode']} | {r['feature_group']} | {r['model']} | "
                     f"{r['rmse']:.3f} | {r['mae']:.3f} | {r['r2']:.3f} |")

    lines += ["", "## Best classification (highest F1) per mode × feature group", "",
              "| Mode | Feature group | Model | F1 | Recall | Precision | Acc |",
              "|------|---------------|-------|-----|--------|-----------|-----|"]
    best_cls = (cls.sort_values("f1", ascending=False).groupby(["mode", "feature_group"], as_index=False).first())
    for _, r in best_cls.sort_values(["mode", "f1"], ascending=[True, False]).iterrows():
        lines.append(f"| {r['mode']} | {r['feature_group']} | {r['model']} | "
                     f"{r['f1']:.3f} | {r['recall']:.3f} | {r['precision']:.3f} | {r['accuracy']:.3f} |")

    # Headline winners
    lines += ["", "## Headlines", ""]
    if len(reg):
        br = reg.sort_values("rmse").iloc[0]
        lines.append(f"- **Lowest RMSE overall**: `{br['model']}` on "
                     f"`{br['feature_group']}` / `{br['mode']}` — RMSE {br['rmse']:.3f}, R² {br['r2']:.3f}")
        pooled_reg = reg[reg["mode"] == "pooled"].sort_values("rmse")
        if len(pooled_reg):
            pr = pooled_reg.iloc[0]
            beats = pr["rmse"] < 12.99
            lines.append(
                f"- **Best POOLED regression** (apples-to-apples vs the pooled "
                f"crop_checkpoint_mean baseline RMSE 12.99): `{pr['model']}` on "
                f"`{pr['feature_group']}` — RMSE {pr['rmse']:.3f} → "
                f"{'**beats**' if beats else 'does NOT beat'} the baseline."
            )
        lines.append(
            "- NOTE: per-crop RMSE (e.g. soybeans_only ≈ 5) is *not* comparable to "
            "the pooled baseline — soybean anomalies live on a much smaller scale "
            "(~±5 bu/acre) than corn (~±18). Compare within the same crop subset."
        )
    if len(cls):
        bc = cls.sort_values("f1", ascending=False).iloc[0]
        lines.append(f"- **Best classification overall**: `{bc['model']}` on "
                     f"`{bc['feature_group']}` / `{bc['mode']}` — F1 {bc['f1']:.3f}, recall {bc['recall']:.3f} "
                     f"(prior best from script 04: F1 0.266, recall 0.357)")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    logger.info("Ablation report saved → %s", path)


def main() -> None:
    args = parse_args()
    logger.info("=== CropShield: Step 5 — Feature Ablation ===")

    panel_path = Path(args.panel)
    if not panel_path.exists():
        logger.error("Panel not found at %s. Run scripts 01–02 first.", panel_path)
        sys.exit(1)

    panel = load_panel(panel_path)
    drought_present = "mean_drought_severity" in panel.columns
    logger.info("Feature groups available: %s", available_groups(panel))

    all_metrics: list[dict] = []
    all_preds: list[pd.DataFrame] = []
    for mode in MODES:
        logger.info("--- Mode: %s ---", mode)
        m, p = run_mode(panel, mode, args)
        all_metrics.extend(m)
        all_preds.extend(p)

    metrics_df = pd.DataFrame(all_metrics)
    metrics_path = Path(args.metrics_out)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(metrics_path, index=False)
    logger.info("Ablation metrics saved → %s (%d rows)", metrics_path, len(metrics_df))

    preds = pd.concat(all_preds, ignore_index=True)
    pred_path = Path(args.predictions_out)
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(pred_path, index=False)
    logger.info("Ablation predictions saved → %s (%d rows)", pred_path, len(preds))

    write_report(metrics_df, Path(args.report_out), drought_present)

    # Console summary
    reg = metrics_df[(metrics_df.task == "regression") & (metrics_df.group == "overall")]
    print("\n── Ablation: regression RMSE by mode × feature group ─────────")
    pivot = reg.pivot_table(index="feature_group", columns="mode", values="rmse", aggfunc="min")
    print(pivot.round(3).to_string())
    cls = metrics_df[(metrics_df.task == "classification") & (metrics_df.group == "overall")]
    print("\n── Ablation: best classification F1 by mode × feature group ──")
    pivotc = cls.pivot_table(index="feature_group", columns="mode", values="f1", aggfunc="max")
    print(pivotc.round(3).to_string())
    print()

    logger.info("=== Step 5 complete ===")


if __name__ == "__main__":
    main()
