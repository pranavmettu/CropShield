"""
Script 04: Train the first leakage-safe ML models for CropShield.

Pipeline
--------
1. Load data/processed/modeling_panel.csv and run a panel sanity check.
2. Temporal split (train on early years, test on latest years).
3. Assign modeling-safe severe_risk labels AFTER the split
   (thresholds computed on training data only).
4. Select leakage-safe features (targets / realised outcomes excluded).
5. Train regression models (Ridge, RandomForest, HistGradientBoosting) and
   classification models (LogisticRegression, RandomForest), each wrapped in a
   Pipeline so preprocessing is fit on training data only.
6. Evaluate overall and by crop / checkpoint / year / crop×checkpoint.
7. Compare against reports/metrics/baseline_metrics.csv.
8. Save ml_metrics.csv, ml_report.md, ml_predictions.csv.

Usage
-----
    python scripts/04_train_ml_models.py
    python scripts/04_train_ml_models.py --test-years 3 --include-year-index

Run from the project root directory.
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

from cropshield.data.fips_utils import normalise_fips_series
from cropshield.evaluation.metrics import classification_metrics, regression_metrics
from cropshield.evaluation.panel_audit import add_checkpoint_column
from cropshield.evaluation.validation_splits import temporal_split
from cropshield.features.yield_targets import assign_modeling_risk_labels
from cropshield.models.ml_models import (
    LEAKAGE_COLUMNS,
    build_classification_models,
    build_regression_models,
    select_feature_columns,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = {
    "county_fips", "state", "county", "crop", "year", "checkpoint",
    "expected_yield", "yield_anomaly", "yield_anomaly_pct",
}
PANEL_KEYS = ["county_fips", "crop", "year", "checkpoint"]
REGRESSION_TARGET = "yield_anomaly"
CLASSIFICATION_TARGET = "severe_risk"
GROUP_COLS = ["crop", "checkpoint", "year"]


# ── CLI ─────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CropShield first ML models.")
    parser.add_argument("--panel", default="data/processed/modeling_panel.csv")
    parser.add_argument("--test-years", type=int, default=3, help="Most recent years held out")
    parser.add_argument("--risk-quantile", type=float, default=0.20)
    parser.add_argument(
        "--include-county", action="store_true",
        help="Use county_fips as a high-cardinality categorical feature",
    )
    parser.add_argument(
        "--include-year-index", action="store_true",
        help="Include year_index as a numeric trend feature (tests temporal shortcut)",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--baseline-metrics", default="reports/metrics/baseline_metrics.csv")
    parser.add_argument("--metrics-out", default="reports/metrics/ml_metrics.csv")
    parser.add_argument("--predictions-out", default="data/processed/ml_predictions.csv")
    parser.add_argument("--report-out", default="reports/ml_report.md")
    return parser.parse_args()


# ── Loading + sanity checks ───────────────────────────────────────────────────

def load_panel(path: Path) -> pd.DataFrame:
    """Load and sanitise the modeling panel."""
    df = pd.read_csv(path, dtype={"county_fips": "string"})
    df["county_fips"] = normalise_fips_series(df["county_fips"])
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df = add_checkpoint_column(df)
    # Descriptive label must never reach modeling/eval.
    if "severe_risk_descriptive" in df.columns:
        df = df.drop(columns=["severe_risk_descriptive"])
    # Existing severe_risk (if any) is dropped; it is re-assigned post-split.
    if "severe_risk" in df.columns:
        df = df.drop(columns=["severe_risk"])
    return df


def panel_sanity_check(df: pd.DataFrame) -> None:
    """Validate the panel before modeling; raise on any violation."""
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise KeyError(f"Panel missing required columns: {sorted(missing)}")

    # Drop rows with NaN checkpoint (unmatched weather) before key uniqueness check
    keyed = df.dropna(subset=["checkpoint"])
    dupes = keyed.duplicated(subset=PANEL_KEYS).sum()
    if dupes > 0:
        raise ValueError(f"Panel has {dupes} duplicate rows on {PANEL_KEYS}")

    if "severe_risk_descriptive" in df.columns:
        raise AssertionError("severe_risk_descriptive must be dropped before modeling")

    logger.info(
        "Panel sanity check passed: %d rows | %d counties | crops=%s | checkpoints=%s",
        len(df),
        df["county_fips"].nunique(),
        sorted(df["crop"].unique()),
        sorted(df["checkpoint"].dropna().unique()),
    )


def assert_temporal_split_safe(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Raise if train and test years overlap."""
    train_years = set(int(y) for y in train["year"].unique())
    test_years = set(int(y) for y in test["year"].unique())
    overlap = train_years & test_years
    if overlap:
        raise AssertionError(f"Train/test year overlap detected: {sorted(overlap)}")
    logger.info(
        "Temporal split safe: train years %s, test years %s",
        sorted(train_years), sorted(test_years),
    )


# ── Evaluation helpers ────────────────────────────────────────────────────────

def _grouped_regression(name, y_true, y_pred, meta, group_cols) -> list[dict]:
    rows = [{
        "model": name, "task": "regression", "group": "overall", "group_value": "all",
        **regression_metrics(y_true, y_pred, model_name=name),
    }]
    # single-column groups
    for col in group_cols:
        for val in sorted(meta[col].dropna().unique()):
            mask = (meta[col] == val).values
            if mask.sum() < 2:
                continue
            rows.append({
                "model": name, "task": "regression", "group": col, "group_value": str(val),
                **regression_metrics(y_true[mask], y_pred[mask]),
            })
    # crop × checkpoint
    if {"crop", "checkpoint"}.issubset(meta.columns):
        for crop in sorted(meta["crop"].dropna().unique()):
            for ckpt in sorted(meta["checkpoint"].dropna().unique()):
                mask = ((meta["crop"] == crop) & (meta["checkpoint"] == ckpt)).values
                if mask.sum() < 2:
                    continue
                rows.append({
                    "model": name, "task": "regression",
                    "group": "crop_checkpoint", "group_value": f"{crop}/{ckpt}",
                    **regression_metrics(y_true[mask], y_pred[mask]),
                })
    return rows


def _grouped_classification(name, y_true, y_pred, meta, group_cols) -> list[dict]:
    rows = [{
        "model": name, "task": "classification", "group": "overall", "group_value": "all",
        **classification_metrics(y_true, y_pred, model_name=name),
    }]
    for col in group_cols:
        for val in sorted(meta[col].dropna().unique()):
            mask = (meta[col] == val).values
            if mask.sum() < 2 or len(np.unique(y_true[mask])) < 2:
                continue
            rows.append({
                "model": name, "task": "classification", "group": col, "group_value": str(val),
                **classification_metrics(y_true[mask], y_pred[mask]),
            })
    if {"crop", "checkpoint"}.issubset(meta.columns):
        for crop in sorted(meta["crop"].dropna().unique()):
            for ckpt in sorted(meta["checkpoint"].dropna().unique()):
                mask = ((meta["crop"] == crop) & (meta["checkpoint"] == ckpt)).values
                if mask.sum() < 2 or len(np.unique(y_true[mask])) < 2:
                    continue
                rows.append({
                    "model": name, "task": "classification",
                    "group": "crop_checkpoint", "group_value": f"{crop}/{ckpt}",
                    **classification_metrics(y_true[mask], y_pred[mask]),
                })
    return rows


# ── Reporting ───────────────────────────────────────────────────────────────────

def _load_baseline_metrics(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        logger.warning("Baseline metrics not found at %s — skipping comparison", path)
        return None
    return pd.read_csv(path)


def write_report(
    ml_metrics: pd.DataFrame,
    baseline_metrics: pd.DataFrame | None,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_info: dict,
    path: Path,
) -> None:
    reg = ml_metrics[(ml_metrics.task == "regression") & (ml_metrics.group == "overall")].sort_values("rmse")
    cls = ml_metrics[(ml_metrics.task == "classification") & (ml_metrics.group == "overall")].sort_values("f1", ascending=False)

    best_reg = reg.iloc[0] if len(reg) else None
    best_cls = cls.iloc[0] if len(cls) else None

    # Baseline references
    base_reg_rmse = base_cls_f1 = base_cls_recall = None
    best_base_reg_name = best_base_cls_name = None
    if baseline_metrics is not None:
        b_reg = baseline_metrics[(baseline_metrics.task == "regression") & (baseline_metrics.group == "overall")].sort_values("rmse")
        b_cls = baseline_metrics[(baseline_metrics.task == "classification") & (baseline_metrics.group == "overall")].sort_values("f1", ascending=False)
        if len(b_reg):
            base_reg_rmse = float(b_reg.iloc[0]["rmse"])
            best_base_reg_name = b_reg.iloc[0]["model"]
        if len(b_cls):
            base_cls_f1 = float(b_cls.iloc[0]["f1"])
            base_cls_recall = float(b_cls.iloc[0]["recall"])
            best_base_cls_name = b_cls.iloc[0]["model"]

    lines = [
        "# CropShield ML Model Report",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Setup",
        "",
        f"- Train rows: {len(train_df):,} (years {int(train_df['year'].min())}–{int(train_df['year'].max())})",
        f"- Test rows: {len(test_df):,} (years {int(test_df['year'].min())}–{int(test_df['year'].max())})",
        f"- Crops: {', '.join(sorted(train_df['crop'].unique()))}",
        f"- Checkpoints: {', '.join(sorted(train_df['checkpoint'].dropna().unique()))}",
        f"- Numeric features ({len(feature_info['numeric'])}): {', '.join(feature_info['numeric'])}",
        f"- Categorical features ({len(feature_info['categorical'])}): {', '.join(feature_info['categorical'])}",
        f"- include_county={feature_info['include_county']}, include_year_index={feature_info['include_year_index']}",
        "",
        "## Regression — overall (sorted by RMSE)",
        "",
        "| Model | RMSE | MAE | R² |",
        "|-------|------|-----|-----|",
    ]
    for _, r in reg.iterrows():
        lines.append(f"| {r['model']} | {r['rmse']:.3f} | {r['mae']:.3f} | {r['r2']:.3f} |")

    lines += ["", "## Regression — by checkpoint (best model)", ""]
    if best_reg is not None:
        by_ckpt = ml_metrics[
            (ml_metrics.task == "regression")
            & (ml_metrics.model == best_reg["model"])
            & (ml_metrics.group == "checkpoint")
        ].sort_values("group_value")
        lines += ["| Checkpoint | RMSE | MAE | R² |", "|-----------|------|-----|-----|"]
        for _, r in by_ckpt.iterrows():
            lines.append(f"| {r['group_value']} | {r['rmse']:.3f} | {r['mae']:.3f} | {r['r2']:.3f} |")

    lines += ["", "## Regression — by crop (best model)", ""]
    if best_reg is not None:
        by_crop = ml_metrics[
            (ml_metrics.task == "regression")
            & (ml_metrics.model == best_reg["model"])
            & (ml_metrics.group == "crop")
        ].sort_values("group_value")
        lines += ["| Crop | RMSE | MAE | R² |", "|------|------|-----|-----|"]
        for _, r in by_crop.iterrows():
            lines.append(f"| {r['group_value']} | {r['rmse']:.3f} | {r['mae']:.3f} | {r['r2']:.3f} |")

    lines += ["", "## Classification — overall (sorted by F1)", "",
              "| Model | F1 | Accuracy | Precision | Recall |",
              "|-------|-----|----------|-----------|--------|"]
    for _, r in cls.iterrows():
        lines.append(
            f"| {r['model']} | {r['f1']:.3f} | {r['accuracy']:.3f} | "
            f"{r['precision']:.3f} | {r['recall']:.3f} |"
        )

    lines += ["", "## Baseline comparison", ""]
    if best_reg is not None and base_reg_rmse is not None:
        beats = best_reg["rmse"] < base_reg_rmse
        lines.append(
            f"- **Best ML regression**: `{best_reg['model']}` RMSE={best_reg['rmse']:.3f}, "
            f"R²={best_reg['r2']:.3f}"
        )
        lines.append(
            f"- Best baseline (`{best_base_reg_name}`) RMSE={base_reg_rmse:.3f} → "
            f"ML {'**beats**' if beats else 'does NOT beat'} the baseline "
            f"({best_reg['rmse']:.3f} vs {base_reg_rmse:.3f})"
        )
    if best_cls is not None and base_cls_f1 is not None:
        beats_f1 = best_cls["f1"] > base_cls_f1
        beats_recall = best_cls["recall"] > base_cls_recall
        lines.append(
            f"- **Best ML classifier**: `{best_cls['model']}` F1={best_cls['f1']:.3f}, "
            f"recall={best_cls['recall']:.3f}"
        )
        lines.append(
            f"- Baseline (`{best_base_cls_name}`) F1={base_cls_f1:.3f}, recall={base_cls_recall:.3f} → "
            f"ML {'**beats**' if beats_f1 else 'does NOT beat'} F1 and "
            f"{'**beats**' if beats_recall else 'does NOT beat'} recall"
        )

    # Checkpoint trend assessment
    lines += ["", "## Checkpoint trend (does later = better?)", ""]
    if best_reg is not None:
        by_ckpt = ml_metrics[
            (ml_metrics.task == "regression")
            & (ml_metrics.model == best_reg["model"])
            & (ml_metrics.group == "checkpoint")
        ]
        order = ["may_31", "june_30", "july_31", "august_31", "full_season"]
        ordered = by_ckpt.set_index("group_value").reindex(order).dropna(subset=["rmse"])
        if len(ordered) >= 2:
            first_rmse = ordered.iloc[0]["rmse"]
            last_rmse = ordered.iloc[-1]["rmse"]
            trend = "improves" if last_rmse < first_rmse else "does NOT clearly improve"
            lines.append(
                f"- For `{best_reg['model']}`, RMSE {trend} from "
                f"{ordered.index[0]} ({first_rmse:.3f}) to "
                f"{ordered.index[-1]} ({last_rmse:.3f})."
            )

    lines += ["", "## Leakage check", "",
              f"- Target/realised-outcome columns excluded from features: {sorted(LEAKAGE_COLUMNS)}",
              "- Preprocessing fit on training split only (sklearn Pipeline).",
              "- severe_risk assigned after temporal split via assign_modeling_risk_labels.",
              ""]

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    logger.info("ML report saved → %s", path)


# ── Main ────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    logger.info("=== CropShield: Step 4 — Train ML Models ===")

    panel_path = Path(args.panel)
    if not panel_path.exists():
        logger.error("Panel not found at %s. Run scripts 01–02 first.", panel_path)
        sys.exit(1)

    panel = load_panel(panel_path)
    panel_sanity_check(panel)

    # Regression needs a present target; drop unmatched-weather rows (NaN checkpoint)
    panel = panel.dropna(subset=[REGRESSION_TARGET, "checkpoint"]).copy()

    # ── Temporal split + modeling-safe labels ────────────────────────────────
    train_raw, test_raw = temporal_split(panel, n_test_years=args.test_years)
    assert_temporal_split_safe(train_raw, test_raw)
    train_df, test_df = assign_modeling_risk_labels(
        train_raw, test_raw, quantile=args.risk_quantile,
    )
    assert "severe_risk" in train_df.columns
    assert "severe_risk_descriptive" not in train_df.columns

    # ── Feature selection ─────────────────────────────────────────────────────
    numeric, categorical = select_feature_columns(
        panel,
        include_county=args.include_county,
        include_year_index=args.include_year_index,
    )
    feature_info = {
        "numeric": numeric, "categorical": categorical,
        "include_county": args.include_county,
        "include_year_index": args.include_year_index,
    }
    feature_cols = numeric + categorical

    metric_rows: list[dict] = []
    pred_frames: list[pd.DataFrame] = []
    meta_cols = ["year", "state", "county_fips", "crop", "checkpoint"]

    # ── Regression ────────────────────────────────────────────────────────────
    X_train = train_df[feature_cols]
    y_train = train_df[REGRESSION_TARGET].values
    X_test = test_df[feature_cols]
    y_test = test_df[REGRESSION_TARGET].values

    reg_models = build_regression_models(numeric, categorical, random_state=args.random_state)
    for name, pipe in reg_models.items():
        logger.info("Training regression model: %s", name)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        metric_rows.extend(_grouped_regression(name, y_test, y_pred, test_df, GROUP_COLS))
        pred_frames.append(test_df[meta_cols].assign(
            model_name=name, task="regression",
            y_true=y_test, y_pred=y_pred, y_proba=np.nan,
        ))

    # ── Classification ────────────────────────────────────────────────────────
    train_cls = train_df.dropna(subset=[CLASSIFICATION_TARGET])
    test_cls = test_df.dropna(subset=[CLASSIFICATION_TARGET])
    Xc_train = train_cls[feature_cols]
    yc_train = train_cls[CLASSIFICATION_TARGET].astype(int).values
    Xc_test = test_cls[feature_cols]
    yc_test = test_cls[CLASSIFICATION_TARGET].astype(int).values

    clf_models = build_classification_models(numeric, categorical, random_state=args.random_state)
    for name, pipe in clf_models.items():
        logger.info("Training classification model: %s", name)
        pipe.fit(Xc_train, yc_train)
        y_pred = pipe.predict(Xc_test).astype(int)
        if hasattr(pipe, "predict_proba"):
            y_proba = pipe.predict_proba(Xc_test)[:, 1]
        else:
            y_proba = np.full(len(Xc_test), np.nan)
        metric_rows.extend(_grouped_classification(name, yc_test, y_pred, test_cls, GROUP_COLS))
        pred_frames.append(test_cls[meta_cols].assign(
            model_name=name, task="classification",
            y_true=yc_test, y_pred=y_pred, y_proba=y_proba,
        ))

    # ── Save outputs ──────────────────────────────────────────────────────────
    ml_metrics = pd.DataFrame(metric_rows)
    metrics_path = Path(args.metrics_out)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    ml_metrics.to_csv(metrics_path, index=False)
    logger.info("ML metrics saved → %s (%d rows)", metrics_path, len(ml_metrics))

    predictions = pd.concat(pred_frames, ignore_index=True)
    pred_path = Path(args.predictions_out)
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(pred_path, index=False)
    logger.info("ML predictions saved → %s (%d rows)", pred_path, len(predictions))

    baseline_metrics = _load_baseline_metrics(Path(args.baseline_metrics))
    write_report(ml_metrics, baseline_metrics, train_df, test_df, feature_info, Path(args.report_out))

    # ── Console summary ───────────────────────────────────────────────────────
    reg = ml_metrics[(ml_metrics.task == "regression") & (ml_metrics.group == "overall")].sort_values("rmse")
    cls = ml_metrics[(ml_metrics.task == "classification") & (ml_metrics.group == "overall")].sort_values("f1", ascending=False)
    print("\n── ML Results (overall) ──────────────────────────────────────")
    print("Regression (sorted by RMSE):")
    print(reg[["model", "rmse", "mae", "r2"]].to_string(index=False))
    print("\nClassification (sorted by F1):")
    print(cls[["model", "f1", "accuracy", "precision", "recall"]].to_string(index=False))
    print()

    logger.info("=== Step 4 complete ===")


if __name__ == "__main__":
    main()
