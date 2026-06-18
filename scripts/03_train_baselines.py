"""
Script 03: Train and evaluate honest baseline models.

Uses temporal validation only.  Risk labels are assigned after splitting
via assign_modeling_risk_labels() so thresholds come from training data.

Usage
-----
    python scripts/03_train_baselines.py
    python scripts/03_train_baselines.py --test-years 3 --risk-quantile 0.20

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
from cropshield.evaluation.panel_audit import add_checkpoint_column, save_panel_audit
from cropshield.evaluation.validation_splits import temporal_split
from cropshield.features.yield_targets import assign_modeling_risk_labels
from cropshield.models.baselines import (
    get_classification_baselines,
    get_regression_baselines,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CropShield baseline models.")
    parser.add_argument(
        "--panel", default="data/processed/modeling_panel.csv",
        help="Path to modeling panel CSV",
    )
    parser.add_argument("--test-years", type=int, default=3, help="Years held out for testing")
    parser.add_argument("--risk-quantile", type=float, default=0.20, help="Severe-risk quantile")
    parser.add_argument(
        "--metrics-out", default="reports/metrics/baseline_metrics.csv",
        help="Output path for metrics CSV",
    )
    parser.add_argument(
        "--predictions-out", default="data/processed/baseline_predictions.csv",
        help="Output path for test-set predictions",
    )
    parser.add_argument(
        "--report-out", default="reports/baseline_report.md",
        help="Output path for markdown summary",
    )
    parser.add_argument(
        "--audit-out", default="reports/panel_audit.md",
        help="Output path for panel audit markdown",
    )
    return parser.parse_args()


def load_panel(path: Path) -> pd.DataFrame:
    """Load and sanitise the modeling panel for baseline training."""
    df = pd.read_csv(path, dtype={"county_fips": "string"})
    df["county_fips"] = normalise_fips_series(df["county_fips"])
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df = add_checkpoint_column(df)
    # Remove legacy / descriptive risk columns — labels assigned after split
    for col in ("severe_risk", "severe_risk_descriptive"):
        if col in df.columns:
            df = df.drop(columns=[col])
    return df


def _eval_regression(
    name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    meta: pd.DataFrame,
    group_cols: list[str],
) -> list[dict]:
    rows = []
    base = {"model": name, "task": "regression", "group": "overall", "group_value": "all"}
    base.update(regression_metrics(y_true, y_pred, model_name=name))
    rows.append(base)

    for col in group_cols:
        for val in meta[col].unique():
            mask = (meta[col] == val).values
            if mask.sum() < 2:
                continue
            row = {"model": name, "task": "regression", "group": col, "group_value": str(val)}
            row.update(regression_metrics(y_true[mask], y_pred[mask], model_name=f"{name}/{col}={val}"))
            rows.append(row)
    return rows


def _eval_classification(
    name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    meta: pd.DataFrame,
    group_cols: list[str],
) -> list[dict]:
    rows = []
    base = {"model": name, "task": "classification", "group": "overall", "group_value": "all"}
    base.update(classification_metrics(y_true, y_pred, model_name=name))
    rows.append(base)

    for col in group_cols:
        for val in meta[col].unique():
            mask = (meta[col] == val).values
            if mask.sum() < 2 or len(np.unique(y_true[mask])) < 2:
                continue
            row = {"model": name, "task": "classification", "group": col, "group_value": str(val)}
            row.update(classification_metrics(y_true[mask], y_pred[mask], model_name=f"{name}/{col}={val}"))
            rows.append(row)
    return rows


def write_report(
    metrics_df: pd.DataFrame,
    audit_summary: dict,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    path: Path,
) -> None:
    """Write a short markdown summary of baseline results."""
    reg_overall = metrics_df[
        (metrics_df["task"] == "regression") & (metrics_df["group"] == "overall")
    ].sort_values("rmse")
    cls_overall = metrics_df[
        (metrics_df["task"] == "classification") & (metrics_df["group"] == "overall")
    ].sort_values("f1", ascending=False)

    best_reg = reg_overall.iloc[0] if len(reg_overall) else None
    best_cls = cls_overall.iloc[0] if len(cls_overall) else None

    lines = [
        "# CropShield Baseline Report",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## Data",
        "",
        f"- Panel rows: {audit_summary['row_count']:,}",
        f"- Train rows: {len(train_df):,} (years {int(train_df['year'].min())}–{int(train_df['year'].max())})",
        f"- Test rows: {len(test_df):,} (years {int(test_df['year'].min())}–{int(test_df['year'].max())})",
        f"- Counties: {audit_summary['n_counties']}",
        f"- Crops: {', '.join(audit_summary['crops'])}",
        "",
        "## Best regression baseline (lowest RMSE)",
        "",
    ]
    if best_reg is not None:
        lines.append(
            f"- **{best_reg['model']}**: RMSE={best_reg['rmse']:.3f}, "
            f"MAE={best_reg['mae']:.3f}, R²={best_reg['r2']:.3f}"
        )
    lines += ["", "## Best classification baseline (highest F1)", ""]
    if best_cls is not None:
        lines.append(
            f"- **{best_cls['model']}**: F1={best_cls['f1']:.3f}, "
            f"accuracy={best_cls['accuracy']:.3f}, recall={best_cls['recall']:.3f}"
        )
    lines += [
        "",
        "## All regression baselines (overall)",
        "",
        "| Model | RMSE | MAE | R² |",
        "|-------|------|-----|-----|",
    ]
    for _, r in reg_overall.iterrows():
        lines.append(f"| {r['model']} | {r['rmse']:.3f} | {r['mae']:.3f} | {r['r2']:.3f} |")
    lines += [
        "",
        "## All classification baselines (overall)",
        "",
        "| Model | F1 | Accuracy | Precision | Recall |",
        "|-------|-----|----------|-----------|--------|",
    ]
    for _, r in cls_overall.iterrows():
        lines.append(
            f"| {r['model']} | {r['f1']:.3f} | {r['accuracy']:.3f} | "
            f"{r['precision']:.3f} | {r['recall']:.3f} |"
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    logger.info("Baseline report saved → %s", path)


def main() -> None:
    args = parse_args()
    logger.info("=== CropShield: Step 3 — Train Baselines ===")

    panel_path = Path(args.panel)
    if not panel_path.exists():
        logger.error("Modeling panel not found at %s. Run script 02 first.", panel_path)
        sys.exit(1)

    panel = load_panel(panel_path)
    audit_summary = save_panel_audit(panel, args.audit_out)
    logger.info(
        "Panel audit: %d rows | %d counties | years %s | checkpoints %s",
        audit_summary["row_count"],
        audit_summary["n_counties"],
        audit_summary["years"],
        audit_summary["checkpoints"],
    )
    if audit_summary["duplicate_rows"] > 0:
        logger.warning("%d duplicate rows detected", audit_summary["duplicate_rows"])

    # ── Temporal split + modeling-safe risk labels ───────────────────────────
    train_raw, test_raw = temporal_split(panel, n_test_years=args.test_years)
    train_df, test_df = assign_modeling_risk_labels(
        train_raw, test_raw, quantile=args.risk_quantile,
    )
    assert "severe_risk" in train_df.columns and "severe_risk" in test_df.columns
    assert "severe_risk_descriptive" not in train_df.columns

    group_cols = ["crop", "checkpoint", "year"]
    metric_rows: list[dict] = []
    pred_frames: list[pd.DataFrame] = []

    # ── Regression baselines ───────────────────────────────────────────────────
    y_test_reg = test_df["yield_anomaly"].values
    for model in get_regression_baselines():
        model.fit(train_df)
        if model.name == "previous_year_anomaly":
            y_pred = model.predict(test_df)
        else:
            y_pred = model.predict(test_df)
        metric_rows.extend(_eval_regression(model.name, y_test_reg, y_pred, test_df, group_cols))
        pred_frames.append(test_df[["year", "state", "county_fips", "crop", "checkpoint"]].assign(
            model=model.name,
            task="regression",
            y_true=y_test_reg,
            y_pred=y_pred,
        ))

    # ── Classification baselines ─────────────────────────────────────────────
    test_cls = test_df.dropna(subset=["severe_risk"])
    y_test_cls = test_cls["severe_risk"].astype(int).values
    for model in get_classification_baselines():
        model.fit(train_df)
        y_pred = model.predict(test_cls).astype(int)
        metric_rows.extend(
            _eval_classification(model.name, y_test_cls, y_pred, test_cls, group_cols)
        )
        pred_frames.append(test_cls[["year", "state", "county_fips", "crop", "checkpoint"]].assign(
            model=model.name,
            task="classification",
            y_true=y_test_cls,
            y_pred=y_pred,
        ))

    # ── Save outputs ─────────────────────────────────────────────────────────
    metrics_df = pd.DataFrame(metric_rows)
    metrics_path = Path(args.metrics_out)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(metrics_path, index=False)
    logger.info("Metrics saved → %s  (%d rows)", metrics_path, len(metrics_df))

    predictions = pd.concat(pred_frames, ignore_index=True)
    pred_path = Path(args.predictions_out)
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(pred_path, index=False)
    logger.info("Predictions saved → %s  (%d rows)", pred_path, len(predictions))

    write_report(metrics_df, audit_summary, train_df, test_df, Path(args.report_out))

    # Print quick summary
    reg = metrics_df[(metrics_df.task == "regression") & (metrics_df.group == "overall")].sort_values("rmse")
    cls = metrics_df[(metrics_df.task == "classification") & (metrics_df.group == "overall")].sort_values("f1", ascending=False)
    print("\n── Baseline Results (overall) ────────────────────────────────")
    print("Regression (sorted by RMSE):")
    print(reg[["model", "rmse", "mae", "r2"]].to_string(index=False))
    print("\nClassification (sorted by F1):")
    print(cls[["model", "f1", "accuracy", "precision", "recall"]].to_string(index=False))
    print()

    logger.info("=== Step 3 complete ===")


if __name__ == "__main__":
    main()
