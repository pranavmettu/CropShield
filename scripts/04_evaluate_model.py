"""
Script 04: Evaluate trained models and generate reports.

Loads model artifacts and test-set predictions, computes metrics,
saves results to reports/metrics.json, and generates diagnostic figures.

Usage
-----
    python scripts/04_evaluate_model.py
    python scripts/04_evaluate_model.py --model xgboost

Run from the project root directory.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# TODO: Import when implemented
# from cropshield.evaluation.metrics import regression_metrics, save_metrics
# from cropshield.evaluation.error_analysis import compute_residuals, residuals_by_group
# from cropshield.visualization.plots import (
#     plot_predicted_vs_actual, plot_residuals_by_year, plot_residuals_by_state
# )

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate CropShield models.")
    parser.add_argument(
        "--model",
        choices=["all", "baseline", "rf", "xgboost", "lightgbm"],
        default="all",
    )
    parser.add_argument(
        "--predictions",
        default="data/processed/predictions.csv",
        help="Path to predictions CSV",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger.info("=== CropShield: Step 4 — Evaluate Models ===")

    predictions_path = Path(args.predictions)
    if not predictions_path.exists():
        logger.error(
            "Predictions file not found at %s. Run script 03 first.", predictions_path
        )
        sys.exit(1)

    # TODO: Implement evaluation pipeline
    # 1. Load predictions
    # 2. Compute regression_metrics per model
    # 3. Compute classification_metrics if severe_risk predictions exist
    # 4. Save to reports/metrics.json
    # 5. Generate and save diagnostic figures
    logger.warning("Model evaluation not yet implemented.")
    logger.info("=== Step 4 complete ===")


if __name__ == "__main__":
    main()
