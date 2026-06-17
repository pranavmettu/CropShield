"""
Script 03: Train baseline and tree-based models.

Loads the modeling panel, applies a temporal validation split, trains
all configured models, and saves artifacts to models/.

Usage
-----
    python scripts/03_train_model.py
    python scripts/03_train_model.py --model xgboost --test-years 3

Run from the project root directory.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# TODO: Import when implemented
# from cropshield.evaluation.validation_splits import temporal_split
# from cropshield.models.train_baseline import train_baselines
# from cropshield.models.train_tree_model import train_random_forest, train_xgboost, save_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CropShield models.")
    parser.add_argument(
        "--model",
        choices=["all", "baseline", "rf", "xgboost", "lightgbm"],
        default="all",
        help="Which model(s) to train (default: all)",
    )
    parser.add_argument("--test-years", type=int, default=3, help="Years to hold out for testing")
    parser.add_argument(
        "--panel",
        default="data/processed/modeling_panel.csv",
        help="Path to modeling panel CSV",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger.info("=== CropShield: Step 3 — Train Models ===")

    panel_path = Path(args.panel)
    if not panel_path.exists():
        logger.error(
            "Modeling panel not found at %s. Run script 02 first.", panel_path
        )
        sys.exit(1)

    # TODO: Implement training pipeline
    # 1. Load panel
    # 2. temporal_split(panel, n_test_years=args.test_years)
    # 3. Define feature_cols from model_config.yaml
    # 4. Train models based on args.model
    # 5. Save artifacts via save_model()
    logger.warning("Model training not yet implemented.")
    logger.info("=== Step 3 complete ===")


if __name__ == "__main__":
    main()
