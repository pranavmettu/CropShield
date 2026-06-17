"""
SHAP analysis for CropShield.

Provides SHAP (SHapley Additive exPlanations) value computation and
visualisation for tree-based models.

Dependency
----------
Requires the ``shap`` package: pip install shap

If shap is not installed, functions raise ImportError with instructions.

Limitations
-----------
SHAP values indicate the marginal contribution of each feature to a
specific prediction relative to the model's base value. They do not
imply causation and are specific to the trained model, not the true
data-generating process.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_shap_values(
    model: Any,
    X: pd.DataFrame,
) -> tuple[Any, Any]:
    """Compute SHAP values for a fitted tree model.

    Parameters
    ----------
    model : sklearn-compatible tree model
        Fitted RandomForest, XGBoost, or LightGBM estimator.
    X : pd.DataFrame
        Feature matrix to explain (typically the test set).

    Returns
    -------
    (shap_values, explainer) : tuple
        The SHAP values array and the shap.Explainer instance.

    Raises
    ------
    ImportError
        If the ``shap`` package is not installed.
    """
    try:
        import shap
    except ImportError as e:
        raise ImportError(
            "The 'shap' package is required for SHAP analysis. "
            "Install it with: pip install shap"
        ) from e

    # TODO: Implement SHAP computation
    # 1. Instantiate shap.TreeExplainer(model)
    # 2. Call explainer(X) to get SHAP values
    # 3. Return shap_values, explainer
    raise NotImplementedError("compute_shap_values is not yet implemented.")


def plot_shap_summary(
    shap_values: Any,
    X: pd.DataFrame,
    output_path: str | Path | None = "reports/figures/shap_summary.png",
    max_display: int = 20,
) -> None:
    """Generate a SHAP beeswarm summary plot.

    Parameters
    ----------
    shap_values : shap.Explanation
        SHAP values from ``compute_shap_values()``.
    X : pd.DataFrame
        Feature matrix used to compute SHAP values.
    output_path : str or Path, optional
        Path to save the figure.
    max_display : int
        Maximum number of features to display.
    """
    # TODO: Call shap.summary_plot() and save figure
    raise NotImplementedError("plot_shap_summary is not yet implemented.")
