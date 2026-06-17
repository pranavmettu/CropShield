"""
Feature importance extraction and visualisation for CropShield.

Extracts and plots model-native feature importances from tree-based
estimators (RandomForest, XGBoost, LightGBM).

Limitations
-----------
- Impurity-based importance (MDI) is biased toward high-cardinality features.
- Permutation importance is more reliable but computationally expensive.
- Neither measure implies causation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)


def get_feature_importance(
    model: Any,
    feature_names: list[str],
) -> pd.DataFrame:
    """Extract feature importances from a fitted tree model.

    Parameters
    ----------
    model : sklearn-compatible estimator
        Fitted RandomForest, XGBoost, or LightGBM model with a
        ``feature_importances_`` attribute.
    feature_names : list[str]
        Names corresponding to the training feature columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``feature`` and ``importance``, sorted by
        importance descending.
    """
    if not hasattr(model, "feature_importances_"):
        raise AttributeError(
            f"Model {type(model).__name__} does not have feature_importances_. "
            "Fit the model first."
        )
    df = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    return df


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 20,
    title: str = "Feature Importance",
    output_path: str | Path | None = "reports/figures/feature_importance.png",
) -> plt.Figure:
    """Plot a horizontal bar chart of the top-N most important features.

    Parameters
    ----------
    importance_df : pd.DataFrame
        DataFrame from ``get_feature_importance()``.
    top_n : int
        Number of features to display.
    title : str
        Plot title.
    output_path : str or Path, optional
        If provided, saves the figure to this path.

    Returns
    -------
    plt.Figure
        The matplotlib Figure object.
    """
    # TODO: Implement feature importance plot
    # 1. Take top_n rows
    # 2. Plot horizontal bar chart
    # 3. Add axis labels and title
    # 4. Add caption note about correlation vs causation
    # 5. Save if output_path provided
    raise NotImplementedError("plot_feature_importance is not yet implemented.")
