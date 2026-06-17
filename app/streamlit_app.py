"""
CropShield Streamlit Dashboard.

Portfolio-ready interactive demo for exploring county-level crop yield
anomaly predictions, model performance, and feature explanations.

Usage
-----
    streamlit run app/streamlit_app.py

Note
----
This dashboard requires the following files to be generated first:
  - data/processed/predictions.csv     (from make evaluate)
  - reports/metrics.json               (from make evaluate)
  - reports/figures/*.png              (from make evaluate)

If these files are not present, the dashboard will display placeholder
messages rather than crashing.

Disclaimer
----------
CropShield is a student portfolio project using public data. It is NOT
a production agronomic forecast tool. Do not use it for farm-level
recommendations, insurance pricing, or financial decisions.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CropShield — Yield Risk Forecasting",
    page_icon="🌽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
PREDICTIONS_PATH = ROOT / "data" / "processed" / "predictions.csv"
METRICS_PATH = ROOT / "reports" / "metrics.json"
FIGURES_DIR = ROOT / "reports" / "figures"


# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_data
def load_predictions() -> pd.DataFrame | None:
    """Load predictions CSV if it exists."""
    if PREDICTIONS_PATH.exists():
        return pd.read_csv(PREDICTIONS_PATH)
    return None


@st.cache_data
def load_metrics() -> dict | None:
    """Load metrics JSON if it exists."""
    if METRICS_PATH.exists():
        with open(METRICS_PATH) as f:
            return json.load(f)
    return None


def figure_or_placeholder(path: Path, caption: str) -> None:
    """Show a figure if available, otherwise a placeholder message."""
    if path.exists():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.info(f"Figure not yet available: `{path.name}`. Run `make evaluate` to generate it.")


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("CropShield 🌽")
st.sidebar.markdown(
    """
    **Drought-Aware Yield Risk Forecasting**

    MVP: Corn · Iowa & Illinois · 2015 onward
    """
)

page = st.sidebar.radio(
    "Navigate",
    [
        "Project Overview",
        "Model Performance",
        "County Explorer",
        "Feature Importance",
        "Limitations",
    ],
)

# ── Disclaimer banner ────────────────────────────────────────────────────────
st.sidebar.divider()
st.sidebar.warning(
    "⚠️ **Disclaimer**\n\n"
    "This is a student portfolio project, NOT a production forecast tool. "
    "Do not use for farm-level recommendations, insurance pricing, or "
    "financial decisions."
)


# ── Pages ─────────────────────────────────────────────────────────────────────

if page == "Project Overview":
    st.title("CropShield: Drought-Aware Yield Risk Forecasting")
    st.markdown(
        """
        CropShield is an interpretable geospatial ML pipeline that integrates
        public agricultural and climate datasets to predict county-level crop
        yield anomalies during the growing season.

        ### What it does
        - Predicts **yield anomaly** (bu/acre above or below expected trend)
        - Classifies counties as **severe yield-risk** when anomaly is in the
          bottom quantile of historical performance

        ### Data sources
        | Source | Variables |
        |---|---|
        | USDA NASS Quick Stats | County corn yield (bu/acre) |
        | NASA POWER API | Daily precip, temperature |
        | U.S. Drought Monitor | Weekly D0–D4 county coverage |

        ### Methods
        - Leakage-safe target engineering (expected yield from prior years only)
        - Temporal validation (train on earlier years, test on later years)
        - RandomForest and XGBoost models
        - Feature importance and SHAP interpretability
        """
    )
    st.info("Pipeline status: Run `make pipeline` to generate predictions and results.")


elif page == "Model Performance":
    st.title("Model Performance")
    st.markdown(
        "Metrics are computed on a **temporal held-out test set** "
        "(the most recent 3 years of data). Random splits are not used."
    )

    metrics = load_metrics()
    if metrics is None:
        st.warning("Metrics not yet available. Run `make evaluate` first.")
    else:
        # TODO: Render metrics table from JSON
        st.json(metrics)

    st.subheader("Predicted vs Actual Yield Anomaly")
    figure_or_placeholder(
        FIGURES_DIR / "predicted_vs_actual.png",
        "Predicted vs actual yield anomaly on held-out test years.",
    )

    st.subheader("Residuals by Year")
    figure_or_placeholder(
        FIGURES_DIR / "residuals_by_year.png",
        "Prediction residuals grouped by year. Systematic patterns may indicate model limitations.",
    )

    st.subheader("Residuals by State")
    figure_or_placeholder(
        FIGURES_DIR / "residuals_by_state.png",
        "Prediction residuals grouped by state.",
    )


elif page == "County Explorer":
    st.title("County Prediction Explorer")

    predictions = load_predictions()
    if predictions is None:
        st.warning(
            "Predictions not yet available. Run `make evaluate` to generate "
            "`data/processed/predictions.csv`."
        )
        st.stop()

    col1, col2, col3 = st.columns(3)
    with col1:
        crop = st.selectbox("Crop", options=sorted(predictions["crop"].unique()) if "crop" in predictions.columns else ["CORN"])
    with col2:
        state = st.selectbox("State", options=sorted(predictions["state"].unique()) if "state" in predictions.columns else [])
    with col3:
        year = st.selectbox("Year", options=sorted(predictions["year"].unique(), reverse=True) if "year" in predictions.columns else [])

    # TODO: Filter predictions and display county-level results
    st.info("County explorer will be implemented after predictions are generated.")


elif page == "Feature Importance":
    st.title("Feature Importance")
    st.markdown(
        """
        Feature importance shows which variables the model relies on most for
        predictions. **Important caveat:** feature importance reflects
        statistical associations in the training data, not causal relationships.
        A high importance score does not mean that variable *causes* yield loss.
        """
    )
    figure_or_placeholder(
        FIGURES_DIR / "feature_importance.png",
        "Top features ranked by model-native importance (RandomForest or XGBoost).",
    )


elif page == "Limitations":
    st.title("Limitations and Appropriate Use")
    st.markdown(
        """
        ### What this project is
        A **portfolio and research-style demonstration** showing how public
        agricultural and climate datasets can be integrated into an interpretable
        early-warning system for crop stress and yield risk.

        ### What this project is NOT
        - A production agronomic forecast system
        - A farm-level recommendation tool
        - A basis for insurance pricing or financial decisions
        - A replacement for USDA or extension agronomist expertise

        ### Known limitations

        | Limitation | Impact |
        |---|---|
        | County-level aggregation | Hides farm-level variation and management differences |
        | Weather at county centroids | Misses spatial heterogeneity within large counties |
        | No management data | Planting date, irrigation, and fertiliser confound results |
        | Suppressed NASS records | Sparse counties may have missing training data |
        | Two-state MVP scope | Model may not generalise to other geographies |
        | Feature importance ≠ causation | High importance does not prove a variable causes yield loss |
        | Historical training window | Performance in extreme or novel climate years is uncertain |

        ### On uncertainty
        Predictions should be treated as **risk signals** for further investigation,
        not as guaranteed yield forecasts.
        """
    )
