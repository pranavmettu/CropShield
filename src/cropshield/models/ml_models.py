"""
First-generation ML models for CropShield.

Leakage-safe scikit-learn pipelines for:
  - Regression of ``yield_anomaly``
  - Classification of ``severe_risk`` (label assigned after the temporal split)

Design principles
------------------
1. **No target leakage.**  ``select_feature_columns`` explicitly removes every
   target / post-outcome column.  ``actual_yield`` and ``yield_anomaly`` (and
   its percentage form) describe the realised outcome and must never be used as
   inputs.  Risk labels (``severe_risk``, ``severe_risk_descriptive``) are
   outcomes too.  ``expected_yield`` *is* allowed because it is built only from
   prior-year yields.

2. **Preprocessing is fit on training data only.**  Every model is a
   ``Pipeline`` whose first step is a ``ColumnTransformer``; calling
   ``pipeline.fit(X_train, y_train)`` fits the imputers / encoder on the
   training split exclusively.

3. **Reproducible.**  All stochastic estimators take a fixed ``random_state``.
"""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingRegressor,  # noqa: F401  (fallback import)
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)

DEFAULT_RANDOM_STATE = 42

# ── Column contracts ────────────────────────────────────────────────────────────

# Columns that encode the realised outcome — using any as a feature is leakage.
LEAKAGE_COLUMNS: set[str] = {
    "actual_yield",          # the realised yield → anomaly = actual - expected
    "yield_anomaly",         # regression target
    "yield_anomaly_pct",     # target, rescaled
    "severe_risk",           # classification target
    "severe_risk_descriptive",  # EDA-only label, never for modeling
}

# Identifier / bookkeeping columns that are not predictive features by default.
IDENTIFIER_COLUMNS: set[str] = {
    "year",        # raw year — only year_index is offered, and only on request
    "county",      # free-text name; county_fips is the canonical id
    "unit",        # stray NASS metadata
    "is_partial_year",  # diagnostic flag
}

# Categorical candidates (kept only if present in the panel).
BASE_CATEGORICAL_FEATURES: list[str] = ["crop", "checkpoint", "state"]

# County FIPS is high-cardinality; treated as categorical only when requested.
COUNTY_CATEGORICAL = "county_fips"

# Year-as-trend feature; offered only when include_year_index=True.
YEAR_INDEX_FEATURE = "year_index"


def select_feature_columns(
    panel: pd.DataFrame,
    *,
    include_county: bool = False,
    include_year_index: bool = False,
) -> tuple[list[str], list[str]]:
    """Return ``(numeric_features, categorical_features)`` for the panel.

    Guarantees that no leakage column (target / realised outcome / risk label)
    is ever returned, and that identifier columns are excluded unless explicitly
    opted in.

    Parameters
    ----------
    panel : pd.DataFrame
        The modeling panel.
    include_county : bool
        When ``True``, add ``county_fips`` as a (high-cardinality) categorical
        feature.  Off by default to avoid one-hot blow-up and overfitting.
    include_year_index : bool
        When ``True``, add ``year_index`` as a numeric trend feature.  Off by
        default so the model cannot lean on a raw temporal shortcut; the
        training script tests both settings.

    Returns
    -------
    (numeric_features, categorical_features) : tuple[list[str], list[str]]
    """
    categorical = [c for c in BASE_CATEGORICAL_FEATURES if c in panel.columns]
    if include_county and COUNTY_CATEGORICAL in panel.columns:
        categorical.append(COUNTY_CATEGORICAL)

    excluded = LEAKAGE_COLUMNS | IDENTIFIER_COLUMNS | set(categorical) | {COUNTY_CATEGORICAL}
    if not include_year_index:
        excluded.add(YEAR_INDEX_FEATURE)

    numeric: list[str] = []
    for col in panel.columns:
        if col in excluded:
            continue
        # Only keep numeric-dtype columns as numeric features
        if pd.api.types.is_numeric_dtype(panel[col]):
            numeric.append(col)

    logger.info(
        "select_feature_columns: %d numeric, %d categorical "
        "(include_county=%s, include_year_index=%s)",
        len(numeric), len(categorical), include_county, include_year_index,
    )
    logger.debug("numeric=%s", numeric)
    logger.debug("categorical=%s", categorical)
    return numeric, categorical


def assert_no_leakage_features(numeric: list[str], categorical: list[str]) -> None:
    """Raise if any selected feature is a known leakage column."""
    selected = set(numeric) | set(categorical)
    leaked = selected & LEAKAGE_COLUMNS
    if leaked:
        raise ValueError(
            f"Leakage columns selected as features: {sorted(leaked)}. "
            "These describe the realised outcome and must not be inputs."
        )


# ── Preprocessing ───────────────────────────────────────────────────────────────

def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    """Build a ColumnTransformer with leakage-safe, train-only preprocessing.

    Numeric: median imputation + standard scaling (scaler is fit on train only,
    and gives linear models — Ridge, LogisticRegression — a fair penalty across
    features on very different scales, e.g. growing_degree_days vs heat_days).
    Categorical: most-frequent imputation + one-hot (ignore unseen categories).
    """
    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_features),
            ("cat", categorical_pipe, categorical_features),
        ],
        remainder="drop",
    )


# ── Model registries ──────────────────────────────────────────────────────────

def build_regression_models(
    numeric_features: list[str],
    categorical_features: list[str],
    random_state: int = DEFAULT_RANDOM_STATE,
) -> dict[str, Pipeline]:
    """Return ``{name: Pipeline}`` for the regression models.

    A — Ridge regression
    B — RandomForestRegressor
    C — HistGradientBoostingRegressor
    """
    assert_no_leakage_features(numeric_features, categorical_features)

    def _pipe(estimator) -> Pipeline:
        return Pipeline(steps=[
            ("preprocessor", build_preprocessor(numeric_features, categorical_features)),
            ("model", estimator),
        ])

    return {
        "ridge": _pipe(Ridge(alpha=1.0, random_state=random_state)),
        "random_forest_reg": _pipe(
            RandomForestRegressor(
                n_estimators=300,
                random_state=random_state,
                n_jobs=-1,
            )
        ),
        "hist_gradient_boosting_reg": _pipe(
            HistGradientBoostingRegressor(random_state=random_state)
        ),
    }


def build_classification_models(
    numeric_features: list[str],
    categorical_features: list[str],
    random_state: int = DEFAULT_RANDOM_STATE,
) -> dict[str, Pipeline]:
    """Return ``{name: Pipeline}`` for the classification models.

    A — LogisticRegression(class_weight="balanced")
    B — RandomForestClassifier(class_weight="balanced")
    """
    assert_no_leakage_features(numeric_features, categorical_features)

    def _pipe(estimator) -> Pipeline:
        return Pipeline(steps=[
            ("preprocessor", build_preprocessor(numeric_features, categorical_features)),
            ("model", estimator),
        ])

    return {
        "logistic_regression": _pipe(
            LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=random_state,
            )
        ),
        "random_forest_clf": _pipe(
            RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=-1,
            )
        ),
    }
