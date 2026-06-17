"""
Validation split utilities for CropShield.

Why temporal validation?
-------------------------
Agricultural yield data is a time series. Using a random train/test split
would allow future-year data to appear in training, giving the model
knowledge it could not have at prediction time. This produces optimistically
biased validation metrics that do not reflect real-world performance.

Temporal validation (training on earlier years, testing on later years)
correctly simulates a forecasting scenario where only historical data is
available when predictions are made.

Spatial validation
------------------
Holding out all counties from one state tests whether the model generalises
to geographies it has not seen. This is relevant if the model will eventually
be deployed to new states.
"""

from __future__ import annotations

import logging
from typing import Iterator

import pandas as pd

logger = logging.getLogger(__name__)


def temporal_split(
    df: pd.DataFrame,
    year_col: str = "year",
    n_test_years: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data temporally: train on earlier years, test on latest years.

    Parameters
    ----------
    df : pd.DataFrame
        Modeling panel with a year column.
    year_col : str
        Name of the year column.
    n_test_years : int
        Number of the most recent years to hold out for testing.

    Returns
    -------
    (train_df, test_df) : tuple of DataFrames

    Notes
    -----
    With n_test_years=3 and data from 2015–2023, the test set would be
    2021, 2022, 2023 and the training set 2015–2020.
    """
    years = sorted(df[year_col].unique())
    if len(years) <= n_test_years:
        raise ValueError(
            f"Not enough years ({len(years)}) for a temporal split with "
            f"n_test_years={n_test_years}. Reduce n_test_years."
        )
    cutoff_year = years[-n_test_years]
    train = df[df[year_col] < cutoff_year].copy()
    test = df[df[year_col] >= cutoff_year].copy()
    logger.info(
        "Temporal split: train years %d–%d (%d rows), test years %d–%d (%d rows)",
        train[year_col].min(),
        train[year_col].max(),
        len(train),
        test[year_col].min(),
        test[year_col].max(),
        len(test),
    )
    return train, test


def spatial_split(
    df: pd.DataFrame,
    holdout_state: str,
    state_col: str = "state",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data spatially: hold out all counties from one state.

    Parameters
    ----------
    df : pd.DataFrame
        Modeling panel with a state column.
    holdout_state : str
        State name (e.g. ``"ILLINOIS"`` or ``"IA"``) to hold out.
    state_col : str
        Name of the state column.

    Returns
    -------
    (train_df, test_df) : tuple of DataFrames
    """
    train = df[df[state_col] != holdout_state].copy()
    test = df[df[state_col] == holdout_state].copy()
    if len(test) == 0:
        raise ValueError(f"No rows found for holdout state '{holdout_state}'.")
    logger.info(
        "Spatial split: training on %d rows (%d states), testing on %d rows (state=%s)",
        len(train),
        train[state_col].nunique(),
        len(test),
        holdout_state,
    )
    return train, test


def expanding_window_splits(
    df: pd.DataFrame,
    year_col: str = "year",
    min_train_years: int = 5,
) -> Iterator[tuple[pd.DataFrame, pd.DataFrame, int]]:
    """Generate expanding-window temporal cross-validation folds.

    Each fold adds one more year to the training set and uses the next year
    as the test set. This is the most realistic CV strategy for time series.

    Parameters
    ----------
    df : pd.DataFrame
        Modeling panel with a year column.
    year_col : str
        Name of the year column.
    min_train_years : int
        Minimum number of training years before the first fold is yielded.

    Yields
    ------
    (train_df, test_df, test_year) : tuple
        Training data up to and including (test_year - 1), test data for
        test_year, and the test year integer.
    """
    years = sorted(df[year_col].unique())
    for i, test_year in enumerate(years[min_train_years:], start=min_train_years):
        train = df[df[year_col] < test_year].copy()
        test = df[df[year_col] == test_year].copy()
        yield train, test, test_year
