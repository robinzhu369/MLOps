"""Data split tools — train/test splitting for modeling."""

import pandas as pd
import numpy as np


def split_train_test(
    df: pd.DataFrame,
    label_col: str,
    test_size: float = 0.2,
    time_col: str | None = None,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into train and test sets.

    If time_col is provided, uses time-based split (earlier data for train).
    Otherwise, uses stratified random split.

    Args:
        df: Input DataFrame.
        label_col: Label column for stratification.
        test_size: Fraction of data for test set.
        time_col: Optional time column for temporal split.
        random_state: Random seed.

    Returns:
        Tuple of (train_df, test_df).
    """
    if time_col and time_col in df.columns:
        return _time_split(df, time_col, test_size)
    return _stratified_split(df, label_col, test_size, random_state)


def _time_split(
    df: pd.DataFrame, time_col: str, test_size: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by time — earlier records for training."""
    df_sorted = df.sort_values(time_col).reset_index(drop=True)
    split_idx = int(len(df_sorted) * (1 - test_size))
    train = df_sorted.iloc[:split_idx].copy()
    test = df_sorted.iloc[split_idx:].copy()
    return train, test


def _stratified_split(
    df: pd.DataFrame, label_col: str, test_size: float, random_state: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified random split preserving label distribution."""
    rng = np.random.RandomState(random_state)

    train_indices = []
    test_indices = []

    for _, group in df.groupby(label_col):
        indices = group.index.tolist()
        rng.shuffle(indices)
        n_test = max(1, int(len(indices) * test_size))
        test_indices.extend(indices[:n_test])
        train_indices.extend(indices[n_test:])

    train = df.loc[train_indices].reset_index(drop=True)
    test = df.loc[test_indices].reset_index(drop=True)
    return train, test
