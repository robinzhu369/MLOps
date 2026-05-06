"""Metadata extraction tools for uploaded data files."""

import hashlib
import os

import numpy as np
import pandas as pd

from tools.file_tools import read_dataframe


def extract_file_metadata(file_path: str, sample_size: int = 5) -> dict:
    """Extract comprehensive metadata from a data file.

    Returns:
        dict with keys: file_name, n_rows, n_cols, columns, column_dtypes,
        column_profiles, sample_values_masked
    """
    df = read_dataframe(file_path)
    file_name = os.path.basename(file_path)

    column_profiles = {}
    sample_values_masked = {}

    for col in df.columns:
        series = df[col]
        profile = _build_column_profile(series)
        column_profiles[col] = profile
        sample_values_masked[col] = _get_masked_samples(series, sample_size)

    return {
        "file_name": file_name,
        "file_path": file_path,
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "columns": list(df.columns),
        "column_dtypes": {col: str(df[col].dtype) for col in df.columns},
        "column_profiles": column_profiles,
        "sample_values_masked": sample_values_masked,
    }


def _build_column_profile(series: pd.Series) -> dict:
    """Build a statistical profile for a single column."""
    total = len(series)
    missing_count = int(series.isna().sum())
    missing_rate = round(missing_count / total, 4) if total > 0 else 0.0
    unique_count = int(series.nunique(dropna=True))
    unique_rate = round(unique_count / total, 4) if total > 0 else 0.0

    profile = {
        "dtype": str(series.dtype),
        "missing_count": missing_count,
        "missing_rate": missing_rate,
        "unique_count": unique_count,
        "unique_rate": unique_rate,
    }

    if pd.api.types.is_numeric_dtype(series):
        desc = series.describe()
        profile.update({
            "mean": _safe_float(desc.get("mean")),
            "std": _safe_float(desc.get("std")),
            "min": _safe_float(desc.get("min")),
            "max": _safe_float(desc.get("max")),
            "median": _safe_float(series.median()),
            "q25": _safe_float(desc.get("25%")),
            "q75": _safe_float(desc.get("75%")),
        })
    elif pd.api.types.is_object_dtype(series) or pd.api.types.is_categorical_dtype(series):
        top_values = series.value_counts(dropna=True).head(5)
        profile["top_values"] = {str(k): int(v) for k, v in top_values.items()}
        avg_len = series.dropna().astype(str).str.len().mean() if not series.dropna().empty else 0.0
        profile["avg_length"] = round(float(avg_len), 1)

    return profile


def _safe_float(val) -> float | None:
    """Convert a value to float safely, handling NaN."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    return round(float(val), 4)


def _get_masked_samples(series: pd.Series, n: int = 5) -> list:
    """Get sample values with basic masking for privacy.

    For string columns longer than 20 chars, mask the middle portion.
    """
    non_null = series.dropna()
    if non_null.empty:
        return []

    samples = non_null.head(n).tolist()
    masked = []
    for val in samples:
        s = str(val)
        if len(s) > 20:
            masked.append(s[:6] + "***" + s[-4:])
        else:
            masked.append(s)
    return masked
