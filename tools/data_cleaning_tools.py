"""Data cleaning execution tools — deterministic cleaning operations."""

import os
import json
from datetime import datetime

import numpy as np
import pandas as pd

from tools.file_tools import read_dataframe
from tools.trace_tools import write_cleaning_log


def execute_cleaning_plan(
    data_path: str,
    cleaning_plan: dict,
    project_name: str,
    output_dir: str | None = None,
) -> dict:
    """Execute a cleaning plan on the given data file.

    The original file is never modified. Output is written to cleaned_{filename}.

    Args:
        data_path: Path to the original data file.
        cleaning_plan: Output from DataCleaningPlannerAgent.
        project_name: Project name for artifact storage.
        output_dir: Optional output directory. Defaults to artifacts/{project_name}/.

    Returns:
        dict with output_path, before_shape, after_shape, executed_steps, cleaning_log.
    """
    df = read_dataframe(data_path)
    before_shape = {"rows": len(df), "cols": len(df.columns)}

    if output_dir is None:
        output_dir = os.path.join("artifacts", project_name)
    os.makedirs(output_dir, exist_ok=True)

    cleaning_steps = cleaning_plan.get("cleaning_steps", [])
    protected_columns = set(cleaning_plan.get("protected_columns", []))
    executed_steps = []

    for step in cleaning_steps:
        action = step["action"]
        params = step.get("params", {})
        before_rows = len(df)
        before_cols = len(df.columns)

        try:
            df = _execute_step(df, action, params, protected_columns)
            after_rows = len(df)
            after_cols = len(df.columns)

            executed_steps.append({
                "action": action,
                "status": "success",
                "before_shape": [before_rows, before_cols],
                "after_shape": [after_rows, after_cols],
                "rows_affected": abs(before_rows - after_rows),
                "cols_affected": abs(before_cols - after_cols),
            })
        except Exception as e:
            executed_steps.append({
                "action": action,
                "status": "error",
                "error": str(e),
            })

    after_shape = {"rows": len(df), "cols": len(df.columns)}

    # Write cleaned file
    base_name = os.path.basename(data_path)
    name_part, ext = os.path.splitext(base_name)
    output_filename = f"cleaned_{name_part}.csv"
    output_path = os.path.join(output_dir, output_filename)
    df.to_csv(output_path, index=False)

    # Write cleaning log
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "input_path": data_path,
        "output_path": output_path,
        "before_shape": before_shape,
        "after_shape": after_shape,
        "n_steps_executed": len(executed_steps),
        "n_steps_success": sum(1 for s in executed_steps if s["status"] == "success"),
        "executed_steps": executed_steps,
    }
    write_cleaning_log(project_name, log_entry)

    return {
        "output_path": output_path,
        "before_shape": before_shape,
        "after_shape": after_shape,
        "executed_steps": executed_steps,
        "cleaning_log_path": os.path.join("artifacts", project_name, "cleaning_log.json"),
    }


def _execute_step(
    df: pd.DataFrame,
    action: str,
    params: dict,
    protected_columns: set,
) -> pd.DataFrame:
    """Execute a single cleaning step."""
    if action == "drop_exact_duplicates":
        return _drop_exact_duplicates(df)
    elif action == "drop_key_duplicates":
        return _drop_key_duplicates(df, params)
    elif action == "drop_high_missing_columns":
        return _drop_high_missing_columns(df, params, protected_columns)
    elif action == "add_missing_indicator":
        return _add_missing_indicator(df, params)
    elif action == "fill_missing_values":
        return _fill_missing_values(df, params, protected_columns)
    elif action == "winsorize_outliers":
        return _winsorize_outliers(df, params, protected_columns)
    elif action == "convert_types":
        return _convert_types(df, params)
    elif action == "drop_constant_columns":
        return _drop_constant_columns(df, protected_columns)
    else:
        raise ValueError(f"Unknown cleaning action: {action}")


def _drop_exact_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate rows."""
    return df.drop_duplicates().reset_index(drop=True)


def _drop_key_duplicates(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Remove duplicates based on key columns, keeping first occurrence."""
    key_columns = params.get("key_columns", [])
    keep = params.get("keep", "first")
    valid_keys = [c for c in key_columns if c in df.columns]
    if not valid_keys:
        return df
    return df.drop_duplicates(subset=valid_keys, keep=keep).reset_index(drop=True)


def _drop_high_missing_columns(
    df: pd.DataFrame, params: dict, protected_columns: set
) -> pd.DataFrame:
    """Drop columns with high missing rates."""
    columns = params.get("columns", [])
    cols_to_drop = [c for c in columns if c in df.columns and c not in protected_columns]
    return df.drop(columns=cols_to_drop)


def _add_missing_indicator(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Add binary indicator columns for missing values."""
    columns = params.get("columns", [])
    for col in columns:
        if col in df.columns:
            indicator_name = f"{col}_missing"
            df[indicator_name] = df[col].isna().astype(int)
    return df


def _fill_missing_values(
    df: pd.DataFrame, params: dict, protected_columns: set
) -> pd.DataFrame:
    """Fill missing values with median (numeric) or mode (categorical)."""
    columns = params.get("columns", [])
    numeric_strategy = params.get("numeric_strategy", "median")
    categorical_strategy = params.get("categorical_strategy", "mode")

    for col in columns:
        if col not in df.columns or col in protected_columns:
            continue

        if df[col].isna().sum() == 0:
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            if numeric_strategy == "median":
                fill_value = df[col].median()
            elif numeric_strategy == "mean":
                fill_value = df[col].mean()
            else:
                fill_value = 0
            df[col] = df[col].fillna(fill_value)
        else:
            if categorical_strategy == "mode":
                mode_values = df[col].mode()
                fill_value = mode_values.iloc[0] if len(mode_values) > 0 else "UNKNOWN"
            else:
                fill_value = "UNKNOWN"
            df[col] = df[col].fillna(fill_value)

    return df


def _winsorize_outliers(
    df: pd.DataFrame, params: dict, protected_columns: set
) -> pd.DataFrame:
    """Clip outliers to specified bounds (Winsorize)."""
    columns = params.get("columns", [])
    for col_info in columns:
        col = col_info["column"]
        if col not in df.columns or col in protected_columns:
            continue
        lower = col_info.get("lower_bound")
        upper = col_info.get("upper_bound")
        if lower is not None and upper is not None:
            df[col] = df[col].clip(lower=lower, upper=upper)
    return df


def _convert_types(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Convert column types based on detected mismatches."""
    conversions = params.get("conversions", [])
    for conv in conversions:
        col = conv.get("column")
        suggested_type = conv.get("suggested_type")
        if col not in df.columns:
            continue
        if suggested_type == "numeric":
            df[col] = pd.to_numeric(df[col], errors="coerce")
        elif suggested_type == "datetime":
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def _drop_constant_columns(df: pd.DataFrame, protected_columns: set) -> pd.DataFrame:
    """Drop columns that have only one unique value."""
    cols_to_drop = []
    for col in df.columns:
        if col in protected_columns:
            continue
        if df[col].nunique(dropna=True) <= 1:
            cols_to_drop.append(col)
    return df.drop(columns=cols_to_drop)
