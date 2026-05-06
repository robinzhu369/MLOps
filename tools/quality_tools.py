"""Data quality analysis tools."""

import numpy as np
import pandas as pd

from core.constants import (
    DEFAULT_HIGH_MISSING_THRESHOLD,
    DEFAULT_OUTLIER_METHOD,
    LEAKAGE_KEYWORDS,
)


def analyze_duplicates(
    df: pd.DataFrame,
    key_columns: list[str] | None = None,
) -> dict:
    """Analyze duplicate rows in the dataset.

    Args:
        df: Input DataFrame.
        key_columns: If provided, check duplicates on these columns only.

    Returns:
        dict with duplicate_rows, duplicate_rate, duplicate_key_rows, suggestion.
    """
    n = len(df)
    exact_dup = int(df.duplicated().sum())
    exact_dup_rate = round(exact_dup / n, 4) if n > 0 else 0.0

    key_dup = 0
    if key_columns:
        valid_keys = [c for c in key_columns if c in df.columns]
        if valid_keys:
            key_dup = int(df.duplicated(subset=valid_keys).sum())

    suggestion = ""
    if exact_dup > 0:
        suggestion = f"发现 {exact_dup} 行完全重复（{exact_dup_rate:.1%}），建议去重"
    if key_dup > 0:
        suggestion += f"；主键重复 {key_dup} 行，需确认是否为数据错误"

    return {
        "duplicate_rows": exact_dup,
        "duplicate_rate": exact_dup_rate,
        "duplicate_key_rows": key_dup,
        "suggestion": suggestion,
    }


def analyze_missing_values(
    df: pd.DataFrame,
    high_missing_threshold: float = DEFAULT_HIGH_MISSING_THRESHOLD,
) -> dict:
    """Analyze missing values for each column.

    Returns:
        dict with columns list (each with missing_count, missing_rate, suggestion),
        summary stats, and high_missing_columns.
    """
    n = len(df)
    columns_info = []
    high_missing_cols = []

    for col in df.columns:
        missing_count = int(df[col].isna().sum())
        missing_rate = round(missing_count / n, 4) if n > 0 else 0.0

        suggestion = ""
        if missing_rate == 0:
            suggestion = "无缺失"
        elif missing_rate < 0.05:
            suggestion = "缺失率低，建议中位数/众数填充"
        elif missing_rate < high_missing_threshold:
            suggestion = "缺失率中等，建议分析缺失模式后决定填充策略"
        else:
            suggestion = "缺失率高，建议考虑删除该列或标记为缺失指示变量"
            high_missing_cols.append(col)

        columns_info.append({
            "column": col,
            "missing_count": missing_count,
            "missing_rate": missing_rate,
            "suggestion": suggestion,
        })

    total_missing = sum(c["missing_count"] for c in columns_info)
    total_cells = n * len(df.columns)

    return {
        "columns": columns_info,
        "total_missing_rate": round(total_missing / total_cells, 4) if total_cells > 0 else 0.0,
        "high_missing_columns": high_missing_cols,
        "high_missing_threshold": high_missing_threshold,
    }


def analyze_outliers(
    df: pd.DataFrame,
    method: str = DEFAULT_OUTLIER_METHOD,
    protected_columns: list[str] | None = None,
) -> dict:
    """Analyze outliers in numeric columns using IQR or Z-score method.

    Args:
        df: Input DataFrame.
        method: "IQR" or "zscore".
        protected_columns: Columns to skip (e.g., label, ID).

    Returns:
        dict with columns list (each with outlier_count, outlier_rate, bounds).
    """
    protected = set(protected_columns or [])
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    columns_info = []

    for col in numeric_cols:
        if col in protected:
            continue

        series = df[col].dropna()
        if len(series) == 0:
            continue

        if method == "IQR":
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
        else:  # zscore
            mean = series.mean()
            std = series.std()
            lower = mean - 3 * std
            upper = mean + 3 * std

        outlier_mask = (series < lower) | (series > upper)
        outlier_count = int(outlier_mask.sum())
        outlier_rate = round(outlier_count / len(series), 4)

        suggestion = ""
        if outlier_rate > 0.1:
            suggestion = "异常值比例高，建议检查数据来源或考虑 Winsorize"
        elif outlier_rate > 0.01:
            suggestion = "存在少量异常值，建议 Winsorize 处理"
        else:
            suggestion = "异常值极少，可忽略"

        columns_info.append({
            "column": col,
            "method": method,
            "outlier_count": outlier_count,
            "outlier_rate": outlier_rate,
            "lower_bound": round(float(lower), 4),
            "upper_bound": round(float(upper), 4),
            "suggestion": suggestion,
        })

    return {"columns": columns_info, "method": method}


def analyze_type_mismatch(df: pd.DataFrame) -> dict:
    """Detect columns where the actual content doesn't match the inferred dtype.

    For example, a column stored as object that contains mostly numeric values.

    Returns:
        dict with mismatched columns and suggested types.
    """
    mismatches = []

    for col in df.columns:
        if df[col].dtype == object:
            non_null = df[col].dropna()
            if len(non_null) == 0:
                continue

            # Check if mostly numeric
            numeric_count = pd.to_numeric(non_null, errors="coerce").notna().sum()
            numeric_rate = numeric_count / len(non_null)

            if numeric_rate > 0.9:
                mismatches.append({
                    "column": col,
                    "current_type": "object",
                    "suggested_type": "numeric",
                    "conversion_rate": round(float(numeric_rate), 4),
                })
                continue

            # Check if mostly datetime
            try:
                datetime_count = pd.to_datetime(non_null, errors="coerce", infer_datetime_format=True).notna().sum()
                datetime_rate = datetime_count / len(non_null)
                if datetime_rate > 0.9:
                    mismatches.append({
                        "column": col,
                        "current_type": "object",
                        "suggested_type": "datetime",
                        "conversion_rate": round(float(datetime_rate), 4),
                    })
            except Exception:
                pass

    return {"mismatches": mismatches}


def analyze_label_quality(
    df: pd.DataFrame,
    label_col: str | None = None,
) -> dict:
    """Analyze label column quality for binary classification.

    Returns:
        dict with label distribution, positive rate, and warnings.
    """
    if not label_col or label_col not in df.columns:
        return {"label_col": label_col, "found": False, "warnings": ["标签列未找到"]}

    series = df[label_col]
    missing = int(series.isna().sum())
    value_counts = series.value_counts(dropna=True).to_dict()
    n_valid = len(series) - missing
    unique_values = sorted(series.dropna().unique().tolist())

    warnings = []

    # Check binary
    if len(unique_values) > 2:
        warnings.append(f"标签列有 {len(unique_values)} 个唯一值，非标准二分类")
    elif len(unique_values) < 2:
        warnings.append("标签列只有一个值，无法建模")

    # Check positive rate
    if len(unique_values) == 2:
        pos_val = max(unique_values)
        pos_count = value_counts.get(pos_val, 0)
        pos_rate = pos_count / n_valid if n_valid > 0 else 0.0

        if pos_rate < 0.01:
            warnings.append(f"正样本率极低 ({pos_rate:.2%})，建议考虑过采样或调整阈值")
        elif pos_rate > 0.5:
            warnings.append(f"正样本率偏高 ({pos_rate:.2%})，请确认正样本定义")
    else:
        pos_rate = None

    if missing > 0:
        warnings.append(f"标签列有 {missing} 个缺失值")

    return {
        "label_col": label_col,
        "found": True,
        "missing_count": missing,
        "value_counts": {str(k): int(v) for k, v in value_counts.items()},
        "unique_values": [str(v) for v in unique_values],
        "positive_rate": round(pos_rate, 4) if pos_rate is not None else None,
        "warnings": warnings,
    }


def analyze_key_quality(
    df: pd.DataFrame,
    key_columns: dict[str, str | None],
) -> dict:
    """Analyze quality of key columns (ID, time, account).

    Args:
        key_columns: dict mapping role to column name, e.g.
            {"id_col": "customer_id", "time_col": "apply_date"}

    Returns:
        dict with analysis for each key column.
    """
    results = {}

    for role, col_name in key_columns.items():
        if not col_name or col_name not in df.columns:
            results[role] = {"column": col_name, "found": False}
            continue

        series = df[col_name]
        missing = int(series.isna().sum())
        unique_count = int(series.nunique(dropna=True))
        unique_rate = round(unique_count / len(df), 4) if len(df) > 0 else 0.0

        info = {
            "column": col_name,
            "found": True,
            "missing_count": missing,
            "unique_count": unique_count,
            "unique_rate": unique_rate,
            "warnings": [],
        }

        if missing > 0:
            info["warnings"].append(f"主键 {col_name} 有 {missing} 个缺失值")

        if role in ("id_col", "customer_key", "account_key") and unique_rate < 1.0:
            dup_count = len(df) - unique_count
            info["warnings"].append(
                f"主键 {col_name} 存在 {dup_count} 个重复值（唯一率 {unique_rate:.1%}）"
            )

        results[role] = info

    return results


def generate_data_quality_report(
    duplicate_analysis: dict,
    missing_analysis: dict,
    outlier_analysis: dict,
    type_analysis: dict,
    label_analysis: dict,
    key_analysis: dict,
) -> dict:
    """Generate an overall data quality report with a quality score.

    Returns:
        dict with all analyses combined and an overall_quality_score (0-100).
    """
    score = 100

    # Deduct for duplicates
    dup_rate = duplicate_analysis.get("duplicate_rate", 0)
    if dup_rate > 0.1:
        score -= 20
    elif dup_rate > 0.01:
        score -= 10
    elif dup_rate > 0:
        score -= 5

    # Deduct for missing values
    total_missing_rate = missing_analysis.get("total_missing_rate", 0)
    high_missing_count = len(missing_analysis.get("high_missing_columns", []))
    if total_missing_rate > 0.2:
        score -= 20
    elif total_missing_rate > 0.05:
        score -= 10
    score -= min(high_missing_count * 3, 15)

    # Deduct for outliers
    outlier_cols = outlier_analysis.get("columns", [])
    high_outlier_count = sum(1 for c in outlier_cols if c.get("outlier_rate", 0) > 0.1)
    score -= min(high_outlier_count * 5, 15)

    # Deduct for type mismatches
    mismatch_count = len(type_analysis.get("mismatches", []))
    score -= min(mismatch_count * 3, 10)

    # Deduct for label issues
    label_warnings = label_analysis.get("warnings", [])
    score -= min(len(label_warnings) * 5, 15)

    # Deduct for key issues
    for role, info in key_analysis.items():
        if isinstance(info, dict):
            key_warnings = info.get("warnings", [])
            score -= min(len(key_warnings) * 5, 10)

    score = max(score, 0)

    need_cleaning_review = score < 80 or dup_rate > 0 or high_missing_count > 0

    return {
        "duplicate_analysis": duplicate_analysis,
        "missing_analysis": missing_analysis,
        "outlier_analysis": outlier_analysis,
        "type_analysis": type_analysis,
        "label_analysis": label_analysis,
        "key_analysis": key_analysis,
        "overall_quality_score": score,
        "need_cleaning_review": need_cleaning_review,
    }
