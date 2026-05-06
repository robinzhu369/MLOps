"""Risk guard tools — detect risky fields before modeling."""

import re

import numpy as np
import pandas as pd

from core.constants import LEAKAGE_KEYWORDS, DEFAULT_HIGH_MISSING_THRESHOLD


def detect_leakage_columns(
    df: pd.DataFrame,
    label_col: str | None = None,
    extra_keywords: list[str] | None = None,
) -> dict:
    """Detect columns that may leak the label (target variable).

    Checks column names against known leakage keywords like "dpd", "overdue",
    "default", "逾期", etc.

    Args:
        df: Input DataFrame.
        label_col: Label column name (excluded from detection).
        extra_keywords: Additional keywords to check.

    Returns:
        dict with leakage_columns list, each with column, matched_keyword, risk_level.
    """
    keywords = LEAKAGE_KEYWORDS.copy()
    if extra_keywords:
        keywords.extend(extra_keywords)

    leakage_columns = []

    for col in df.columns:
        if col == label_col:
            continue

        col_lower = col.lower()
        for keyword in keywords:
            if keyword.lower() in col_lower:
                leakage_columns.append({
                    "column": col,
                    "matched_keyword": keyword,
                    "risk_level": "high",
                    "reason": f"字段名包含泄露关键词 '{keyword}'",
                })
                break

    return {
        "leakage_columns": leakage_columns,
        "n_detected": len(leakage_columns),
    }


def detect_id_columns(
    df: pd.DataFrame,
    protected_columns: list[str] | None = None,
) -> dict:
    """Detect ID-like columns that should not be used as features.

    Heuristics:
    - Column name contains "id", "key", "no", "号" patterns
    - High cardinality (unique rate > 0.9)
    - String/object type with unique patterns

    Args:
        df: Input DataFrame.
        protected_columns: Columns to skip (already known IDs).

    Returns:
        dict with id_columns list, each with column, reason, unique_rate.
    """
    protected = set(protected_columns or [])
    id_patterns = [
        r"_id$", r"^id_", r"_key$", r"_no$", r"_num$",
        r"编号$", r"号码$", r"^id$",
        r"customer_id", r"account_id", r"loan_id", r"app_id",
    ]

    id_columns = []

    for col in df.columns:
        if col in protected:
            continue

        col_lower = col.lower()
        n = len(df)
        unique_rate = df[col].nunique() / n if n > 0 else 0

        # Check name patterns
        name_match = any(re.search(p, col_lower) for p in id_patterns)

        # Check high cardinality for string columns
        is_high_cardinality = (
            df[col].dtype == object and unique_rate > 0.9
        )

        if name_match:
            id_columns.append({
                "column": col,
                "reason": "字段名匹配 ID 模式",
                "unique_rate": round(unique_rate, 4),
                "risk_level": "high",
            })
        elif is_high_cardinality:
            id_columns.append({
                "column": col,
                "reason": f"高基数字符串列（唯一率={unique_rate:.1%}）",
                "unique_rate": round(unique_rate, 4),
                "risk_level": "medium",
            })

    return {
        "id_columns": id_columns,
        "n_detected": len(id_columns),
    }


def detect_high_missing_columns(
    df: pd.DataFrame,
    threshold: float = DEFAULT_HIGH_MISSING_THRESHOLD,
    protected_columns: list[str] | None = None,
) -> dict:
    """Detect columns with missing rate above threshold.

    Args:
        df: Input DataFrame.
        threshold: Missing rate threshold (default 0.3).
        protected_columns: Columns to skip.

    Returns:
        dict with high_missing_columns list.
    """
    protected = set(protected_columns or [])
    n = len(df)
    high_missing = []

    for col in df.columns:
        if col in protected:
            continue

        missing_rate = df[col].isna().sum() / n if n > 0 else 0

        if missing_rate >= threshold:
            high_missing.append({
                "column": col,
                "missing_rate": round(float(missing_rate), 4),
                "risk_level": "medium",
                "reason": f"缺失率 {missing_rate:.1%} 超过阈值 {threshold:.0%}",
            })

    return {
        "high_missing_columns": high_missing,
        "n_detected": len(high_missing),
        "threshold": threshold,
    }


def detect_time_leakage_candidates(
    df: pd.DataFrame,
    observation_time_col: str | None = None,
    protected_columns: list[str] | None = None,
) -> dict:
    """Detect columns that may contain post-observation-point information.

    Checks for:
    - Date/time columns that are after the observation time
    - Column names suggesting post-event data (e.g., "还款日期", "结清日期")

    Args:
        df: Input DataFrame.
        observation_time_col: The observation time column.
        protected_columns: Columns to skip.

    Returns:
        dict with time_leakage_candidates list.
    """
    protected = set(protected_columns or [])

    post_event_keywords = [
        "还款日期", "结清日期", "催收日期", "核销日期",
        "repay_date", "settle_date", "close_date", "collection_date",
        "end_date", "maturity_date", "completion_date",
    ]

    candidates = []

    for col in df.columns:
        if col in protected or col == observation_time_col:
            continue

        col_lower = col.lower()

        # Check name patterns
        for keyword in post_event_keywords:
            if keyword.lower() in col_lower:
                candidates.append({
                    "column": col,
                    "reason": f"字段名包含贷后时间关键词 '{keyword}'",
                    "risk_level": "high",
                })
                break
        else:
            # Check if it's a date column that's consistently after observation time
            if observation_time_col and observation_time_col in df.columns:
                try:
                    col_dates = pd.to_datetime(df[col], errors="coerce")
                    obs_dates = pd.to_datetime(df[observation_time_col], errors="coerce")

                    valid_mask = col_dates.notna() & obs_dates.notna()
                    if valid_mask.sum() > 10:
                        after_rate = (col_dates[valid_mask] > obs_dates[valid_mask]).mean()
                        if after_rate > 0.8:
                            candidates.append({
                                "column": col,
                                "reason": f"该日期列 {after_rate:.0%} 的值晚于观察点，疑似贷后信息",
                                "risk_level": "medium",
                            })
                except (TypeError, ValueError):
                    pass

    return {
        "time_leakage_candidates": candidates,
        "n_detected": len(candidates),
    }
