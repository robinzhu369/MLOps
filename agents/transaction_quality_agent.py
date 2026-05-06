"""TransactionQualityAgent — specialized quality analysis for transaction flow tables."""

from core.permissions import check_tool_permission
from core.exceptions import PermissionDeniedError
from core.state import RiskModelingProjectState
from tools.file_tools import read_dataframe
from tools.trace_tools import write_agent_trace

import numpy as np
import pandas as pd

AGENT_NAME = "TransactionQualityAgent"


def _check_permission(tool_name: str) -> None:
    if not check_tool_permission(AGENT_NAME, tool_name):
        raise PermissionDeniedError(AGENT_NAME, tool_name)


def analyze_transaction_quality(
    df: pd.DataFrame,
    account_col: str,
    time_col: str,
    amount_col: str | None = None,
    direction_col: str | None = None,
) -> dict:
    """Analyze quality issues specific to transaction flow tables.

    Checks:
    - Time column validity and range
    - Account coverage and activity distribution
    - Amount distribution and anomalies
    - Direction field validity
    - Temporal gaps and patterns

    Returns:
        dict with transaction-specific quality metrics.
    """
    n = len(df)
    issues = []
    warnings = []

    # Time column analysis
    time_analysis = _analyze_time_column(df, time_col)
    if time_analysis.get("invalid_count", 0) > 0:
        issues.append(f"时间列 {time_col} 有 {time_analysis['invalid_count']} 个无效值")

    # Account coverage
    account_analysis = _analyze_account_coverage(df, account_col)

    # Amount analysis
    amount_analysis = {}
    if amount_col and amount_col in df.columns:
        amount_analysis = _analyze_amount(df, amount_col)
        if amount_analysis.get("negative_count", 0) > 0 and not direction_col:
            warnings.append("金额列存在负值但未检测到借贷方向字段，请确认金额含义")

    # Direction analysis
    direction_analysis = {}
    if direction_col and direction_col in df.columns:
        direction_analysis = _analyze_direction(df, direction_col)

    # Temporal pattern
    temporal_analysis = _analyze_temporal_pattern(df, time_col, account_col)

    overall_score = _compute_transaction_quality_score(
        time_analysis, account_analysis, amount_analysis, issues
    )

    return {
        "n_rows": n,
        "n_accounts": account_analysis.get("n_accounts", 0),
        "time_analysis": time_analysis,
        "account_analysis": account_analysis,
        "amount_analysis": amount_analysis,
        "direction_analysis": direction_analysis,
        "temporal_analysis": temporal_analysis,
        "issues": issues,
        "warnings": warnings,
        "overall_quality_score": overall_score,
        "need_cleaning_review": len(issues) > 0 or overall_score < 80,
    }


def _analyze_time_column(df: pd.DataFrame, time_col: str) -> dict:
    """Analyze the transaction time column."""
    if time_col not in df.columns:
        return {"error": f"时间列 {time_col} 不存在", "invalid_count": len(df)}

    time_series = pd.to_datetime(df[time_col], errors="coerce")
    invalid_count = int(time_series.isna().sum()) - int(df[time_col].isna().sum())
    valid_times = time_series.dropna()

    result = {
        "invalid_count": invalid_count,
        "null_count": int(df[time_col].isna().sum()),
    }

    if not valid_times.empty:
        result["min_time"] = str(valid_times.min())
        result["max_time"] = str(valid_times.max())
        result["time_span_days"] = (valid_times.max() - valid_times.min()).days
        result["is_sorted"] = bool(valid_times.is_monotonic_increasing)

    return result


def _analyze_account_coverage(df: pd.DataFrame, account_col: str) -> dict:
    """Analyze account coverage and activity distribution."""
    if account_col not in df.columns:
        return {"error": f"账户列 {account_col} 不存在", "n_accounts": 0}

    account_counts = df[account_col].value_counts()
    n_accounts = len(account_counts)

    return {
        "n_accounts": n_accounts,
        "avg_txn_per_account": round(float(account_counts.mean()), 1),
        "median_txn_per_account": round(float(account_counts.median()), 1),
        "max_txn_per_account": int(account_counts.max()),
        "min_txn_per_account": int(account_counts.min()),
        "single_txn_accounts": int((account_counts == 1).sum()),
        "single_txn_rate": round(float((account_counts == 1).sum()) / n_accounts, 4) if n_accounts > 0 else 0.0,
    }


def _analyze_amount(df: pd.DataFrame, amount_col: str) -> dict:
    """Analyze transaction amount distribution."""
    series = df[amount_col].dropna()
    if series.empty:
        return {"error": "金额列全部为空"}

    return {
        "mean": round(float(series.mean()), 2),
        "median": round(float(series.median()), 2),
        "std": round(float(series.std()), 2),
        "min": round(float(series.min()), 2),
        "max": round(float(series.max()), 2),
        "zero_count": int((series == 0).sum()),
        "negative_count": int((series < 0).sum()),
        "zero_rate": round(float((series == 0).sum()) / len(series), 4),
    }


def _analyze_direction(df: pd.DataFrame, direction_col: str) -> dict:
    """Analyze debit/credit direction field."""
    series = df[direction_col].dropna()
    value_counts = series.value_counts()

    return {
        "unique_values": list(value_counts.index[:10]),
        "value_distribution": {str(k): int(v) for k, v in value_counts.head(10).items()},
        "n_unique": int(series.nunique()),
        "is_binary": series.nunique() == 2,
    }


def _analyze_temporal_pattern(df: pd.DataFrame, time_col: str, account_col: str) -> dict:
    """Analyze temporal patterns in the transaction data."""
    time_series = pd.to_datetime(df[time_col], errors="coerce")
    valid_mask = time_series.notna()

    if valid_mask.sum() < 2:
        return {"error": "有效时间记录不足"}

    valid_df = df[valid_mask].copy()
    valid_df["_parsed_time"] = time_series[valid_mask]

    # Daily transaction volume
    daily_counts = valid_df.groupby(valid_df["_parsed_time"].dt.date).size()

    result = {
        "total_days": len(daily_counts),
        "avg_daily_txn": round(float(daily_counts.mean()), 1),
        "max_daily_txn": int(daily_counts.max()),
        "min_daily_txn": int(daily_counts.min()),
    }

    # Check for gaps (days with zero transactions)
    if len(daily_counts) > 1:
        date_range = pd.date_range(daily_counts.index.min(), daily_counts.index.max())
        gap_days = len(date_range) - len(daily_counts)
        result["gap_days"] = gap_days
        result["gap_rate"] = round(gap_days / len(date_range), 4) if len(date_range) > 0 else 0.0

    return result


def _compute_transaction_quality_score(
    time_analysis: dict,
    account_analysis: dict,
    amount_analysis: dict,
    issues: list,
) -> int:
    """Compute an overall quality score (0-100) for transaction data."""
    score = 100

    # Deduct for time issues
    invalid_time = time_analysis.get("invalid_count", 0)
    null_time = time_analysis.get("null_count", 0)
    if invalid_time > 0:
        score -= min(20, invalid_time // 10)
    if null_time > 0:
        score -= min(10, null_time // 100)

    # Deduct for single-txn accounts
    single_rate = account_analysis.get("single_txn_rate", 0)
    if single_rate > 0.5:
        score -= 15
    elif single_rate > 0.3:
        score -= 10

    # Deduct for amount issues
    zero_rate = amount_analysis.get("zero_rate", 0)
    if zero_rate > 0.1:
        score -= 10

    # Deduct for each issue
    score -= len(issues) * 5

    return max(0, min(100, score))


def run(state: RiskModelingProjectState) -> dict:
    """Run transaction quality analysis.

    Expects state to contain:
        - project_name
        - transaction_data_path
        - account_col or customer_col
        - time_col (transaction time)
        - Optional: amount_col, direction_col from detected_roles

    Returns:
        Transaction quality report dict.
    """
    project_name = state["project_name"]
    data_path = state.get("transaction_data_path")

    if not data_path:
        write_agent_trace(
            project_name=project_name,
            agent_name=AGENT_NAME,
            reasoning_summary="没有交易流水数据文件",
            action="skip",
            observation_summary="transaction_data_path 为空",
            decision="跳过交易质量分析",
            next_node="DataCleaningPlannerAgent",
            status="skipped",
        )
        return {"skipped": True}

    df = read_dataframe(data_path)

    account_col = state.get("account_col") or state.get("customer_col", "")
    time_col = state.get("time_col", "")

    # Try to get from transaction_schema
    txn_schema = state.get("transaction_schema", {})
    if not account_col:
        account_col = txn_schema.get("account_col", "")
    if not time_col:
        time_col = txn_schema.get("time_col", "")

    if not account_col or not time_col:
        write_agent_trace(
            project_name=project_name,
            agent_name=AGENT_NAME,
            reasoning_summary="缺少必要的账户列或时间列信息",
            action="skip",
            observation_summary=f"account_col={account_col}, time_col={time_col}",
            decision="无法进行交易质量分析，需要人工指定关键列",
            next_node="HumanReviewGate_TransactionSchema",
            status="need_human_review",
        )
        return {"error": "missing_key_columns", "account_col": account_col, "time_col": time_col}

    # Determine amount and direction columns
    amount_col = txn_schema.get("amount_col")
    direction_col = txn_schema.get("direction_col")

    _check_permission("analyze_transaction_quality")
    report = analyze_transaction_quality(
        df=df,
        account_col=account_col,
        time_col=time_col,
        amount_col=amount_col,
        direction_col=direction_col,
    )

    write_agent_trace(
        project_name=project_name,
        agent_name=AGENT_NAME,
        reasoning_summary="对交易流水表进行专项质量分析",
        action="analyze_transaction_quality",
        action_input_summary={
            "data_path": data_path,
            "account_col": account_col,
            "time_col": time_col,
            "n_rows": len(df),
        },
        observation_summary=(
            f"质量评分={report['overall_quality_score']}, "
            f"账户数={report['n_accounts']}, "
            f"问题数={len(report['issues'])}"
        ),
        decision="交易质量分析完成",
        next_node="DataCleaningPlannerAgent",
        status="completed",
    )

    return report
