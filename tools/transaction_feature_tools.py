"""Transaction feature engineering tools.

Implements:
- infer_transaction_schema: auto-detect key columns
- build_account_daily_features: daily-level aggregation per account
- build_account_window_features: observation-point window features
- validate_transaction_feature_cutoff: time leakage detection
- profile_generated_features: feature profiling
"""

import numpy as np
import pandas as pd


# ============================================================
# Schema inference
# ============================================================

def infer_transaction_schema(
    df: pd.DataFrame,
    detected_roles: dict | None = None,
) -> dict:
    """Infer transaction table schema (key columns).

    Args:
        df: Transaction DataFrame.
        detected_roles: Optional pre-detected roles from DataTypeClassifierAgent.

    Returns:
        dict with account_col, time_col, amount_col, direction_col, inferred flag.
    """
    schema = {
        "account_col": None,
        "time_col": None,
        "amount_col": None,
        "direction_col": None,
        "inferred": True,
    }

    # Use detected_roles if available
    if detected_roles:
        schema["account_col"] = detected_roles.get("account_key") or detected_roles.get("customer_key")
        schema["time_col"] = detected_roles.get("transaction_time_col")
        schema["amount_col"] = detected_roles.get("amount_col")
        schema["direction_col"] = detected_roles.get("direction_col")

    # Fallback: infer from column names
    cols_lower = {c.lower(): c for c in df.columns}

    if not schema["account_col"]:
        for pattern in ["account_id", "acct_id", "account_no", "customer_id", "cust_id"]:
            if pattern in cols_lower:
                schema["account_col"] = cols_lower[pattern]
                break

    if not schema["time_col"]:
        for pattern in ["transaction_time", "txn_time", "trans_time", "txn_date", "trans_date"]:
            if pattern in cols_lower:
                schema["time_col"] = cols_lower[pattern]
                break

    if not schema["amount_col"]:
        for pattern in ["transaction_amount", "txn_amt", "trans_amt", "amount", "amt"]:
            if pattern in cols_lower:
                schema["amount_col"] = cols_lower[pattern]
                break

    if not schema["direction_col"]:
        for pattern in ["debit_credit_flag", "dc_flag", "debit_credit", "direction", "dr_cr"]:
            if pattern in cols_lower:
                schema["direction_col"] = cols_lower[pattern]
                break

    return schema


# ============================================================
# Daily features
# ============================================================

def build_account_daily_features(
    df: pd.DataFrame,
    account_col: str,
    time_col: str,
    amount_col: str,
    direction_col: str | None = None,
) -> pd.DataFrame:
    """Build daily-level features per account.

    Generates 15 daily features:
    1. txn_count: number of transactions
    2. txn_amount_sum: total transaction amount
    3. txn_amount_mean: average transaction amount
    4. txn_amount_max: max transaction amount
    5. txn_amount_min: min transaction amount
    6. txn_amount_std: std of transaction amount
    7. debit_count: number of debit transactions
    8. credit_count: number of credit transactions
    9. debit_amount_sum: total debit amount
    10. credit_amount_sum: total credit amount
    11. net_amount: credit - debit (net inflow)
    12. large_txn_count: transactions > 2x daily mean
    13. unique_channels: number of unique channels (if available)
    14. max_single_ratio: max single txn / daily total
    15. txn_time_span_hours: time span of transactions in the day

    Args:
        df: Transaction DataFrame.
        account_col: Account ID column.
        time_col: Transaction time column.
        amount_col: Transaction amount column.
        direction_col: Optional debit/credit direction column.

    Returns:
        DataFrame with account_col, transaction_date, and 15 feature columns.
    """
    df = df.copy()
    df["_txn_time"] = pd.to_datetime(df[time_col], errors="coerce")
    df["_txn_date"] = df["_txn_time"].dt.date
    df["_amount"] = pd.to_numeric(df[amount_col], errors="coerce").fillna(0)

    # Determine debit/credit
    if direction_col and direction_col in df.columns:
        dc_series = df[direction_col].astype(str).str.upper()
        df["_is_debit"] = dc_series.isin(["D", "DEBIT", "借", "1"])
        df["_is_credit"] = dc_series.isin(["C", "CREDIT", "贷", "0", "2"])
    else:
        # Assume positive = credit, negative = debit
        df["_is_debit"] = df["_amount"] < 0
        df["_is_credit"] = df["_amount"] >= 0

    df["_abs_amount"] = df["_amount"].abs()

    grouped = df.groupby([account_col, "_txn_date"])

    # Basic aggregations
    agg_result = grouped.agg(
        txn_count=("_abs_amount", "count"),
        txn_amount_sum=("_abs_amount", "sum"),
        txn_amount_mean=("_abs_amount", "mean"),
        txn_amount_max=("_abs_amount", "max"),
        txn_amount_min=("_abs_amount", "min"),
        txn_amount_std=("_abs_amount", "std"),
        debit_count=("_is_debit", "sum"),
        credit_count=("_is_credit", "sum"),
    ).reset_index()

    # Debit/credit amount sums
    debit_sums = df[df["_is_debit"]].groupby([account_col, "_txn_date"])["_abs_amount"].sum().reset_index()
    debit_sums.columns = [account_col, "_txn_date", "debit_amount_sum"]

    credit_sums = df[df["_is_credit"]].groupby([account_col, "_txn_date"])["_abs_amount"].sum().reset_index()
    credit_sums.columns = [account_col, "_txn_date", "credit_amount_sum"]

    agg_result = agg_result.merge(debit_sums, on=[account_col, "_txn_date"], how="left")
    agg_result = agg_result.merge(credit_sums, on=[account_col, "_txn_date"], how="left")
    agg_result["debit_amount_sum"] = agg_result["debit_amount_sum"].fillna(0)
    agg_result["credit_amount_sum"] = agg_result["credit_amount_sum"].fillna(0)

    # Net amount
    agg_result["net_amount"] = agg_result["credit_amount_sum"] - agg_result["debit_amount_sum"]

    # Large transaction count (> 2x daily mean)
    daily_means = agg_result[[account_col, "_txn_date", "txn_amount_mean"]].copy()
    df_with_mean = df.merge(daily_means, on=[account_col, "_txn_date"], how="left")
    large_txn = df_with_mean[df_with_mean["_abs_amount"] > 2 * df_with_mean["txn_amount_mean"]]
    large_counts = large_txn.groupby([account_col, "_txn_date"]).size().reset_index(name="large_txn_count")
    agg_result = agg_result.merge(large_counts, on=[account_col, "_txn_date"], how="left")
    agg_result["large_txn_count"] = agg_result["large_txn_count"].fillna(0).astype(int)

    # Unique channels
    if "channel" in df.columns:
        channel_counts = df.groupby([account_col, "_txn_date"])["channel"].nunique().reset_index()
        channel_counts.columns = [account_col, "_txn_date", "unique_channels"]
        agg_result = agg_result.merge(channel_counts, on=[account_col, "_txn_date"], how="left")
    else:
        agg_result["unique_channels"] = 1

    # Max single ratio
    agg_result["max_single_ratio"] = np.where(
        agg_result["txn_amount_sum"] > 0,
        agg_result["txn_amount_max"] / agg_result["txn_amount_sum"],
        0,
    )

    # Time span in hours
    time_spans = grouped["_txn_time"].agg(lambda x: (x.max() - x.min()).total_seconds() / 3600 if len(x) > 1 else 0)
    time_spans = time_spans.reset_index()
    time_spans.columns = [account_col, "_txn_date", "txn_time_span_hours"]
    agg_result = agg_result.merge(time_spans, on=[account_col, "_txn_date"], how="left")

    # Fill NaN std with 0
    agg_result["txn_amount_std"] = agg_result["txn_amount_std"].fillna(0)

    # Rename date column
    agg_result = agg_result.rename(columns={"_txn_date": "transaction_date"})

    return agg_result


# ============================================================
# Window features
# ============================================================

_WINDOWS = [7, 14, 30, 60, 90, 180]


def build_account_window_features(
    daily_features: pd.DataFrame,
    main_df: pd.DataFrame,
    account_col: str,
    base_time_col: str,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Build observation-point window features from daily features.

    For each account in main_df, aggregates daily features over historical windows
    (7d, 14d, 30d, 60d, 90d, 180d) strictly before the observation date.

    7 feature groups per window:
    1. Activity: txn_count_sum, active_days
    2. Amount scale: amount_sum, amount_mean, amount_max
    3. Income/expense: credit_sum, debit_sum, net_sum
    4. Volatility: amount_std, amount_cv
    5. Behavior: large_txn_ratio, max_single_ratio_mean
    6. Large transactions: large_txn_count_sum
    7. Counterparty diversity: unique_channels_mean

    Args:
        daily_features: Output from build_account_daily_features.
        main_df: Main table with account_col and base_time_col.
        account_col: Account/customer ID column.
        base_time_col: Observation date column in main_df.
        windows: List of window sizes in days. Defaults to [7,14,30,60,90,180].

    Returns:
        DataFrame with account_col, base_time_col, and window features.
    """
    if windows is None:
        windows = _WINDOWS

    daily_features = daily_features.copy()
    daily_features["transaction_date"] = pd.to_datetime(daily_features["transaction_date"])

    main_df = main_df.copy()
    main_df["_base_time"] = pd.to_datetime(main_df[base_time_col], errors="coerce")

    results = main_df[[account_col, base_time_col, "_base_time"]].copy()

    for w in windows:
        prefix = f"w{w}d_"
        window_feats = _compute_window_features(
            daily_features, main_df, account_col, w, prefix
        )
        results = results.merge(window_feats, on=[account_col, "_base_time"], how="left")

    results = results.drop(columns=["_base_time"])
    return results


def _compute_window_features(
    daily_features: pd.DataFrame,
    main_df: pd.DataFrame,
    account_col: str,
    window_days: int,
    prefix: str,
) -> pd.DataFrame:
    """Compute features for a single window size."""
    records = []

    for _, row in main_df.iterrows():
        acct = row[account_col]
        base_time = row["_base_time"]

        if pd.isna(base_time):
            records.append({account_col: acct, "_base_time": base_time})
            continue

        # Strict: transaction_date < base_time (not <=)
        window_start = base_time - pd.Timedelta(days=window_days)
        mask = (
            (daily_features[account_col] == acct)
            & (daily_features["transaction_date"] >= window_start)
            & (daily_features["transaction_date"] < base_time)
        )
        window_data = daily_features[mask]

        feat = {account_col: acct, "_base_time": base_time}

        if window_data.empty:
            # All features are 0/NaN
            feat[f"{prefix}txn_count_sum"] = 0
            feat[f"{prefix}active_days"] = 0
            feat[f"{prefix}amount_sum"] = 0.0
            feat[f"{prefix}amount_mean"] = 0.0
            feat[f"{prefix}amount_max"] = 0.0
            feat[f"{prefix}credit_sum"] = 0.0
            feat[f"{prefix}debit_sum"] = 0.0
            feat[f"{prefix}net_sum"] = 0.0
            feat[f"{prefix}amount_std"] = 0.0
            feat[f"{prefix}amount_cv"] = 0.0
            feat[f"{prefix}large_txn_ratio"] = 0.0
            feat[f"{prefix}max_single_ratio_mean"] = 0.0
            feat[f"{prefix}large_txn_count_sum"] = 0
            feat[f"{prefix}unique_channels_mean"] = 0.0
        else:
            total_txn = window_data["txn_count"].sum()
            feat[f"{prefix}txn_count_sum"] = int(total_txn)
            feat[f"{prefix}active_days"] = len(window_data)
            feat[f"{prefix}amount_sum"] = float(window_data["txn_amount_sum"].sum())
            feat[f"{prefix}amount_mean"] = float(window_data["txn_amount_mean"].mean())
            feat[f"{prefix}amount_max"] = float(window_data["txn_amount_max"].max())
            feat[f"{prefix}credit_sum"] = float(window_data["credit_amount_sum"].sum())
            feat[f"{prefix}debit_sum"] = float(window_data["debit_amount_sum"].sum())
            feat[f"{prefix}net_sum"] = float(window_data["net_amount"].sum())

            amount_std = window_data["txn_amount_sum"].std()
            amount_mean = window_data["txn_amount_sum"].mean()
            feat[f"{prefix}amount_std"] = float(amount_std) if not pd.isna(amount_std) else 0.0
            feat[f"{prefix}amount_cv"] = float(amount_std / amount_mean) if amount_mean > 0 and not pd.isna(amount_std) else 0.0

            large_total = window_data["large_txn_count"].sum()
            feat[f"{prefix}large_txn_ratio"] = float(large_total / total_txn) if total_txn > 0 else 0.0
            feat[f"{prefix}max_single_ratio_mean"] = float(window_data["max_single_ratio"].mean())
            feat[f"{prefix}large_txn_count_sum"] = int(large_total)
            feat[f"{prefix}unique_channels_mean"] = float(window_data["unique_channels"].mean())

        records.append(feat)

    return pd.DataFrame(records)


# ============================================================
# Time cutoff validation
# ============================================================

def validate_transaction_feature_cutoff(
    daily_features: pd.DataFrame,
    main_df: pd.DataFrame,
    account_col: str,
    base_time_col: str,
) -> dict:
    """Validate that no transaction features use data on or after the observation date.

    Args:
        daily_features: Daily features with transaction_date.
        main_df: Main table with base_time_col.
        account_col: Account column.
        base_time_col: Observation date column.

    Returns:
        dict with is_valid, violations_count, sample_violations.
    """
    daily = daily_features.copy()
    daily["transaction_date"] = pd.to_datetime(daily["transaction_date"])

    main = main_df[[account_col, base_time_col]].copy()
    main["_base_time"] = pd.to_datetime(main[base_time_col], errors="coerce")

    # Join to check
    merged = daily.merge(main, on=account_col, how="inner")
    violations = merged[merged["transaction_date"] >= merged["_base_time"]]

    sample = []
    if not violations.empty:
        sample_rows = violations.head(5)
        for _, row in sample_rows.iterrows():
            sample.append({
                "account": str(row[account_col]),
                "transaction_date": str(row["transaction_date"].date()),
                "base_time": str(row["_base_time"].date()),
            })

    return {
        "is_valid": len(violations) == 0,
        "violations_count": len(violations),
        "total_checked": len(merged),
        "sample_violations": sample,
    }


# ============================================================
# Feature profiling
# ============================================================

def profile_generated_features(df: pd.DataFrame, exclude_cols: list[str] | None = None) -> dict:
    """Profile generated features for quality check.

    Args:
        df: Feature DataFrame.
        exclude_cols: Columns to exclude from profiling (e.g., ID, time).

    Returns:
        dict with feature_count, profiles per feature, and summary stats.
    """
    exclude = set(exclude_cols or [])
    feature_cols = [c for c in df.columns if c not in exclude]

    profiles = []
    for col in feature_cols:
        series = df[col]
        profile = {
            "column": col,
            "dtype": str(series.dtype),
            "missing_rate": round(float(series.isna().mean()), 4),
            "unique_count": int(series.nunique()),
        }

        if np.issubdtype(series.dtype, np.number):
            profile["mean"] = round(float(series.mean()), 4) if not series.isna().all() else None
            profile["std"] = round(float(series.std()), 4) if not series.isna().all() else None
            profile["min"] = round(float(series.min()), 4) if not series.isna().all() else None
            profile["max"] = round(float(series.max()), 4) if not series.isna().all() else None
            profile["zero_rate"] = round(float((series == 0).mean()), 4)

        profiles.append(profile)

    # Summary
    numeric_profiles = [p for p in profiles if "mean" in p]
    high_zero_features = [p["column"] for p in numeric_profiles if p.get("zero_rate", 0) > 0.9]
    high_missing_features = [p["column"] for p in profiles if p["missing_rate"] > 0.5]

    return {
        "feature_count": len(feature_cols),
        "profiles": profiles,
        "high_zero_features": high_zero_features,
        "high_missing_features": high_missing_features,
        "summary": {
            "total_features": len(feature_cols),
            "numeric_features": len(numeric_profiles),
            "high_zero_count": len(high_zero_features),
            "high_missing_count": len(high_missing_features),
        },
    }
