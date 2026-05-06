"""TransactionFeatureAgent — orchestrates transaction feature engineering pipeline."""

import os
from typing import Callable

from core.permissions import check_tool_permission
from core.exceptions import PermissionDeniedError
from core.state import RiskModelingProjectState
from tools.file_tools import read_dataframe
from tools.transaction_feature_tools import (
    infer_transaction_schema,
    build_account_daily_features,
    build_account_window_features,
    validate_transaction_feature_cutoff,
    profile_generated_features,
)
from tools.trace_tools import write_agent_trace

AGENT_NAME = "TransactionFeatureAgent"


def _check_permission(tool_name: str) -> None:
    if not check_tool_permission(AGENT_NAME, tool_name):
        raise PermissionDeniedError(AGENT_NAME, tool_name)


def run(
    state: RiskModelingProjectState,
    output_dir: str | None = None,
) -> dict:
    """Run the full transaction feature engineering pipeline.

    Steps:
    1. Infer transaction schema
    2. Build daily features
    3. Build window features (if main table with observation dates available)
    4. Validate time cutoff
    5. Profile generated features

    Args:
        state: Project state with transaction_data_path, main_data_path, etc.
        output_dir: Optional output directory.

    Returns:
        dict with daily_features_path, window_features_path, schema, profile, validation.
    """
    project_name = state["project_name"]
    txn_path = state.get("transaction_data_path")

    if not txn_path:
        write_agent_trace(
            project_name=project_name,
            agent_name=AGENT_NAME,
            reasoning_summary="没有交易流水数据",
            action="skip",
            observation_summary="transaction_data_path 为空",
            decision="跳过交易特征工程",
            next_node="RiskGuardAgent",
            status="skipped",
        )
        return {"skipped": True}

    if output_dir is None:
        output_dir = os.path.join("artifacts", project_name)
    os.makedirs(output_dir, exist_ok=True)

    txn_df = read_dataframe(txn_path)

    # Step 1: Infer schema
    _check_permission("infer_transaction_schema")
    detected_roles = state.get("transaction_schema", {})
    schema = infer_transaction_schema(txn_df, detected_roles=detected_roles)

    account_col = schema["account_col"]
    time_col = schema["time_col"]
    amount_col = schema["amount_col"]
    direction_col = schema["direction_col"]

    if not account_col or not time_col or not amount_col:
        write_agent_trace(
            project_name=project_name,
            agent_name=AGENT_NAME,
            reasoning_summary="无法推断交易表关键列",
            action="infer_transaction_schema",
            action_input_summary=schema,
            observation_summary="缺少 account_col/time_col/amount_col",
            decision="需要人工指定交易表 schema",
            next_node="HumanReviewGate_TransactionSchema",
            status="need_human_review",
        )
        return {"error": "incomplete_schema", "schema": schema}

    # Step 2: Build daily features
    _check_permission("build_account_daily_features")
    daily_df = build_account_daily_features(
        txn_df, account_col, time_col, amount_col, direction_col
    )
    daily_path = os.path.join(output_dir, "transaction_daily_features.csv")
    daily_df.to_csv(daily_path, index=False)

    # Step 3: Build window features (if main table available)
    window_path = None
    validation_result = None
    main_path = state.get("main_data_path")

    if main_path:
        main_df = read_dataframe(main_path)
        base_time_col = state.get("base_time_col") or state.get("time_col")
        id_col = state.get("id_col") or account_col

        if base_time_col and base_time_col in main_df.columns:
            # Use id_col as account_col in main_df for join
            main_for_window = main_df.copy()
            if id_col != account_col and id_col in main_for_window.columns:
                main_for_window = main_for_window.rename(columns={id_col: account_col})

            _check_permission("build_account_window_features")
            window_df = build_account_window_features(
                daily_features=daily_df,
                main_df=main_for_window,
                account_col=account_col,
                base_time_col=base_time_col,
            )
            window_path = os.path.join(output_dir, "transaction_window_features.csv")
            window_df.to_csv(window_path, index=False)

            # Step 4: Validate cutoff
            _check_permission("validate_transaction_feature_cutoff")
            validation_result = validate_transaction_feature_cutoff(
                daily_features=daily_df,
                main_df=main_for_window,
                account_col=account_col,
                base_time_col=base_time_col,
            )

    # Step 5: Profile features
    _check_permission("profile_generated_features")
    profile_df = daily_df
    if window_path:
        window_df_loaded = read_dataframe(window_path)
        profile_df = window_df_loaded

    profile = profile_generated_features(profile_df)

    write_agent_trace(
        project_name=project_name,
        agent_name=AGENT_NAME,
        reasoning_summary="完成交易流水特征工程",
        action="build_account_window_features",
        action_input_summary={
            "account_col": account_col,
            "time_col": time_col,
            "amount_col": amount_col,
            "daily_features_shape": list(daily_df.shape),
            "window_features_generated": window_path is not None,
        },
        observation_summary=(
            f"日维度特征: {daily_df.shape}, "
            f"窗口特征: {'已生成' if window_path else '未生成（无主表）'}"
        ),
        decision="交易特征工程完成",
        next_node="RiskGuardAgent",
        status="completed",
    )

    return {
        "schema": schema,
        "daily_features_path": daily_path,
        "daily_features_shape": list(daily_df.shape),
        "window_features_path": window_path,
        "validation_result": validation_result,
        "feature_profile": profile,
    }
