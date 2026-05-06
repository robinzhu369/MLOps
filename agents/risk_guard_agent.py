"""RiskGuardAgent — pre-modeling risk field detection and exclusion."""

from core.permissions import check_tool_permission
from core.exceptions import PermissionDeniedError
from core.state import RiskModelingProjectState
from tools.file_tools import read_dataframe
from tools.risk_guard_tools import (
    detect_leakage_columns,
    detect_id_columns,
    detect_high_missing_columns,
    detect_time_leakage_candidates,
)
from tools.trace_tools import write_agent_trace

AGENT_NAME = "RiskGuardAgent"


def _check_permission(tool_name: str) -> None:
    if not check_tool_permission(AGENT_NAME, tool_name):
        raise PermissionDeniedError(AGENT_NAME, tool_name)


def run(state: RiskModelingProjectState) -> dict:
    """Run risk guard checks on the modeling-ready dataset.

    Expects state to contain:
        - project_name
        - cleaned_data_path or main_data_path
        - label_col, id_col (optional)
        - base_time_col (optional)

    Returns:
        dict with drop_recommendations, warnings, safe_columns.
    """
    project_name = state["project_name"]

    data_path = state.get("cleaned_data_path") or state.get("main_data_path")
    if not data_path:
        uploaded = state.get("uploaded_files", [])
        if uploaded:
            data_path = uploaded[0].get("file_path")

    if not data_path:
        write_agent_trace(
            project_name=project_name,
            agent_name=AGENT_NAME,
            reasoning_summary="没有可检查的数据文件",
            action="skip",
            observation_summary="cleaned_data_path 和 main_data_path 均为空",
            decision="无法进行风险字段检查",
            next_node="error_handling",
            status="error",
        )
        return {"error": "no_data_file"}

    df = read_dataframe(data_path)

    label_col = state.get("label_col")
    id_col = state.get("id_col")
    base_time_col = state.get("base_time_col")

    protected_columns = []
    if label_col:
        protected_columns.append(label_col)
    if id_col:
        protected_columns.append(id_col)
    if base_time_col:
        protected_columns.append(base_time_col)

    # Step 1: Leakage detection
    _check_permission("detect_leakage_columns")
    leakage_result = detect_leakage_columns(df, label_col=label_col)

    # Step 2: ID column detection
    _check_permission("detect_id_columns")
    id_result = detect_id_columns(df, protected_columns=protected_columns)

    # Step 3: High missing columns
    _check_permission("detect_high_missing_columns")
    missing_result = detect_high_missing_columns(df, protected_columns=protected_columns)

    # Step 4: Time leakage candidates
    _check_permission("detect_time_leakage_candidates")
    time_result = detect_time_leakage_candidates(
        df, observation_time_col=base_time_col, protected_columns=protected_columns
    )

    # Compile drop recommendations
    drop_recommendations = []
    warnings = []

    for item in leakage_result["leakage_columns"]:
        drop_recommendations.append({
            "column": item["column"],
            "reason": item["reason"],
            "risk_level": item["risk_level"],
            "category": "leakage",
        })

    for item in id_result["id_columns"]:
        drop_recommendations.append({
            "column": item["column"],
            "reason": item["reason"],
            "risk_level": item["risk_level"],
            "category": "id_field",
        })

    for item in missing_result["high_missing_columns"]:
        drop_recommendations.append({
            "column": item["column"],
            "reason": item["reason"],
            "risk_level": item["risk_level"],
            "category": "high_missing",
        })

    for item in time_result["time_leakage_candidates"]:
        drop_recommendations.append({
            "column": item["column"],
            "reason": item["reason"],
            "risk_level": item["risk_level"],
            "category": "time_leakage",
        })

    # Determine safe columns
    drop_cols = set(r["column"] for r in drop_recommendations)
    safe_columns = [c for c in df.columns if c not in drop_cols and c not in protected_columns]

    # Generate warnings
    if leakage_result["n_detected"] > 0:
        warnings.append(f"检测到 {leakage_result['n_detected']} 个疑似标签泄露字段")
    if time_result["n_detected"] > 0:
        warnings.append(f"检测到 {time_result['n_detected']} 个疑似时间穿越字段")
    if id_result["n_detected"] > 0:
        warnings.append(f"检测到 {id_result['n_detected']} 个 ID 类字段")

    need_human_review = any(r["risk_level"] == "high" for r in drop_recommendations)

    write_agent_trace(
        project_name=project_name,
        agent_name=AGENT_NAME,
        reasoning_summary="建模前风险字段检查",
        action="risk_guard_check",
        action_input_summary={
            "data_path": data_path,
            "n_columns": len(df.columns),
            "label_col": label_col,
        },
        observation_summary=(
            f"建议删除 {len(drop_recommendations)} 个字段, "
            f"安全字段 {len(safe_columns)} 个, "
            f"需人工确认={need_human_review}"
        ),
        decision="风险字段检查完成",
        next_node="HumanReviewGate_RiskGuard" if need_human_review else "ModelingAgent",
        status="completed",
    )

    return {
        "drop_recommendations": drop_recommendations,
        "warnings": warnings,
        "safe_columns": safe_columns,
        "n_safe_columns": len(safe_columns),
        "n_drop_recommended": len(drop_recommendations),
        "need_human_review": need_human_review,
        "details": {
            "leakage": leakage_result,
            "id_fields": id_result,
            "high_missing": missing_result,
            "time_leakage": time_result,
        },
    }
