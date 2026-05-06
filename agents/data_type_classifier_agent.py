"""DataTypeClassifierAgent — classifies uploaded files into data types and recommends pipelines."""

from typing import Callable

from core.permissions import check_tool_permission
from core.exceptions import PermissionDeniedError
from core.state import RiskModelingProjectState
from tools.data_type_tools import (
    classify_data_type_by_rules,
    classify_data_type_by_llm,
    merge_data_type_classification,
)
from tools.trace_tools import write_agent_trace

AGENT_NAME = "DataTypeClassifierAgent"


def _check_permission(tool_name: str) -> None:
    if not check_tool_permission(AGENT_NAME, tool_name):
        raise PermissionDeniedError(AGENT_NAME, tool_name)


def run(
    state: RiskModelingProjectState,
    llm_call: Callable[[str], str] | None = None,
) -> dict:
    """Classify all uploaded files and recommend pipelines.

    Args:
        state: Project state with uploaded_files and field_semantics populated.
        llm_call: Optional LLM callable for enhanced classification.

    Returns:
        dict with classifications list (one per file) and overall pipeline recommendation.
    """
    project_name = state["project_name"]
    uploaded_files = state.get("uploaded_files", [])
    field_semantics = state.get("field_semantics", {})

    if not uploaded_files:
        write_agent_trace(
            project_name=project_name,
            agent_name=AGENT_NAME,
            reasoning_summary="没有已上传文件可供分类",
            action="skip",
            observation_summary="uploaded_files 为空",
            decision="无法进行数据类型分类",
            next_node="error_handling",
            status="error",
        )
        return {"classifications": [], "need_human_review": False}

    classifications = []

    for file_state in uploaded_files:
        file_name = file_state.get("file_name", "")
        columns = file_state.get("columns", [])
        column_profiles = file_state.get("column_profiles", {})
        n_rows = file_state.get("n_rows", 0)
        n_cols = file_state.get("n_cols", 0)

        # Filter field_semantics for this file
        file_semantics = {}
        for key, info in field_semantics.items():
            if key.startswith(f"{file_name}:"):
                col_name = key.split(":", 1)[1]
                file_semantics[col_name] = info
            elif key in columns:
                file_semantics[key] = info

        # Step 1: Rule-based classification
        _check_permission("classify_data_type_by_rules")
        rule_result = classify_data_type_by_rules(
            file_name=file_name,
            columns=columns,
            column_profiles=column_profiles,
            field_semantics=file_semantics,
        )

        # Step 2: LLM-based classification (if available)
        _check_permission("classify_data_type_by_llm")
        llm_result = classify_data_type_by_llm(
            file_name=file_name,
            n_rows=n_rows,
            n_cols=n_cols,
            columns=columns,
            field_semantics=file_semantics,
            llm_call=llm_call,
        )

        # Step 3: Merge results
        _check_permission("merge_data_type_classification")
        merged = merge_data_type_classification(rule_result, llm_result)

        classifications.append(merged)

    # Determine overall need for human review
    need_human_review = any(c.get("need_human_review", False) for c in classifications)

    write_agent_trace(
        project_name=project_name,
        agent_name=AGENT_NAME,
        reasoning_summary="对所有上传文件进行数据类型分类",
        action="merge_data_type_classification",
        action_input_summary={
            "files": [c["file_name"] for c in classifications],
            "types": [c["detected_data_type"] for c in classifications],
        },
        observation_summary=(
            f"分类完成: "
            + ", ".join(f"{c['file_name']}={c['detected_data_type']}" for c in classifications)
        ),
        decision="数据类型分类完成，进入人工确认",
        next_node="HumanReviewGate_DataType",
        status="completed",
    )

    return {
        "classifications": classifications,
        "need_human_review": need_human_review,
    }
