"""FieldSemanticParserAgent — LLM-based field semantic parsing when no dictionary is provided."""

from typing import Any, Callable

from core.permissions import check_tool_permission
from core.exceptions import PermissionDeniedError
from core.state import RiskModelingProjectState
from tools.field_semantic_tools import (
    parse_field_semantics_by_llm,
    merge_dictionary_and_llm_semantics,
)
from tools.trace_tools import write_agent_trace

AGENT_NAME = "FieldSemanticParserAgent"


def _check_permission(tool_name: str) -> None:
    if not check_tool_permission(AGENT_NAME, tool_name):
        raise PermissionDeniedError(AGENT_NAME, tool_name)


def run(
    state: RiskModelingProjectState,
    llm_call: Callable[[str], str] | None = None,
) -> dict:
    """Parse field semantics using LLM for all uploaded files.

    Args:
        state: Current project state with uploaded_files populated.
        llm_call: Callable that takes a prompt and returns LLM response.
                  If None, uses rule-based fallback.

    Returns:
        dict with dictionary_uploaded=False, field_semantics, need_human_review.
    """
    project_name = state["project_name"]
    uploaded_files = state.get("uploaded_files", [])

    if not uploaded_files:
        write_agent_trace(
            project_name=project_name,
            agent_name=AGENT_NAME,
            reasoning_summary="没有已上传的文件可供解析",
            action="skip",
            observation_summary="uploaded_files 为空",
            decision="无法进行字段语义解析",
            next_node="error_handling",
            status="error",
        )
        return {"dictionary_uploaded": False, "field_semantics": {}, "need_human_review": False}

    # Parse each file's fields
    _check_permission("parse_field_semantics_by_llm")
    all_field_semantics = {}

    for file_state in uploaded_files:
        file_name = file_state.get("file_name", "")
        n_rows = file_state.get("n_rows", 0)
        n_cols = file_state.get("n_cols", 0)
        column_profiles = file_state.get("column_profiles", {})
        sample_values_masked = file_state.get("sample_values_masked", {})

        result = parse_field_semantics_by_llm(
            file_name=file_name,
            n_rows=n_rows,
            n_cols=n_cols,
            column_profiles=column_profiles,
            sample_values_masked=sample_values_masked,
            llm_call=llm_call,
        )

        # Prefix with table name for multi-file scenarios
        for col, info in result.get("field_semantics", {}).items():
            info["table_name"] = file_name
            all_field_semantics[f"{file_name}:{col}"] = info

    # Determine if human review is needed
    need_human_review = any(
        info.get("confidence", 0) < 0.8 or info.get("risk_level") == "high"
        for info in all_field_semantics.values()
    )

    final_result = {
        "dictionary_uploaded": False,
        "field_semantics": all_field_semantics,
        "need_human_review": need_human_review,
    }

    write_agent_trace(
        project_name=project_name,
        agent_name=AGENT_NAME,
        reasoning_summary="使用 LLM/规则对所有上传文件进行字段语义解析",
        action="parse_field_semantics_by_llm",
        action_input_summary={
            "files": [f.get("file_name") for f in uploaded_files],
            "total_fields": len(all_field_semantics),
        },
        observation_summary=f"解析 {len(all_field_semantics)} 个字段，需人工确认={need_human_review}",
        decision="字段语义解析完成，进入人工确认",
        next_node="HumanReviewGate_FieldSemantics",
        status="completed",
    )

    return final_result


def run_with_dictionary_merge(
    state: RiskModelingProjectState,
    dictionary_result: dict,
    llm_call: Callable[[str], str] | None = None,
) -> dict:
    """Parse fields with LLM and merge with partial dictionary results.

    Used when dictionary coverage is incomplete.
    """
    project_name = state["project_name"]

    # Get LLM results
    llm_result = run(state, llm_call)

    # Merge
    _check_permission("merge_dictionary_and_llm_semantics")
    merged = merge_dictionary_and_llm_semantics(dictionary_result, llm_result)

    write_agent_trace(
        project_name=project_name,
        agent_name=AGENT_NAME,
        reasoning_summary="合并数据字典和 LLM 字段解析结果",
        action="merge_dictionary_and_llm_semantics",
        action_input_summary={
            "dictionary_fields": len(dictionary_result.get("parsed_columns", [])),
            "llm_fields": len(llm_result.get("field_semantics", {})),
        },
        observation_summary=f"合并后共 {len(merged)} 个字段语义",
        decision="合并完成，进入人工确认",
        next_node="HumanReviewGate_FieldSemantics",
        status="completed",
    )

    return {
        "dictionary_uploaded": True,
        "field_semantics": merged,
        "need_human_review": True,
    }
