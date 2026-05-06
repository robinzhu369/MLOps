"""DataDictionaryParserAgent — parses uploaded data dictionary to extract field semantics."""

from core.permissions import check_tool_permission
from core.exceptions import PermissionDeniedError
from core.state import RiskModelingProjectState
from tools.dictionary_tools import (
    parse_data_dictionary,
    validate_dictionary_columns,
    map_dictionary_to_dataset,
)
from tools.trace_tools import write_agent_trace

AGENT_NAME = "DataDictionaryParserAgent"


def _check_permission(tool_name: str) -> None:
    if not check_tool_permission(AGENT_NAME, tool_name):
        raise PermissionDeniedError(AGENT_NAME, tool_name)


def run(state: RiskModelingProjectState) -> dict:
    """Parse the uploaded data dictionary and produce field semantics.

    Expects state to contain:
        - project_name
        - data_dictionary_path
        - uploaded_files (to map dictionary to dataset columns)

    Returns:
        FieldSemanticResult-like dict with dictionary_uploaded, parsed_columns, warnings.
    """
    project_name = state["project_name"]
    dictionary_path = state.get("data_dictionary_path")

    if not dictionary_path:
        write_agent_trace(
            project_name=project_name,
            agent_name=AGENT_NAME,
            reasoning_summary="未提供数据字典路径",
            action="skip",
            observation_summary="数据字典路径为空",
            decision="跳过数据字典解析，转入 LLM 字段语义解析",
            next_node="FieldSemanticParserAgent",
            status="skipped",
        )
        return {"dictionary_uploaded": False, "parsed_columns": [], "warnings": []}

    # Step 1: Validate dictionary structure
    _check_permission("validate_dictionary_columns")
    validation = validate_dictionary_columns(dictionary_path)

    if not validation["is_valid"]:
        warning_msg = f"数据字典缺少必要列: {validation['missing_required']}"
        write_agent_trace(
            project_name=project_name,
            agent_name=AGENT_NAME,
            reasoning_summary="验证数据字典结构",
            action="validate_dictionary_columns",
            action_input_summary={"path": dictionary_path},
            observation_summary=warning_msg,
            decision="数据字典格式不合规，需要用户修正或转入 LLM 解析",
            next_node="HumanReviewGate_FieldSemantics",
            status="validation_failed",
        )
        return {
            "dictionary_uploaded": True,
            "parsed_columns": [],
            "warnings": [warning_msg],
        }

    # Step 2: Parse dictionary
    _check_permission("parse_data_dictionary")
    parse_result = parse_data_dictionary(dictionary_path)

    # Step 3: Map to dataset columns
    _check_permission("map_dictionary_to_dataset")
    uploaded_files = state.get("uploaded_files", [])
    all_dataset_columns = []
    for f in uploaded_files:
        all_dataset_columns.extend(f.get("columns", []))

    mapping = map_dictionary_to_dataset(
        parse_result["parsed_columns"],
        list(set(all_dataset_columns)),
    )

    # Add mapping warnings
    warnings = list(parse_result["warnings"])
    if mapping["unmatched_in_dataset"]:
        warnings.append(
            f"以下数据集字段未在数据字典中找到: {mapping['unmatched_in_dataset']}"
        )
    if mapping["unmatched_in_dictionary"]:
        warnings.append(
            f"以下数据字典字段未在数据集中找到: {mapping['unmatched_in_dictionary']}"
        )

    result = {
        "dictionary_uploaded": True,
        "parsed_columns": parse_result["parsed_columns"],
        "warnings": warnings,
        "coverage_rate": mapping["coverage_rate"],
        "need_human_review": mapping["coverage_rate"] < 1.0 or len(warnings) > 0,
    }

    write_agent_trace(
        project_name=project_name,
        agent_name=AGENT_NAME,
        reasoning_summary="解析数据字典并映射到数据集字段",
        action="parse_data_dictionary",
        action_input_summary={
            "dictionary_path": dictionary_path,
            "total_entries": len(parse_result["parsed_columns"]),
        },
        observation_summary=(
            f"解析 {len(parse_result['parsed_columns'])} 个字段定义，"
            f"覆盖率 {mapping['coverage_rate']:.1%}"
        ),
        decision="字段语义解析完成，进入人工确认",
        next_node="HumanReviewGate_FieldSemantics",
        status="completed",
    )

    return result
