"""DataQualityAgent — orchestrates data quality analysis and generates reports."""

from core.permissions import check_tool_permission
from core.exceptions import PermissionDeniedError
from core.state import RiskModelingProjectState
from tools.file_tools import read_dataframe
from tools.quality_tools import (
    analyze_duplicates,
    analyze_missing_values,
    analyze_outliers,
    analyze_type_mismatch,
    analyze_label_quality,
    analyze_key_quality,
    generate_data_quality_report,
)
from tools.trace_tools import write_agent_trace

AGENT_NAME = "DataQualityAgent"


def _check_permission(tool_name: str) -> None:
    if not check_tool_permission(AGENT_NAME, tool_name):
        raise PermissionDeniedError(AGENT_NAME, tool_name)


def run(state: RiskModelingProjectState) -> dict:
    """Run full data quality analysis on the main data file.

    Expects state to contain:
        - project_name
        - main_data_path (or first uploaded file)
        - label_col, id_col (optional)

    Returns:
        DataQualityReport-like dict.
    """
    project_name = state["project_name"]

    # Determine which file to analyze
    data_path = state.get("main_data_path")
    if not data_path:
        uploaded_files = state.get("uploaded_files", [])
        if uploaded_files:
            data_path = uploaded_files[0].get("file_path")

    if not data_path:
        write_agent_trace(
            project_name=project_name,
            agent_name=AGENT_NAME,
            reasoning_summary="没有可分析的数据文件",
            action="skip",
            observation_summary="main_data_path 和 uploaded_files 均为空",
            decision="无法进行数据质量分析",
            next_node="error_handling",
            status="error",
        )
        return {"error": "no_data_file"}

    df = read_dataframe(data_path)

    label_col = state.get("label_col")
    id_col = state.get("id_col")
    key_columns = []
    if id_col:
        key_columns.append(id_col)

    protected_columns = []
    if label_col:
        protected_columns.append(label_col)
    if id_col:
        protected_columns.append(id_col)

    # Step 1: Duplicates
    _check_permission("analyze_duplicates")
    dup_result = analyze_duplicates(df, key_columns=key_columns or None)

    # Step 2: Missing values
    _check_permission("analyze_missing_values")
    missing_result = analyze_missing_values(df)

    # Step 3: Outliers
    _check_permission("analyze_outliers")
    outlier_result = analyze_outliers(df, protected_columns=protected_columns)

    # Step 4: Type mismatch
    _check_permission("analyze_type_mismatch")
    type_result = analyze_type_mismatch(df)

    # Step 5: Label quality
    _check_permission("analyze_label_quality")
    label_result = {}
    if label_col:
        label_result = analyze_label_quality(df, label_col)

    # Step 6: Key quality
    _check_permission("analyze_key_quality")
    key_result = {}
    key_dict = {}
    if id_col:
        key_dict["id_col"] = id_col
    if key_dict:
        key_result = analyze_key_quality(df, key_dict)

    # Step 7: Generate report
    _check_permission("generate_data_quality_report")
    report = generate_data_quality_report(
        duplicate_analysis=dup_result,
        missing_analysis=missing_result,
        outlier_analysis=outlier_result,
        type_analysis=type_result,
        label_analysis=label_result,
        key_analysis=key_result,
    )

    write_agent_trace(
        project_name=project_name,
        agent_name=AGENT_NAME,
        reasoning_summary="对主数据文件进行全面质量分析",
        action="generate_data_quality_report",
        action_input_summary={
            "data_path": data_path,
            "rows": len(df),
            "cols": len(df.columns),
        },
        observation_summary=(
            f"质量评分={report['overall_quality_score']}, "
            f"重复行={dup_result['duplicate_rows']}, "
            f"高缺失列={len(missing_result['high_missing_columns'])}, "
            f"需清洗确认={report['need_cleaning_review']}"
        ),
        decision="数据质量分析完成，进入人工确认",
        next_node="HumanReviewGate_DataQuality",
        status="completed",
    )

    return report
