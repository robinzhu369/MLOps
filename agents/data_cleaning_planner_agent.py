"""DataCleaningPlannerAgent — generates a cleaning plan based on data quality report."""

from core.constants import DEFAULT_HIGH_MISSING_THRESHOLD
from core.permissions import check_tool_permission
from core.exceptions import PermissionDeniedError
from core.state import RiskModelingProjectState
from tools.trace_tools import write_agent_trace

AGENT_NAME = "DataCleaningPlannerAgent"


def _check_permission(tool_name: str) -> None:
    if not check_tool_permission(AGENT_NAME, tool_name):
        raise PermissionDeniedError(AGENT_NAME, tool_name)


def generate_cleaning_plan(
    quality_report: dict,
    label_col: str | None = None,
    id_col: str | None = None,
    time_col: str | None = None,
) -> dict:
    """Generate a cleaning plan based on the data quality report.

    Args:
        quality_report: Output from DataQualityAgent.
        label_col: Label column name (protected).
        id_col: ID column name (protected).
        time_col: Time column name (protected).

    Returns:
        dict with cleaning_steps list, protected_columns, and summary.
    """
    protected_columns = []
    if label_col:
        protected_columns.append(label_col)
    if id_col:
        protected_columns.append(id_col)
    if time_col:
        protected_columns.append(time_col)

    cleaning_steps = []

    # 1. Duplicate handling
    dup_analysis = quality_report.get("duplicate_analysis", {})
    if dup_analysis.get("duplicate_rows", 0) > 0:
        cleaning_steps.append({
            "action": "drop_exact_duplicates",
            "priority": 1,
            "reason": f"发现 {dup_analysis['duplicate_rows']} 行完全重复",
            "params": {},
        })

    if dup_analysis.get("duplicate_key_rows", 0) > 0 and id_col:
        cleaning_steps.append({
            "action": "drop_key_duplicates",
            "priority": 2,
            "reason": f"主键 {id_col} 存在 {dup_analysis['duplicate_key_rows']} 行重复",
            "params": {"key_columns": [id_col], "keep": "first"},
        })

    # 2. High missing columns — drop
    missing_analysis = quality_report.get("missing_analysis", {})
    high_missing_cols = missing_analysis.get("high_missing_columns", [])
    cols_to_drop = [c for c in high_missing_cols if c not in protected_columns]
    if cols_to_drop:
        cleaning_steps.append({
            "action": "drop_high_missing_columns",
            "priority": 3,
            "reason": f"以下列缺失率超过阈值: {cols_to_drop}",
            "params": {"columns": cols_to_drop},
        })

    # 3. Missing value imputation for remaining columns
    columns_info = missing_analysis.get("columns", [])
    fill_numeric = []
    fill_categorical = []
    add_indicator = []

    for col_info in columns_info:
        col = col_info["column"]
        rate = col_info["missing_rate"]
        if rate == 0 or col in protected_columns or col in cols_to_drop:
            continue

        if rate > 0.1:
            add_indicator.append(col)

        # Determine fill strategy based on column type (heuristic from rate)
        if rate < DEFAULT_HIGH_MISSING_THRESHOLD:
            fill_numeric.append(col)  # Will be filtered at execution time

    if add_indicator:
        cleaning_steps.append({
            "action": "add_missing_indicator",
            "priority": 4,
            "reason": f"为缺失率>10%的列添加缺失指示变量: {add_indicator}",
            "params": {"columns": add_indicator},
        })

    if fill_numeric:
        cleaning_steps.append({
            "action": "fill_missing_values",
            "priority": 5,
            "reason": "对缺失列进行填充（数值列用中位数，分类列用众数）",
            "params": {"columns": fill_numeric, "numeric_strategy": "median", "categorical_strategy": "mode"},
        })

    # 4. Outlier treatment
    outlier_analysis = quality_report.get("outlier_analysis", {})
    outlier_cols = outlier_analysis.get("columns", [])
    winsorize_cols = []
    for col_info in outlier_cols:
        col = col_info["column"]
        if col in protected_columns:
            continue
        if col_info.get("outlier_rate", 0) > 0.01:
            winsorize_cols.append({
                "column": col,
                "lower_bound": col_info.get("lower_bound"),
                "upper_bound": col_info.get("upper_bound"),
            })

    if winsorize_cols:
        cleaning_steps.append({
            "action": "winsorize_outliers",
            "priority": 6,
            "reason": f"对 {len(winsorize_cols)} 个列进行异常值缩尾处理",
            "params": {"columns": winsorize_cols},
        })

    # 5. Type conversion
    type_analysis = quality_report.get("type_analysis", {})
    mismatches = type_analysis.get("mismatches", [])
    if mismatches:
        cleaning_steps.append({
            "action": "convert_types",
            "priority": 7,
            "reason": f"修正 {len(mismatches)} 个列的数据类型",
            "params": {"conversions": mismatches},
        })

    # 6. Drop constant columns
    cleaning_steps.append({
        "action": "drop_constant_columns",
        "priority": 8,
        "reason": "删除常数列（唯一值=1）",
        "params": {"protected_columns": protected_columns},
    })

    # Sort by priority
    cleaning_steps.sort(key=lambda x: x["priority"])

    return {
        "cleaning_steps": cleaning_steps,
        "protected_columns": protected_columns,
        "n_steps": len(cleaning_steps),
        "summary": f"共 {len(cleaning_steps)} 个清洗步骤",
    }


def run(state: RiskModelingProjectState) -> dict:
    """Generate cleaning plan from project state.

    Expects state to contain:
        - project_name
        - data_quality_report
        - label_col, id_col (optional)

    Returns:
        Cleaning plan dict.
    """
    project_name = state["project_name"]
    quality_report = state.get("data_quality_report", {})

    if not quality_report:
        write_agent_trace(
            project_name=project_name,
            agent_name=AGENT_NAME,
            reasoning_summary="没有数据质量报告",
            action="skip",
            observation_summary="data_quality_report 为空",
            decision="无法生成清洗方案",
            next_node="error_handling",
            status="error",
        )
        return {"error": "no_quality_report"}

    label_col = state.get("label_col")
    id_col = state.get("id_col")
    time_col = state.get("time_col")

    _check_permission("generate_cleaning_plan")
    plan = generate_cleaning_plan(
        quality_report=quality_report,
        label_col=label_col,
        id_col=id_col,
        time_col=time_col,
    )

    write_agent_trace(
        project_name=project_name,
        agent_name=AGENT_NAME,
        reasoning_summary="根据数据质量报告生成清洗方案",
        action="generate_cleaning_plan",
        action_input_summary={
            "quality_score": quality_report.get("overall_quality_score"),
            "protected_columns": plan["protected_columns"],
        },
        observation_summary=f"生成 {plan['n_steps']} 个清洗步骤",
        decision="清洗方案生成完成，等待人工确认后执行",
        next_node="HumanReviewGate_CleaningPlan",
        status="completed",
    )

    return plan
