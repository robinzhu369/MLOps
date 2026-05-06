"""ReportAgent — generates Markdown model evaluation report."""

import os

from core.permissions import check_tool_permission
from core.exceptions import PermissionDeniedError
from core.state import RiskModelingProjectState
from tools.report_tools import generate_markdown_report
from tools.trace_tools import write_agent_trace

AGENT_NAME = "ReportAgent"


def _check_permission(tool_name: str) -> None:
    if not check_tool_permission(AGENT_NAME, tool_name):
        raise PermissionDeniedError(AGENT_NAME, tool_name)


def run(
    state: RiskModelingProjectState,
    output_dir: str | None = None,
) -> dict:
    """Generate a comprehensive model evaluation report.

    Collects data from state (quality, cleaning, metrics, strategy,
    feature importance, risk warnings) and produces a Markdown report.

    Returns:
        dict with report_path and report content summary.
    """
    project_name = state["project_name"]

    if output_dir is None:
        output_dir = os.path.join("artifacts", project_name, "reports")
    os.makedirs(output_dir, exist_ok=True)

    # Gather data summary
    data_summary = _build_data_summary(state)

    # Gather quality summary
    quality_summary = state.get("data_quality_report")

    # Gather cleaning summary
    cleaning_summary = _build_cleaning_summary(state)

    # Gather model metrics
    model_metrics = state.get("metrics")

    # Gather threshold table
    threshold_table = state.get("threshold_table")

    # Gather feature importance
    feature_importance = state.get("feature_importance")

    # Gather risk warnings
    risk_warnings = state.get("warnings", [])

    # Gather recommendations
    recommendations = state.get("recommendations")

    _check_permission("generate_markdown_report")
    report_md = generate_markdown_report(
        project_name=project_name,
        data_summary=data_summary,
        quality_summary=quality_summary,
        cleaning_summary=cleaning_summary,
        model_metrics=model_metrics,
        threshold_table=threshold_table,
        feature_importance=feature_importance,
        risk_warnings=risk_warnings,
        recommendations=recommendations,
    )

    report_path = os.path.join(output_dir, "model_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    write_agent_trace(
        project_name=project_name,
        agent_name=AGENT_NAME,
        reasoning_summary="汇总各阶段结果生成模型评估报告",
        action="generate_markdown_report",
        action_input_summary={
            "has_metrics": model_metrics is not None,
            "has_threshold_table": threshold_table is not None,
            "has_feature_importance": feature_importance is not None,
            "n_warnings": len(risk_warnings),
        },
        observation_summary=f"报告已生成: {report_path}",
        decision="报告生成完成",
        next_node="END",
        status="completed",
    )

    return {
        "report_path": report_path,
        "report_length": len(report_md),
    }


def _build_data_summary(state: RiskModelingProjectState) -> dict | None:
    """Build data summary from state."""
    uploaded_files = state.get("uploaded_files", [])
    if not uploaded_files:
        return None

    first_file = uploaded_files[0]
    n_rows = first_file.get("n_rows", 0)
    n_cols = first_file.get("n_cols", 0)

    # Positive rate from quality report or metrics
    quality = state.get("data_quality_report") or {}
    label_quality = quality.get("label_quality", {})
    positive_rate = label_quality.get("positive_rate")

    return {
        "n_rows": n_rows,
        "n_features": n_cols,
        "positive_rate": positive_rate,
        "train_size": state.get("train_size"),
        "test_size": state.get("test_size"),
    }


def _build_cleaning_summary(state: RiskModelingProjectState) -> dict | None:
    """Build cleaning summary from state."""
    cleaning_plan = state.get("cleaning_plan")
    if not cleaning_plan:
        return None

    return {
        "before_shape": cleaning_plan.get("before_shape"),
        "after_shape": cleaning_plan.get("after_shape"),
        "n_steps": len(cleaning_plan.get("steps", [])),
        "steps": cleaning_plan.get("steps", []),
    }
