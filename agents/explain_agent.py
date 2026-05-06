"""ExplainAgent — computes and outputs feature importance."""

import os

import pandas as pd

from core.permissions import check_tool_permission
from core.exceptions import PermissionDeniedError
from core.state import RiskModelingProjectState
from tools.file_tools import read_dataframe
from tools.explain_tools import compute_feature_importance
from tools.trace_tools import write_agent_trace

AGENT_NAME = "ExplainAgent"


def _check_permission(tool_name: str) -> None:
    if not check_tool_permission(AGENT_NAME, tool_name):
        raise PermissionDeniedError(AGENT_NAME, tool_name)


def run(
    state: RiskModelingProjectState,
    output_dir: str | None = None,
) -> dict:
    """Compute feature importance.

    Returns:
        dict with feature_importance_path and top features.
    """
    project_name = state["project_name"]

    if output_dir is None:
        output_dir = os.path.join("artifacts", project_name)
    os.makedirs(output_dir, exist_ok=True)

    model_path = state.get("model_path")
    feature_columns = state.get("feature_columns", [])

    if not model_path or not feature_columns:
        return {"error": "no_model_or_features"}

    # Load test data if available
    test_df = None
    label_col = state.get("label_col")
    test_path = state.get("test_path")
    if test_path:
        test_df = read_dataframe(test_path)

    _check_permission("compute_feature_importance")
    importance_df = compute_feature_importance(
        model_path=model_path,
        feature_columns=feature_columns,
        test_df=test_df,
        label_col=label_col,
    )

    importance_path = os.path.join(output_dir, "feature_importance.csv")
    importance_df.to_csv(importance_path, index=False)

    write_agent_trace(
        project_name=project_name,
        agent_name=AGENT_NAME,
        reasoning_summary="计算特征重要性",
        action="compute_feature_importance",
        action_input_summary={"n_features": len(feature_columns)},
        observation_summary=f"Top 5: {importance_df.head(5)['feature_name'].tolist()}",
        decision="特征重要性计算完成",
        next_node="ReportAgent",
        status="completed",
    )

    return {
        "feature_importance_path": importance_path,
        "top_features": importance_df.head(20).to_dict(orient="records"),
        "n_features": len(importance_df),
    }
