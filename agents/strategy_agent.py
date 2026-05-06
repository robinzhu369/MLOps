"""StrategyAgent — generates threshold strategy table."""

import os

import numpy as np
import pandas as pd

from core.permissions import check_tool_permission
from core.exceptions import PermissionDeniedError
from core.state import RiskModelingProjectState
from tools.file_tools import read_dataframe
from tools.strategy_tools import build_threshold_table
from tools.trace_tools import write_agent_trace

AGENT_NAME = "StrategyAgent"


def _check_permission(tool_name: str) -> None:
    if not check_tool_permission(AGENT_NAME, tool_name):
        raise PermissionDeniedError(AGENT_NAME, tool_name)


def run(
    state: RiskModelingProjectState,
    y_true: np.ndarray | None = None,
    y_proba: np.ndarray | None = None,
    output_dir: str | None = None,
) -> dict:
    """Generate threshold strategy table.

    Returns:
        dict with threshold_table_path and table data.
    """
    project_name = state["project_name"]
    label_col = state.get("label_col")

    if output_dir is None:
        output_dir = os.path.join("artifacts", project_name)
    os.makedirs(output_dir, exist_ok=True)

    if y_true is None:
        test_path = state.get("test_path")
        if test_path:
            test_df = read_dataframe(test_path)
            y_true = test_df[label_col].values
        else:
            return {"error": "no_test_data"}

    if y_proba is None:
        predictions = state.get("predictions")
        if predictions is not None:
            y_proba = np.asarray(predictions)
        else:
            return {"error": "no_predictions"}

    _check_permission("build_threshold_table")
    table = build_threshold_table(y_true, y_proba)

    table_path = os.path.join(output_dir, "threshold_table.csv")
    table.to_csv(table_path, index=False)

    write_agent_trace(
        project_name=project_name,
        agent_name=AGENT_NAME,
        reasoning_summary="生成阈值策略分析表",
        action="build_threshold_table",
        action_input_summary={"n_samples": len(y_true), "n_thresholds": len(table)},
        observation_summary=f"生成 {len(table)} 行阈值策略表",
        decision="策略分析完成",
        next_node="ExplainAgent",
        status="completed",
    )

    return {
        "threshold_table_path": table_path,
        "threshold_table": table.to_dict(orient="records"),
        "n_thresholds": len(table),
    }
