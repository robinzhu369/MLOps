"""EvaluationAgent — computes model evaluation metrics."""

import os
import json

import numpy as np

from core.permissions import check_tool_permission
from core.exceptions import PermissionDeniedError
from core.state import RiskModelingProjectState
from tools.file_tools import read_dataframe
from tools.metric_tools import evaluate_binary_model
from tools.trace_tools import write_agent_trace

AGENT_NAME = "EvaluationAgent"


def _check_permission(tool_name: str) -> None:
    if not check_tool_permission(AGENT_NAME, tool_name):
        raise PermissionDeniedError(AGENT_NAME, tool_name)


def run(
    state: RiskModelingProjectState,
    y_true: np.ndarray | None = None,
    y_proba: np.ndarray | None = None,
    output_dir: str | None = None,
) -> dict:
    """Run model evaluation.

    Args:
        state: Project state.
        y_true: True labels (if not provided, loads from test.csv).
        y_proba: Predicted probabilities (if not provided, uses state predictions).
        output_dir: Output directory for metrics.json.

    Returns:
        Evaluation metrics dict.
    """
    project_name = state["project_name"]
    label_col = state.get("label_col")

    if output_dir is None:
        output_dir = os.path.join("artifacts", project_name)
    os.makedirs(output_dir, exist_ok=True)

    # Get y_true
    if y_true is None:
        test_path = state.get("test_path")
        if test_path:
            test_df = read_dataframe(test_path)
            y_true = test_df[label_col].values
        else:
            return {"error": "no_test_data"}

    # Get y_proba
    if y_proba is None:
        predictions = state.get("predictions")
        if predictions is not None:
            y_proba = np.asarray(predictions)
        else:
            return {"error": "no_predictions"}

    _check_permission("evaluate_binary_model")
    metrics = evaluate_binary_model(y_true, y_proba)

    # Save metrics
    metrics_path = os.path.join(output_dir, "metrics.json")
    serializable_metrics = {k: v for k, v in metrics.items()}
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(serializable_metrics, f, ensure_ascii=False, indent=2)

    write_agent_trace(
        project_name=project_name,
        agent_name=AGENT_NAME,
        reasoning_summary="计算模型评估指标",
        action="evaluate_binary_model",
        action_input_summary={"n_samples": len(y_true), "threshold": 0.5},
        observation_summary=f"AUC={metrics['auc']}, KS={metrics['ks']}",
        decision="评估完成",
        next_node="StrategyAgent",
        status="completed",
    )

    metrics["metrics_path"] = metrics_path
    return metrics
