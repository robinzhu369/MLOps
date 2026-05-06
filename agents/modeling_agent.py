"""ModelingAgent — orchestrates data preparation and model training."""

import os

import pandas as pd

from core.permissions import check_tool_permission
from core.exceptions import PermissionDeniedError
from core.state import RiskModelingProjectState
from tools.file_tools import read_dataframe
from tools.split_tools import split_train_test
from tools.modeling_tools import train_autogluon_binary
from tools.trace_tools import write_agent_trace

AGENT_NAME = "ModelingAgent"


def _check_permission(tool_name: str) -> None:
    if not check_tool_permission(AGENT_NAME, tool_name):
        raise PermissionDeniedError(AGENT_NAME, tool_name)


def run(
    state: RiskModelingProjectState,
    output_dir: str | None = None,
    time_limit: int = 300,
) -> dict:
    """Run the modeling pipeline.

    Steps:
    1. Load cleaned data
    2. Exclude risk columns
    3. Split train/test
    4. Train model

    Returns:
        dict with model_path, leaderboard_path, train/test shapes, metrics.
    """
    project_name = state["project_name"]

    data_path = state.get("cleaned_data_path") or state.get("main_data_path")
    if not data_path:
        return {"error": "no_data_file"}

    label_col = state.get("label_col")
    if not label_col:
        return {"error": "no_label_col"}

    if output_dir is None:
        output_dir = os.path.join("artifacts", project_name)
    os.makedirs(output_dir, exist_ok=True)

    df = read_dataframe(data_path)

    # Determine columns to exclude
    drop_cols = set(state.get("drop_columns", []))
    id_col = state.get("id_col")
    base_time_col = state.get("base_time_col")
    if id_col:
        drop_cols.add(id_col)
    if base_time_col:
        drop_cols.add(base_time_col)

    # Remove non-feature columns
    excluded = [c for c in drop_cols if c in df.columns]

    # Split
    time_col = base_time_col if base_time_col and base_time_col in df.columns else None
    train_df, test_df = split_train_test(
        df, label_col=label_col, time_col=time_col
    )

    # Save splits
    train_path = os.path.join(output_dir, "train.csv")
    test_path = os.path.join(output_dir, "test.csv")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    # Train
    _check_permission("train_autogluon_binary")
    result = train_autogluon_binary(
        train_df=train_df,
        test_df=test_df,
        label_col=label_col,
        output_dir=output_dir,
        time_limit=time_limit,
        excluded_columns=list(excluded),
    )

    write_agent_trace(
        project_name=project_name,
        agent_name=AGENT_NAME,
        reasoning_summary="执行 AutoML 建模",
        action="train_autogluon_binary",
        action_input_summary={
            "train_shape": list(train_df.shape),
            "test_shape": list(test_df.shape),
            "n_features": len(result.get("feature_columns", [])),
            "time_limit": time_limit,
        },
        observation_summary=(
            f"最佳模型: {result.get('best_model')}, "
            f"测试集 AUC: {result.get('test_score')}"
        ),
        decision="建模完成",
        next_node="EvaluationAgent",
        status="completed",
    )

    return {
        "model_path": result["model_path"],
        "leaderboard_path": result["leaderboard_path"],
        "best_model": result["best_model"],
        "train_score": result["train_score"],
        "test_score": result["test_score"],
        "feature_columns": result["feature_columns"],
        "train_path": train_path,
        "test_path": test_path,
        "train_shape": list(train_df.shape),
        "test_shape": list(test_df.shape),
        "predictions": result["predictions"],
    }
