"""Main + Transaction Pipeline (Scenario C) — LangGraph orchestration.

Pipeline for main table + transaction flow table joint modeling.
Flow: DataIntake → FieldSemantic → DataType → DataQuality → CleaningPlan →
      CleaningExecute → TransactionFeature → FeatureMerge → TimeLeakageCheck →
      RiskGuard → Modeling → Evaluation → Strategy → Explain → Report

Combines transaction feature engineering with structured modeling.
"""

import os

import pandas as pd
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver

from core.constants import PipelineType
from pipelines.state import PipelineState
from pipelines.utils import checkpoint_safe


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------


@checkpoint_safe
def node_data_intake(state: dict) -> dict:
    """Process uploaded files (main table + transaction table).

    Expects _pending_files to be a list of file paths (strings).
    """
    import os
    import agents.data_intake_agent as agent

    file_paths = state.get("_pending_files", [])
    if not file_paths:
        return {"errors": state.get("errors", []) + ["no_pending_files"]}

    uploaded_files = []
    for file_path in file_paths:
        file_name = os.path.basename(file_path)
        with open(file_path, "rb") as f:
            file_state = agent.run(state, f, file_name)
        uploaded_files.append(file_state)

    return {"uploaded_files": uploaded_files}


@checkpoint_safe
def node_field_semantic(state: dict) -> dict:
    """Parse field semantics via LLM or rules."""
    import agents.field_semantic_parser_agent as agent

    llm_call = state.get("_llm_call")
    result = agent.run(state, llm_call=llm_call)
    return {"field_semantics": result.get("field_semantics", {})}


@checkpoint_safe
def node_data_type(state: dict) -> dict:
    """Classify data types and identify main vs transaction tables."""
    import agents.data_type_classifier_agent as agent
    import agents.pipeline_router_agent as router

    llm_call = state.get("_llm_call")
    result = agent.run(state, llm_call=llm_call)

    # Determine file roles
    classifications = result.get("classifications", [])
    route_result = router.route_pipeline(classifications)
    file_roles = route_result.get("file_roles", {})

    # Assign main_data_path and transaction_data_path based on roles
    updates = {"data_type_classification_result": result}

    for c in classifications:
        file_name = c.get("file_name", "")
        detected_type = c.get("detected_data_type", "")
        file_path = None
        # Find file path from uploaded_files
        for uf in state.get("uploaded_files", []):
            if uf.get("file_name") == file_name:
                file_path = uf.get("file_path")
                break

        if file_path:
            if detected_type in ("main_table", "structured_modeling_table"):
                updates["main_data_path"] = file_path
            elif detected_type == "transaction_flow_table":
                updates["transaction_data_path"] = file_path

    return updates


@checkpoint_safe
def node_data_quality(state: dict) -> dict:
    """Run data quality analysis on main table."""
    import agents.data_quality_agent as agent

    report = agent.run(state)
    return {"data_quality_report": report}


@checkpoint_safe
def node_cleaning_plan(state: dict) -> dict:
    """Generate cleaning plan."""
    import agents.data_cleaning_planner_agent as agent

    plan = agent.run(state)
    return {"cleaning_plan": plan}


@checkpoint_safe
def node_cleaning_execute(state: dict) -> dict:
    """Execute cleaning on main table."""
    from tools.data_cleaning_tools import execute_cleaning_plan

    data_path = state.get("main_data_path")
    cleaning_plan = state.get("cleaning_plan", {})
    project_name = state["project_name"]

    if not data_path or not cleaning_plan:
        return {"errors": state.get("errors", []) + ["no_data_or_plan_for_cleaning"]}

    result = execute_cleaning_plan(
        data_path=data_path,
        cleaning_plan=cleaning_plan,
        project_name=project_name,
    )

    return {
        "cleaned_data_path": result.get("output_path"),
        "cleaning_log_path": result.get("cleaning_log_path"),
    }


@checkpoint_safe
def node_transaction_feature(state: dict) -> dict:
    """Build transaction features."""
    import agents.transaction_feature_agent as agent

    result = agent.run(state)
    if "error" in result:
        return {"errors": state.get("errors", []) + [result["error"]]}

    return {
        "transaction_daily_feature_path": result.get("daily_features_path"),
        "transaction_window_feature_path": result.get("window_features_path"),
        "transaction_schema": result.get("schema", {}),
    }


@checkpoint_safe
def node_feature_merge(state: dict) -> dict:
    """Merge transaction features back to main table.

    Joins on account/customer key, producing a modeling-ready dataset.
    """
    from tools.file_tools import read_dataframe

    project_name = state["project_name"]
    main_path = state.get("cleaned_data_path") or state.get("main_data_path")
    window_feature_path = state.get("transaction_window_feature_path")
    daily_feature_path = state.get("transaction_daily_feature_path")

    if not main_path:
        return {"errors": state.get("errors", []) + ["no_main_data_for_merge"]}

    main_df = read_dataframe(main_path)

    # Determine join key
    join_key = (
        state.get("account_col")
        or state.get("customer_col")
        or state.get("id_col")
    )

    if not join_key or join_key not in main_df.columns:
        # No join key available, skip merge
        return {"modeling_data_path": main_path}

    # Merge window features (preferred) or daily features
    feature_path = window_feature_path or daily_feature_path
    if feature_path and os.path.exists(feature_path):
        feature_df = read_dataframe(feature_path)

        # Find matching join key in feature table
        feature_join_key = None
        for candidate in [join_key, "account_id", "customer_id"]:
            if candidate in feature_df.columns:
                feature_join_key = candidate
                break

        if feature_join_key:
            # Avoid duplicate columns
            overlap_cols = set(main_df.columns) & set(feature_df.columns) - {feature_join_key}
            if overlap_cols:
                feature_df = feature_df.drop(columns=list(overlap_cols))

            merged_df = main_df.merge(
                feature_df,
                left_on=join_key,
                right_on=feature_join_key,
                how="left",
            )
        else:
            merged_df = main_df
    else:
        merged_df = main_df

    # Save merged data
    output_dir = os.path.join("artifacts", project_name)
    os.makedirs(output_dir, exist_ok=True)
    merged_path = os.path.join(output_dir, "merged_modeling_data.csv")
    merged_df.to_csv(merged_path, index=False)

    return {"modeling_data_path": merged_path, "cleaned_data_path": merged_path}


@checkpoint_safe
def node_time_leakage_check(state: dict) -> dict:
    """Validate no time leakage in merged features.

    Ensures transaction features only use data before the observation point.
    """
    from tools.risk_guard_tools import detect_time_leakage_candidates
    from tools.file_tools import read_dataframe

    data_path = state.get("modeling_data_path") or state.get("cleaned_data_path")
    if not data_path:
        return {}

    df = read_dataframe(data_path)
    base_time_col = state.get("base_time_col") or state.get("time_col")
    label_col = state.get("label_col")

    protected = []
    if label_col:
        protected.append(label_col)
    if base_time_col:
        protected.append(base_time_col)

    result = detect_time_leakage_candidates(
        df=df,
        observation_time_col=base_time_col,
        protected_columns=protected,
    )

    warnings = state.get("warnings", [])
    if result.get("candidates"):
        for c in result["candidates"]:
            warnings.append(f"时间穿越风险: {c['column']} — {c['reason']}")

    return {"warnings": warnings}


@checkpoint_safe
def node_risk_guard(state: dict) -> dict:
    """Detect risk fields."""
    import agents.risk_guard_agent as agent

    result = agent.run(state)
    drop_recommendations = result.get("drop_recommendations", [])
    drop_columns = [
        r["column"] if isinstance(r, dict) else r
        for r in drop_recommendations
    ]
    warnings = state.get("warnings", []) + result.get("warnings", [])
    return {"drop_columns": drop_columns, "warnings": warnings}


@checkpoint_safe
def node_modeling(state: dict) -> dict:
    """Train AutoML model on merged data."""
    import agents.modeling_agent as agent

    time_limit = state.get("_time_limit", 300)
    result = agent.run(state, time_limit=time_limit)

    if "error" in result:
        return {"errors": state.get("errors", []) + [result["error"]]}

    return {
        "model_path": result["model_path"],
        "leaderboard_path": result["leaderboard_path"],
        "train_path": result["train_path"],
        "test_path": result["test_path"],
        "predictions": result["predictions"],
        "feature_columns": result["feature_columns"],
    }


@checkpoint_safe
def node_evaluation(state: dict) -> dict:
    """Compute model evaluation metrics."""
    import agents.evaluation_agent as agent

    result = agent.run(state)
    if "error" in result:
        return {"errors": state.get("errors", []) + [result["error"]]}
    return {"metrics": result}


@checkpoint_safe
def node_strategy(state: dict) -> dict:
    """Generate threshold strategy table."""
    import agents.strategy_agent as agent

    result = agent.run(state)
    if "error" in result:
        return {"errors": state.get("errors", []) + [result["error"]]}
    return {
        "threshold_table_path": result["threshold_table_path"],
        "threshold_table": result["threshold_table"],
    }


@checkpoint_safe
def node_explain(state: dict) -> dict:
    """Compute feature importance."""
    import agents.explain_agent as agent

    result = agent.run(state)
    if "error" in result:
        return {"errors": state.get("errors", []) + [result["error"]]}
    return {
        "feature_importance_path": result["feature_importance_path"],
        "feature_importance": result["top_features"],
    }


@checkpoint_safe
def node_report(state: dict) -> dict:
    """Generate final evaluation report."""
    import agents.report_agent as agent

    result = agent.run(state)
    return {"report_path": result.get("report_path")}


# ---------------------------------------------------------------------------
# Human review gate nodes
# ---------------------------------------------------------------------------


def gate_field_semantics(state: dict) -> dict:
    return {}


def gate_data_type(state: dict) -> dict:
    return {}


def gate_data_quality(state: dict) -> dict:
    return {}


def gate_cleaning_plan(state: dict) -> dict:
    return {}


def gate_table_role(state: dict) -> dict:
    """Human review gate for table role assignment (main vs transaction)."""
    return {}


def gate_feature(state: dict) -> dict:
    """Human review gate for transaction feature plan."""
    return {}


def gate_risk(state: dict) -> dict:
    return {}


def gate_result(state: dict) -> dict:
    return {}


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

GATE_FIELD_SEMANTICS = "HumanReviewGate_FieldSemantics"
GATE_DATA_TYPE = "HumanReviewGate_DataType"
GATE_TABLE_ROLE = "HumanReviewGate_TableRole"
GATE_DATA_QUALITY = "HumanReviewGate_DataQuality"
GATE_CLEANING_PLAN = "HumanReviewGate_CleaningPlan"
GATE_FEATURE = "HumanReviewGate_Feature"
GATE_RISK = "HumanReviewGate_Risk"
GATE_RESULT = "HumanReviewGate_Result"

ALL_GATES = [
    GATE_FIELD_SEMANTICS,
    GATE_DATA_TYPE,
    GATE_TABLE_ROLE,
    GATE_DATA_QUALITY,
    GATE_CLEANING_PLAN,
    GATE_FEATURE,
    GATE_RISK,
    GATE_RESULT,
]


def build_graph() -> StateGraph:
    """Build the main + transaction pipeline graph.

    Returns:
        Uncompiled StateGraph.
    """
    graph = StateGraph(PipelineState)

    # Agent nodes
    graph.add_node("DataIntake", node_data_intake)
    graph.add_node("FieldSemantic", node_field_semantic)
    graph.add_node("DataType", node_data_type)
    graph.add_node("DataQuality", node_data_quality)
    graph.add_node("CleaningPlan", node_cleaning_plan)
    graph.add_node("CleaningExecute", node_cleaning_execute)
    graph.add_node("TransactionFeature", node_transaction_feature)
    graph.add_node("FeatureMerge", node_feature_merge)
    graph.add_node("TimeLeakageCheck", node_time_leakage_check)
    graph.add_node("RiskGuard", node_risk_guard)
    graph.add_node("Modeling", node_modeling)
    graph.add_node("Evaluation", node_evaluation)
    graph.add_node("Strategy", node_strategy)
    graph.add_node("Explain", node_explain)
    graph.add_node("Report", node_report)

    # Gate nodes
    graph.add_node(GATE_FIELD_SEMANTICS, gate_field_semantics)
    graph.add_node(GATE_DATA_TYPE, gate_data_type)
    graph.add_node(GATE_TABLE_ROLE, gate_table_role)
    graph.add_node(GATE_DATA_QUALITY, gate_data_quality)
    graph.add_node(GATE_CLEANING_PLAN, gate_cleaning_plan)
    graph.add_node(GATE_FEATURE, gate_feature)
    graph.add_node(GATE_RISK, gate_risk)
    graph.add_node(GATE_RESULT, gate_result)

    # Edges
    graph.add_edge(START, "DataIntake")
    graph.add_edge("DataIntake", "FieldSemantic")
    graph.add_edge("FieldSemantic", GATE_FIELD_SEMANTICS)
    graph.add_edge(GATE_FIELD_SEMANTICS, "DataType")
    graph.add_edge("DataType", GATE_DATA_TYPE)
    graph.add_edge(GATE_DATA_TYPE, GATE_TABLE_ROLE)
    graph.add_edge(GATE_TABLE_ROLE, "DataQuality")
    graph.add_edge("DataQuality", GATE_DATA_QUALITY)
    graph.add_edge(GATE_DATA_QUALITY, "CleaningPlan")
    graph.add_edge("CleaningPlan", GATE_CLEANING_PLAN)
    graph.add_edge(GATE_CLEANING_PLAN, "CleaningExecute")
    graph.add_edge("CleaningExecute", "TransactionFeature")
    graph.add_edge("TransactionFeature", GATE_FEATURE)
    graph.add_edge(GATE_FEATURE, "FeatureMerge")
    graph.add_edge("FeatureMerge", "TimeLeakageCheck")
    graph.add_edge("TimeLeakageCheck", "RiskGuard")
    graph.add_edge("RiskGuard", GATE_RISK)
    graph.add_edge(GATE_RISK, "Modeling")
    graph.add_edge("Modeling", "Evaluation")
    graph.add_edge("Evaluation", "Strategy")
    graph.add_edge("Strategy", "Explain")
    graph.add_edge("Explain", "Report")
    graph.add_edge("Report", GATE_RESULT)
    graph.add_edge(GATE_RESULT, END)

    return graph


def compile_pipeline(checkpointer=None):
    """Compile the pipeline with human-in-the-loop interrupt points.

    Args:
        checkpointer: LangGraph checkpointer. If None, uses MemorySaver.

    Returns:
        Compiled LangGraph application.
    """
    if checkpointer is None:
        checkpointer = MemorySaver()

    graph = build_graph()
    return graph.compile(
        checkpointer=checkpointer,
        interrupt_before=ALL_GATES,
    )


PIPELINE_TYPE = PipelineType.MAIN_PLUS_TRANSACTION
