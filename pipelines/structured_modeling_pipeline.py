"""Structured Modeling Pipeline (Scenario A) — LangGraph orchestration.

Pipeline for structured modeling tables with a label column.
Flow: DataIntake → FieldSemantic → DataType → DataQuality → CleaningPlan →
      CleaningExecute → RiskGuard → Modeling → Evaluation → Strategy → Explain → Report

Human review gates pause execution at critical decision points.
"""

from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver

from core.constants import PipelineType
from pipelines.state import PipelineState
from pipelines.utils import checkpoint_safe


# ---------------------------------------------------------------------------
# Node functions — each wraps an agent's run() and merges results into state
# ---------------------------------------------------------------------------


@checkpoint_safe
def node_data_intake(state: dict) -> dict:
    """Process uploaded files and extract metadata.

    Expects _pending_files to be a list of file paths (strings).
    """
    import agents.data_intake_agent as agent

    file_paths = state.get("_pending_files", [])
    if not file_paths:
        return {"errors": state.get("errors", []) + ["no_pending_files"]}

    uploaded_files = []
    for file_path in file_paths:
        import os
        file_name = os.path.basename(file_path)
        with open(file_path, "rb") as f:
            file_state = agent.run(state, f, file_name)
        uploaded_files.append(file_state)

    # Set main_data_path from first file
    main_data_path = uploaded_files[0].get("file_path") if uploaded_files else None

    return {
        "uploaded_files": uploaded_files,
        "main_data_path": main_data_path,
    }


@checkpoint_safe
def node_field_semantic(state: dict) -> dict:
    """Parse field semantics via LLM or rules."""
    import agents.field_semantic_parser_agent as agent

    llm_call = state.get("_llm_call")
    result = agent.run(state, llm_call=llm_call)
    return {"field_semantics": result.get("field_semantics", {})}


@checkpoint_safe
def node_data_type(state: dict) -> dict:
    """Classify data types and determine pipeline routing."""
    import agents.data_type_classifier_agent as agent

    llm_call = state.get("_llm_call")
    result = agent.run(state, llm_call=llm_call)
    return {"data_type_classification_result": result}


@checkpoint_safe
def node_data_quality(state: dict) -> dict:
    """Run data quality analysis."""
    import agents.data_quality_agent as agent

    report = agent.run(state)
    return {"data_quality_report": report}


@checkpoint_safe
def node_cleaning_plan(state: dict) -> dict:
    """Generate cleaning plan based on quality report."""
    import agents.data_cleaning_planner_agent as agent

    plan = agent.run(state)
    return {"cleaning_plan": plan}


@checkpoint_safe
def node_cleaning_execute(state: dict) -> dict:
    """Execute the approved cleaning plan."""
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
def node_risk_guard(state: dict) -> dict:
    """Detect risk fields (leakage, IDs, high missing)."""
    import agents.risk_guard_agent as agent

    result = agent.run(state)
    # Extract column names from drop_recommendations (list of dicts)
    drop_recommendations = result.get("drop_recommendations", [])
    drop_columns = [
        r["column"] if isinstance(r, dict) else r
        for r in drop_recommendations
    ]
    warnings = state.get("warnings", []) + result.get("warnings", [])
    return {
        "drop_columns": drop_columns,
        "warnings": warnings,
    }


@checkpoint_safe
def node_modeling(state: dict) -> dict:
    """Train AutoML model."""
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
# Human review gate nodes — these are no-ops; the graph pauses before them
# ---------------------------------------------------------------------------


def gate_field_semantics(state: dict) -> dict:
    """Human review gate for field semantics. Graph pauses here."""
    return {}


def gate_data_type(state: dict) -> dict:
    """Human review gate for data type classification."""
    return {}


def gate_data_quality(state: dict) -> dict:
    """Human review gate for data quality results."""
    return {}


def gate_cleaning_plan(state: dict) -> dict:
    """Human review gate for cleaning plan approval."""
    return {}


def gate_risk(state: dict) -> dict:
    """Human review gate for risk field exclusion."""
    return {}


def gate_result(state: dict) -> dict:
    """Human review gate for final results."""
    return {}


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

# Human review gate node names
GATE_FIELD_SEMANTICS = "HumanReviewGate_FieldSemantics"
GATE_DATA_TYPE = "HumanReviewGate_DataType"
GATE_DATA_QUALITY = "HumanReviewGate_DataQuality"
GATE_CLEANING_PLAN = "HumanReviewGate_CleaningPlan"
GATE_RISK = "HumanReviewGate_Risk"
GATE_RESULT = "HumanReviewGate_Result"

ALL_GATES = [
    GATE_FIELD_SEMANTICS,
    GATE_DATA_TYPE,
    GATE_DATA_QUALITY,
    GATE_CLEANING_PLAN,
    GATE_RISK,
    GATE_RESULT,
]


def build_graph() -> StateGraph:
    """Build the structured modeling pipeline graph."""
    graph = StateGraph(PipelineState)

    # Add nodes
    graph.add_node("DataIntake", node_data_intake)
    graph.add_node("FieldSemantic", node_field_semantic)
    graph.add_node(GATE_FIELD_SEMANTICS, gate_field_semantics)
    graph.add_node("DataType", node_data_type)
    graph.add_node(GATE_DATA_TYPE, gate_data_type)
    graph.add_node("DataQuality", node_data_quality)
    graph.add_node(GATE_DATA_QUALITY, gate_data_quality)
    graph.add_node("CleaningPlan", node_cleaning_plan)
    graph.add_node(GATE_CLEANING_PLAN, gate_cleaning_plan)
    graph.add_node("CleaningExecute", node_cleaning_execute)
    graph.add_node("RiskGuard", node_risk_guard)
    graph.add_node(GATE_RISK, gate_risk)
    graph.add_node("Modeling", node_modeling)
    graph.add_node("Evaluation", node_evaluation)
    graph.add_node("Strategy", node_strategy)
    graph.add_node("Explain", node_explain)
    graph.add_node("Report", node_report)
    graph.add_node(GATE_RESULT, gate_result)

    # Add edges
    graph.add_edge(START, "DataIntake")
    graph.add_edge("DataIntake", "FieldSemantic")
    graph.add_edge("FieldSemantic", GATE_FIELD_SEMANTICS)
    graph.add_edge(GATE_FIELD_SEMANTICS, "DataType")
    graph.add_edge("DataType", GATE_DATA_TYPE)
    graph.add_edge(GATE_DATA_TYPE, "DataQuality")
    graph.add_edge("DataQuality", GATE_DATA_QUALITY)
    graph.add_edge(GATE_DATA_QUALITY, "CleaningPlan")
    graph.add_edge("CleaningPlan", GATE_CLEANING_PLAN)
    graph.add_edge(GATE_CLEANING_PLAN, "CleaningExecute")
    graph.add_edge("CleaningExecute", "RiskGuard")
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
        checkpointer: LangGraph checkpointer for state persistence.
                      If None, uses MemorySaver.

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


PIPELINE_TYPE = PipelineType.STRUCTURED_MODELING
