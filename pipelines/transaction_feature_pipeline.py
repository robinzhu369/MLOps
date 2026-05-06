"""Transaction Feature Pipeline (Scenario B) — LangGraph orchestration.

Pipeline for transaction flow tables without a label column.
Flow: DataIntake → FieldSemantic → DataType → TransactionQuality → CleaningPlan →
      CleaningExecute → TransactionFeature → Report

No modeling step — outputs feature tables and a profile report only.
"""

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
    """Process uploaded files and extract metadata.

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

    # For transaction pipeline, the first file is the transaction data
    txn_path = uploaded_files[0].get("file_path") if uploaded_files else None

    return {
        "uploaded_files": uploaded_files,
        "transaction_data_path": txn_path,
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
    """Classify data types."""
    import agents.data_type_classifier_agent as agent

    llm_call = state.get("_llm_call")
    result = agent.run(state, llm_call=llm_call)
    return {"data_type_classification_result": result}


@checkpoint_safe
def node_transaction_quality(state: dict) -> dict:
    """Run transaction-specific quality analysis."""
    import agents.transaction_quality_agent as agent

    result = agent.run(state)
    return {"transaction_quality_report": result}


@checkpoint_safe
def node_cleaning_plan(state: dict) -> dict:
    """Generate cleaning plan for transaction data."""
    import agents.data_cleaning_planner_agent as agent

    # Use transaction quality report as the quality report
    txn_quality = state.get("transaction_quality_report", {})
    if txn_quality and "error" not in txn_quality:
        state_with_quality = dict(state)
        state_with_quality["data_quality_report"] = txn_quality
        plan = agent.run(state_with_quality)
    else:
        plan = agent.run(state)
    return {"cleaning_plan": plan}


@checkpoint_safe
def node_cleaning_execute(state: dict) -> dict:
    """Execute the approved cleaning plan on transaction data."""
    from tools.data_cleaning_tools import execute_cleaning_plan

    data_path = state.get("transaction_data_path")
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
        "cleaned_transaction_data_path": result.get("output_path"),
        "cleaning_log_path": result.get("cleaning_log_path"),
    }


@checkpoint_safe
def node_transaction_feature(state: dict) -> dict:
    """Build transaction features (daily + window)."""
    import agents.transaction_feature_agent as agent

    # Use cleaned path if available
    cleaned_path = state.get("cleaned_transaction_data_path")
    if cleaned_path:
        state_copy = dict(state)
        state_copy["transaction_data_path"] = cleaned_path
        result = agent.run(state_copy)
    else:
        result = agent.run(state)

    if "error" in result:
        return {"errors": state.get("errors", []) + [result["error"]]}

    return {
        "transaction_daily_feature_path": result.get("daily_features_path"),
        "transaction_window_feature_path": result.get("window_features_path"),
        "transaction_schema": result.get("schema", {}),
    }


@checkpoint_safe
def node_report(state: dict) -> dict:
    """Generate feature profile report."""
    import agents.report_agent as agent

    result = agent.run(state)
    return {"report_path": result.get("report_path")}


# ---------------------------------------------------------------------------
# Human review gate nodes
# ---------------------------------------------------------------------------


def gate_field_semantics(state: dict) -> dict:
    """Human review gate for field semantics."""
    return {}


def gate_data_type(state: dict) -> dict:
    """Human review gate for data type classification."""
    return {}


def gate_data_quality(state: dict) -> dict:
    """Human review gate for transaction quality."""
    return {}


def gate_cleaning_plan(state: dict) -> dict:
    """Human review gate for cleaning plan."""
    return {}


def gate_feature(state: dict) -> dict:
    """Human review gate for transaction feature plan."""
    return {}


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

GATE_FIELD_SEMANTICS = "HumanReviewGate_FieldSemantics"
GATE_DATA_TYPE = "HumanReviewGate_DataType"
GATE_DATA_QUALITY = "HumanReviewGate_DataQuality"
GATE_CLEANING_PLAN = "HumanReviewGate_CleaningPlan"
GATE_FEATURE = "HumanReviewGate_Feature"

ALL_GATES = [
    GATE_FIELD_SEMANTICS,
    GATE_DATA_TYPE,
    GATE_DATA_QUALITY,
    GATE_CLEANING_PLAN,
    GATE_FEATURE,
]


def build_graph() -> StateGraph:
    """Build the transaction feature pipeline graph.

    Returns:
        Uncompiled StateGraph.
    """
    graph = StateGraph(PipelineState)

    # Agent nodes
    graph.add_node("DataIntake", node_data_intake)
    graph.add_node("FieldSemantic", node_field_semantic)
    graph.add_node("DataType", node_data_type)
    graph.add_node("TransactionQuality", node_transaction_quality)
    graph.add_node("CleaningPlan", node_cleaning_plan)
    graph.add_node("CleaningExecute", node_cleaning_execute)
    graph.add_node("TransactionFeature", node_transaction_feature)
    graph.add_node("Report", node_report)

    # Human review gates
    graph.add_node(GATE_FIELD_SEMANTICS, gate_field_semantics)
    graph.add_node(GATE_DATA_TYPE, gate_data_type)
    graph.add_node(GATE_DATA_QUALITY, gate_data_quality)
    graph.add_node(GATE_CLEANING_PLAN, gate_cleaning_plan)
    graph.add_node(GATE_FEATURE, gate_feature)

    # Edges
    graph.add_edge(START, "DataIntake")
    graph.add_edge("DataIntake", "FieldSemantic")
    graph.add_edge("FieldSemantic", GATE_FIELD_SEMANTICS)
    graph.add_edge(GATE_FIELD_SEMANTICS, "DataType")
    graph.add_edge("DataType", GATE_DATA_TYPE)
    graph.add_edge(GATE_DATA_TYPE, "TransactionQuality")
    graph.add_edge("TransactionQuality", GATE_DATA_QUALITY)
    graph.add_edge(GATE_DATA_QUALITY, "CleaningPlan")
    graph.add_edge("CleaningPlan", GATE_CLEANING_PLAN)
    graph.add_edge(GATE_CLEANING_PLAN, "CleaningExecute")
    graph.add_edge("CleaningExecute", "TransactionFeature")
    graph.add_edge("TransactionFeature", GATE_FEATURE)
    graph.add_edge(GATE_FEATURE, "Report")
    graph.add_edge("Report", END)

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


PIPELINE_TYPE = PipelineType.TRANSACTION_FEATURE
