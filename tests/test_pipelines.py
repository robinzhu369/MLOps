"""Tests for Stage 10: LangGraph Pipeline Orchestration (Tasks 10.1–10.3)."""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from langgraph.checkpoint.memory import MemorySaver


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def structured_csv(tmp_dir):
    """Create a structured modeling table CSV (Scenario A)."""
    rng = np.random.RandomState(42)
    n = 100
    df = pd.DataFrame({
        "customer_id": [f"C{i:04d}" for i in range(n)],
        "age": rng.randint(20, 65, n),
        "income": rng.uniform(3000, 50000, n).round(2),
        "loan_amount": rng.uniform(1000, 100000, n).round(2),
        "credit_score": rng.randint(300, 850, n),
        "apply_date": pd.date_range("2023-01-01", periods=n, freq="D"),
        "bad_flag": rng.choice([0, 1], size=n, p=[0.85, 0.15]),
    })
    path = os.path.join(tmp_dir, "loan_application.csv")
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def transaction_csv(tmp_dir):
    """Create a transaction flow table CSV (Scenario B)."""
    rng = np.random.RandomState(42)
    n = 500
    df = pd.DataFrame({
        "transaction_id": [f"T{i:06d}" for i in range(n)],
        "account_id": [f"A{rng.randint(0, 20):04d}" for _ in range(n)],
        "transaction_time": pd.date_range("2023-01-01", periods=n, freq="2h"),
        "transaction_amount": rng.uniform(10, 5000, n).round(2),
        "debit_credit_flag": rng.choice(["D", "C"], n),
        "channel": rng.choice(["ATM", "POS", "ONLINE", "TRANSFER"], n),
        "balance_after_txn": rng.uniform(0, 50000, n).round(2),
    })
    path = os.path.join(tmp_dir, "transaction_flow.csv")
    df.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Task 10.1: Structured Modeling Pipeline Tests
# ---------------------------------------------------------------------------


class TestStructuredModelingPipeline:
    def test_graph_structure(self):
        """Verify graph has correct nodes and edges."""
        from pipelines.structured_modeling_pipeline import build_graph, ALL_GATES

        graph = build_graph()
        node_names = set(graph.nodes.keys())

        # All expected nodes present
        expected_nodes = {
            "DataIntake", "FieldSemantic", "DataType", "DataQuality",
            "CleaningPlan", "CleaningExecute", "RiskGuard",
            "Modeling", "Evaluation", "Strategy", "Explain", "Report",
        }
        for node in expected_nodes:
            assert node in node_names, f"Missing node: {node}"

        # All gates present
        for gate in ALL_GATES:
            assert gate in node_names, f"Missing gate: {gate}"

    def test_compile_with_checkpointer(self):
        """Pipeline compiles with MemorySaver checkpointer."""
        from pipelines.structured_modeling_pipeline import compile_pipeline

        memory = MemorySaver()
        app = compile_pipeline(checkpointer=memory)
        assert app is not None

    def test_interrupt_at_first_gate(self, structured_csv, monkeypatch, tmp_dir):
        """Pipeline pauses at first human review gate."""
        from pipelines.structured_modeling_pipeline import compile_pipeline

        monkeypatch.chdir(tmp_dir)

        memory = MemorySaver()
        app = compile_pipeline(checkpointer=memory)

        initial_state = {
            "project_name": "test_structured",
            "_pending_files": [structured_csv],
        }

        config = {"configurable": {"thread_id": "test_structured_1"}}
        result = app.invoke(initial_state, config)

        # Should have processed data intake and field semantics
        assert "uploaded_files" in result
        assert len(result["uploaded_files"]) == 1
        assert result["uploaded_files"][0]["file_name"] == "loan_application.csv"

    def test_end_to_end_no_gates(self, structured_csv, monkeypatch, tmp_dir):
        """Pipeline runs end-to-end without interrupt gates (for testing)."""
        from pipelines.structured_modeling_pipeline import build_graph

        monkeypatch.chdir(tmp_dir)

        # Compile without interrupt_before to run straight through
        graph = build_graph()
        app = graph.compile()

        initial_state = {
            "project_name": "test_e2e",
            "_pending_files": [structured_csv],
            "label_col": "bad_flag",
            "id_col": "customer_id",
            "base_time_col": "apply_date",
            "_time_limit": 60,
        }

        result = app.invoke(initial_state)

        # Verify full pipeline completed
        assert "uploaded_files" in result
        assert "field_semantics" in result
        assert "data_quality_report" in result
        assert "cleaning_plan" in result
        assert "model_path" in result
        assert "metrics" in result
        assert "threshold_table" in result
        assert "feature_importance" in result
        assert "report_path" in result
        assert os.path.exists(result["report_path"])

    def test_resume_after_gate(self, structured_csv, monkeypatch, tmp_dir):
        """Pipeline resumes correctly after human review gate."""
        from pipelines.structured_modeling_pipeline import compile_pipeline

        monkeypatch.chdir(tmp_dir)

        memory = MemorySaver()
        app = compile_pipeline(checkpointer=memory)

        initial_state = {
            "project_name": "test_resume",
            "_pending_files": [structured_csv],
            "label_col": "bad_flag",
            "id_col": "customer_id",
        }

        config = {"configurable": {"thread_id": "test_resume_1"}}

        # First invoke — pauses at first gate
        result1 = app.invoke(initial_state, config)
        assert "uploaded_files" in result1

        # Resume — should proceed to next gate
        result2 = app.invoke(None, config)
        # After field semantics gate, should have field_semantics
        assert "field_semantics" in result2


# ---------------------------------------------------------------------------
# Task 10.2: Transaction Feature Pipeline Tests
# ---------------------------------------------------------------------------


class TestTransactionFeaturePipeline:
    def test_graph_structure(self):
        """Verify graph has correct nodes."""
        from pipelines.transaction_feature_pipeline import build_graph, ALL_GATES

        graph = build_graph()
        node_names = set(graph.nodes.keys())

        expected_nodes = {
            "DataIntake", "FieldSemantic", "DataType",
            "TransactionQuality", "CleaningPlan", "CleaningExecute",
            "TransactionFeature", "Report",
        }
        for node in expected_nodes:
            assert node in node_names, f"Missing node: {node}"

        # No modeling nodes
        assert "Modeling" not in node_names
        assert "Evaluation" not in node_names
        assert "Strategy" not in node_names

        # All gates present
        for gate in ALL_GATES:
            assert gate in node_names

    def test_compile(self):
        """Pipeline compiles successfully."""
        from pipelines.transaction_feature_pipeline import compile_pipeline

        app = compile_pipeline()
        assert app is not None

    def test_interrupt_at_first_gate(self, transaction_csv, monkeypatch, tmp_dir):
        """Pipeline pauses at first human review gate."""
        from pipelines.transaction_feature_pipeline import compile_pipeline

        monkeypatch.chdir(tmp_dir)

        memory = MemorySaver()
        app = compile_pipeline(checkpointer=memory)

        initial_state = {
            "project_name": "test_txn",
            "_pending_files": [transaction_csv],
        }

        config = {"configurable": {"thread_id": "test_txn_1"}}
        result = app.invoke(initial_state, config)

        assert "uploaded_files" in result
        assert result["uploaded_files"][0]["file_name"] == "transaction_flow.csv"
        assert "transaction_data_path" in result


# ---------------------------------------------------------------------------
# Task 10.3: Main + Transaction Pipeline Tests
# ---------------------------------------------------------------------------


class TestMainPlusTransactionPipeline:
    def test_graph_structure(self):
        """Verify graph has correct nodes including merge and time check."""
        from pipelines.main_plus_transaction_pipeline import build_graph, ALL_GATES

        graph = build_graph()
        node_names = set(graph.nodes.keys())

        expected_nodes = {
            "DataIntake", "FieldSemantic", "DataType",
            "DataQuality", "CleaningPlan", "CleaningExecute",
            "TransactionFeature", "FeatureMerge", "TimeLeakageCheck",
            "RiskGuard", "Modeling", "Evaluation", "Strategy",
            "Explain", "Report",
        }
        for node in expected_nodes:
            assert node in node_names, f"Missing node: {node}"

        # All gates present
        for gate in ALL_GATES:
            assert gate in node_names

    def test_compile(self):
        """Pipeline compiles successfully."""
        from pipelines.main_plus_transaction_pipeline import compile_pipeline

        app = compile_pipeline()
        assert app is not None

    def test_has_table_role_gate(self):
        """Pipeline C has the TableRole gate for multi-table confirmation."""
        from pipelines.main_plus_transaction_pipeline import ALL_GATES

        assert "HumanReviewGate_TableRole" in ALL_GATES

    def test_interrupt_at_first_gate(
        self, structured_csv, transaction_csv, monkeypatch, tmp_dir
    ):
        """Pipeline pauses at first human review gate with multi-file input."""
        from pipelines.main_plus_transaction_pipeline import compile_pipeline

        monkeypatch.chdir(tmp_dir)

        memory = MemorySaver()
        app = compile_pipeline(checkpointer=memory)

        initial_state = {
            "project_name": "test_combined",
            "_pending_files": [structured_csv, transaction_csv],
        }

        config = {"configurable": {"thread_id": "test_combined_1"}}
        result = app.invoke(initial_state, config)

        assert "uploaded_files" in result
        assert len(result["uploaded_files"]) == 2


# ---------------------------------------------------------------------------
# Cross-pipeline tests
# ---------------------------------------------------------------------------


class TestPipelineCommon:
    def test_all_pipelines_importable(self):
        """All three pipelines can be imported."""
        from pipelines.structured_modeling_pipeline import PIPELINE_TYPE as pt_a
        from pipelines.transaction_feature_pipeline import PIPELINE_TYPE as pt_b
        from pipelines.main_plus_transaction_pipeline import PIPELINE_TYPE as pt_c

        assert pt_a == "structured_modeling_pipeline"
        assert pt_b == "transaction_feature_pipeline"
        assert pt_c == "main_plus_transaction_pipeline"

    def test_pipeline_selection_by_router(self):
        """PipelineRouterAgent correctly selects pipelines."""
        from agents.pipeline_router_agent import route_pipeline
        from core.constants import DataType, PipelineType

        # Scenario A
        result_a = route_pipeline([{
            "file_name": "loan.csv",
            "detected_data_type": DataType.STRUCTURED_MODELING_TABLE,
        }])
        assert result_a["pipeline_type"] == PipelineType.STRUCTURED_MODELING

        # Scenario B
        result_b = route_pipeline([{
            "file_name": "txn.csv",
            "detected_data_type": DataType.TRANSACTION_FLOW_TABLE,
        }])
        assert result_b["pipeline_type"] == PipelineType.TRANSACTION_FEATURE

        # Scenario C
        result_c = route_pipeline([
            {"file_name": "main.csv", "detected_data_type": DataType.MAIN_TABLE},
            {"file_name": "txn.csv", "detected_data_type": DataType.TRANSACTION_FLOW_TABLE},
        ])
        assert result_c["pipeline_type"] == PipelineType.MAIN_PLUS_TRANSACTION
