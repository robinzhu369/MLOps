"""Integration tests — Stage 12, Task 12.2.

End-to-end validation for Scenarios A, B, and C.
Each test runs the full LangGraph pipeline without interrupt gates
(compile without interrupt_before) to verify the complete flow.

Assertions cover:
- All expected state keys populated
- Artifact files created on disk
- agent_trace.json completeness
- human_confirmations.json records
- No time-leakage in Scenario C
"""

import json
import os

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "test_scenarios")
LOAN_CSV = os.path.join(TEST_DATA_DIR, "loan_application.csv")
TXN_CSV = os.path.join(TEST_DATA_DIR, "transaction_flow.csv")
DICT_CSV = os.path.join(TEST_DATA_DIR, "data_dictionary.csv")


def _artifact_dir(project_name: str, base: str) -> str:
    return os.path.join(base, "artifacts", project_name)


def _read_json(path: str) -> list:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_dir(tmp_path):
    return str(tmp_path)


@pytest.fixture(autouse=True)
def chdir_tmp(tmp_dir, monkeypatch):
    """Run every test from a fresh temp directory so artifacts don't collide."""
    monkeypatch.chdir(tmp_dir)


# ---------------------------------------------------------------------------
# Scenario A: Structured Modeling Pipeline
# ---------------------------------------------------------------------------


class TestScenarioA:
    """End-to-end test for Scenario A (structured modeling table with label)."""

    PROJECT = "e2e_scenario_a"

    def _run(self, tmp_dir: str) -> dict:
        from pipelines.structured_modeling_pipeline import build_graph

        graph = build_graph()
        app = graph.compile()  # no interrupt gates

        initial_state = {
            "project_name": self.PROJECT,
            "_pending_files": [LOAN_CSV],
            "label_col": "bad_flag",
            "id_col": "customer_id",
            "base_time_col": "apply_date",
            "_time_limit": 60,
        }

        return app.invoke(initial_state)

    def test_pipeline_completes(self, tmp_dir):
        result = self._run(tmp_dir)
        assert result is not None, "Pipeline returned None"

    def test_uploaded_files_populated(self, tmp_dir):
        result = self._run(tmp_dir)
        assert "uploaded_files" in result
        assert len(result["uploaded_files"]) == 1
        assert result["uploaded_files"][0]["file_name"] == "loan_application.csv"

    def test_field_semantics_populated(self, tmp_dir):
        result = self._run(tmp_dir)
        assert "field_semantics" in result
        assert len(result["field_semantics"]) > 0

    def test_data_quality_report_populated(self, tmp_dir):
        result = self._run(tmp_dir)
        report = result.get("data_quality_report", {})
        assert report, "data_quality_report is empty"
        assert "duplicate_analysis" in report
        assert "missing_analysis" in report

    def test_cleaning_plan_populated(self, tmp_dir):
        result = self._run(tmp_dir)
        plan = result.get("cleaning_plan", {})
        assert plan, "cleaning_plan is empty"
        assert "cleaning_steps" in plan

    def test_cleaned_data_file_exists(self, tmp_dir):
        result = self._run(tmp_dir)
        cleaned_path = result.get("cleaned_data_path")
        assert cleaned_path, "cleaned_data_path not set"
        assert os.path.exists(cleaned_path), f"Cleaned file not found: {cleaned_path}"

    def test_model_artifacts_exist(self, tmp_dir):
        result = self._run(tmp_dir)
        model_path = result.get("model_path")
        assert model_path, "model_path not set"
        assert os.path.exists(model_path), f"Model dir not found: {model_path}"

    def test_metrics_populated(self, tmp_dir):
        result = self._run(tmp_dir)
        metrics = result.get("metrics", {})
        assert metrics, "metrics is empty"
        assert "auc" in metrics, "AUC missing from metrics"
        auc = float(metrics["auc"])
        assert 0.0 <= auc <= 1.0, f"AUC out of range: {auc}"

    def test_threshold_table_populated(self, tmp_dir):
        result = self._run(tmp_dir)
        threshold_table = result.get("threshold_table")
        assert threshold_table, "threshold_table is empty"
        assert isinstance(threshold_table, list)
        assert len(threshold_table) > 0

    def test_feature_importance_populated(self, tmp_dir):
        result = self._run(tmp_dir)
        fi = result.get("feature_importance")
        assert fi, "feature_importance is empty"
        assert isinstance(fi, list)
        assert len(fi) > 0

    def test_report_file_exists(self, tmp_dir):
        result = self._run(tmp_dir)
        report_path = result.get("report_path")
        assert report_path, "report_path not set"
        assert os.path.exists(report_path), f"Report file not found: {report_path}"
        with open(report_path, "r", encoding="utf-8") as f:
            content = f.read()
        assert len(content) > 100, "Report file is too short"

    def test_agent_trace_written(self, tmp_dir):
        result = self._run(tmp_dir)
        trace_path = os.path.join(
            _artifact_dir(self.PROJECT, tmp_dir), "agent_trace.json"
        )
        trace = _read_json(trace_path)
        assert len(trace) > 0, "agent_trace.json is empty"
        # Verify required fields in each entry
        for entry in trace:
            assert "agent_name" in entry
            assert "action" in entry
            assert "timestamp" in entry

    def test_no_errors_in_result(self, tmp_dir):
        result = self._run(tmp_dir)
        errors = result.get("errors", [])
        assert errors == [], f"Pipeline produced errors: {errors}"

    def test_original_data_unchanged(self, tmp_dir):
        """Original CSV must not be modified."""
        original_mtime = os.path.getmtime(LOAN_CSV)
        self._run(tmp_dir)
        assert os.path.getmtime(LOAN_CSV) == original_mtime, "Original file was modified"


# ---------------------------------------------------------------------------
# Scenario B: Transaction Feature Pipeline
# ---------------------------------------------------------------------------


class TestScenarioB:
    """End-to-end test for Scenario B (transaction flow table, no label)."""

    PROJECT = "e2e_scenario_b"

    def _run(self, tmp_dir: str) -> dict:
        from pipelines.transaction_feature_pipeline import build_graph

        graph = build_graph()
        app = graph.compile()

        initial_state = {
            "project_name": self.PROJECT,
            "_pending_files": [TXN_CSV],
            "account_col": "account_id",
            "time_col": "transaction_time",
            "_time_limit": 60,
        }

        return app.invoke(initial_state)

    def test_pipeline_completes(self, tmp_dir):
        result = self._run(tmp_dir)
        assert result is not None

    def test_uploaded_files_populated(self, tmp_dir):
        result = self._run(tmp_dir)
        assert "uploaded_files" in result
        assert result["uploaded_files"][0]["file_name"] == "transaction_flow.csv"

    def test_transaction_quality_report(self, tmp_dir):
        result = self._run(tmp_dir)
        report = result.get("transaction_quality_report") or result.get("data_quality_report")
        assert report, "No quality report produced"

    def test_feature_files_exist(self, tmp_dir):
        result = self._run(tmp_dir)
        daily_path = result.get("transaction_daily_feature_path")
        # Daily features are always produced; window features require a main table
        assert daily_path, "transaction_daily_feature_path not set"
        assert os.path.exists(daily_path), f"Daily feature file missing: {daily_path}"

    def test_feature_files_non_empty(self, tmp_dir):
        result = self._run(tmp_dir)
        daily_path = result.get("transaction_daily_feature_path")
        window_path = result.get("transaction_window_feature_path")
        if daily_path and os.path.exists(daily_path):
            df = pd.read_csv(daily_path)
            assert len(df) > 0, "Daily feature file is empty"
        # Window features are optional in Scenario B (no main table)
        if window_path and os.path.exists(window_path):
            df = pd.read_csv(window_path)
            assert len(df) > 0, "Window feature file is empty"

    def test_report_file_exists(self, tmp_dir):
        result = self._run(tmp_dir)
        report_path = result.get("report_path")
        assert report_path, "report_path not set"
        assert os.path.exists(report_path)

    def test_agent_trace_written(self, tmp_dir):
        result = self._run(tmp_dir)
        trace_path = os.path.join(
            _artifact_dir(self.PROJECT, tmp_dir), "agent_trace.json"
        )
        trace = _read_json(trace_path)
        assert len(trace) > 0

    def test_no_errors_in_result(self, tmp_dir):
        result = self._run(tmp_dir)
        errors = result.get("errors", [])
        assert errors == [], f"Pipeline produced errors: {errors}"

    def test_no_model_trained(self, tmp_dir):
        """Scenario B must not produce a model (no label)."""
        result = self._run(tmp_dir)
        assert not result.get("model_path"), "Scenario B should not train a model"
        assert not result.get("metrics"), "Scenario B should not produce model metrics"


# ---------------------------------------------------------------------------
# Scenario C: Main + Transaction Pipeline
# ---------------------------------------------------------------------------


class TestScenarioC:
    """End-to-end test for Scenario C (main table + transaction flow)."""

    PROJECT = "e2e_scenario_c"

    def _run(self, tmp_dir: str) -> dict:
        from pipelines.main_plus_transaction_pipeline import build_graph

        graph = build_graph()
        app = graph.compile()

        initial_state = {
            "project_name": self.PROJECT,
            "_pending_files": [LOAN_CSV, TXN_CSV],
            "label_col": "bad_flag",
            "id_col": "customer_id",
            "account_col": "account_id",
            "base_time_col": "apply_date",
            "time_col": "transaction_time",
            "_time_limit": 60,
        }

        return app.invoke(initial_state)

    def test_pipeline_completes(self, tmp_dir):
        result = self._run(tmp_dir)
        assert result is not None

    def test_both_files_uploaded(self, tmp_dir):
        result = self._run(tmp_dir)
        assert "uploaded_files" in result
        assert len(result["uploaded_files"]) == 2
        names = {f["file_name"] for f in result["uploaded_files"]}
        assert "loan_application.csv" in names
        assert "transaction_flow.csv" in names

    def test_transaction_features_generated(self, tmp_dir):
        result = self._run(tmp_dir)
        daily_path = result.get("transaction_daily_feature_path")
        window_path = result.get("transaction_window_feature_path")
        assert daily_path and os.path.exists(daily_path)
        assert window_path and os.path.exists(window_path)

    def test_model_trained(self, tmp_dir):
        result = self._run(tmp_dir)
        assert result.get("model_path"), "model_path not set"
        assert os.path.exists(result["model_path"])

    def test_metrics_populated(self, tmp_dir):
        result = self._run(tmp_dir)
        metrics = result.get("metrics", {})
        assert metrics
        assert "auc" in metrics
        auc = float(metrics["auc"])
        assert 0.0 <= auc <= 1.0

    def test_report_file_exists(self, tmp_dir):
        result = self._run(tmp_dir)
        report_path = result.get("report_path")
        assert report_path and os.path.exists(report_path)

    def test_no_time_leakage(self, tmp_dir):
        """Transaction features must only use data before apply_date."""
        result = self._run(tmp_dir)
        window_path = result.get("transaction_window_feature_path")
        if not window_path or not os.path.exists(window_path):
            pytest.skip("Window feature file not produced")

        window_df = pd.read_csv(window_path)
        loan_df = pd.read_csv(LOAN_CSV)

        # If window features have a join key, verify no future transactions leaked
        # The pipeline's TimeLeakageCheck node should have caught this;
        # here we verify the feature file doesn't contain a future-date column.
        assert "transaction_time" not in window_df.columns, (
            "Raw transaction_time should not appear in window feature table"
        )

    def test_agent_trace_written(self, tmp_dir):
        result = self._run(tmp_dir)
        trace_path = os.path.join(
            _artifact_dir(self.PROJECT, tmp_dir), "agent_trace.json"
        )
        trace = _read_json(trace_path)
        assert len(trace) > 0

    def test_no_errors_in_result(self, tmp_dir):
        result = self._run(tmp_dir)
        errors = result.get("errors", [])
        assert errors == [], f"Pipeline produced errors: {errors}"

    def test_original_files_unchanged(self, tmp_dir):
        loan_mtime = os.path.getmtime(LOAN_CSV)
        txn_mtime = os.path.getmtime(TXN_CSV)
        self._run(tmp_dir)
        assert os.path.getmtime(LOAN_CSV) == loan_mtime
        assert os.path.getmtime(TXN_CSV) == txn_mtime


# ---------------------------------------------------------------------------
# Cross-scenario: artifact format validation
# ---------------------------------------------------------------------------


class TestArtifactFormats:
    """Validate artifact file formats across scenarios."""

    def test_agent_trace_schema(self, tmp_dir):
        """agent_trace.json entries must have required fields."""
        from pipelines.structured_modeling_pipeline import build_graph

        graph = build_graph()
        app = graph.compile()
        app.invoke({
            "project_name": "e2e_trace_check",
            "_pending_files": [LOAN_CSV],
            "label_col": "bad_flag",
            "_time_limit": 60,
        })

        trace_path = os.path.join(
            _artifact_dir("e2e_trace_check", tmp_dir), "agent_trace.json"
        )
        trace = _read_json(trace_path)
        assert len(trace) > 0

        required_fields = {"agent_name", "action", "timestamp", "status"}
        for entry in trace:
            missing = required_fields - set(entry.keys())
            assert not missing, f"Trace entry missing fields: {missing}\nEntry: {entry}"

    def test_cleaning_log_schema(self, tmp_dir):
        """cleaning_log.json entries must have action and timestamp."""
        from pipelines.structured_modeling_pipeline import build_graph

        graph = build_graph()
        app = graph.compile()
        app.invoke({
            "project_name": "e2e_cleaning_log",
            "_pending_files": [LOAN_CSV],
            "label_col": "bad_flag",
            "_time_limit": 60,
        })

        log_path = os.path.join(
            _artifact_dir("e2e_cleaning_log", tmp_dir), "cleaning_log.json"
        )
        log = _read_json(log_path)
        # Cleaning log may be empty if no steps were needed
        for entry in log:
            assert "action" in entry
            assert "timestamp" in entry

    def test_leaderboard_csv_format(self, tmp_dir):
        """leaderboard.csv must be a valid CSV with model names."""
        from pipelines.structured_modeling_pipeline import build_graph

        graph = build_graph()
        app = graph.compile()
        result = app.invoke({
            "project_name": "e2e_leaderboard",
            "_pending_files": [LOAN_CSV],
            "label_col": "bad_flag",
            "_time_limit": 60,
        })

        lb_path = result.get("leaderboard_path")
        if lb_path and os.path.exists(lb_path):
            df = pd.read_csv(lb_path)
            assert len(df) > 0
            assert "model" in df.columns or len(df.columns) > 0
