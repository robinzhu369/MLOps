"""Tests for agents/data_intake_agent.py."""

import io
import os

import pandas as pd
import pytest

from agents.data_intake_agent import run, run_batch
from core.state import RiskModelingProjectState


@pytest.fixture
def project_state(tmp_path, monkeypatch):
    """Create a project state with working directory set to tmp_path."""
    monkeypatch.chdir(tmp_path)
    return RiskModelingProjectState(project_name="test_project")


@pytest.fixture
def csv_file():
    """Create a CSV file-like object."""
    content = "customer_id,age,income,bad_flag\n1,25,50000,0\n2,30,60000,1\n3,35,75000,0\n"
    return io.BytesIO(content.encode())


class TestDataIntakeAgent:
    def test_run_single_file(self, project_state, csv_file):
        result = run(project_state, csv_file, "loan.csv")

        assert result["file_name"] == "loan.csv"
        assert result["file_format"] == "csv"
        assert result["n_rows"] == 3
        assert result["n_cols"] == 4
        assert "customer_id" in result["columns"]
        assert "bad_flag" in result["columns"]
        assert result["column_profiles"]["age"]["dtype"] == "int64"
        assert os.path.exists(result["file_path"])

    def test_run_writes_trace(self, project_state, csv_file):
        run(project_state, csv_file, "loan.csv")

        trace_path = os.path.join("artifacts", "test_project", "agent_trace.json")
        assert os.path.exists(trace_path)

        import json
        with open(trace_path) as f:
            traces = json.load(f)
        assert len(traces) >= 1
        assert traces[-1]["agent_name"] == "DataIntakeAgent"
        assert traces[-1]["status"] == "completed"

    def test_run_batch(self, project_state):
        csv1 = io.BytesIO(b"id,val\n1,a\n2,b\n")
        csv2 = io.BytesIO(b"id,score\n1,0.5\n2,0.8\n3,0.3\n")

        results = run_batch(project_state, [(csv1, "file1.csv"), (csv2, "file2.csv")])

        assert len(results) == 2
        assert results[0]["file_name"] == "file1.csv"
        assert results[0]["n_rows"] == 2
        assert results[1]["file_name"] == "file2.csv"
        assert results[1]["n_rows"] == 3

    def test_run_unsupported_format(self, project_state):
        content = io.BytesIO(b"some data")
        with pytest.raises(ValueError, match="Unsupported file format"):
            run(project_state, content, "bad.txt")
