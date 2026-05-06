"""Tests for tools/trace_tools.py."""

import json
import os
import tempfile
from unittest.mock import patch

from tools.trace_tools import (
    write_agent_trace,
    read_agent_trace,
    write_human_confirmation,
    read_human_confirmations,
    write_cleaning_log,
    read_cleaning_log,
)


def test_write_and_read_agent_trace(tmp_path):
    project = "test_project"
    with patch("tools.trace_tools._get_artifact_path") as mock_path:
        file_path = str(tmp_path / "agent_trace.json")
        mock_path.return_value = file_path

        write_agent_trace(
            project_name=project,
            agent_name="DataQualityAgent",
            reasoning_summary="检查数据质量",
            action="analyze_data_quality",
            action_input_summary={"file": "test.csv"},
            observation_summary="发现重复行",
            decision="进入人工确认",
            next_node="HumanReviewGate_DataQuality",
            status="need_human_review",
        )

        records = read_agent_trace(project)

    assert len(records) == 1
    assert records[0]["agent_name"] == "DataQualityAgent"
    assert records[0]["action"] == "analyze_data_quality"
    assert records[0]["status"] == "need_human_review"
    assert records[0]["timestamp"] != ""


def test_append_multiple_traces(tmp_path):
    project = "test_project"
    with patch("tools.trace_tools._get_artifact_path") as mock_path:
        file_path = str(tmp_path / "agent_trace.json")
        mock_path.return_value = file_path

        write_agent_trace(project, "Agent1", "reason1", "action1")
        write_agent_trace(project, "Agent2", "reason2", "action2")

        records = read_agent_trace(project)

    assert len(records) == 2
    assert records[0]["agent_name"] == "Agent1"
    assert records[1]["agent_name"] == "Agent2"


def test_write_and_read_human_confirmation(tmp_path):
    project = "test_project"
    with patch("tools.trace_tools._get_artifact_path") as mock_path:
        file_path = str(tmp_path / "human_confirmations.json")
        mock_path.return_value = file_path

        write_human_confirmation(
            project_name=project,
            confirmation_type="cleaning_plan",
            user_decision="accept_with_modification",
            details={"duplicate_strategy": "drop_exact_duplicate_rows"},
        )

        records = read_human_confirmations(project)

    assert len(records) == 1
    assert records[0]["confirmation_type"] == "cleaning_plan"
    assert records[0]["user_decision"] == "accept_with_modification"
    assert records[0]["details"]["duplicate_strategy"] == "drop_exact_duplicate_rows"


def test_write_and_read_cleaning_log(tmp_path):
    project = "test_project"
    with patch("tools.trace_tools._get_artifact_path") as mock_path:
        file_path = str(tmp_path / "cleaning_log.json")
        mock_path.return_value = file_path

        write_cleaning_log(
            project_name=project,
            action="drop_exact_duplicate_rows",
            details={"rows_removed": 120},
        )

        records = read_cleaning_log(project)

    assert len(records) == 1
    assert records[0]["action"] == "drop_exact_duplicate_rows"
    assert records[0]["details"]["rows_removed"] == 120


def test_read_nonexistent_file(tmp_path):
    project = "test_project"
    with patch("tools.trace_tools._get_artifact_path") as mock_path:
        mock_path.return_value = str(tmp_path / "nonexistent.json")
        records = read_agent_trace(project)

    assert records == []
