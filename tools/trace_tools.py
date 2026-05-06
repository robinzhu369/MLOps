"""Trace and logging tools for agent actions, human confirmations, and cleaning logs."""

import json
import os
from datetime import datetime
from typing import Any

from core.schemas import AgentTraceEntry, HumanConfirmation


def _get_artifact_path(project_name: str, filename: str) -> str:
    """Get the full path for an artifact file."""
    dir_path = os.path.join("artifacts", project_name)
    os.makedirs(dir_path, exist_ok=True)
    return os.path.join(dir_path, filename)


def _append_json_record(file_path: str, record: dict) -> None:
    """Append a record to a JSON array file."""
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = []
    data.append(record)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _read_json_records(file_path: str) -> list[dict]:
    """Read all records from a JSON array file."""
    if not os.path.exists(file_path):
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_agent_trace(
    project_name: str,
    agent_name: str,
    reasoning_summary: str,
    action: str,
    action_input_summary: dict[str, Any] | None = None,
    observation_summary: str = "",
    decision: str = "",
    next_node: str = "",
    status: str = "",
) -> str:
    """Write an agent trace entry to agent_trace.json."""
    file_path = _get_artifact_path(project_name, "agent_trace.json")
    entry = AgentTraceEntry(
        agent_name=agent_name,
        reasoning_summary=reasoning_summary,
        action=action,
        action_input_summary=action_input_summary or {},
        observation_summary=observation_summary,
        decision=decision,
        next_node=next_node,
        status=status,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )
    _append_json_record(file_path, entry.model_dump())
    return file_path


def read_agent_trace(project_name: str) -> list[dict]:
    """Read all agent trace entries."""
    file_path = _get_artifact_path(project_name, "agent_trace.json")
    return _read_json_records(file_path)


def write_human_confirmation(
    project_name: str,
    confirmation_type: str,
    user_decision: str,
    details: dict[str, Any] | None = None,
) -> str:
    """Write a human confirmation record to human_confirmations.json."""
    file_path = _get_artifact_path(project_name, "human_confirmations.json")
    entry = HumanConfirmation(
        confirmation_type=confirmation_type,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        user_decision=user_decision,
        details=details or {},
    )
    _append_json_record(file_path, entry.model_dump())
    return file_path


def read_human_confirmations(project_name: str) -> list[dict]:
    """Read all human confirmation records."""
    file_path = _get_artifact_path(project_name, "human_confirmations.json")
    return _read_json_records(file_path)


def write_cleaning_log(
    project_name: str,
    action: str,
    details: dict[str, Any] | None = None,
) -> str:
    """Write a cleaning action record to cleaning_log.json."""
    file_path = _get_artifact_path(project_name, "cleaning_log.json")
    record = {
        "action": action,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "details": details or {},
    }
    _append_json_record(file_path, record)
    return file_path


def read_cleaning_log(project_name: str) -> list[dict]:
    """Read all cleaning log records."""
    file_path = _get_artifact_path(project_name, "cleaning_log.json")
    return _read_json_records(file_path)
