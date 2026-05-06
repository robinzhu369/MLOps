"""Tests for core/permissions.py."""

from core.permissions import check_tool_permission, AGENT_TOOL_PERMISSIONS


def test_permitted_tool():
    assert check_tool_permission("DataIntakeAgent", "save_uploaded_file") is True


def test_denied_tool():
    assert check_tool_permission("DataIntakeAgent", "train_autogluon_binary") is False


def test_unknown_agent():
    assert check_tool_permission("NonExistentAgent", "save_uploaded_file") is False


def test_all_agents_have_tools():
    for agent_name, tools in AGENT_TOOL_PERMISSIONS.items():
        assert len(tools) > 0, f"{agent_name} has no permitted tools"


def test_modeling_agent_only_has_train():
    assert AGENT_TOOL_PERMISSIONS["ModelingAgent"] == ["train_autogluon_binary"]
