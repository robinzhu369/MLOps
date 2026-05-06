"""Tests for agents/pipeline_router_agent.py."""

import pytest

from core.constants import DataType, PipelineType
from agents.pipeline_router_agent import route_pipeline, run as run_router_agent
from core.state import RiskModelingProjectState


class TestRoutePipeline:
    def test_single_structured_table(self):
        classifications = [{
            "file_name": "loan.csv",
            "detected_data_type": DataType.STRUCTURED_MODELING_TABLE,
            "detected_roles": {"label_col": "bad_flag"},
        }]
        result = route_pipeline(classifications)
        assert result["pipeline_type"] == PipelineType.STRUCTURED_MODELING

    def test_single_transaction_table(self):
        classifications = [{
            "file_name": "txn.csv",
            "detected_data_type": DataType.TRANSACTION_FLOW_TABLE,
            "detected_roles": {"transaction_time_col": "txn_time"},
        }]
        result = route_pipeline(classifications)
        assert result["pipeline_type"] == PipelineType.TRANSACTION_FEATURE

    def test_single_main_table_no_txn(self):
        classifications = [{
            "file_name": "main.csv",
            "detected_data_type": DataType.MAIN_TABLE,
            "detected_roles": {"label_col": "bad_flag"},
        }]
        result = route_pipeline(classifications)
        assert result["pipeline_type"] == PipelineType.STRUCTURED_MODELING

    def test_main_plus_transaction(self):
        classifications = [
            {
                "file_name": "loan.csv",
                "detected_data_type": DataType.MAIN_TABLE,
                "detected_roles": {"label_col": "bad_flag"},
            },
            {
                "file_name": "txn.csv",
                "detected_data_type": DataType.TRANSACTION_FLOW_TABLE,
                "detected_roles": {"transaction_time_col": "txn_time"},
            },
        ]
        result = route_pipeline(classifications)
        assert result["pipeline_type"] == PipelineType.MAIN_PLUS_TRANSACTION

    def test_structured_plus_transaction(self):
        classifications = [
            {
                "file_name": "loan.csv",
                "detected_data_type": DataType.STRUCTURED_MODELING_TABLE,
                "detected_roles": {"label_col": "bad_flag"},
            },
            {
                "file_name": "txn.csv",
                "detected_data_type": DataType.TRANSACTION_FLOW_TABLE,
                "detected_roles": {},
            },
        ]
        result = route_pipeline(classifications)
        assert result["pipeline_type"] == PipelineType.MAIN_PLUS_TRANSACTION

    def test_unknown_single_file(self):
        classifications = [{
            "file_name": "mystery.csv",
            "detected_data_type": DataType.UNKNOWN,
            "detected_roles": {},
        }]
        result = route_pipeline(classifications)
        assert result["pipeline_type"] == PipelineType.MANUAL_CONFIGURATION

    def test_empty_classifications(self):
        result = route_pipeline([])
        assert result["pipeline_type"] == PipelineType.MANUAL_CONFIGURATION

    def test_multiple_transaction_only(self):
        classifications = [
            {
                "file_name": "txn1.csv",
                "detected_data_type": DataType.TRANSACTION_FLOW_TABLE,
                "detected_roles": {},
            },
            {
                "file_name": "txn2.csv",
                "detected_data_type": DataType.TRANSACTION_FLOW_TABLE,
                "detected_roles": {},
            },
        ]
        result = route_pipeline(classifications)
        assert result["pipeline_type"] == PipelineType.TRANSACTION_FEATURE


class TestPipelineRouterAgent:
    def test_run(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        state = RiskModelingProjectState(
            project_name="test_proj",
            data_type_classification_result={
                "classifications": [
                    {
                        "file_name": "loan.csv",
                        "detected_data_type": DataType.STRUCTURED_MODELING_TABLE,
                        "detected_roles": {"label_col": "bad_flag"},
                    },
                ],
            },
        )

        result = run_router_agent(state)
        assert result["pipeline_type"] == PipelineType.STRUCTURED_MODELING
        assert "file_roles" in result

    def test_run_multi_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        state = RiskModelingProjectState(
            project_name="test_proj",
            data_type_classification_result={
                "classifications": [
                    {
                        "file_name": "loan.csv",
                        "detected_data_type": DataType.MAIN_TABLE,
                        "detected_roles": {"label_col": "bad_flag"},
                    },
                    {
                        "file_name": "txn.csv",
                        "detected_data_type": DataType.TRANSACTION_FLOW_TABLE,
                        "detected_roles": {"transaction_time_col": "txn_time"},
                    },
                ],
            },
        )

        result = run_router_agent(state)
        assert result["pipeline_type"] == PipelineType.MAIN_PLUS_TRANSACTION
