"""Tests for tools/data_type_tools.py and agents/data_type_classifier_agent.py."""

import json

import pytest

from core.constants import DataType, PipelineType
from tools.data_type_tools import (
    classify_data_type_by_rules,
    classify_data_type_by_llm,
    merge_data_type_classification,
)
from agents.data_type_classifier_agent import run as run_classifier_agent
from core.state import RiskModelingProjectState


@pytest.fixture
def transaction_columns():
    return [
        "transaction_id", "account_id", "customer_id",
        "transaction_time", "transaction_amount", "debit_credit_flag",
        "merchant_category", "channel", "balance_after_txn",
    ]


@pytest.fixture
def transaction_profiles():
    return {
        "transaction_id": {"dtype": "object", "unique_rate": 0.99, "missing_rate": 0.0, "unique_count": 9900},
        "account_id": {"dtype": "object", "unique_rate": 0.1, "missing_rate": 0.0, "unique_count": 1000},
        "customer_id": {"dtype": "object", "unique_rate": 0.08, "missing_rate": 0.0, "unique_count": 800},
        "transaction_time": {"dtype": "object", "unique_rate": 0.95, "missing_rate": 0.0, "unique_count": 9500},
        "transaction_amount": {"dtype": "float64", "unique_rate": 0.85, "missing_rate": 0.0, "unique_count": 8500},
        "debit_credit_flag": {"dtype": "object", "unique_rate": 0.0002, "missing_rate": 0.0, "unique_count": 2},
        "merchant_category": {"dtype": "object", "unique_rate": 0.005, "missing_rate": 0.02, "unique_count": 50},
        "channel": {"dtype": "object", "unique_rate": 0.0005, "missing_rate": 0.0, "unique_count": 5},
        "balance_after_txn": {"dtype": "float64", "unique_rate": 0.9, "missing_rate": 0.01, "unique_count": 9000},
    }


@pytest.fixture
def modeling_columns():
    return [
        "customer_id", "age", "income", "loan_amount",
        "apply_date", "bad_flag",
    ]


@pytest.fixture
def modeling_profiles():
    return {
        "customer_id": {"dtype": "int64", "unique_rate": 1.0, "missing_rate": 0.0, "unique_count": 10000},
        "age": {"dtype": "int64", "unique_rate": 0.05, "missing_rate": 0.02, "unique_count": 50},
        "income": {"dtype": "float64", "unique_rate": 0.8, "missing_rate": 0.05, "unique_count": 8000},
        "loan_amount": {"dtype": "float64", "unique_rate": 0.6, "missing_rate": 0.0, "unique_count": 6000},
        "apply_date": {"dtype": "object", "unique_rate": 0.2, "missing_rate": 0.0, "unique_count": 200},
        "bad_flag": {"dtype": "int64", "unique_rate": 0.0002, "missing_rate": 0.0, "unique_count": 2},
    }


class TestClassifyByRules:
    def test_transaction_flow(self, transaction_columns, transaction_profiles):
        result = classify_data_type_by_rules(
            "transaction_flow.csv", transaction_columns, transaction_profiles
        )
        assert result["detected_data_type"] == DataType.TRANSACTION_FLOW_TABLE
        assert result["confidence"] > 0.5
        assert result["detected_roles"]["transaction_time_col"] is not None
        assert result["detected_roles"]["amount_col"] is not None

    def test_structured_modeling(self, modeling_columns, modeling_profiles):
        result = classify_data_type_by_rules(
            "loan_application.csv", modeling_columns, modeling_profiles
        )
        assert result["detected_data_type"] == DataType.STRUCTURED_MODELING_TABLE
        assert result["detected_roles"]["label_col"] == "bad_flag"

    def test_with_field_semantics(self, modeling_columns, modeling_profiles):
        semantics = {
            "bad_flag": {"role": "label"},
            "apply_date": {"role": "observation_time"},
            "customer_id": {"role": "customer_key"},
        }
        result = classify_data_type_by_rules(
            "loan.csv", modeling_columns, modeling_profiles, field_semantics=semantics
        )
        assert result["detected_data_type"] == DataType.STRUCTURED_MODELING_TABLE
        assert result["detected_roles"]["label_col"] == "bad_flag"
        assert result["detected_roles"]["base_time_col"] == "apply_date"

    def test_unknown_type(self):
        columns = ["col_a", "col_b", "col_c"]
        profiles = {
            "col_a": {"dtype": "object", "unique_rate": 0.5, "missing_rate": 0.0, "unique_count": 50},
            "col_b": {"dtype": "float64", "unique_rate": 0.8, "missing_rate": 0.0, "unique_count": 80},
            "col_c": {"dtype": "int64", "unique_rate": 0.3, "missing_rate": 0.0, "unique_count": 30},
        }
        result = classify_data_type_by_rules("unknown.csv", columns, profiles)
        # Should still return something, likely auxiliary or unknown
        assert result["detected_data_type"] in (
            DataType.AUXILIARY_TABLE, DataType.UNKNOWN,
            DataType.STRUCTURED_MODELING_TABLE, DataType.TRANSACTION_FLOW_TABLE,
        )


class TestClassifyByLlm:
    def test_with_mock_llm(self, transaction_columns, transaction_profiles):
        mock_response = json.dumps({
            "detected_data_type": "transaction_flow_table",
            "confidence": 0.92,
            "reasoning_summary": "包含交易时间、金额、借贷标志等典型流水字段",
        })

        def mock_llm(prompt):
            return mock_response

        result = classify_data_type_by_llm(
            file_name="txn.csv",
            n_rows=10000,
            n_cols=len(transaction_columns),
            columns=transaction_columns,
            llm_call=mock_llm,
        )
        assert result["detected_data_type"] == "transaction_flow_table"
        assert result["confidence"] == 0.92

    def test_without_llm(self, modeling_columns, modeling_profiles):
        result = classify_data_type_by_llm(
            file_name="loan.csv",
            n_rows=10000,
            n_cols=len(modeling_columns),
            columns=modeling_columns,
            llm_call=None,
        )
        # Without LLM, returns unknown with low confidence
        assert result["confidence"] == 0.0


class TestMergeClassification:
    def test_agreement(self):
        rule = {
            "file_name": "txn.csv",
            "detected_data_type": DataType.TRANSACTION_FLOW_TABLE,
            "confidence": 0.8,
            "reasoning_summary": "规则识别",
            "detected_roles": {"amount_col": "txn_amt"},
        }
        llm = {
            "file_name": "txn.csv",
            "detected_data_type": "transaction_flow_table",
            "confidence": 0.9,
            "reasoning": "LLM识别",
            "detected_roles": {},
        }
        merged = merge_data_type_classification(rule, llm)
        assert merged["detected_data_type"] == DataType.TRANSACTION_FLOW_TABLE
        assert merged["confidence"] >= 0.8
        assert merged["need_human_review"] is False

    def test_disagreement(self):
        rule = {
            "file_name": "data.csv",
            "detected_data_type": DataType.STRUCTURED_MODELING_TABLE,
            "confidence": 0.6,
            "reasoning_summary": "规则识别",
            "detected_roles": {},
        }
        llm = {
            "file_name": "data.csv",
            "detected_data_type": "transaction_flow_table",
            "confidence": 0.7,
            "reasoning": "LLM识别",
            "detected_roles": {},
        }
        merged = merge_data_type_classification(rule, llm)
        assert merged["need_human_review"] is True

    def test_low_confidence(self):
        rule = {
            "file_name": "data.csv",
            "detected_data_type": DataType.AUXILIARY_TABLE,
            "confidence": 0.4,
            "reasoning_summary": "不确定",
            "detected_roles": {},
        }
        llm = {
            "file_name": "data.csv",
            "detected_data_type": "auxiliary_table",
            "confidence": 0.3,
            "reasoning": "不确定",
            "detected_roles": {},
        }
        merged = merge_data_type_classification(rule, llm)
        assert merged["need_human_review"] is True


class TestDataTypeClassifierAgent:
    def test_run_transaction_file(self, tmp_path, monkeypatch, transaction_columns, transaction_profiles):
        monkeypatch.chdir(tmp_path)
        state = RiskModelingProjectState(
            project_name="test_proj",
            uploaded_files=[{
                "file_name": "txn.csv",
                "columns": transaction_columns,
                "column_profiles": transaction_profiles,
                "n_rows": 10000,
                "n_cols": len(transaction_columns),
            }],
            field_semantics={},
        )

        result = run_classifier_agent(state)

        assert len(result["classifications"]) == 1
        assert result["classifications"][0]["detected_data_type"] == DataType.TRANSACTION_FLOW_TABLE

    def test_run_modeling_file(self, tmp_path, monkeypatch, modeling_columns, modeling_profiles):
        monkeypatch.chdir(tmp_path)
        state = RiskModelingProjectState(
            project_name="test_proj",
            uploaded_files=[{
                "file_name": "loan.csv",
                "columns": modeling_columns,
                "column_profiles": modeling_profiles,
                "n_rows": 10000,
                "n_cols": len(modeling_columns),
            }],
            field_semantics={},
        )

        result = run_classifier_agent(state)

        assert len(result["classifications"]) == 1
        assert result["classifications"][0]["detected_data_type"] == DataType.STRUCTURED_MODELING_TABLE

    def test_run_no_files(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        state = RiskModelingProjectState(
            project_name="test_proj",
            uploaded_files=[],
        )

        result = run_classifier_agent(state)
        assert result["classifications"] == []
