"""Tests for tools/dictionary_tools.py and agents/data_dictionary_parser_agent.py."""

import os

import pandas as pd
import pytest

from tools.dictionary_tools import (
    parse_data_dictionary,
    validate_dictionary_columns,
    map_dictionary_to_dataset,
)
from agents.data_dictionary_parser_agent import run as run_dictionary_agent
from core.state import RiskModelingProjectState


@pytest.fixture
def dictionary_csv(tmp_path):
    """Create a sample data dictionary CSV."""
    df = pd.DataFrame({
        "table_name": ["loan_app", "loan_app", "loan_app", "loan_app", "txn_flow"],
        "column_name": ["bad_flag", "customer_id", "apply_date", "dpd30", "txn_time"],
        "column_cn_name": ["是否逾期", "客户ID", "申请日期", "30天逾期", "交易时间"],
        "business_meaning": ["放款后90天是否逾期", "客户唯一标识", "申请日期", "30天逾期天数", "交易发生时间"],
        "data_type": ["int", "string", "date", "int", "datetime"],
        "is_label": ["yes", "no", "no", "no", "no"],
        "is_id": ["no", "yes", "no", "no", "no"],
        "is_time": ["no", "no", "yes", "no", "yes"],
        "is_sensitive": ["no", "no", "no", "no", "no"],
        "available_time": ["after_loan", "application_time", "application_time", "after_loan", "transaction_time"],
    })
    path = str(tmp_path / "data_dictionary.csv")
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def invalid_dictionary_csv(tmp_path):
    """Create a dictionary CSV missing required columns."""
    df = pd.DataFrame({
        "field": ["bad_flag", "customer_id"],
        "description": ["label", "id"],
    })
    path = str(tmp_path / "bad_dict.csv")
    df.to_csv(path, index=False)
    return path


class TestValidateDictionaryColumns:
    def test_valid_dictionary(self, dictionary_csv):
        result = validate_dictionary_columns(dictionary_csv)
        assert result["is_valid"] is True
        assert result["missing_required"] == []
        assert "is_label" in result["available_optional"]

    def test_invalid_dictionary(self, invalid_dictionary_csv):
        result = validate_dictionary_columns(invalid_dictionary_csv)
        assert result["is_valid"] is False
        assert "table_name" in result["missing_required"]
        assert "column_name" in result["missing_required"]


class TestParseDictionary:
    def test_parse_roles(self, dictionary_csv):
        result = parse_data_dictionary(dictionary_csv)

        assert result["dictionary_uploaded"] is True
        columns = result["parsed_columns"]
        assert len(columns) == 5

        # Check label detection
        bad_flag = next(c for c in columns if c["column_name"] == "bad_flag")
        assert bad_flag["role"] == "label"
        assert bad_flag["confidence"] == 1.0
        assert bad_flag["from_dictionary"] is True

        # Check ID detection
        cust_id = next(c for c in columns if c["column_name"] == "customer_id")
        assert cust_id["role"] == "customer_key"

        # Check time detection
        apply_date = next(c for c in columns if c["column_name"] == "apply_date")
        assert apply_date["role"] == "observation_time"

        # Check transaction time
        txn_time = next(c for c in columns if c["column_name"] == "txn_time")
        assert txn_time["role"] == "transaction_time"

    def test_risk_detection(self, dictionary_csv):
        result = parse_data_dictionary(dictionary_csv)
        columns = result["parsed_columns"]

        dpd30 = next(c for c in columns if c["column_name"] == "dpd30")
        assert dpd30["risk_level"] == "high"

        # Should have warnings about high-risk fields
        assert len(result["warnings"]) > 0


class TestMapDictionaryToDataset:
    def test_full_coverage(self):
        parsed = [
            {"column_name": "a", "role": "normal"},
            {"column_name": "b", "role": "label"},
        ]
        result = map_dictionary_to_dataset(parsed, ["a", "b"])
        assert result["coverage_rate"] == 1.0
        assert result["unmatched_in_dataset"] == []
        assert result["unmatched_in_dictionary"] == []

    def test_partial_coverage(self):
        parsed = [
            {"column_name": "a", "role": "normal"},
        ]
        result = map_dictionary_to_dataset(parsed, ["a", "b", "c"])
        assert len(result["matched"]) == 1
        assert set(result["unmatched_in_dataset"]) == {"b", "c"}
        assert result["coverage_rate"] == pytest.approx(1 / 3)

    def test_extra_in_dictionary(self):
        parsed = [
            {"column_name": "a", "role": "normal"},
            {"column_name": "x", "role": "normal"},
        ]
        result = map_dictionary_to_dataset(parsed, ["a"])
        assert result["unmatched_in_dictionary"] == ["x"]


class TestDataDictionaryParserAgent:
    def test_run_with_dictionary(self, tmp_path, monkeypatch, dictionary_csv):
        monkeypatch.chdir(tmp_path)
        state = RiskModelingProjectState(
            project_name="test_proj",
            data_dictionary_path=dictionary_csv,
            uploaded_files=[{
                "file_name": "loan_app.csv",
                "columns": ["bad_flag", "customer_id", "apply_date", "income"],
            }],
        )

        result = run_dictionary_agent(state)

        assert result["dictionary_uploaded"] is True
        assert len(result["parsed_columns"]) == 5
        assert result.get("coverage_rate") is not None

    def test_run_without_dictionary(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        state = RiskModelingProjectState(
            project_name="test_proj",
            uploaded_files=[],
        )

        result = run_dictionary_agent(state)

        assert result["dictionary_uploaded"] is False
        assert result["parsed_columns"] == []
