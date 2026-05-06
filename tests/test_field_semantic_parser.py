"""Tests for tools/field_semantic_tools.py and agents/field_semantic_parser_agent.py."""

import json

import pytest

from tools.field_semantic_tools import (
    build_field_info_text,
    build_semantic_prompt,
    parse_llm_response,
    parse_field_semantics_by_llm,
    merge_dictionary_and_llm_semantics,
    _rule_based_field_parsing,
)
from agents.field_semantic_parser_agent import run as run_semantic_agent
from core.state import RiskModelingProjectState


@pytest.fixture
def sample_profiles():
    return {
        "customer_id": {
            "dtype": "int64",
            "missing_rate": 0.0,
            "unique_count": 1000,
            "unique_rate": 1.0,
        },
        "bad_flag": {
            "dtype": "int64",
            "missing_rate": 0.0,
            "unique_count": 2,
            "unique_rate": 0.002,
        },
        "apply_date": {
            "dtype": "object",
            "missing_rate": 0.0,
            "unique_count": 200,
            "unique_rate": 0.2,
        },
        "income": {
            "dtype": "float64",
            "missing_rate": 0.05,
            "unique_count": 800,
            "unique_rate": 0.8,
        },
        "transaction_amount": {
            "dtype": "float64",
            "missing_rate": 0.0,
            "unique_count": 950,
            "unique_rate": 0.95,
        },
        "dpd30": {
            "dtype": "int64",
            "missing_rate": 0.0,
            "unique_count": 10,
            "unique_rate": 0.01,
        },
    }


@pytest.fixture
def sample_values():
    return {
        "customer_id": ["1001", "1002", "1003"],
        "bad_flag": ["0", "1", "0"],
        "apply_date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "income": ["50000.0", "60000.0", "75000.0"],
        "transaction_amount": ["100.5", "2500.0", "50.0"],
        "dpd30": ["0", "5", "30"],
    }


class TestBuildFieldInfo:
    def test_basic_output(self, sample_profiles, sample_values):
        text = build_field_info_text(sample_profiles, sample_values)
        assert "customer_id" in text
        assert "bad_flag" in text
        assert "类型=" in text
        assert "缺失率=" in text

    def test_includes_samples(self, sample_profiles, sample_values):
        text = build_field_info_text(sample_profiles, sample_values)
        assert "样例值=" in text


class TestBuildSemanticPrompt:
    def test_prompt_structure(self, sample_profiles, sample_values):
        prompt = build_semantic_prompt("test.csv", 1000, 6, sample_profiles, sample_values)
        assert "test.csv" in prompt
        assert "1000" in prompt
        assert "customer_id" in prompt
        assert "JSON" in prompt


class TestParseLlmResponse:
    def test_valid_json(self):
        response = json.dumps({
            "field_semantics": {
                "bad_flag": {"role": "label", "confidence": 0.95, "business_meaning": "是否逾期"}
            }
        })
        result = parse_llm_response(response)
        assert "bad_flag" in result["field_semantics"]
        assert result["field_semantics"]["bad_flag"]["role"] == "label"

    def test_json_in_code_block(self):
        response = '```json\n{"field_semantics": {"col1": {"role": "normal", "confidence": 0.8}}}\n```'
        result = parse_llm_response(response)
        assert "col1" in result["field_semantics"]

    def test_invalid_json(self):
        result = parse_llm_response("not json at all")
        assert result["field_semantics"] == {}

    def test_json_without_wrapper(self):
        response = json.dumps({
            "bad_flag": {"role": "label", "confidence": 0.9}
        })
        result = parse_llm_response(response)
        assert "bad_flag" in result["field_semantics"]


class TestRuleBasedParsing:
    def test_detects_label(self, sample_profiles, sample_values):
        result = _rule_based_field_parsing(sample_profiles, sample_values)
        assert result["bad_flag"]["role"] == "label"

    def test_detects_id(self, sample_profiles, sample_values):
        result = _rule_based_field_parsing(sample_profiles, sample_values)
        assert result["customer_id"]["role"] == "customer_key"

    def test_detects_time(self, sample_profiles, sample_values):
        result = _rule_based_field_parsing(sample_profiles, sample_values)
        assert result["apply_date"]["role"] == "observation_time"

    def test_detects_amount(self, sample_profiles, sample_values):
        result = _rule_based_field_parsing(sample_profiles, sample_values)
        assert result["transaction_amount"]["role"] == "amount"

    def test_detects_leakage(self, sample_profiles, sample_values):
        result = _rule_based_field_parsing(sample_profiles, sample_values)
        assert result["dpd30"]["role"] == "possible_leakage_feature"


class TestParseFieldSemanticsByLlm:
    def test_with_mock_llm(self, sample_profiles, sample_values):
        mock_response = json.dumps({
            "field_semantics": {
                "bad_flag": {"role": "label", "confidence": 0.95, "business_meaning": "逾期标签"},
                "customer_id": {"role": "customer_key", "confidence": 0.99, "business_meaning": "客户ID"},
            }
        })

        def mock_llm(prompt):
            return mock_response

        result = parse_field_semantics_by_llm(
            "test.csv", 1000, 6, sample_profiles, sample_values, llm_call=mock_llm
        )

        assert result["dictionary_uploaded"] is False
        assert "bad_flag" in result["field_semantics"]
        assert result["field_semantics"]["bad_flag"]["role"] == "label"

    def test_without_llm_uses_rules(self, sample_profiles, sample_values):
        result = parse_field_semantics_by_llm(
            "test.csv", 1000, 6, sample_profiles, sample_values, llm_call=None
        )

        assert result["dictionary_uploaded"] is False
        assert len(result["field_semantics"]) > 0
        assert result["field_semantics"]["bad_flag"]["role"] == "label"


class TestMergeDictionaryAndLlm:
    def test_dictionary_takes_priority(self):
        dict_result = {
            "parsed_columns": [
                {"column_name": "bad_flag", "role": "label", "confidence": 1.0, "risk_level": "normal"},
            ]
        }
        llm_result = {
            "field_semantics": {
                "bad_flag": {"role": "normal", "confidence": 0.5},
                "income": {"role": "normal", "confidence": 0.9},
            }
        }

        merged = merge_dictionary_and_llm_semantics(dict_result, llm_result)
        semantics = merged["field_semantics"]

        assert semantics["bad_flag"]["role"] == "label"
        assert semantics["bad_flag"]["from_dictionary"] is True
        assert semantics["income"]["role"] == "normal"
        assert semantics["income"]["from_dictionary"] is False


class TestFieldSemanticParserAgent:
    def test_run_with_files(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        state = RiskModelingProjectState(
            project_name="test_proj",
            uploaded_files=[{
                "file_name": "loan.csv",
                "n_rows": 100,
                "n_cols": 4,
                "column_profiles": {
                    "customer_id": {"dtype": "int64", "missing_rate": 0.0, "unique_count": 100, "unique_rate": 1.0},
                    "bad_flag": {"dtype": "int64", "missing_rate": 0.0, "unique_count": 2, "unique_rate": 0.02},
                },
                "sample_values_masked": {
                    "customer_id": ["1", "2", "3"],
                    "bad_flag": ["0", "1", "0"],
                },
            }],
        )

        result = run_semantic_agent(state, llm_call=None)

        assert result["dictionary_uploaded"] is False
        assert len(result["field_semantics"]) > 0

    def test_run_no_files(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        state = RiskModelingProjectState(
            project_name="test_proj",
            uploaded_files=[],
        )

        result = run_semantic_agent(state, llm_call=None)

        assert result["field_semantics"] == {}
