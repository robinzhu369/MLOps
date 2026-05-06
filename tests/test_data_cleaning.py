"""Tests for agents/data_cleaning_planner_agent.py and tools/data_cleaning_tools.py."""

import json
import os

import numpy as np
import pandas as pd
import pytest

from agents.data_cleaning_planner_agent import generate_cleaning_plan, run as run_planner
from tools.data_cleaning_tools import execute_cleaning_plan
from core.state import RiskModelingProjectState


@pytest.fixture
def quality_report():
    """A sample data quality report."""
    return {
        "duplicate_analysis": {
            "duplicate_rows": 10,
            "duplicate_rate": 0.01,
            "duplicate_key_rows": 3,
        },
        "missing_analysis": {
            "total_missing_rate": 0.05,
            "high_missing_columns": ["col_high_miss"],
            "columns": [
                {"column": "age", "missing_rate": 0.08, "missing_count": 80},
                {"column": "income", "missing_rate": 0.15, "missing_count": 150},
                {"column": "col_high_miss", "missing_rate": 0.6, "missing_count": 600},
                {"column": "bad_flag", "missing_rate": 0.0, "missing_count": 0},
            ],
        },
        "outlier_analysis": {
            "columns": [
                {"column": "income", "outlier_rate": 0.05, "lower_bound": 0, "upper_bound": 200000},
                {"column": "age", "outlier_rate": 0.001, "lower_bound": 18, "upper_bound": 80},
            ],
        },
        "type_analysis": {
            "mismatches": [
                {"column": "zip_code", "current_type": "object", "suggested_type": "numeric", "conversion_rate": 0.95},
            ],
        },
        "label_analysis": {"found": True, "positive_rate": 0.15},
        "key_analysis": {},
        "overall_quality_score": 65,
        "need_cleaning_review": True,
    }


@pytest.fixture
def sample_dirty_df():
    """Create a dirty DataFrame for cleaning tests."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "customer_id": list(range(1, n + 1)),
        "age": np.random.randint(18, 70, n).astype(float),
        "income": np.random.normal(50000, 20000, n),
        "zip_code": [str(x) for x in np.random.randint(10000, 99999, n)],
        "col_high_miss": [None] * 60 + list(range(40)),
        "constant_col": [1] * n,
        "bad_flag": np.random.choice([0, 1], n, p=[0.8, 0.2]),
    })
    # Inject issues
    df.loc[0:4, "age"] = np.nan
    df.loc[10:14, "income"] = np.nan
    df.loc[50, "income"] = 999999  # outlier
    # Add duplicate rows
    df = pd.concat([df, df.iloc[[0, 1]]], ignore_index=True)
    return df


@pytest.fixture
def dirty_csv_path(tmp_path, sample_dirty_df):
    path = str(tmp_path / "dirty_data.csv")
    sample_dirty_df.to_csv(path, index=False)
    return path


class TestGenerateCleaningPlan:
    def test_basic_plan(self, quality_report):
        plan = generate_cleaning_plan(
            quality_report, label_col="bad_flag", id_col="customer_id"
        )

        assert plan["n_steps"] > 0
        assert "bad_flag" in plan["protected_columns"]
        assert "customer_id" in plan["protected_columns"]

        actions = [s["action"] for s in plan["cleaning_steps"]]
        assert "drop_exact_duplicates" in actions
        assert "drop_high_missing_columns" in actions

    def test_protected_columns_not_dropped(self, quality_report):
        # Even if label has high missing, it should not be dropped
        quality_report["missing_analysis"]["high_missing_columns"].append("bad_flag")
        plan = generate_cleaning_plan(
            quality_report, label_col="bad_flag"
        )

        drop_step = next(
            (s for s in plan["cleaning_steps"] if s["action"] == "drop_high_missing_columns"),
            None,
        )
        if drop_step:
            assert "bad_flag" not in drop_step["params"]["columns"]

    def test_no_issues_minimal_plan(self):
        clean_report = {
            "duplicate_analysis": {"duplicate_rows": 0, "duplicate_rate": 0.0, "duplicate_key_rows": 0},
            "missing_analysis": {"total_missing_rate": 0.0, "high_missing_columns": [], "columns": []},
            "outlier_analysis": {"columns": []},
            "type_analysis": {"mismatches": []},
            "label_analysis": {},
            "key_analysis": {},
            "overall_quality_score": 95,
            "need_cleaning_review": False,
        }
        plan = generate_cleaning_plan(clean_report)
        # Should still have drop_constant_columns as a safety step
        actions = [s["action"] for s in plan["cleaning_steps"]]
        assert "drop_constant_columns" in actions

    def test_outlier_winsorize(self, quality_report):
        plan = generate_cleaning_plan(quality_report, label_col="bad_flag")
        winsorize_step = next(
            (s for s in plan["cleaning_steps"] if s["action"] == "winsorize_outliers"),
            None,
        )
        assert winsorize_step is not None
        # income has outlier_rate > 0.01, age does not
        cols = [c["column"] for c in winsorize_step["params"]["columns"]]
        assert "income" in cols

    def test_steps_sorted_by_priority(self, quality_report):
        plan = generate_cleaning_plan(quality_report, label_col="bad_flag")
        priorities = [s["priority"] for s in plan["cleaning_steps"]]
        assert priorities == sorted(priorities)


class TestExecuteCleaningPlan:
    def test_full_execution(self, dirty_csv_path, quality_report, tmp_path):
        plan = generate_cleaning_plan(
            quality_report, label_col="bad_flag", id_col="customer_id"
        )

        result = execute_cleaning_plan(
            data_path=dirty_csv_path,
            cleaning_plan=plan,
            project_name="test_proj",
            output_dir=str(tmp_path / "output"),
        )

        assert os.path.exists(result["output_path"])
        assert result["before_shape"]["rows"] == 102  # 100 + 2 duplicates
        assert result["after_shape"]["rows"] <= result["before_shape"]["rows"]
        assert result["after_shape"]["cols"] <= result["before_shape"]["cols"] + 5  # indicators added

        # Original file unchanged
        original_df = pd.read_csv(dirty_csv_path)
        assert len(original_df) == 102

    def test_drop_exact_duplicates(self, dirty_csv_path, tmp_path):
        plan = {
            "cleaning_steps": [{"action": "drop_exact_duplicates", "priority": 1, "params": {}}],
            "protected_columns": [],
        }
        result = execute_cleaning_plan(dirty_csv_path, plan, "test", str(tmp_path / "out"))
        assert result["after_shape"]["rows"] < result["before_shape"]["rows"]

    def test_drop_key_duplicates(self, dirty_csv_path, tmp_path):
        plan = {
            "cleaning_steps": [{
                "action": "drop_key_duplicates",
                "priority": 1,
                "params": {"key_columns": ["customer_id"], "keep": "first"},
            }],
            "protected_columns": [],
        }
        result = execute_cleaning_plan(dirty_csv_path, plan, "test", str(tmp_path / "out"))
        cleaned_df = pd.read_csv(result["output_path"])
        assert cleaned_df["customer_id"].is_unique

    def test_fill_missing_values(self, dirty_csv_path, tmp_path):
        plan = {
            "cleaning_steps": [{
                "action": "fill_missing_values",
                "priority": 1,
                "params": {"columns": ["age", "income"], "numeric_strategy": "median", "categorical_strategy": "mode"},
            }],
            "protected_columns": ["bad_flag"],
        }
        result = execute_cleaning_plan(dirty_csv_path, plan, "test", str(tmp_path / "out"))
        cleaned_df = pd.read_csv(result["output_path"])
        assert cleaned_df["age"].isna().sum() == 0
        assert cleaned_df["income"].isna().sum() == 0

    def test_add_missing_indicator(self, dirty_csv_path, tmp_path):
        plan = {
            "cleaning_steps": [{
                "action": "add_missing_indicator",
                "priority": 1,
                "params": {"columns": ["age", "income"]},
            }],
            "protected_columns": [],
        }
        result = execute_cleaning_plan(dirty_csv_path, plan, "test", str(tmp_path / "out"))
        cleaned_df = pd.read_csv(result["output_path"])
        assert "age_missing" in cleaned_df.columns
        assert "income_missing" in cleaned_df.columns
        assert cleaned_df["age_missing"].sum() > 0

    def test_winsorize_outliers(self, dirty_csv_path, tmp_path):
        plan = {
            "cleaning_steps": [{
                "action": "winsorize_outliers",
                "priority": 1,
                "params": {"columns": [{"column": "income", "lower_bound": 0, "upper_bound": 200000}]},
            }],
            "protected_columns": [],
        }
        result = execute_cleaning_plan(dirty_csv_path, plan, "test", str(tmp_path / "out"))
        cleaned_df = pd.read_csv(result["output_path"])
        assert cleaned_df["income"].max() <= 200000

    def test_drop_constant_columns(self, dirty_csv_path, tmp_path):
        plan = {
            "cleaning_steps": [{
                "action": "drop_constant_columns",
                "priority": 1,
                "params": {"protected_columns": ["bad_flag"]},
            }],
            "protected_columns": ["bad_flag"],
        }
        result = execute_cleaning_plan(dirty_csv_path, plan, "test", str(tmp_path / "out"))
        cleaned_df = pd.read_csv(result["output_path"])
        assert "constant_col" not in cleaned_df.columns

    def test_convert_types(self, dirty_csv_path, tmp_path):
        plan = {
            "cleaning_steps": [{
                "action": "convert_types",
                "priority": 1,
                "params": {"conversions": [{"column": "zip_code", "suggested_type": "numeric"}]},
            }],
            "protected_columns": [],
        }
        result = execute_cleaning_plan(dirty_csv_path, plan, "test", str(tmp_path / "out"))
        cleaned_df = pd.read_csv(result["output_path"])
        assert cleaned_df["zip_code"].dtype in (np.float64, np.int64)

    def test_cleaning_log_written(self, dirty_csv_path, tmp_path):
        plan = {
            "cleaning_steps": [{"action": "drop_exact_duplicates", "priority": 1, "params": {}}],
            "protected_columns": [],
        }
        result = execute_cleaning_plan(dirty_csv_path, plan, "test_proj", str(tmp_path / "out"))

        log_path = os.path.join("artifacts", "test_proj", "cleaning_log.json")
        assert os.path.exists(log_path)

    def test_unknown_action_error(self, dirty_csv_path, tmp_path):
        plan = {
            "cleaning_steps": [{"action": "unknown_action", "priority": 1, "params": {}}],
            "protected_columns": [],
        }
        result = execute_cleaning_plan(dirty_csv_path, plan, "test", str(tmp_path / "out"))
        assert result["executed_steps"][0]["status"] == "error"


class TestDataCleaningPlannerAgent:
    def test_run(self, tmp_path, monkeypatch, quality_report):
        monkeypatch.chdir(tmp_path)
        state = RiskModelingProjectState(
            project_name="test_proj",
            data_quality_report=quality_report,
            label_col="bad_flag",
            id_col="customer_id",
        )

        result = run_planner(state)
        assert result["n_steps"] > 0
        assert "bad_flag" in result["protected_columns"]

    def test_run_no_report(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        state = RiskModelingProjectState(project_name="test_proj")

        result = run_planner(state)
        assert result.get("error") == "no_quality_report"
