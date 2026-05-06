"""Tests for tools/quality_tools.py and agents/data_quality_agent.py."""

import numpy as np
import pandas as pd
import pytest

from tools.quality_tools import (
    analyze_duplicates,
    analyze_missing_values,
    analyze_outliers,
    analyze_type_mismatch,
    analyze_label_quality,
    analyze_key_quality,
    generate_data_quality_report,
)
from agents.data_quality_agent import run as run_quality_agent
from core.state import RiskModelingProjectState


@pytest.fixture
def sample_df():
    """Create a sample DataFrame with various quality issues."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "customer_id": list(range(1, n + 1)),
        "age": np.random.randint(18, 70, n).astype(float),
        "income": np.random.normal(50000, 20000, n),
        "score": np.random.uniform(300, 850, n),
        "apply_date": ["2024-01-01"] * n,
        "bad_flag": np.random.choice([0, 1], n, p=[0.8, 0.2]),
    })
    # Inject issues
    df.loc[0:2, "age"] = np.nan  # 3 missing
    df.loc[50, "income"] = 999999  # outlier
    df.loc[51, "income"] = -50000  # outlier
    # Add a duplicate row
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    return df


@pytest.fixture
def sample_csv_path(tmp_path, sample_df):
    path = str(tmp_path / "sample.csv")
    sample_df.to_csv(path, index=False)
    return path


class TestAnalyzeDuplicates:
    def test_with_duplicates(self, sample_df):
        result = analyze_duplicates(sample_df)
        assert result["duplicate_rows"] >= 1
        assert result["duplicate_rate"] > 0

    def test_no_duplicates(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = analyze_duplicates(df)
        assert result["duplicate_rows"] == 0
        assert result["duplicate_rate"] == 0.0

    def test_key_duplicates(self):
        df = pd.DataFrame({"id": [1, 1, 2, 3], "val": [10, 20, 30, 40]})
        result = analyze_duplicates(df, key_columns=["id"])
        assert result["duplicate_key_rows"] == 1


class TestAnalyzeMissingValues:
    def test_with_missing(self, sample_df):
        result = analyze_missing_values(sample_df)
        age_info = next(c for c in result["columns"] if c["column"] == "age")
        # sample_df has 101 rows (100 + 1 duplicate of row[0] which has NaN age)
        assert age_info["missing_count"] == 4
        assert age_info["missing_rate"] > 0

    def test_no_missing(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = analyze_missing_values(df)
        assert result["total_missing_rate"] == 0.0
        assert result["high_missing_columns"] == []

    def test_high_missing(self):
        df = pd.DataFrame({"a": [1, None, None, None, None]})
        result = analyze_missing_values(df, high_missing_threshold=0.3)
        assert "a" in result["high_missing_columns"]


class TestAnalyzeOutliers:
    def test_iqr_method(self, sample_df):
        result = analyze_outliers(sample_df, method="IQR")
        assert len(result["columns"]) > 0
        income_info = next((c for c in result["columns"] if c["column"] == "income"), None)
        assert income_info is not None
        assert income_info["outlier_count"] > 0

    def test_protected_columns(self, sample_df):
        result = analyze_outliers(sample_df, protected_columns=["bad_flag", "customer_id"])
        col_names = [c["column"] for c in result["columns"]]
        assert "bad_flag" not in col_names
        assert "customer_id" not in col_names

    def test_zscore_method(self, sample_df):
        result = analyze_outliers(sample_df, method="zscore")
        assert result["method"] == "zscore"


class TestAnalyzeTypeMismatch:
    def test_numeric_as_string(self):
        df = pd.DataFrame({"num_col": ["1", "2", "3", "4", "5"]})
        result = analyze_type_mismatch(df)
        assert len(result["mismatches"]) == 1
        assert result["mismatches"][0]["suggested_type"] == "numeric"

    def test_no_mismatch(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        result = analyze_type_mismatch(df)
        assert len(result["mismatches"]) == 0


class TestAnalyzeLabelQuality:
    def test_binary_label(self, sample_df):
        result = analyze_label_quality(sample_df, "bad_flag")
        assert result["found"] is True
        assert len(result["unique_values"]) == 2
        assert 0.1 < result["positive_rate"] < 0.4

    def test_missing_label(self):
        df = pd.DataFrame({"label": [0, 1, None, 0, 1]})
        result = analyze_label_quality(df, "label")
        assert result["missing_count"] == 1

    def test_nonexistent_label(self):
        df = pd.DataFrame({"a": [1, 2, 3]})
        result = analyze_label_quality(df, "nonexistent")
        assert result["found"] is False


class TestAnalyzeKeyQuality:
    def test_unique_key(self):
        df = pd.DataFrame({"id": [1, 2, 3, 4, 5], "val": [10, 20, 30, 40, 50]})
        result = analyze_key_quality(df, {"id_col": "id"})
        assert result["id_col"]["found"] is True
        assert result["id_col"]["unique_rate"] == 1.0

    def test_non_unique_key(self):
        df = pd.DataFrame({"id": [1, 1, 2, 3, 3], "val": [10, 20, 30, 40, 50]})
        result = analyze_key_quality(df, {"id_col": "id"})
        assert result["id_col"]["unique_rate"] < 1.0
        assert len(result["id_col"]["warnings"]) > 0

    def test_null_in_key(self):
        df = pd.DataFrame({"id": [1, 2, None, 4, 5]})
        result = analyze_key_quality(df, {"id_col": "id"})
        assert result["id_col"]["missing_count"] == 1


class TestGenerateReport:
    def test_report_structure(self):
        report = generate_data_quality_report(
            duplicate_analysis={"duplicate_rows": 5, "duplicate_rate": 0.05},
            missing_analysis={"total_missing_rate": 0.02, "high_missing_columns": []},
            outlier_analysis={"columns": [{"column": "x", "outlier_rate": 0.01}]},
            type_analysis={"mismatches": []},
            label_analysis={"is_binary": True, "positive_rate": 0.15},
            key_analysis={"is_unique": True},
        )
        assert "overall_quality_score" in report
        assert "need_cleaning_review" in report
        assert 0 <= report["overall_quality_score"] <= 100

    def test_low_quality_score(self):
        report = generate_data_quality_report(
            duplicate_analysis={"duplicate_rows": 500, "duplicate_rate": 0.5},
            missing_analysis={"total_missing_rate": 0.3, "high_missing_columns": ["a", "b", "c"]},
            outlier_analysis={"columns": [{"column": "x", "outlier_rate": 0.2}]},
            type_analysis={"mismatches": [{"column": "y"}]},
            label_analysis={"is_binary": True, "positive_rate": 0.01},
            key_analysis={"is_unique": False},
        )
        assert report["overall_quality_score"] < 70
        assert report["need_cleaning_review"] is True


class TestDataQualityAgent:
    def test_run(self, tmp_path, monkeypatch, sample_csv_path):
        monkeypatch.chdir(tmp_path)
        state = RiskModelingProjectState(
            project_name="test_proj",
            main_data_path=sample_csv_path,
            label_col="bad_flag",
            id_col="customer_id",
        )

        result = run_quality_agent(state)

        assert "overall_quality_score" in result
        assert "need_cleaning_review" in result
        assert "duplicate_analysis" in result
        assert "missing_analysis" in result
        assert "outlier_analysis" in result

    def test_run_from_uploaded_files(self, tmp_path, monkeypatch, sample_csv_path):
        monkeypatch.chdir(tmp_path)
        state = RiskModelingProjectState(
            project_name="test_proj",
            uploaded_files=[{"file_path": sample_csv_path, "file_name": "sample.csv"}],
        )

        result = run_quality_agent(state)
        assert "overall_quality_score" in result

    def test_run_no_data(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        state = RiskModelingProjectState(
            project_name="test_proj",
        )

        result = run_quality_agent(state)
        assert result.get("error") == "no_data_file"
