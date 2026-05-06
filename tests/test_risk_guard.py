"""Tests for tools/risk_guard_tools.py and agents/risk_guard_agent.py."""

import numpy as np
import pandas as pd
import pytest

from tools.risk_guard_tools import (
    detect_leakage_columns,
    detect_id_columns,
    detect_high_missing_columns,
    detect_time_leakage_candidates,
)
from agents.risk_guard_agent import run as run_risk_guard
from core.state import RiskModelingProjectState


@pytest.fixture
def risky_df():
    """DataFrame with various risky columns."""
    np.random.seed(42)
    n = 100
    return pd.DataFrame({
        "customer_id": [f"C{i:05d}" for i in range(n)],
        "apply_date": pd.date_range("2024-01-01", periods=n).astype(str),
        "age": np.random.randint(18, 65, n),
        "income": np.random.normal(50000, 15000, n),
        "bad_flag": np.random.choice([0, 1], n),
        # Leakage columns
        "dpd_30": np.random.randint(0, 5, n),
        "overdue_days": np.random.randint(0, 90, n),
        "逾期金额": np.random.uniform(0, 10000, n),
        # ID columns
        "loan_id": [f"L{i:08d}" for i in range(n)],
        "app_id": [f"APP{i}" for i in range(n)],
        # High missing
        "sparse_col": [None] * 80 + list(range(20)),
        # Time leakage
        "还款日期": pd.date_range("2024-06-01", periods=n).astype(str),
        "settle_date": pd.date_range("2024-07-01", periods=n).astype(str),
    })


@pytest.fixture
def risky_csv_path(tmp_path, risky_df):
    path = str(tmp_path / "risky_data.csv")
    risky_df.to_csv(path, index=False)
    return path


class TestDetectLeakageColumns:
    def test_detects_dpd(self, risky_df):
        result = detect_leakage_columns(risky_df, label_col="bad_flag")
        cols = [c["column"] for c in result["leakage_columns"]]
        assert "dpd_30" in cols

    def test_detects_overdue(self, risky_df):
        result = detect_leakage_columns(risky_df, label_col="bad_flag")
        cols = [c["column"] for c in result["leakage_columns"]]
        assert "overdue_days" in cols

    def test_detects_chinese_keywords(self, risky_df):
        result = detect_leakage_columns(risky_df, label_col="bad_flag")
        cols = [c["column"] for c in result["leakage_columns"]]
        assert "逾期金额" in cols

    def test_excludes_label_col(self, risky_df):
        # If label_col itself contains a keyword, it should not be flagged
        df = risky_df.rename(columns={"bad_flag": "default_flag"})
        result = detect_leakage_columns(df, label_col="default_flag")
        cols = [c["column"] for c in result["leakage_columns"]]
        assert "default_flag" not in cols

    def test_extra_keywords(self, risky_df):
        risky_df["custom_risk_field"] = 0
        result = detect_leakage_columns(
            risky_df, label_col="bad_flag", extra_keywords=["custom_risk"]
        )
        cols = [c["column"] for c in result["leakage_columns"]]
        assert "custom_risk_field" in cols

    def test_no_leakage(self):
        df = pd.DataFrame({"age": [25, 30], "income": [50000, 60000], "label": [0, 1]})
        result = detect_leakage_columns(df, label_col="label")
        assert result["n_detected"] == 0


class TestDetectIdColumns:
    def test_detects_id_by_name(self, risky_df):
        result = detect_id_columns(risky_df, protected_columns=["customer_id"])
        cols = [c["column"] for c in result["id_columns"]]
        assert "loan_id" in cols
        assert "app_id" in cols

    def test_excludes_protected(self, risky_df):
        result = detect_id_columns(risky_df, protected_columns=["customer_id", "loan_id"])
        cols = [c["column"] for c in result["id_columns"]]
        assert "customer_id" not in cols
        assert "loan_id" not in cols

    def test_detects_high_cardinality_string(self):
        n = 100
        df = pd.DataFrame({
            "mystery_col": [f"unique_{i}" for i in range(n)],
            "normal_col": np.random.choice(["A", "B", "C"], n),
        })
        result = detect_id_columns(df)
        cols = [c["column"] for c in result["id_columns"]]
        assert "mystery_col" in cols
        assert "normal_col" not in cols


class TestDetectHighMissingColumns:
    def test_detects_high_missing(self, risky_df):
        result = detect_high_missing_columns(risky_df, threshold=0.3)
        cols = [c["column"] for c in result["high_missing_columns"]]
        assert "sparse_col" in cols

    def test_respects_threshold(self, risky_df):
        # With very high threshold, nothing should be flagged
        result = detect_high_missing_columns(risky_df, threshold=0.99)
        assert result["n_detected"] == 0

    def test_excludes_protected(self, risky_df):
        risky_df["bad_flag"] = None  # Make label all missing
        result = detect_high_missing_columns(
            risky_df, threshold=0.3, protected_columns=["bad_flag"]
        )
        cols = [c["column"] for c in result["high_missing_columns"]]
        assert "bad_flag" not in cols


class TestDetectTimeLeakageCandidates:
    def test_detects_post_event_columns(self, risky_df):
        result = detect_time_leakage_candidates(risky_df, observation_time_col="apply_date")
        cols = [c["column"] for c in result["time_leakage_candidates"]]
        assert "还款日期" in cols
        assert "settle_date" in cols

    def test_excludes_protected(self, risky_df):
        result = detect_time_leakage_candidates(
            risky_df,
            observation_time_col="apply_date",
            protected_columns=["还款日期"],
        )
        cols = [c["column"] for c in result["time_leakage_candidates"]]
        assert "还款日期" not in cols

    def test_no_time_leakage(self):
        df = pd.DataFrame({
            "apply_date": ["2024-01-01"],
            "age": [30],
            "income": [50000],
        })
        result = detect_time_leakage_candidates(df, observation_time_col="apply_date")
        assert result["n_detected"] == 0


class TestRiskGuardAgent:
    def test_run_full(self, tmp_path, monkeypatch, risky_csv_path):
        monkeypatch.chdir(tmp_path)
        state = RiskModelingProjectState(
            project_name="test_proj",
            cleaned_data_path=risky_csv_path,
            label_col="bad_flag",
            id_col="customer_id",
            base_time_col="apply_date",
        )

        result = run_risk_guard(state)

        assert result["n_drop_recommended"] > 0
        assert result["n_safe_columns"] > 0
        assert len(result["warnings"]) > 0
        assert result["need_human_review"] is True

        # Verify specific detections
        drop_cols = [r["column"] for r in result["drop_recommendations"]]
        assert "dpd_30" in drop_cols
        assert "overdue_days" in drop_cols
        assert "loan_id" in drop_cols

        # Safe columns should not include risky ones
        assert "dpd_30" not in result["safe_columns"]
        assert "age" in result["safe_columns"]
        assert "income" in result["safe_columns"]

    def test_run_clean_data(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        df = pd.DataFrame({
            "age": [25, 30, 35],
            "income": [50000, 60000, 70000],
            "bad_flag": [0, 1, 0],
        })
        path = str(tmp_path / "clean.csv")
        df.to_csv(path, index=False)

        state = RiskModelingProjectState(
            project_name="test_proj",
            cleaned_data_path=path,
            label_col="bad_flag",
        )

        result = run_risk_guard(state)
        assert result["n_drop_recommended"] == 0
        assert result["need_human_review"] is False

    def test_run_no_data(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        state = RiskModelingProjectState(project_name="test_proj")

        result = run_risk_guard(state)
        assert result.get("error") == "no_data_file"
