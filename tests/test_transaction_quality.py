"""Tests for agents/transaction_quality_agent.py."""

import numpy as np
import pandas as pd
import pytest

from agents.transaction_quality_agent import (
    analyze_transaction_quality,
    run as run_txn_quality_agent,
)
from core.state import RiskModelingProjectState


@pytest.fixture
def txn_df():
    """Create a sample transaction DataFrame."""
    np.random.seed(42)
    n = 500
    accounts = [f"A{i:04d}" for i in range(50)]
    df = pd.DataFrame({
        "account_id": np.random.choice(accounts, n),
        "txn_time": pd.date_range("2024-01-01", periods=n, freq="h").astype(str),
        "amount": np.random.exponential(1000, n).round(2),
        "dc_flag": np.random.choice(["D", "C"], n),
        "channel": np.random.choice(["ATM", "POS", "ONLINE", "TRANSFER"], n),
    })
    return df


@pytest.fixture
def txn_csv_path(tmp_path, txn_df):
    path = str(tmp_path / "transactions.csv")
    txn_df.to_csv(path, index=False)
    return path


class TestAnalyzeTransactionQuality:
    def test_basic_analysis(self, txn_df):
        result = analyze_transaction_quality(
            txn_df,
            account_col="account_id",
            time_col="txn_time",
            amount_col="amount",
            direction_col="dc_flag",
        )

        assert result["n_rows"] == 500
        assert result["n_accounts"] == 50
        assert result["overall_quality_score"] > 0
        assert "time_analysis" in result
        assert "account_analysis" in result
        assert "amount_analysis" in result
        assert "direction_analysis" in result

    def test_time_analysis(self, txn_df):
        result = analyze_transaction_quality(
            txn_df, account_col="account_id", time_col="txn_time"
        )
        time_info = result["time_analysis"]
        assert time_info["invalid_count"] == 0
        assert time_info["time_span_days"] > 0

    def test_account_analysis(self, txn_df):
        result = analyze_transaction_quality(
            txn_df, account_col="account_id", time_col="txn_time"
        )
        acct_info = result["account_analysis"]
        assert acct_info["n_accounts"] == 50
        assert acct_info["avg_txn_per_account"] == pytest.approx(10.0, abs=2)

    def test_amount_analysis(self, txn_df):
        result = analyze_transaction_quality(
            txn_df, account_col="account_id", time_col="txn_time", amount_col="amount"
        )
        amt_info = result["amount_analysis"]
        assert amt_info["mean"] > 0
        assert amt_info["negative_count"] == 0

    def test_direction_analysis(self, txn_df):
        result = analyze_transaction_quality(
            txn_df, account_col="account_id", time_col="txn_time", direction_col="dc_flag"
        )
        dir_info = result["direction_analysis"]
        assert dir_info["is_binary"] is True
        assert set(dir_info["unique_values"]) == {"D", "C"}

    def test_invalid_time_column(self, txn_df):
        txn_df["txn_time"] = "not_a_date"
        result = analyze_transaction_quality(
            txn_df, account_col="account_id", time_col="txn_time"
        )
        # Should still complete without error
        assert result["n_rows"] == 500

    def test_missing_account_col(self, txn_df):
        result = analyze_transaction_quality(
            txn_df, account_col="nonexistent", time_col="txn_time"
        )
        assert result["account_analysis"].get("error") is not None


class TestTransactionQualityAgent:
    def test_run(self, tmp_path, monkeypatch, txn_csv_path):
        monkeypatch.chdir(tmp_path)
        state = RiskModelingProjectState(
            project_name="test_proj",
            transaction_data_path=txn_csv_path,
            account_col="account_id",
            time_col="txn_time",
            transaction_schema={
                "account_col": "account_id",
                "time_col": "txn_time",
                "amount_col": "amount",
                "direction_col": "dc_flag",
            },
        )

        result = run_txn_quality_agent(state)
        assert result["n_rows"] == 500
        assert result["overall_quality_score"] > 0

    def test_run_no_transaction_file(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        state = RiskModelingProjectState(
            project_name="test_proj",
        )

        result = run_txn_quality_agent(state)
        assert result.get("skipped") is True
