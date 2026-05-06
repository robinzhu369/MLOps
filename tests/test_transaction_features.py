"""Tests for tools/transaction_feature_tools.py and agents/transaction_feature_agent.py."""

import numpy as np
import pandas as pd
import pytest

from tools.transaction_feature_tools import (
    infer_transaction_schema,
    build_account_daily_features,
    build_account_window_features,
    validate_transaction_feature_cutoff,
    profile_generated_features,
)
from agents.transaction_feature_agent import run as run_feature_agent
from core.state import RiskModelingProjectState


@pytest.fixture
def txn_df():
    """Create a transaction DataFrame spanning multiple days and accounts."""
    np.random.seed(42)
    records = []
    accounts = ["A001", "A002", "A003"]
    for acct in accounts:
        for day in range(1, 31):  # 30 days
            n_txn = np.random.randint(1, 5)
            for _ in range(n_txn):
                hour = np.random.randint(8, 22)
                minute = np.random.randint(0, 60)
                records.append({
                    "account_id": acct,
                    "transaction_time": f"2024-01-{day:02d} {hour:02d}:{minute:02d}:00",
                    "transaction_amount": round(np.random.exponential(500), 2),
                    "dc_flag": np.random.choice(["D", "C"]),
                    "channel": np.random.choice(["ATM", "POS", "ONLINE"]),
                })
    return pd.DataFrame(records)


@pytest.fixture
def main_df():
    """Create a main table with observation dates."""
    return pd.DataFrame({
        "customer_id": ["A001", "A002", "A003"],
        "apply_date": ["2024-01-20", "2024-01-25", "2024-01-28"],
        "bad_flag": [0, 1, 0],
    })


@pytest.fixture
def txn_csv_path(tmp_path, txn_df):
    path = str(tmp_path / "transactions.csv")
    txn_df.to_csv(path, index=False)
    return path


@pytest.fixture
def main_csv_path(tmp_path, main_df):
    path = str(tmp_path / "main.csv")
    main_df.to_csv(path, index=False)
    return path


class TestInferTransactionSchema:
    def test_infer_from_columns(self, txn_df):
        schema = infer_transaction_schema(txn_df)
        assert schema["account_col"] == "account_id"
        assert schema["time_col"] == "transaction_time"
        assert schema["amount_col"] == "transaction_amount"
        assert schema["direction_col"] == "dc_flag"

    def test_infer_with_detected_roles(self, txn_df):
        roles = {"account_key": "account_id", "transaction_time_col": "transaction_time"}
        schema = infer_transaction_schema(txn_df, detected_roles=roles)
        assert schema["account_col"] == "account_id"
        assert schema["time_col"] == "transaction_time"

    def test_infer_unknown_columns(self):
        df = pd.DataFrame({"col_a": [1], "col_b": [2], "col_c": [3]})
        schema = infer_transaction_schema(df)
        assert schema["account_col"] is None
        assert schema["time_col"] is None


class TestBuildAccountDailyFeatures:
    def test_output_shape(self, txn_df):
        daily = build_account_daily_features(
            txn_df, "account_id", "transaction_time", "transaction_amount", "dc_flag"
        )
        # Should have one row per account per day
        assert "account_id" in daily.columns
        assert "transaction_date" in daily.columns
        # 3 accounts x up to 30 days
        assert len(daily) <= 3 * 30
        assert len(daily) > 0

    def test_15_features(self, txn_df):
        daily = build_account_daily_features(
            txn_df, "account_id", "transaction_time", "transaction_amount", "dc_flag"
        )
        expected_features = [
            "txn_count", "txn_amount_sum", "txn_amount_mean",
            "txn_amount_max", "txn_amount_min", "txn_amount_std",
            "debit_count", "credit_count",
            "debit_amount_sum", "credit_amount_sum", "net_amount",
            "large_txn_count", "unique_channels",
            "max_single_ratio", "txn_time_span_hours",
        ]
        for feat in expected_features:
            assert feat in daily.columns, f"Missing feature: {feat}"

    def test_txn_count_correct(self, txn_df):
        daily = build_account_daily_features(
            txn_df, "account_id", "transaction_time", "transaction_amount", "dc_flag"
        )
        # Verify against manual count for one account-day
        a001_day1 = txn_df[
            (txn_df["account_id"] == "A001") &
            (txn_df["transaction_time"].str.startswith("2024-01-01"))
        ]
        daily_a001_d1 = daily[
            (daily["account_id"] == "A001") &
            (daily["transaction_date"] == pd.Timestamp("2024-01-01").date())
        ]
        if not daily_a001_d1.empty:
            assert daily_a001_d1["txn_count"].iloc[0] == len(a001_day1)

    def test_net_amount(self, txn_df):
        daily = build_account_daily_features(
            txn_df, "account_id", "transaction_time", "transaction_amount", "dc_flag"
        )
        # net_amount = credit_amount_sum - debit_amount_sum
        row = daily.iloc[0]
        assert abs(row["net_amount"] - (row["credit_amount_sum"] - row["debit_amount_sum"])) < 0.01

    def test_without_direction_col(self, txn_df):
        daily = build_account_daily_features(
            txn_df, "account_id", "transaction_time", "transaction_amount"
        )
        assert "debit_count" in daily.columns
        assert len(daily) > 0


class TestBuildAccountWindowFeatures:
    def test_output_shape(self, txn_df, main_df):
        daily = build_account_daily_features(
            txn_df, "account_id", "transaction_time", "transaction_amount", "dc_flag"
        )
        # Rename main_df's customer_id to account_id for join
        main_renamed = main_df.rename(columns={"customer_id": "account_id"})
        window = build_account_window_features(
            daily_features=daily,
            main_df=main_renamed,
            account_col="account_id",
            base_time_col="apply_date",
        )
        # One row per main table row
        assert len(window) == len(main_df)
        assert "account_id" in window.columns

    def test_window_features_generated(self, txn_df, main_df):
        daily = build_account_daily_features(
            txn_df, "account_id", "transaction_time", "transaction_amount", "dc_flag"
        )
        main_renamed = main_df.rename(columns={"customer_id": "account_id"})
        window = build_account_window_features(
            daily_features=daily,
            main_df=main_renamed,
            account_col="account_id",
            base_time_col="apply_date",
        )
        # Should have features for multiple windows (w7d_, w30d_, w90d_)
        window_cols = [c for c in window.columns if "w7d_" in c or "w30d_" in c or "w90d_" in c]
        assert len(window_cols) > 0

    def test_time_cutoff_respected(self, txn_df, main_df):
        """Verify that window features only use data before apply_date."""
        daily = build_account_daily_features(
            txn_df, "account_id", "transaction_time", "transaction_amount", "dc_flag"
        )
        main_renamed = main_df.rename(columns={"customer_id": "account_id"})
        window = build_account_window_features(
            daily_features=daily,
            main_df=main_renamed,
            account_col="account_id",
            base_time_col="apply_date",
        )
        # A001 apply_date=2024-01-20, A003 apply_date=2024-01-28
        # A003 has later apply_date so more historical data in 30d window
        a001 = window[window["account_id"] == "A001"]
        a003 = window[window["account_id"] == "A003"]
        assert a003["w30d_txn_count_sum"].iloc[0] >= a001["w30d_txn_count_sum"].iloc[0]

    def test_with_custom_windows(self, txn_df, main_df):
        daily = build_account_daily_features(
            txn_df, "account_id", "transaction_time", "transaction_amount", "dc_flag"
        )
        main_renamed = main_df.rename(columns={"customer_id": "account_id"})
        window = build_account_window_features(
            daily_features=daily,
            main_df=main_renamed,
            account_col="account_id",
            base_time_col="apply_date",
            windows=[7, 30],
        )
        # Should only have w7d_ and w30d_ prefixes
        w7_cols = [c for c in window.columns if "w7d_" in c]
        w30_cols = [c for c in window.columns if "w30d_" in c]
        w90_cols = [c for c in window.columns if "w90d_" in c]
        assert len(w7_cols) > 0
        assert len(w30_cols) > 0
        assert len(w90_cols) == 0


class TestValidateTransactionFeatureCutoff:
    def test_no_leakage(self, txn_df, main_df):
        daily = build_account_daily_features(
            txn_df, "account_id", "transaction_time", "transaction_amount", "dc_flag"
        )
        main_renamed = main_df.rename(columns={"customer_id": "account_id"})
        # Filter daily to only before apply_date (simulating correct cutoff)
        daily_before = daily[daily["transaction_date"] < pd.Timestamp("2024-01-20").date()]
        result = validate_transaction_feature_cutoff(
            daily_features=daily_before,
            main_df=main_renamed,
            account_col="account_id",
            base_time_col="apply_date",
        )
        assert result["is_valid"] is True
        assert result["violations_count"] == 0

    def test_detects_leakage(self, txn_df, main_df):
        daily = build_account_daily_features(
            txn_df, "account_id", "transaction_time", "transaction_amount", "dc_flag"
        )
        main_renamed = main_df.rename(columns={"customer_id": "account_id"})
        # Daily features include dates on/after apply_date — should detect violations
        result = validate_transaction_feature_cutoff(
            daily_features=daily,
            main_df=main_renamed,
            account_col="account_id",
            base_time_col="apply_date",
        )
        # A001 apply_date=2024-01-20, daily has data up to 2024-01-30
        assert result["is_valid"] is False
        assert result["violations_count"] > 0


class TestProfileGeneratedFeatures:
    def test_basic_profile(self, txn_df):
        daily = build_account_daily_features(
            txn_df, "account_id", "transaction_time", "transaction_amount", "dc_flag"
        )
        profile = profile_generated_features(daily)
        assert "feature_count" in profile
        assert profile["feature_count"] > 0
        assert "profiles" in profile
        assert len(profile["profiles"]) > 0


class TestTransactionFeatureAgent:
    def test_run_full_pipeline(self, tmp_path, monkeypatch, txn_csv_path, main_csv_path):
        monkeypatch.chdir(tmp_path)
        state = RiskModelingProjectState(
            project_name="test_proj",
            transaction_data_path=txn_csv_path,
            main_data_path=main_csv_path,
            transaction_schema={
                "account_key": "account_id",
                "transaction_time_col": "transaction_time",
                "amount_col": "transaction_amount",
                "direction_col": "dc_flag",
            },
            base_time_col="apply_date",
            id_col="customer_id",
        )

        result = run_feature_agent(state, output_dir=str(tmp_path / "output"))

        assert result.get("daily_features_path") is not None
        assert result.get("window_features_path") is not None
        assert result["schema"]["account_col"] == "account_id"

    def test_run_no_txn_data(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        state = RiskModelingProjectState(project_name="test_proj")

        result = run_feature_agent(state)
        assert result.get("skipped") is True

    def test_run_txn_only_no_main(self, tmp_path, monkeypatch, txn_csv_path):
        monkeypatch.chdir(tmp_path)
        state = RiskModelingProjectState(
            project_name="test_proj",
            transaction_data_path=txn_csv_path,
            transaction_schema={
                "account_key": "account_id",
                "transaction_time_col": "transaction_time",
                "amount_col": "transaction_amount",
                "direction_col": "dc_flag",
            },
        )

        result = run_feature_agent(state, output_dir=str(tmp_path / "output"))

        assert result.get("daily_features_path") is not None
        # No window features without main table
        assert result.get("window_features_path") is None
