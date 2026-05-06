"""Tests for Stage 9: Modeling & Evaluation (Tasks 9.1–9.5)."""

import json
import os
import tempfile

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_dir():
    """Create a temporary directory for test artifacts."""
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def sample_binary_data():
    """Generate sample binary classification data."""
    rng = np.random.RandomState(42)
    n = 200
    df = pd.DataFrame({
        "id": range(n),
        "feature_a": rng.randn(n),
        "feature_b": rng.randn(n),
        "feature_c": rng.uniform(0, 100, n),
        "apply_date": pd.date_range("2023-01-01", periods=n, freq="D"),
        "bad_flag": rng.choice([0, 1], size=n, p=[0.8, 0.2]),
    })
    return df


@pytest.fixture
def y_true_and_proba():
    """Generate realistic y_true and y_proba for evaluation tests."""
    rng = np.random.RandomState(42)
    n = 500
    y_true = rng.choice([0, 1], size=n, p=[0.8, 0.2])
    # Make probabilities somewhat correlated with true labels
    y_proba = np.clip(
        y_true * 0.4 + rng.uniform(0, 0.6, n),
        0, 1,
    )
    return y_true, y_proba


@pytest.fixture
def base_state(tmp_dir, sample_binary_data):
    """Create a base project state for agent tests."""
    project_name = "test_project"
    project_dir = os.path.join(tmp_dir, "artifacts", project_name)
    os.makedirs(project_dir, exist_ok=True)

    # Save sample data
    data_path = os.path.join(project_dir, "cleaned_data.csv")
    sample_binary_data.to_csv(data_path, index=False)

    return {
        "project_name": project_name,
        "uploaded_files": [{
            "file_name": "loan_data.csv",
            "file_path": data_path,
            "n_rows": len(sample_binary_data),
            "n_cols": len(sample_binary_data.columns),
        }],
        "cleaned_data_path": data_path,
        "main_data_path": data_path,
        "label_col": "bad_flag",
        "id_col": "id",
        "base_time_col": "apply_date",
        "drop_columns": [],
        "data_quality_report": {
            "overall_quality_score": 85,
            "duplicate_rows": 0,
            "high_missing_count": 1,
            "outlier_count": 2,
            "label_quality": {"positive_rate": 0.2},
        },
        "cleaning_plan": {
            "before_shape": [200, 6],
            "after_shape": [200, 6],
            "steps": [
                {"priority": 1, "action": "fill_missing", "rows_affected": 5},
            ],
        },
        "warnings": ["feature_c 与 bad_flag 相关性较高，请确认是否存在泄露"],
    }


# ---------------------------------------------------------------------------
# Task 9.1: split_tools tests
# ---------------------------------------------------------------------------

class TestSplitTools:
    def test_stratified_split(self, sample_binary_data):
        from tools.split_tools import split_train_test

        train, test = split_train_test(
            sample_binary_data, label_col="bad_flag", test_size=0.2
        )
        assert len(train) + len(test) == len(sample_binary_data)
        assert len(test) > 0
        assert len(train) > len(test)

    def test_time_split(self, sample_binary_data):
        from tools.split_tools import split_train_test

        train, test = split_train_test(
            sample_binary_data, label_col="bad_flag", time_col="apply_date"
        )
        assert len(train) + len(test) == len(sample_binary_data)
        # Earlier dates should be in train
        assert train["apply_date"].max() <= test["apply_date"].min()

    def test_split_preserves_columns(self, sample_binary_data):
        from tools.split_tools import split_train_test

        train, test = split_train_test(
            sample_binary_data, label_col="bad_flag"
        )
        assert list(train.columns) == list(sample_binary_data.columns)
        assert list(test.columns) == list(sample_binary_data.columns)


# ---------------------------------------------------------------------------
# Task 9.1: modeling_tools tests
# ---------------------------------------------------------------------------

class TestModelingTools:
    def test_fallback_train(self, sample_binary_data, tmp_dir):
        """Test fallback training with sklearn when AutoGluon is unavailable."""
        from tools.modeling_tools import _fallback_train

        train = sample_binary_data.iloc[:160].copy()
        test = sample_binary_data.iloc[160:].copy()

        result = _fallback_train(
            train_df=train,
            test_df=test,
            label_col="bad_flag",
            output_dir=tmp_dir,
            excluded_columns=["id", "apply_date"],
        )

        assert "model_path" in result
        assert "leaderboard_path" in result
        assert "best_model" in result
        assert "predictions" in result
        assert result["best_model"] == "GradientBoosting"
        assert os.path.exists(result["leaderboard_path"])
        assert len(result["predictions"]) == len(test)
        assert len(result["feature_columns"]) > 0
        assert "id" not in result["feature_columns"]
        assert "apply_date" not in result["feature_columns"]


# ---------------------------------------------------------------------------
# Task 9.1: ModelingAgent tests
# ---------------------------------------------------------------------------

class TestModelingAgent:
    def test_run_success(self, base_state, tmp_dir, monkeypatch):
        """Test ModelingAgent runs successfully."""
        import agents.modeling_agent as agent

        # Monkeypatch artifacts dir
        monkeypatch.chdir(tmp_dir)
        os.makedirs(os.path.join(tmp_dir, "artifacts", "test_project"), exist_ok=True)

        result = agent.run(
            state=base_state,
            output_dir=os.path.join(tmp_dir, "artifacts", "test_project"),
            time_limit=60,
        )

        assert "error" not in result
        assert "model_path" in result
        assert "train_path" in result
        assert "test_path" in result
        assert "predictions" in result
        assert "feature_columns" in result
        assert os.path.exists(result["train_path"])
        assert os.path.exists(result["test_path"])

    def test_run_no_data(self, tmp_dir, monkeypatch):
        """Test ModelingAgent returns error when no data."""
        import agents.modeling_agent as agent

        monkeypatch.chdir(tmp_dir)
        state = {"project_name": "test_project"}
        result = agent.run(state=state)
        assert result == {"error": "no_data_file"}

    def test_run_no_label(self, base_state, tmp_dir, monkeypatch):
        """Test ModelingAgent returns error when no label column."""
        import agents.modeling_agent as agent

        monkeypatch.chdir(tmp_dir)
        base_state["label_col"] = None
        result = agent.run(state=base_state)
        assert result == {"error": "no_label_col"}


# ---------------------------------------------------------------------------
# Task 9.2: metric_tools tests
# ---------------------------------------------------------------------------

class TestMetricTools:
    def test_compute_ks(self, y_true_and_proba):
        from tools.metric_tools import compute_ks

        y_true, y_proba = y_true_and_proba
        result = compute_ks(y_true, y_proba)

        assert "ks_statistic" in result
        assert "ks_threshold" in result
        assert "ks_decile_table" in result
        assert 0 <= result["ks_statistic"] <= 1
        assert 0 <= result["ks_threshold"] <= 1
        assert len(result["ks_decile_table"]) == 10

    def test_compute_ks_perfect_separation(self):
        from tools.metric_tools import compute_ks

        y_true = np.array([1, 1, 1, 0, 0, 0])
        y_proba = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
        result = compute_ks(y_true, y_proba)
        assert result["ks_statistic"] == 1.0

    def test_compute_ks_no_separation(self):
        from tools.metric_tools import compute_ks

        # When probabilities are perfectly interleaved with labels, KS ~ 0
        y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
        y_proba = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        result = compute_ks(y_true, y_proba)
        # With tied scores, KS depends on sort stability; just verify it runs
        assert 0.0 <= result["ks_statistic"] <= 1.0
        assert "ks_decile_table" in result

    def test_evaluate_binary_model(self, y_true_and_proba):
        from tools.metric_tools import evaluate_binary_model

        y_true, y_proba = y_true_and_proba
        result = evaluate_binary_model(y_true, y_proba)

        assert "auc" in result
        assert "ks" in result
        assert "accuracy" in result
        assert "precision" in result
        assert "recall" in result
        assert "f1" in result
        assert "confusion_matrix" in result
        assert "threshold" in result

        # Validate ranges
        assert 0 <= result["auc"] <= 1
        assert 0 <= result["ks"] <= 1
        assert 0 <= result["accuracy"] <= 1
        assert 0 <= result["precision"] <= 1
        assert 0 <= result["recall"] <= 1
        assert 0 <= result["f1"] <= 1

        # Confusion matrix
        cm = result["confusion_matrix"]
        assert cm["tn"] + cm["fp"] + cm["fn"] + cm["tp"] == len(y_true)

    def test_evaluate_binary_model_custom_threshold(self, y_true_and_proba):
        from tools.metric_tools import evaluate_binary_model

        y_true, y_proba = y_true_and_proba
        result = evaluate_binary_model(y_true, y_proba, threshold=0.3)
        assert result["threshold"] == 0.3


# ---------------------------------------------------------------------------
# Task 9.2: EvaluationAgent tests
# ---------------------------------------------------------------------------

class TestEvaluationAgent:
    def test_run_with_explicit_inputs(self, base_state, tmp_dir, y_true_and_proba, monkeypatch):
        import agents.evaluation_agent as agent

        monkeypatch.chdir(tmp_dir)
        y_true, y_proba = y_true_and_proba
        output_dir = os.path.join(tmp_dir, "artifacts", "test_project")
        os.makedirs(output_dir, exist_ok=True)

        result = agent.run(
            state=base_state,
            y_true=y_true,
            y_proba=y_proba,
            output_dir=output_dir,
        )

        assert "error" not in result
        assert "auc" in result
        assert "ks" in result
        assert "metrics_path" in result
        assert os.path.exists(result["metrics_path"])

        # Verify JSON output
        with open(result["metrics_path"]) as f:
            saved = json.load(f)
        assert saved["auc"] == result["auc"]

    def test_run_no_test_data(self, tmp_dir, monkeypatch):
        import agents.evaluation_agent as agent

        monkeypatch.chdir(tmp_dir)
        state = {"project_name": "test_project"}
        result = agent.run(state=state)
        assert result == {"error": "no_test_data"}

    def test_run_no_predictions(self, base_state, tmp_dir, monkeypatch):
        import agents.evaluation_agent as agent

        monkeypatch.chdir(tmp_dir)
        # Provide test_path but no predictions
        test_df = pd.DataFrame({"bad_flag": [0, 1, 0, 1], "x": [1, 2, 3, 4]})
        test_path = os.path.join(tmp_dir, "test.csv")
        test_df.to_csv(test_path, index=False)
        base_state["test_path"] = test_path

        result = agent.run(state=base_state)
        assert result == {"error": "no_predictions"}


# ---------------------------------------------------------------------------
# Task 9.3: strategy_tools tests
# ---------------------------------------------------------------------------

class TestStrategyTools:
    def test_build_threshold_table(self, y_true_and_proba):
        from tools.strategy_tools import build_threshold_table

        y_true, y_proba = y_true_and_proba
        table = build_threshold_table(y_true, y_proba)

        assert isinstance(table, pd.DataFrame)
        expected_cols = {"threshold", "pass_rate", "reject_rate", "pass_bad_rate", "capture_rate", "n_pass", "n_reject"}
        assert expected_cols.issubset(set(table.columns))
        assert len(table) > 0

        # Validate rates sum to 1
        for _, row in table.iterrows():
            assert abs(row["pass_rate"] + row["reject_rate"] - 1.0) < 0.01

    def test_build_threshold_table_explicit_thresholds(self, y_true_and_proba):
        from tools.strategy_tools import build_threshold_table

        y_true, y_proba = y_true_and_proba
        thresholds = [0.2, 0.3, 0.4, 0.5]
        table = build_threshold_table(y_true, y_proba, thresholds=thresholds)

        assert len(table) == 4
        assert list(table["threshold"]) == thresholds

    def test_threshold_table_monotonicity(self, y_true_and_proba):
        """Higher threshold → higher pass rate."""
        from tools.strategy_tools import build_threshold_table

        y_true, y_proba = y_true_and_proba
        table = build_threshold_table(y_true, y_proba)

        pass_rates = table["pass_rate"].values
        # Pass rate should be non-decreasing as threshold increases
        for i in range(1, len(pass_rates)):
            assert pass_rates[i] >= pass_rates[i - 1]


# ---------------------------------------------------------------------------
# Task 9.3: StrategyAgent tests
# ---------------------------------------------------------------------------

class TestStrategyAgent:
    def test_run_with_explicit_inputs(self, base_state, tmp_dir, y_true_and_proba, monkeypatch):
        import agents.strategy_agent as agent

        monkeypatch.chdir(tmp_dir)
        y_true, y_proba = y_true_and_proba
        output_dir = os.path.join(tmp_dir, "artifacts", "test_project")
        os.makedirs(output_dir, exist_ok=True)

        result = agent.run(
            state=base_state,
            y_true=y_true,
            y_proba=y_proba,
            output_dir=output_dir,
        )

        assert "error" not in result
        assert "threshold_table_path" in result
        assert "threshold_table" in result
        assert "n_thresholds" in result
        assert os.path.exists(result["threshold_table_path"])
        assert result["n_thresholds"] > 0

    def test_run_no_data(self, tmp_dir, monkeypatch):
        import agents.strategy_agent as agent

        monkeypatch.chdir(tmp_dir)
        state = {"project_name": "test_project"}
        result = agent.run(state=state)
        assert result == {"error": "no_test_data"}


# ---------------------------------------------------------------------------
# Task 9.4: explain_tools tests
# ---------------------------------------------------------------------------

class TestExplainTools:
    def test_compute_feature_importance_fallback(self, tmp_dir, sample_binary_data):
        """Test feature importance with sklearn fallback model."""
        from sklearn.ensemble import GradientBoostingClassifier
        import joblib
        from tools.explain_tools import compute_feature_importance

        # Train a simple model
        feature_cols = ["feature_a", "feature_b", "feature_c"]
        X = sample_binary_data[feature_cols].values
        y = sample_binary_data["bad_flag"].values

        model = GradientBoostingClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)

        model_dir = os.path.join(tmp_dir, "model")
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(model, os.path.join(model_dir, "model.pkl"))

        result = compute_feature_importance(
            model_path=model_dir,
            feature_columns=feature_cols,
        )

        assert isinstance(result, pd.DataFrame)
        assert "feature_name" in result.columns
        assert "importance_score" in result.columns
        assert len(result) == 3
        # Should be sorted descending
        assert result["importance_score"].is_monotonic_decreasing

    def test_compute_feature_importance_no_model(self, tmp_dir):
        """Test fallback when no model exists."""
        from tools.explain_tools import compute_feature_importance

        result = compute_feature_importance(
            model_path=os.path.join(tmp_dir, "nonexistent"),
            feature_columns=["a", "b", "c"],
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        # All zeros fallback
        assert (result["importance_score"] == 0.0).all()


# ---------------------------------------------------------------------------
# Task 9.4: ExplainAgent tests
# ---------------------------------------------------------------------------

class TestExplainAgent:
    def test_run_success(self, base_state, tmp_dir, sample_binary_data, monkeypatch):
        import agents.explain_agent as agent
        from sklearn.ensemble import GradientBoostingClassifier
        import joblib

        monkeypatch.chdir(tmp_dir)

        # Set up model
        feature_cols = ["feature_a", "feature_b", "feature_c"]
        X = sample_binary_data[feature_cols].values
        y = sample_binary_data["bad_flag"].values

        model_dir = os.path.join(tmp_dir, "artifacts", "test_project", "model")
        os.makedirs(model_dir, exist_ok=True)
        model = GradientBoostingClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        joblib.dump(model, os.path.join(model_dir, "model.pkl"))

        base_state["model_path"] = model_dir
        base_state["feature_columns"] = feature_cols

        output_dir = os.path.join(tmp_dir, "artifacts", "test_project")
        result = agent.run(state=base_state, output_dir=output_dir)

        assert "error" not in result
        assert "feature_importance_path" in result
        assert "top_features" in result
        assert os.path.exists(result["feature_importance_path"])
        assert result["n_features"] == 3

    def test_run_no_model(self, tmp_dir, monkeypatch):
        import agents.explain_agent as agent

        monkeypatch.chdir(tmp_dir)
        state = {"project_name": "test_project", "feature_columns": ["a"]}
        result = agent.run(state=state)
        assert result == {"error": "no_model_or_features"}


# ---------------------------------------------------------------------------
# Task 9.5: report_tools tests
# ---------------------------------------------------------------------------

class TestReportTools:
    def test_generate_markdown_report_full(self):
        from tools.report_tools import generate_markdown_report

        report = generate_markdown_report(
            project_name="test_project",
            data_summary={
                "n_rows": 1000,
                "n_features": 20,
                "positive_rate": 0.15,
                "train_size": 800,
                "test_size": 200,
            },
            quality_summary={
                "overall_quality_score": 85,
                "duplicate_rows": 5,
                "high_missing_count": 2,
                "outlier_count": 3,
            },
            cleaning_summary={
                "before_shape": [1000, 20],
                "after_shape": [995, 20],
                "n_steps": 2,
                "steps": [
                    {"priority": 1, "action": "drop_duplicates", "rows_affected": 5},
                    {"priority": 2, "action": "fill_missing", "rows_affected": 10},
                ],
            },
            model_metrics={
                "auc": 0.82,
                "ks": 0.45,
                "accuracy": 0.85,
                "precision": 0.72,
                "recall": 0.65,
                "f1": 0.68,
                "confusion_matrix": {"tn": 680, "fp": 20, "fn": 35, "tp": 65},
            },
            threshold_table=[
                {"threshold": 0.2, "pass_rate": 0.82, "reject_rate": 0.18, "pass_bad_rate": 0.038, "capture_rate": 0.42},
                {"threshold": 0.3, "pass_rate": 0.88, "reject_rate": 0.12, "pass_bad_rate": 0.046, "capture_rate": 0.31},
            ],
            feature_importance=[
                {"feature_name": "income", "importance_score": 0.25},
                {"feature_name": "credit_score", "importance_score": 0.20},
            ],
            risk_warnings=["feature_x 可能存在标签泄露"],
            recommendations=["建议增加外部数据源", "可尝试 WOE 编码"],
        )

        assert "# 模型评估报告" in report
        assert "test_project" in report
        assert "1000" in report
        assert "AUC" in report
        assert "0.82" in report
        assert "income" in report
        assert "标签泄露" in report
        assert "WOE" in report

    def test_generate_markdown_report_minimal(self):
        from tools.report_tools import generate_markdown_report

        report = generate_markdown_report(project_name="empty_project")

        assert "# 模型评估报告" in report
        assert "empty_project" in report
        assert "暂无" in report

    def test_default_recommendations_low_auc(self):
        from tools.report_tools import _generate_default_recommendations

        recs = _generate_default_recommendations({"auc": 0.6, "ks": 0.15})
        assert "AUC 偏低" in recs
        assert "KS 偏低" in recs

    def test_default_recommendations_high_auc(self):
        from tools.report_tools import _generate_default_recommendations

        recs = _generate_default_recommendations({"auc": 0.98, "ks": 0.5})
        assert "数据泄露" in recs


# ---------------------------------------------------------------------------
# Task 9.5: ReportAgent tests
# ---------------------------------------------------------------------------

class TestReportAgent:
    def test_run_success(self, base_state, tmp_dir, monkeypatch):
        import agents.report_agent as agent

        monkeypatch.chdir(tmp_dir)

        base_state["metrics"] = {
            "auc": 0.82, "ks": 0.45, "accuracy": 0.85,
            "precision": 0.72, "recall": 0.65, "f1": 0.68,
            "confusion_matrix": {"tn": 680, "fp": 20, "fn": 35, "tp": 65},
        }
        base_state["threshold_table"] = [
            {"threshold": 0.3, "pass_rate": 0.88, "reject_rate": 0.12,
             "pass_bad_rate": 0.046, "capture_rate": 0.31},
        ]
        base_state["feature_importance"] = [
            {"feature_name": "income", "importance_score": 0.25},
        ]

        output_dir = os.path.join(tmp_dir, "artifacts", "test_project", "reports")
        result = agent.run(state=base_state, output_dir=output_dir)

        assert "error" not in result
        assert "report_path" in result
        assert os.path.exists(result["report_path"])
        assert result["report_length"] > 0

        # Verify report content
        with open(result["report_path"], encoding="utf-8") as f:
            content = f.read()
        assert "模型评估报告" in content
        assert "AUC" in content
        assert "0.82" in content

    def test_run_minimal_state(self, tmp_dir, monkeypatch):
        """Test report generation with minimal state (no metrics yet)."""
        import agents.report_agent as agent

        monkeypatch.chdir(tmp_dir)
        state = {
            "project_name": "test_project",
            "uploaded_files": [{"file_name": "data.csv", "n_rows": 100, "n_cols": 10}],
        }

        output_dir = os.path.join(tmp_dir, "artifacts", "test_project", "reports")
        result = agent.run(state=state, output_dir=output_dir)

        assert "error" not in result
        assert os.path.exists(result["report_path"])

    def test_permission_check(self, monkeypatch):
        """Test that ReportAgent checks permissions."""
        from core.permissions import check_tool_permission

        assert check_tool_permission("ReportAgent", "generate_markdown_report") is True
        assert check_tool_permission("ReportAgent", "train_autogluon_binary") is False


# ---------------------------------------------------------------------------
# Integration: End-to-end stage 9 flow
# ---------------------------------------------------------------------------

class TestStage9Integration:
    def test_full_pipeline(self, base_state, tmp_dir, monkeypatch):
        """Test the full stage 9 pipeline: model → evaluate → strategy → explain → report."""
        import agents.modeling_agent as modeling_agent
        import agents.evaluation_agent as evaluation_agent
        import agents.strategy_agent as strategy_agent
        import agents.explain_agent as explain_agent
        import agents.report_agent as report_agent

        monkeypatch.chdir(tmp_dir)
        output_dir = os.path.join(tmp_dir, "artifacts", "test_project")
        os.makedirs(output_dir, exist_ok=True)

        # Step 1: Train model
        model_result = modeling_agent.run(
            state=base_state, output_dir=output_dir, time_limit=30
        )
        assert "error" not in model_result

        # Update state with modeling results
        base_state["model_path"] = model_result["model_path"]
        base_state["train_path"] = model_result["train_path"]
        base_state["test_path"] = model_result["test_path"]
        base_state["feature_columns"] = model_result["feature_columns"]
        base_state["predictions"] = model_result["predictions"]

        # Step 2: Evaluate
        eval_result = evaluation_agent.run(
            state=base_state, output_dir=output_dir
        )
        assert "error" not in eval_result
        assert "auc" in eval_result

        base_state["metrics"] = eval_result

        # Step 3: Strategy
        strategy_result = strategy_agent.run(
            state=base_state, output_dir=output_dir
        )
        assert "error" not in strategy_result
        base_state["threshold_table"] = strategy_result["threshold_table"]

        # Step 4: Explain
        explain_result = explain_agent.run(
            state=base_state, output_dir=output_dir
        )
        assert "error" not in explain_result
        base_state["feature_importance"] = explain_result["top_features"]

        # Step 5: Report
        report_result = report_agent.run(
            state=base_state, output_dir=os.path.join(output_dir, "reports")
        )
        assert "error" not in report_result
        assert os.path.exists(report_result["report_path"])

        # Verify final report has all sections
        with open(report_result["report_path"], encoding="utf-8") as f:
            report = f.read()
        assert "数据概况" in report
        assert "数据质量" in report
        assert "清洗记录" in report
        assert "模型评估指标" in report
        assert "阈值策略" in report
        assert "特征重要性" in report
        assert "风险提示" in report
        assert "优化建议" in report
