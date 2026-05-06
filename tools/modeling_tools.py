"""Modeling tools — AutoGluon binary classification training."""

import os

import pandas as pd


def train_autogluon_binary(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_col: str,
    output_dir: str,
    time_limit: int = 300,
    presets: str = "medium_quality",
    excluded_columns: list[str] | None = None,
) -> dict:
    """Train a binary classification model using AutoGluon.

    Args:
        train_df: Training DataFrame.
        test_df: Test DataFrame.
        label_col: Target column name.
        output_dir: Directory to save model artifacts.
        time_limit: Training time limit in seconds.
        presets: AutoGluon presets (e.g., "best_quality", "medium_quality").
        excluded_columns: Columns to exclude from features.

    Returns:
        dict with model_path, leaderboard, best_model, train_score, test_score.
    """
    try:
        from autogluon.tabular import TabularPredictor
    except ImportError:
        return _fallback_train(train_df, test_df, label_col, output_dir, excluded_columns)

    model_dir = os.path.join(output_dir, "model")
    os.makedirs(model_dir, exist_ok=True)

    # Prepare data
    exclude = set(excluded_columns or [])
    feature_cols = [c for c in train_df.columns if c not in exclude and c != label_col]
    train_data = train_df[feature_cols + [label_col]]
    test_data = test_df[feature_cols + [label_col]]

    predictor = TabularPredictor(
        label=label_col,
        path=model_dir,
        eval_metric="roc_auc",
        problem_type="binary",
    )

    predictor.fit(
        train_data=train_data,
        time_limit=time_limit,
        presets=presets,
    )

    leaderboard = predictor.leaderboard(test_data, silent=True)
    leaderboard_path = os.path.join(output_dir, "leaderboard.csv")
    leaderboard.to_csv(leaderboard_path, index=False)

    # Predictions
    test_pred_proba = predictor.predict_proba(test_data.drop(columns=[label_col]))
    if isinstance(test_pred_proba, pd.DataFrame) and 1 in test_pred_proba.columns:
        test_pred_proba = test_pred_proba[1]

    train_score = predictor.evaluate(train_data)
    test_score = predictor.evaluate(test_data)

    return {
        "model_path": model_dir,
        "leaderboard_path": leaderboard_path,
        "best_model": predictor.get_model_best(),
        "train_score": train_score,
        "test_score": test_score,
        "feature_columns": feature_cols,
        "n_models_trained": len(leaderboard),
        "predictions": test_pred_proba.values if hasattr(test_pred_proba, "values") else test_pred_proba,
    }


def _fallback_train(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_col: str,
    output_dir: str,
    excluded_columns: list[str] | None = None,
) -> dict:
    """Fallback training using sklearn when AutoGluon is not available."""
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    import numpy as np
    import joblib

    model_dir = os.path.join(output_dir, "model")
    os.makedirs(model_dir, exist_ok=True)

    exclude = set(excluded_columns or [])
    feature_cols = [
        c for c in train_df.columns
        if c not in exclude and c != label_col
        and train_df[c].dtype in [np.float64, np.int64, np.float32, np.int32]
    ]

    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df[label_col]
    X_test = test_df[feature_cols].fillna(0)
    y_test = test_df[label_col]

    model = GradientBoostingClassifier(
        n_estimators=100, max_depth=4, random_state=42
    )
    model.fit(X_train, y_train)

    train_proba = model.predict_proba(X_train)[:, 1]
    test_proba = model.predict_proba(X_test)[:, 1]

    train_auc = roc_auc_score(y_train, train_proba)
    test_auc = roc_auc_score(y_test, test_proba)

    # Save model
    model_path = os.path.join(model_dir, "model.pkl")
    joblib.dump(model, model_path)

    # Leaderboard
    leaderboard = pd.DataFrame([{
        "model": "GradientBoosting",
        "score_val": test_auc,
        "fit_time": None,
    }])
    leaderboard_path = os.path.join(output_dir, "leaderboard.csv")
    leaderboard.to_csv(leaderboard_path, index=False)

    return {
        "model_path": model_dir,
        "leaderboard_path": leaderboard_path,
        "best_model": "GradientBoosting",
        "train_score": train_auc,
        "test_score": test_auc,
        "feature_columns": feature_cols,
        "n_models_trained": 1,
        "predictions": test_proba,
    }
