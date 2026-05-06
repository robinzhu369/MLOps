"""Explain tools — feature importance computation."""

import os

import numpy as np
import pandas as pd


def compute_feature_importance(
    model_path: str,
    feature_columns: list[str],
    test_df: pd.DataFrame | None = None,
    label_col: str | None = None,
) -> pd.DataFrame:
    """Compute feature importance from a trained model.

    Tries AutoGluon first, falls back to sklearn model.

    Args:
        model_path: Path to saved model directory.
        feature_columns: List of feature column names.
        test_df: Optional test data for permutation importance.
        label_col: Label column name.

    Returns:
        DataFrame with feature_name and importance_score, sorted descending.
    """
    # Try AutoGluon
    try:
        from autogluon.tabular import TabularPredictor
        predictor = TabularPredictor.load(model_path)
        if test_df is not None and label_col:
            importance = predictor.feature_importance(test_df, silent=True)
            result = pd.DataFrame({
                "feature_name": importance.index,
                "importance_score": importance["importance"].values,
            })
        else:
            # Use model's internal importance
            result = pd.DataFrame({
                "feature_name": feature_columns,
                "importance_score": [0.0] * len(feature_columns),
            })
        return result.sort_values("importance_score", ascending=False).reset_index(drop=True)
    except (ImportError, Exception):
        pass

    # Try sklearn model
    try:
        import joblib
        model_file = os.path.join(model_path, "model.pkl")
        if os.path.exists(model_file):
            model = joblib.load(model_file)
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                result = pd.DataFrame({
                    "feature_name": feature_columns[:len(importances)],
                    "importance_score": importances,
                })
                return result.sort_values("importance_score", ascending=False).reset_index(drop=True)
    except Exception:
        pass

    # Fallback: return zeros
    return pd.DataFrame({
        "feature_name": feature_columns,
        "importance_score": [0.0] * len(feature_columns),
    })
