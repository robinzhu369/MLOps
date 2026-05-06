"""Metric tools — binary classification evaluation metrics."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


def compute_ks(y_true: np.ndarray, y_proba: np.ndarray) -> dict:
    """Compute KS (Kolmogorov-Smirnov) statistic.

    KS = max|TPR - FPR| across all thresholds.

    Args:
        y_true: True binary labels.
        y_proba: Predicted probabilities for positive class.

    Returns:
        dict with ks_statistic, ks_threshold, ks_decile_table.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_proba = np.asarray(y_proba, dtype=float)

    # Sort by predicted probability descending
    sorted_idx = np.argsort(-y_proba)
    y_true_sorted = y_true[sorted_idx]
    y_proba_sorted = y_proba[sorted_idx]

    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos

    if n_pos == 0 or n_neg == 0:
        return {"ks_statistic": 0.0, "ks_threshold": 0.5, "ks_decile_table": []}

    # Cumulative rates
    cum_pos = np.cumsum(y_true_sorted) / n_pos
    cum_neg = np.cumsum(1 - y_true_sorted) / n_neg

    ks_values = np.abs(cum_pos - cum_neg)
    ks_idx = np.argmax(ks_values)
    ks_statistic = float(ks_values[ks_idx])
    ks_threshold = float(y_proba_sorted[ks_idx])

    # Decile table
    n = len(y_true)
    n_bins = min(10, n)
    decile_size = n // n_bins if n_bins > 0 else 0
    decile_table = []

    for i in range(n_bins):
        start = i * decile_size
        end = (i + 1) * decile_size if i < n_bins - 1 else n
        decile_true = y_true_sorted[start:end]
        decile_proba = y_proba_sorted[start:end]

        if len(decile_true) == 0:
            continue

        decile_table.append({
            "decile": i + 1,
            "count": int(end - start),
            "n_positive": int(decile_true.sum()),
            "positive_rate": round(float(decile_true.mean()), 4),
            "min_proba": round(float(decile_proba.min()), 4),
            "max_proba": round(float(decile_proba.max()), 4),
            "cum_positive_rate": round(float(y_true_sorted[:end].sum() / n_pos), 4),
            "cum_negative_rate": round(float((1 - y_true_sorted[:end]).sum() / n_neg), 4),
        })

    return {
        "ks_statistic": round(ks_statistic, 4),
        "ks_threshold": round(ks_threshold, 4),
        "ks_decile_table": decile_table,
    }


def evaluate_binary_model(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """Compute all binary classification metrics.

    Args:
        y_true: True binary labels.
        y_proba: Predicted probabilities for positive class.
        threshold: Classification threshold.

    Returns:
        dict with auc, ks, accuracy, precision, recall, f1, confusion_matrix.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_proba = np.asarray(y_proba, dtype=float)
    y_pred = (y_proba >= threshold).astype(int)

    auc = float(roc_auc_score(y_true, y_proba))
    ks_result = compute_ks(y_true, y_proba)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    return {
        "auc": round(auc, 4),
        "ks": ks_result["ks_statistic"],
        "ks_threshold": ks_result["ks_threshold"],
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        },
        "threshold": threshold,
        "ks_decile_table": ks_result["ks_decile_table"],
    }
