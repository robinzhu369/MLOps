"""Strategy tools — threshold analysis for business decision making."""

import numpy as np
import pandas as pd


def build_threshold_table(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    thresholds: list[float] | None = None,
    n_bins: int = 20,
) -> pd.DataFrame:
    """Build a threshold strategy table.

    For each threshold, computes:
    - pass_rate: fraction of samples below threshold (approved)
    - reject_rate: fraction above threshold (rejected)
    - pass_bad_rate: bad rate among passed samples
    - capture_rate: fraction of bad samples captured (rejected)

    Args:
        y_true: True binary labels.
        y_proba: Predicted probabilities for positive class.
        thresholds: Explicit thresholds. If None, generates n_bins evenly spaced.
        n_bins: Number of threshold bins if thresholds not provided.

    Returns:
        DataFrame with columns: threshold, pass_rate, reject_rate,
        pass_bad_rate, capture_rate, n_pass, n_reject.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_proba = np.asarray(y_proba, dtype=float)

    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, n_bins + 1)[1:-1].tolist()

    n_total = len(y_true)
    n_bad_total = int(y_true.sum())

    records = []
    for t in thresholds:
        # Samples with proba < threshold are "passed" (low risk)
        pass_mask = y_proba < t
        reject_mask = ~pass_mask

        n_pass = int(pass_mask.sum())
        n_reject = int(reject_mask.sum())

        pass_rate = n_pass / n_total if n_total > 0 else 0.0
        reject_rate = n_reject / n_total if n_total > 0 else 0.0

        # Bad rate among passed
        pass_bad = int(y_true[pass_mask].sum()) if n_pass > 0 else 0
        pass_bad_rate = pass_bad / n_pass if n_pass > 0 else 0.0

        # Capture rate: bad samples in reject / total bad
        reject_bad = int(y_true[reject_mask].sum()) if n_reject > 0 else 0
        capture_rate = reject_bad / n_bad_total if n_bad_total > 0 else 0.0

        records.append({
            "threshold": round(t, 4),
            "pass_rate": round(pass_rate, 4),
            "reject_rate": round(reject_rate, 4),
            "pass_bad_rate": round(pass_bad_rate, 4),
            "capture_rate": round(capture_rate, 4),
            "n_pass": n_pass,
            "n_reject": n_reject,
        })

    return pd.DataFrame(records)
