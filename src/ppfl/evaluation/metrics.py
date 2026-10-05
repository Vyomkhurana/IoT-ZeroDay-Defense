"""Binary detection metrics (attack = positive class)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b > 0 else float("nan")


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    y_true, y_pred = np.asarray(y_true).astype(bool), np.asarray(y_pred).astype(bool)
    return {
        "tp": int(np.sum(y_true & y_pred)),
        "fp": int(np.sum(~y_true & y_pred)),
        "tn": int(np.sum(~y_true & ~y_pred)),
        "fn": int(np.sum(y_true & ~y_pred)),
    }


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray | None = None) -> dict[str, float]:
    """Accuracy, precision, recall (detection rate), F1, FPR, TNR and, if scores are given, ROC/PR AUC.

    Undefined quantities (e.g. precision with no positive predictions) are NaN rather
    than silently 0, so that they are visible in reports.
    """
    c = confusion_counts(y_true, y_pred)
    tp, fp, tn, fn = c["tp"], c["fp"], c["tn"], c["fn"]
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * tp, 2 * tp + fp + fn)
    out = {
        **{k: float(v) for k, v in c.items()},
        "n": float(tp + fp + tn + fn),
        "accuracy": _safe_div(tp + tn, tp + fp + tn + fn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "fpr": _safe_div(fp, fp + tn),
        "tnr": _safe_div(tn, fp + tn),
        "balanced_accuracy": float(np.nanmean([recall, _safe_div(tn, fp + tn)])) if (tp + fn) and (tn + fp) else float("nan"),
    }
    if scores is not None:
        y = np.asarray(y_true).astype(int)
        both = 0 < y.sum() < len(y)
        out["roc_auc"] = float(roc_auc_score(y, scores)) if both else float("nan")
        out["pr_auc"] = float(average_precision_score(y, scores)) if both else float("nan")
    return out
