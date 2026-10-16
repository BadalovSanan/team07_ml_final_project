"""
evaluate.py — metrics + comparison helpers (FROM SCRATCH where reasonable).

Implement the metrics yourself in NumPy. scikit-learn may be used ONLY to
sanity-check your numbers or for the baseline models you compare against.
"""

from __future__ import annotations

import numpy as np


# --- Regression (Task A: price) --------------------------------------------
def rmse(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean(np.abs(y_true - y_pred)))


def r2(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    return float(1.0 - ss_res / ss_tot)


# --- Classification (Task B: price tier) -----------------------------------
def confusion_matrix(y_true, y_pred):
    """Return the 2x2 matrix [[TN, FP], [FN, TP]]."""
    y_true, y_pred = np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int)
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    return np.array([[tn, fp], [fn, tp]])


def precision_recall_f1(y_true, y_pred):
    """Return (precision, recall, f1) for the positive (premium) class."""
    (tn, fp), (fn, tp) = confusion_matrix(y_true, y_pred)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return float(precision), float(recall), float(f1)


def roc_auc(y_true, scores) -> float:
    """
    AUC from decision scores (e.g. SVM margins or tree probabilities).
    Rank-based (Mann-Whitney) computation is fine.
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, float)
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    # average ranks (ties share the mean rank)
    uniq, inv, cnt = np.unique(scores, return_inverse=True, return_counts=True)
    ranks = (np.cumsum(cnt) - (cnt - 1) / 2.0)[inv]
    u = ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


# --- Comparison helper ------------------------------------------------------
def compare(name_to_metrics: dict) -> None:
    """
    Pretty-print a table: your models vs the scikit-learn baselines.
    Keep the SAME preprocessing/split for a fair comparison.
    """
    # values can be a number or a (mean, std) tuple
    cols = []
    for m in name_to_metrics.values():
        cols += [k for k in m if k not in cols]

    def fmt(v):
        if v is None:
            return "-"
        return f"{v[0]:.4g} ± {v[1]:.2g}" if isinstance(v, (tuple, list)) else f"{v:.4g}"

    nw = max(len(n) for n in name_to_metrics) + 2
    cw = {c: max([len(c)] + [len(fmt(m.get(c))) for m in name_to_metrics.values()]) + 2 for c in cols}
    print("".ljust(nw) + "".join(c.rjust(cw[c]) for c in cols))
    print("-" * (nw + sum(cw.values())))
    for name, m in name_to_metrics.items():
        print(name.ljust(nw) + "".join(fmt(m.get(c)).rjust(cw[c]) for c in cols))