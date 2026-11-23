"""Core multi-class metrics and closed-form mirage threshold calculations."""

from __future__ import annotations

from typing import Dict, Optional, Sequence
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


def _validate_inputs(
    y_true: Sequence[int],
    y_prob: np.ndarray,
    y_pred: Optional[Sequence[int]] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    yt = np.asarray(y_true, dtype=int)
    probs = np.asarray(y_prob, dtype=float)
    if probs.ndim != 2:
        raise ValueError(f"y_prob must be 2D (n_samples, n_classes), got shape {probs.shape}")
    if len(yt) != len(probs):
        raise ValueError(f"Length mismatch: len(y_true)={len(yt)} vs len(y_prob)={len(probs)}")
    yp = np.asarray(y_pred, dtype=int) if y_pred is not None else np.argmax(probs, axis=1)
    if len(yp) != len(yt):
        raise ValueError(f"Length mismatch: len(y_pred)={len(yp)} vs len(y_true)={len(yt)}")
    return yt, probs, yp


def compute_split_metrics(
    y_true: Sequence[int],
    y_prob: np.ndarray,
    y_pred: Optional[Sequence[int]] = None,
    class_names: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    """Compute percentage accuracy, BAcc, Macro-F1, MCC, AUROC, AUPRC, and confusion matrix."""
    yt, probs, yp = _validate_inputs(y_true, y_prob, y_pred)
    n_classes = probs.shape[1]
    labels = list(range(n_classes))
    names = list(class_names) if class_names is not None else [f"Class_{i}" for i in labels]
    if len(names) != n_classes:
        raise ValueError(f"Expected {n_classes} class names, got {len(names)}")

    row_sums = probs.sum(axis=1, keepdims=True)
    probs = np.where(row_sums > 0, probs / np.maximum(row_sums, 1e-12), 1.0 / n_classes)

    y_bin = label_binarize(yt, classes=labels)
    if n_classes == 2 and y_bin.shape[1] == 1:
        y_bin = np.hstack([1 - y_bin, y_bin])

    per_class_ap: Dict[str, float] = {}
    per_class_rec: Dict[str, float] = {}
    for c, cname in enumerate(names):
        mask_c = yt == c
        per_class_rec[cname] = float((yp[mask_c] == c).mean() * 100.0) if mask_c.any() else float("nan")
        if y_bin[:, c].sum() > 0:
            per_class_ap[cname] = float(average_precision_score(y_bin[:, c], probs[:, c]) * 100.0)
        else:
            per_class_ap[cname] = float("nan")

    try:
        macro_auroc = float(
            roc_auc_score(yt, probs, labels=labels, multi_class="ovr", average="macro") * 100.0
        )
    except ValueError:
        macro_auroc = float("nan")

    valid_aps = [v for v in per_class_ap.values() if not np.isnan(v)]
    macro_auprc = float(np.mean(valid_aps)) if valid_aps else float("nan")

    cm = confusion_matrix(yt, yp, labels=labels)
    return {
        "acc": float(accuracy_score(yt, yp) * 100.0),
        "bacc": float(balanced_accuracy_score(yt, yp) * 100.0),
        "f1": float(f1_score(yt, yp, average="macro", zero_division=0) * 100.0),
        "mcc": float(matthews_corrcoef(yt, yp) * 100.0),
        "auroc": macro_auroc,
        "auprc": macro_auprc,
        "per_class_ap": per_class_ap,
        "per_class_recall": per_class_rec,
        "confusion_matrix": cm.tolist(),
    }


def theoretical_mirage_threshold(
    lab_acc: float,
    dominant_recall: float,
    other_recall: float,
) -> float:
    """Compute exact critical prevalence p* = (Acc_L - r_{-1}) / (r_1 - r_{-1}) in % units.

    Inputs may all be fractions in [0, 1] or percentages in [0, 100]. Returns same scale.
    Negative p* means r_{-1} >= Acc_L so raw accuracy rises immediately for any positive
    dominant-class shift if CMD recall exceeds other-class recall.
    """
    denom = float(dominant_recall) - float(other_recall)
    if abs(denom) < 1e-12:
        return float("inf") if lab_acc > other_recall else float("-inf")
    ratio = (float(lab_acc) - float(other_recall)) / denom
    scale = 100.0 if max(abs(lab_acc), abs(dominant_recall), abs(other_recall)) > 1.5 else 1.0
    return float(ratio * scale)
