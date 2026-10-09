"""Shared evaluation metrics. Spoiler = positive class (1)."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate(y_true, y_pred, y_score=None) -> dict:
    """Return spoiler-class metrics for one set of predictions.

    y_true:  the real labels (1 = spoiler, 0 = not a spoiler)
    y_pred:  the model's predicted labels
    y_score: the model's score for "spoiler" (needed for PR-AUC)
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    results = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }

    if y_score is not None:
        results["pr_auc"] = average_precision_score(y_true, y_score)

    return results