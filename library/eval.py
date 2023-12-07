"""This module contains functions for evaluating predictions."""

import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix
)

import numpy.typing as npt

def eval_preds(y: npt.ArrayLike, probs: npt.ArrayLike) -> dict[str, float]:
    pred = probs.argmax(axis=1)
    return dict(
        acc = accuracy_score(y, pred),
        bal_acc = balanced_accuracy_score(y, pred),
        f1_macro = f1_score(y, pred, average="macro"),
        f1_micro = f1_score(y, pred, average="micro"),
        precision_macro = precision_score(y, pred, average="macro"),
        precision_micro = precision_score(y, pred, average="micro"),
        recall_macro = recall_score(y, pred, average="macro"),
        recall_micro = recall_score(y, pred, average="micro"),
        auc = roc_auc_score(y, probs, multi_class="ovr")
    )

def confusion_matrix_plot(y: npt.ArrayLike, probs: npt.ArrayLike, labels: list[str]) -> "pyplot.figure":
    cm = confusion_matrix(y, probs.argmax(axis=1))
    fig, ax = plt.subplots(figsize=(12, 10))
    ax = sb.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    return fig

