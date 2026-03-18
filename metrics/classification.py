"""Classification-level evaluation metrics."""

from typing import Dict, List, Optional, Union

import numpy as np
from sklearn.metrics import (
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_classification_metrics(
    human_labels: Union[List[str], np.ndarray],
    llm_labels: Union[List[str], np.ndarray],
    labels: Optional[List[str]] = None,
    average: str = "weighted",
) -> Dict:
    """
    Compute standard classification quality metrics.

    Parameters
    ----------
    human_labels : array-like
        Ground-truth labels from human annotators.
    llm_labels : array-like
        Predicted labels from the LLM.
    labels : list of str, optional
        Explicit label ordering for confusion matrix and per-class metrics.
    average : str
        Averaging method for precision/recall/F1 ('weighted', 'macro', 'micro').

    Returns
    -------
    dict with keys: precision, recall, f1, cohen_kappa, per_class_report,
    confusion_matrix.
    """
    return {
        "precision": float(precision_score(human_labels, llm_labels, average=average, labels=labels, zero_division=0)),
        "recall": float(recall_score(human_labels, llm_labels, average=average, labels=labels, zero_division=0)),
        "f1": float(f1_score(human_labels, llm_labels, average=average, labels=labels, zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(human_labels, llm_labels)),
        "per_class_report": classification_report(human_labels, llm_labels, labels=labels, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(human_labels, llm_labels, labels=labels).tolist(),
    }


def compute_krippendorff_alpha(
    annotations_matrix: np.ndarray,
    level_of_measurement: str = "nominal",
) -> float:
    """
    Compute Krippendorff's Alpha for inter-rater reliability.

    Parameters
    ----------
    annotations_matrix : np.ndarray
        Shape (n_raters, n_items). Use np.nan for missing annotations.
    level_of_measurement : str
        One of 'nominal', 'ordinal', 'interval', 'ratio'.

    Returns
    -------
    float — alpha coefficient.
    """
    import krippendorff

    return float(
        krippendorff.alpha(
            reliability_data=annotations_matrix,
            level_of_measurement=level_of_measurement,
        )
    )
