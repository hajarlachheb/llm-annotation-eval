"""Bootstrap resampling for confidence intervals on evaluation metrics."""

from typing import Callable, Dict, Union

import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score, precision_score, recall_score


def bootstrap_metric(
    human_labels: np.ndarray,
    llm_labels: np.ndarray,
    metric_fn: Callable = f1_score,
    n_iterations: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 42,
    **metric_kwargs,
) -> Dict[str, float]:
    """
    Estimate a metric's distribution via bootstrap resampling.

    Parameters
    ----------
    human_labels, llm_labels : array-like
    metric_fn : callable
        sklearn-compatible metric(y_true, y_pred, **kwargs).
    n_iterations : int
    confidence_level : float
    random_state : int
    **metric_kwargs : passed to metric_fn.

    Returns
    -------
    dict with mean, std, ci_lower, ci_upper.
    """
    rng = np.random.RandomState(random_state)
    human_arr = np.asarray(human_labels)
    llm_arr = np.asarray(llm_labels)
    n = len(human_arr)

    scores = []
    for _ in range(n_iterations):
        idx = rng.choice(n, size=n, replace=True)
        try:
            score = metric_fn(human_arr[idx], llm_arr[idx], **metric_kwargs)
        except Exception:
            continue
        scores.append(score)

    scores = np.array(scores)
    alpha = 1 - confidence_level
    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "ci_lower": float(np.percentile(scores, alpha / 2 * 100)),
        "ci_upper": float(np.percentile(scores, (1 - alpha / 2) * 100)),
        "n_successful": len(scores),
    }


def bootstrap_all_metrics(
    human_labels: np.ndarray,
    llm_labels: np.ndarray,
    n_iterations: int = 1000,
    confidence_level: float = 0.95,
    average: str = "weighted",
) -> Dict[str, Dict[str, float]]:
    """Compute bootstrap CIs for precision, recall, F1, and Cohen's Kappa."""
    common = dict(
        human_labels=human_labels,
        llm_labels=llm_labels,
        n_iterations=n_iterations,
        confidence_level=confidence_level,
    )
    return {
        "precision": bootstrap_metric(**common, metric_fn=precision_score, average=average, zero_division=0),
        "recall": bootstrap_metric(**common, metric_fn=recall_score, average=average, zero_division=0),
        "f1": bootstrap_metric(**common, metric_fn=f1_score, average=average, zero_division=0),
        "cohen_kappa": bootstrap_metric(**common, metric_fn=cohen_kappa_score),
    }
