"""Visualization helpers: confusion matrices, per-class bar charts, heatmaps, calibration, SHAP."""

from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_confusion_matrix(
    cm: Union[np.ndarray, List[List[int]]],
    labels: List[str],
    title: str = "Confusion Matrix",
    save_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (10, 8),
    cmap: str = "Blues",
) -> plt.Figure:
    """Plot an annotated confusion-matrix heatmap."""
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap=cmap,
        xticklabels=labels, yticklabels=labels, ax=ax,
    )
    ax.set_xlabel("LLM Predicted")
    ax.set_ylabel("Human Ground Truth")
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_per_class_f1(
    per_class_report: Dict,
    title: str = "Per-Class F1 Score",
    save_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (12, 6),
) -> plt.Figure:
    """Horizontal bar chart of F1 scores per class."""
    skip = {"accuracy", "macro avg", "weighted avg", "micro avg"}
    classes = [k for k in per_class_report if k not in skip]
    f1s = [per_class_report[c]["f1-score"] for c in classes]

    sorted_pairs = sorted(zip(classes, f1s), key=lambda x: x[1])
    classes, f1s = zip(*sorted_pairs) if sorted_pairs else ([], [])

    fig, ax = plt.subplots(figsize=figsize)
    colors = sns.color_palette("viridis", len(classes))
    ax.barh(list(classes), list(f1s), color=colors)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("F1 Score")
    ax.set_title(title)

    for i, v in enumerate(f1s):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=9)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_agreement_heatmap(
    agreement_scores: Dict[str, float],
    title: str = "Inter-LLM Agreement (Cohen's Kappa)",
    save_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (8, 6),
) -> plt.Figure:
    """Heatmap of pairwise agreement between models."""
    models = set()
    for key in agreement_scores:
        parts = key.split(" vs ")
        models.update(parts)
    models = sorted(models)

    n = len(models)
    matrix = np.eye(n)
    model_idx = {m: i for i, m in enumerate(models)}

    for key, val in agreement_scores.items():
        a, b = key.split(" vs ")
        i, j = model_idx[a], model_idx[b]
        matrix[i, j] = val
        matrix[j, i] = val

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        matrix, annot=True, fmt=".3f", cmap="YlOrRd",
        xticklabels=models, yticklabels=models, ax=ax,
        vmin=0, vmax=1,
    )
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


# ── Calibration Plots ────────────────────────────────────────────────────────


def plot_reliability_diagram(
    cal_df: pd.DataFrame,
    ece: float,
    title: str = "Reliability Diagram",
    save_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (8, 7),
) -> plt.Figure:
    """
    Plot a reliability diagram (calibration curve).

    Shows how well the LLM's confidence aligns with actual accuracy.
    The diagonal represents perfect calibration.

    Parameters
    ----------
    cal_df : pd.DataFrame
        Output of calibration.calibration_curve(). Needs columns
        'mean_confidence' and 'actual_accuracy'.
    ece : float
        Expected Calibration Error.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, gridspec_kw={"height_ratios": [3, 1]})

    confs = cal_df["mean_confidence"].values
    accs = cal_df["actual_accuracy"].values
    counts = cal_df["count"].values

    ax1.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
    ax1.bar(confs, accs, width=0.08, alpha=0.7, color=sns.color_palette("deep")[0],
            edgecolor="white", label="Model")
    gaps = accs - confs
    for c, a, g in zip(confs, accs, gaps):
        if abs(g) > 0.02:
            color = "#d32f2f" if g < 0 else "#388e3c"
            ax1.annotate("", xy=(c, a), xytext=(c, c),
                         arrowprops=dict(arrowstyle="->", color=color, lw=1.5))

    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("Actual Accuracy")
    ax1.set_title(f"{title}  (ECE = {ece:.4f})")
    ax1.legend(loc="upper left")

    ax2.bar(confs, counts, width=0.08, color=sns.color_palette("deep")[1], alpha=0.7)
    ax2.set_xlabel("Mean Predicted Confidence")
    ax2.set_ylabel("Count")
    ax2.set_xlim(0, 1)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


# ── Feature Importance Plots ─────────────────────────────────────────────────


def plot_feature_importance(
    feature_result: Dict,
    title: str = "Text Features Driving LLM Errors",
    save_path: Optional[Union[str, Path]] = None,
    top_n: int = 15,
    figsize: tuple = (12, 8),
) -> plt.Figure:
    """
    Two-panel bar chart: features predicting errors (left) and correct (right).

    Parameters
    ----------
    feature_result : dict
        Output of explainability.text_features_driving_errors().
    """
    error_feats = feature_result.get("error_predictors", [])[:top_n]
    correct_feats = feature_result.get("correct_predictors", [])[:top_n]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    if error_feats:
        names, vals = zip(*reversed(error_feats))
        ax1.barh(range(len(names)), vals, color="#d32f2f", alpha=0.8)
        ax1.set_yticks(range(len(names)))
        ax1.set_yticklabels(names, fontsize=9)
        ax1.set_xlabel("Coefficient (higher = more error-prone)")
        ax1.set_title("Features Predicting ERRORS")

    if correct_feats:
        names, vals = zip(*reversed(correct_feats))
        abs_vals = [abs(v) for v in vals]
        ax2.barh(range(len(names)), abs_vals, color="#388e3c", alpha=0.8)
        ax2.set_yticks(range(len(names)))
        ax2.set_yticklabels(names, fontsize=9)
        ax2.set_xlabel("Coefficient (higher = more correct)")
        ax2.set_title("Features Predicting CORRECT")

    acc = feature_result.get("cv_accuracy")
    if acc is not None:
        fig.suptitle(f"{title}\n(Error predictor CV accuracy: {acc:.1%})", fontsize=12, y=1.02)
    else:
        fig.suptitle(title, fontsize=12, y=1.02)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig
