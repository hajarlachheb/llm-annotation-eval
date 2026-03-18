"""
Explainability analysis for LLM annotation quality.

Provides:
- TF-IDF feature attribution (what text features predict LLM errors)
- SHAP values for fine-grained, per-sample explanations
- Rationale keyword profiling (when LLM rationales are available)
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

log = logging.getLogger(__name__)


# ── TF-IDF Feature Attribution ───────────────────────────────────────────────


def text_features_driving_errors(
    df: pd.DataFrame,
    text_col: str,
    human_col: str,
    llm_col: str,
    top_n: int = 20,
    max_features: int = 5000,
    ngram_range: Tuple[int, int] = (1, 2),
) -> Dict:
    """
    Train a lightweight model to predict LLM errors from text features.

    Fits TF-IDF + Logistic Regression to predict whether the LLM got
    each sample wrong, then extracts the features most associated with
    errors vs. correct predictions.

    Parameters
    ----------
    df : pd.DataFrame
    text_col : str
    human_col, llm_col : str
    top_n : int
        Number of top features to return per direction.
    max_features : int
        TF-IDF vocabulary cap.
    ngram_range : tuple

    Returns
    -------
    dict with error_predictors, correct_predictors, model_accuracy, cv_accuracy.
    """
    df = df.copy()
    df["_is_error"] = (df[human_col] != df[llm_col]).astype(int)

    if df["_is_error"].nunique() < 2:
        log.warning("All predictions are %s — cannot fit error model.",
                     "correct" if df["_is_error"].sum() == 0 else "incorrect")
        return {"error_predictors": [], "correct_predictors": [], "model_accuracy": None}

    tfidf = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words="english",
        sublinear_tf=True,
    )
    X = tfidf.fit_transform(df[text_col].fillna(""))
    y = df["_is_error"].values

    model = LogisticRegression(max_iter=1000, C=1.0, solver="liblinear")
    cv_scores = cross_val_score(model, X, y, cv=min(5, len(df)), scoring="accuracy")
    model.fit(X, y)

    feature_names = tfidf.get_feature_names_out()
    coefs = model.coef_[0]

    error_idx = coefs.argsort()[-top_n:][::-1]
    correct_idx = coefs.argsort()[:top_n]

    return {
        "error_predictors": [(feature_names[i], round(float(coefs[i]), 4)) for i in error_idx],
        "correct_predictors": [(feature_names[i], round(float(coefs[i]), 4)) for i in correct_idx],
        "model_accuracy": round(float(model.score(X, y)), 4),
        "cv_accuracy": round(float(cv_scores.mean()), 4),
        "cv_std": round(float(cv_scores.std()), 4),
        "error_rate": round(float(y.mean()), 4),
    }


# ── SHAP Values ─────────────────────────────────────────────────────────────


def compute_shap_explanations(
    df: pd.DataFrame,
    text_col: str,
    human_col: str,
    llm_col: str,
    max_features: int = 3000,
    ngram_range: Tuple[int, int] = (1, 2),
    n_background: int = 100,
    n_explain: int = 200,
) -> Dict:
    """
    Compute SHAP values to explain which text features drive LLM errors.

    Uses a TF-IDF + LogisticRegression pipeline with a SHAP LinearExplainer
    for fast, exact Shapley values.

    Parameters
    ----------
    df : pd.DataFrame
    text_col : str
    human_col, llm_col : str
    max_features : int
    ngram_range : tuple
    n_background : int
        Number of background samples for the SHAP explainer.
    n_explain : int
        Number of samples to compute SHAP values for.

    Returns
    -------
    dict with:
        - global_shap: list of (feature, mean_abs_shap) sorted descending
        - shap_values: np.ndarray of shape (n_explain, n_features)
        - feature_names: list of str
        - expected_value: float
        - error_sample_ids: index of explained samples
    """
    import shap

    df = df.copy()
    df["_is_error"] = (df[human_col] != df[llm_col]).astype(int)

    if df["_is_error"].nunique() < 2:
        log.warning("Cannot compute SHAP — no variance in error labels.")
        return {"global_shap": [], "shap_values": None}

    tfidf = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words="english",
        sublinear_tf=True,
    )
    X = tfidf.fit_transform(df[text_col].fillna(""))
    y = df["_is_error"].values
    feature_names = list(tfidf.get_feature_names_out())

    model = LogisticRegression(max_iter=1000, C=1.0, solver="liblinear")
    model.fit(X, y)

    n_bg = min(n_background, X.shape[0])
    n_exp = min(n_explain, X.shape[0])

    bg_idx = np.random.RandomState(42).choice(X.shape[0], n_bg, replace=False)
    exp_idx = np.random.RandomState(43).choice(X.shape[0], n_exp, replace=False)

    explainer = shap.LinearExplainer(model, X[bg_idx], feature_perturbation="interventional")
    shap_values = explainer.shap_values(X[exp_idx])

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    sorted_idx = mean_abs_shap.argsort()[::-1]
    global_importance = [
        (feature_names[i], round(float(mean_abs_shap[i]), 6))
        for i in sorted_idx[:50]
    ]

    return {
        "global_shap": global_importance,
        "shap_values": shap_values,
        "feature_names": feature_names,
        "expected_value": float(explainer.expected_value),
        "explained_indices": exp_idx.tolist(),
        "model_accuracy": round(float(model.score(X, y)), 4),
    }


def save_shap_summary_plot(
    shap_result: Dict,
    save_path: Union[str, Path],
    max_display: int = 20,
    title: str = "SHAP Feature Importance — LLM Error Prediction",
) -> None:
    """
    Generate and save a SHAP summary bar plot.

    Parameters
    ----------
    shap_result : dict
        Output of compute_shap_explanations().
    save_path : str or Path
    max_display : int
    title : str
    """
    import matplotlib.pyplot as plt

    global_shap = shap_result["global_shap"][:max_display]
    if not global_shap:
        return

    features, importances = zip(*reversed(global_shap))

    fig, ax = plt.subplots(figsize=(10, max(6, len(features) * 0.35)))
    colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(features)))
    ax.barh(range(len(features)), importances, color=colors)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features, fontsize=9)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(title)

    for i, v in enumerate(importances):
        ax.text(v + max(importances) * 0.01, i, f"{v:.4f}", va="center", fontsize=8)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved SHAP summary plot to %s", save_path)


def save_shap_beeswarm_plot(
    shap_result: Dict,
    save_path: Union[str, Path],
    max_display: int = 20,
) -> None:
    """
    Generate and save a SHAP beeswarm plot showing per-sample feature effects.

    Parameters
    ----------
    shap_result : dict
        Output of compute_shap_explanations().
    save_path : str or Path
    max_display : int
    """
    import shap
    import matplotlib.pyplot as plt

    sv = shap_result.get("shap_values")
    if sv is None:
        return

    explanation = shap.Explanation(
        values=sv,
        base_values=shap_result["expected_value"],
        feature_names=shap_result["feature_names"],
    )

    fig = plt.figure(figsize=(10, max(6, max_display * 0.4)))
    shap.plots.beeswarm(explanation, max_display=max_display, show=False)
    plt.title("SHAP Beeswarm — Per-Sample Feature Effects on Error Prediction")
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved SHAP beeswarm plot to %s", save_path)


# ── Confidence-Based Explanations ────────────────────────────────────────────


def high_confidence_errors(
    df: pd.DataFrame,
    human_col: str,
    llm_col: str,
    confidence_col: str = "confidence",
    text_col: str = "text",
    threshold: float = 0.8,
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Find cases where the LLM was highly confident but wrong.

    These are the most dangerous failure modes — the model doesn't
    know what it doesn't know.
    """
    df = df.copy()
    df["_correct"] = df[human_col] == df[llm_col]
    mask = (df[confidence_col] >= threshold) & (~df["_correct"])
    result = (
        df[mask]
        .sort_values(confidence_col, ascending=False)
        .head(top_n)
        [[text_col, human_col, llm_col, confidence_col]]
    )
    return result


# ── Rationale Analysis ───────────────────────────────────────────────────────


def extract_rationale_keywords(
    df: pd.DataFrame,
    rationale_col: str,
    llm_col: str,
    top_n: int = 20,
) -> Dict[str, List[Tuple[str, int]]]:
    """
    For each predicted label, find the most common words in rationales.

    Reveals what textual cues the LLM relies on for each class decision.
    """
    from collections import Counter

    STOPWORDS = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "to", "of", "in", "for",
        "on", "with", "at", "by", "from", "as", "into", "through", "during",
        "before", "after", "and", "but", "or", "nor", "not", "so", "yet",
        "both", "either", "neither", "each", "every", "all", "any", "few",
        "more", "most", "other", "some", "such", "no", "only", "own", "same",
        "than", "too", "very", "just", "because", "if", "when", "where",
        "how", "what", "which", "who", "whom", "this", "that", "these",
        "those", "it", "its", "i", "me", "my", "we", "our", "you", "your",
        "he", "him", "his", "she", "her", "they", "them", "their", "text",
        "indicates", "suggests", "contains", "uses", "overall",
    }

    profiles = {}
    for label, group in df.groupby(llm_col):
        texts = " ".join(group[rationale_col].fillna("")).lower()
        words = [w.strip(".,!?;:\"'()[]") for w in texts.split()]
        words = [w for w in words if len(w) > 2 and w not in STOPWORDS]
        profiles[label] = Counter(words).most_common(top_n)

    return profiles
