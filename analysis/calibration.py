"""
Confidence calibration analysis.

Measures whether the LLM's reported confidence aligns with actual accuracy.
A well-calibrated model should be right 80% of the time when it says 0.80.
"""

import logging
from typing import Dict, Optional, Tuple, Union
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def calibration_curve(
    df: pd.DataFrame,
    confidence_col: str,
    human_col: str,
    llm_col: str,
    n_bins: int = 10,
) -> Tuple[pd.DataFrame, float]:
    """
    Compute calibration curve and Expected Calibration Error (ECE).

    Parameters
    ----------
    df : pd.DataFrame
    confidence_col : str
    human_col, llm_col : str
    n_bins : int

    Returns
    -------
    (calibration_df, ece)
        calibration_df: DataFrame with columns mean_confidence, actual_accuracy, count.
        ece: Expected Calibration Error (lower is better, 0 = perfect calibration).
    """
    df = df.copy()
    df["_correct"] = (df[human_col] == df[llm_col]).astype(int)
    df["_conf_bin"] = pd.cut(df[confidence_col], bins=n_bins, include_lowest=True)

    cal = (
        df.groupby("_conf_bin", observed=False)
        .agg(
            mean_confidence=(confidence_col, "mean"),
            actual_accuracy=("_correct", "mean"),
            count=("_correct", "count"),
        )
        .dropna(subset=["mean_confidence"])
        .reset_index()
    )

    total = cal["count"].sum()
    if total == 0:
        return cal, 0.0

    ece = float(
        (cal["count"] / total * (cal["mean_confidence"] - cal["actual_accuracy"]).abs()).sum()
    )

    return cal, ece


def confidence_stratified_metrics(
    df: pd.DataFrame,
    confidence_col: str,
    human_col: str,
    llm_col: str,
    bins: int = 4,
) -> pd.DataFrame:
    """
    Break down accuracy, error count, and mean confidence per confidence quartile.

    Parameters
    ----------
    df : pd.DataFrame
    confidence_col, human_col, llm_col : str
    bins : int

    Returns
    -------
    pd.DataFrame per confidence tier.
    """
    df = df.copy()
    df["_correct"] = (df[human_col] == df[llm_col]).astype(int)
    df["_conf_tier"] = pd.qcut(df[confidence_col], q=bins, duplicates="drop")

    return (
        df.groupby("_conf_tier", observed=False)
        .agg(
            n_samples=("_correct", "count"),
            accuracy=("_correct", "mean"),
            n_errors=("_correct", lambda x: (x == 0).sum()),
            mean_confidence=(confidence_col, "mean"),
        )
        .reset_index()
    )


def find_overconfident_errors(
    df: pd.DataFrame,
    confidence_col: str,
    human_col: str,
    llm_col: str,
    text_col: str = "text",
    threshold: float = 0.85,
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Find samples where the LLM was highly confident but wrong.

    These represent the model's blind spots — it doesn't know what it
    doesn't know.
    """
    df = df.copy()
    df["_correct"] = df[human_col] == df[llm_col]
    mask = (~df["_correct"]) & (df[confidence_col] >= threshold)

    cols = [text_col, human_col, llm_col, confidence_col]
    cols = [c for c in cols if c in df.columns]

    return (
        df[mask]
        .sort_values(confidence_col, ascending=False)
        .head(top_n)[cols]
        .reset_index(drop=True)
    )


def find_underconfident_correct(
    df: pd.DataFrame,
    confidence_col: str,
    human_col: str,
    llm_col: str,
    text_col: str = "text",
    threshold: float = 0.4,
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Find samples where the LLM had low confidence but was actually correct.

    These indicate the model is being overly cautious on certain patterns.
    """
    df = df.copy()
    df["_correct"] = df[human_col] == df[llm_col]
    mask = df["_correct"] & (df[confidence_col] <= threshold)

    cols = [text_col, human_col, llm_col, confidence_col]
    cols = [c for c in cols if c in df.columns]

    return (
        df[mask]
        .sort_values(confidence_col, ascending=True)
        .head(top_n)[cols]
        .reset_index(drop=True)
    )
