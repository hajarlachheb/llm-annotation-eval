"""Systematic error detection and sampling for annotation quality."""

from typing import Dict, List, Optional, Tuple

import pandas as pd


def find_systematic_errors(
    df: pd.DataFrame,
    human_col: str = "label_human",
    llm_col: str = "label_llm",
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Identify the most frequent (human_label, llm_label) mismatch patterns.

    Parameters
    ----------
    df : pd.DataFrame
    human_col, llm_col : str
    top_n : int
        Number of top error patterns to return.

    Returns
    -------
    pd.DataFrame with columns: label_human, label_llm, count, pct_of_errors.
    """
    errors = df[df[human_col] != df[llm_col]].copy()
    if errors.empty:
        return pd.DataFrame(columns=[human_col, llm_col, "count", "pct_of_errors"])

    pattern_counts = (
        errors.groupby([human_col, llm_col])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(top_n)
    )
    pattern_counts["pct_of_errors"] = (
        pattern_counts["count"] / len(errors) * 100
    ).round(2)

    return pattern_counts


def sample_error_examples(
    df: pd.DataFrame,
    human_col: str = "label_human",
    llm_col: str = "label_llm",
    text_col: str = "text",
    n_per_pattern: int = 3,
    top_patterns: int = 5,
    random_state: int = 42,
) -> Dict[Tuple[str, str], List[str]]:
    """
    Sample example texts for the top error patterns.

    Parameters
    ----------
    df : pd.DataFrame
    human_col, llm_col, text_col : str
    n_per_pattern : int
        Number of example texts to sample per error pattern.
    top_patterns : int
        Number of top error patterns to include.
    random_state : int

    Returns
    -------
    dict mapping (human_label, llm_label) -> list of example texts.
    """
    errors = df[df[human_col] != df[llm_col]].copy()
    if errors.empty:
        return {}

    top = (
        errors.groupby([human_col, llm_col])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(top_patterns)
    )

    result = {}
    for _, row in top.iterrows():
        mask = (errors[human_col] == row[human_col]) & (errors[llm_col] == row[llm_col])
        subset = errors[mask]
        sample = subset.sample(n=min(n_per_pattern, len(subset)), random_state=random_state)
        key = (row[human_col], row[llm_col])
        result[key] = sample[text_col].tolist()

    return result
