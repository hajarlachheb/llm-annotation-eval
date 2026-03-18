"""Slice-based evaluation — compute metrics across data subsets."""

from typing import Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd


def bin_continuous_column(
    df: pd.DataFrame,
    column: str,
    bins: List[Union[int, float]],
    bin_labels: Optional[List[str]] = None,
    target_column: str = None,
) -> pd.Series:
    """
    Bin a continuous column into discrete buckets.

    Parameters
    ----------
    df : pd.DataFrame
    column : str
        Source column (e.g. 'text_length').
    bins : list of numbers
        Bin edges.
    bin_labels : list of str, optional
        Labels for each bin.
    target_column : str, optional
        If provided, the binned column name; otherwise returns the Series.
    """
    binned = pd.cut(df[column], bins=bins, labels=bin_labels, include_lowest=True)
    if target_column:
        df[target_column] = binned
    return binned


def evaluate_by_slice(
    df: pd.DataFrame,
    slice_column: str,
    human_col: str,
    llm_col: str,
    metric_fn: Callable,
    min_samples: int = 10,
    **metric_kwargs,
) -> Dict[str, Dict]:
    """
    Compute a metric for each unique value in `slice_column`.

    Parameters
    ----------
    df : pd.DataFrame
    slice_column : str
        Column defining the slices (e.g. 'domain', 'text_length_bin').
    human_col : str
        Column with human labels.
    llm_col : str
        Column with LLM labels.
    metric_fn : callable
        Function(human_labels, llm_labels, **kwargs) -> dict or float.
    min_samples : int
        Skip slices with fewer samples than this.
    **metric_kwargs : passed through to metric_fn.

    Returns
    -------
    dict mapping slice_value -> metric result.
    """
    results = {}
    for slice_val, group in df.groupby(slice_column, observed=False):
        if len(group) < min_samples:
            continue
        result = metric_fn(
            group[human_col].values,
            group[llm_col].values,
            **metric_kwargs,
        )
        results[str(slice_val)] = result
    return results
