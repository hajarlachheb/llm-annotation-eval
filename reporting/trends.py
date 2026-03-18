"""Multi-model / multi-version trend tracking."""

from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def build_trend_table(
    history: List[Dict],
) -> pd.DataFrame:
    """
    Convert a list of evaluation snapshots into a trend DataFrame.

    Parameters
    ----------
    history : list of dict
        Each dict should have at least 'model', 'version' (or 'timestamp'),
        and metric fields like 'f1', 'precision', 'recall', 'cohen_kappa'.

    Returns
    -------
    pd.DataFrame
    """
    return pd.DataFrame(history)


def plot_metric_trends(
    trend_df: pd.DataFrame,
    metric: str = "f1",
    group_by: str = "model",
    x_axis: str = "version",
    title: Optional[str] = None,
    save_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (12, 6),
) -> plt.Figure:
    """
    Line plot showing a metric over versions, grouped by model.

    Parameters
    ----------
    trend_df : pd.DataFrame
        Output of build_trend_table().
    metric : str
        Column name for the metric to plot.
    group_by : str
        Column used to color separate lines (e.g. 'model').
    x_axis : str
        Column for the x-axis (e.g. 'version', 'timestamp').
    title : str, optional
    save_path : path, optional
    figsize : tuple
    """
    fig, ax = plt.subplots(figsize=figsize)
    sns.lineplot(data=trend_df, x=x_axis, y=metric, hue=group_by, marker="o", ax=ax)
    ax.set_title(title or f"{metric.upper()} Over Versions")
    ax.set_ylabel(metric.upper())
    ax.set_ylim(0, 1.05)
    ax.legend(title=group_by.title())
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
