"""Summary tables for evaluation results."""

from pathlib import Path
from typing import Dict, Optional, Union

import pandas as pd


def build_summary_table(
    all_results: Dict[str, Dict],
    metrics_keys: tuple = ("precision", "recall", "f1", "cohen_kappa"),
) -> pd.DataFrame:
    """
    Build a summary DataFrame from per-model metric dicts.

    Parameters
    ----------
    all_results : dict
        Mapping model_name -> metrics dict (output of compute_classification_metrics).
    metrics_keys : tuple of str
        Which top-level metric keys to include.

    Returns
    -------
    pd.DataFrame indexed by model name.
    """
    rows = []
    for model, metrics in all_results.items():
        row = {"model": model}
        for key in metrics_keys:
            if key in metrics:
                val = metrics[key]
                if isinstance(val, dict) and "mean" in val:
                    row[key] = val["mean"]
                    row[f"{key}_ci_lower"] = val.get("ci_lower")
                    row[f"{key}_ci_upper"] = val.get("ci_upper")
                else:
                    row[key] = val
        rows.append(row)

    return pd.DataFrame(rows).set_index("model")


def save_summary(
    df: pd.DataFrame,
    output_dir: Union[str, Path],
    filename: str = "summary",
    formats: tuple = ("csv", "json"),
) -> None:
    """Write summary table to disk in one or more formats."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if "csv" in formats:
        df.to_csv(output_dir / f"{filename}.csv")
    if "json" in formats:
        df.to_json(output_dir / f"{filename}.json", indent=2, orient="index")
