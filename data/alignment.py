"""Utilities for aligning and normalizing human vs. LLM annotations."""

from typing import Dict, List, Optional

import pandas as pd


def normalize_labels(
    labels: List[str],
    label_map: Optional[Dict[str, str]] = None,
) -> List[str]:
    """
    Normalize labels: strip whitespace, lowercase, and optionally remap.

    Parameters
    ----------
    labels : list of str
    label_map : dict, optional
        Mapping from raw label to canonical label (applied after lowercasing).
    """
    normalized = [str(l).strip().lower() for l in labels]
    if label_map:
        normalized = [label_map.get(l, l) for l in normalized]
    return normalized


def validate_alignment(
    human_df: pd.DataFrame,
    llm_df: pd.DataFrame,
    id_column: str = "sample_id",
) -> None:
    """
    Validate that human and LLM DataFrames cover the same sample IDs.

    Raises ValueError with details on mismatches.
    """
    human_ids = set(human_df[id_column])
    llm_ids = set(llm_df[id_column])

    missing_in_llm = human_ids - llm_ids
    extra_in_llm = llm_ids - human_ids

    issues = []
    if missing_in_llm:
        issues.append(
            f"{len(missing_in_llm)} samples in human data missing from LLM output "
            f"(e.g. {sorted(missing_in_llm)[:5]})"
        )
    if extra_in_llm:
        issues.append(
            f"{len(extra_in_llm)} samples in LLM output not in human data "
            f"(e.g. {sorted(extra_in_llm)[:5]})"
        )
    if issues:
        raise ValueError("Alignment issues:\n  - " + "\n  - ".join(issues))


def align_datasets(
    human_df: pd.DataFrame,
    llm_df: pd.DataFrame,
    id_column: str = "sample_id",
    label_column: str = "label",
    normalize: bool = True,
    label_map: Optional[Dict[str, str]] = None,
    extra_llm_columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Merge human and LLM annotations on sample ID, normalize labels, and return
    a unified DataFrame with columns: sample_id, text, label_human, label_llm.

    Parameters
    ----------
    human_df, llm_df : pd.DataFrame
    id_column : str
    label_column : str
    normalize : bool
        Whether to normalize labels (lowercase, strip).
    label_map : dict, optional
        Optional label remapping applied during normalization.
    extra_llm_columns : list of str, optional
        Additional columns from the LLM DataFrame to carry through
        (e.g. 'confidence', 'rationale').

    Returns
    -------
    pd.DataFrame with columns [id_column, 'label_human', 'label_llm'] plus any
    extra columns from the human DataFrame and requested LLM columns.
    """
    validate_alignment(human_df, llm_df, id_column)

    llm_cols = [id_column, label_column]
    if extra_llm_columns:
        llm_cols += [c for c in extra_llm_columns if c in llm_df.columns]

    merged = human_df.merge(
        llm_df[llm_cols],
        on=id_column,
        suffixes=("_human", "_llm"),
    )

    human_col = f"{label_column}_human"
    llm_col = f"{label_column}_llm"

    if normalize:
        merged[human_col] = normalize_labels(merged[human_col].tolist(), label_map)
        merged[llm_col] = normalize_labels(merged[llm_col].tolist(), label_map)

    return merged
