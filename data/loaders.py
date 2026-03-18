"""Data loading utilities for human and LLM annotations."""

import json
from pathlib import Path
from typing import Union

import pandas as pd
import yaml


def load_config(config_path: Union[str, Path]) -> dict:
    """Load a YAML evaluation config file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_annotations(filepath: Union[str, Path], file_format: str = "auto") -> pd.DataFrame:
    """
    Load annotation data from JSONL, CSV, or JSON files.

    Parameters
    ----------
    filepath : str or Path
        Path to the annotation file.
    file_format : str
        One of 'jsonl', 'csv', 'json', or 'auto' (inferred from extension).

    Returns
    -------
    pd.DataFrame
    """
    filepath = Path(filepath)

    if file_format == "auto":
        suffix = filepath.suffix.lower()
        format_map = {".jsonl": "jsonl", ".csv": "csv", ".json": "json"}
        file_format = format_map.get(suffix)
        if file_format is None:
            raise ValueError(f"Cannot infer format from extension '{suffix}'. Specify file_format explicitly.")

    if file_format == "jsonl":
        records = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return pd.DataFrame(records)

    if file_format == "csv":
        return pd.read_csv(filepath)

    if file_format == "json":
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return pd.DataFrame(data)
        return pd.DataFrame.from_dict(data, orient="index")

    raise ValueError(f"Unsupported format: {file_format}")


def load_all_annotations(config: dict, base_dir: Union[str, Path] = ".") -> dict:
    """
    Load human and all LLM annotation files specified in config.

    Returns
    -------
    dict with keys 'human' and one key per LLM model name, each holding a DataFrame.
    """
    base_dir = Path(base_dir)
    ds = config["dataset"]

    result = {
        "human": load_annotations(base_dir / ds["human_labels"]),
    }
    for model_name, path in ds["llm_labels"].items():
        result[model_name] = load_annotations(base_dir / path)

    return result
