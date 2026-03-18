"""Token-level / sequence-labeling evaluation metrics (NER, POS, etc.)."""

from typing import Dict, List

from seqeval.metrics import (
    classification_report as seq_classification_report,
    f1_score as seq_f1,
    precision_score as seq_precision,
    recall_score as seq_recall,
)
from seqeval.scheme import IOB2


def compute_ner_metrics(
    human_sequences: List[List[str]],
    llm_sequences: List[List[str]],
    mode: str = "strict",
    scheme: type = IOB2,
) -> Dict:
    """
    Compute entity-level precision, recall, and F1 for sequence labeling.

    Parameters
    ----------
    human_sequences : list of list of str
        Ground-truth tag sequences, e.g. [["B-PER", "I-PER", "O"], ...].
    llm_sequences : list of list of str
        Predicted tag sequences in the same format.
    mode : str
        Evaluation mode — 'strict' requires exact span + type match.
    scheme : seqeval scheme class
        Tagging scheme (IOB2, IOB1, IOBES, etc.).

    Returns
    -------
    dict with precision, recall, f1, and per_entity_report.
    """
    return {
        "precision": float(seq_precision(human_sequences, llm_sequences, mode=mode, scheme=scheme)),
        "recall": float(seq_recall(human_sequences, llm_sequences, mode=mode, scheme=scheme)),
        "f1": float(seq_f1(human_sequences, llm_sequences, mode=mode, scheme=scheme)),
        "per_entity_report": seq_classification_report(
            human_sequences, llm_sequences, mode=mode, scheme=scheme, output_dict=True
        ),
    }
