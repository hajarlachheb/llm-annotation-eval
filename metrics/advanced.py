"""Advanced evaluation metrics: soft scoring, inter-LLM agreement, etc."""

from itertools import combinations
from typing import Dict, List, Tuple, Union

import numpy as np
from sklearn.metrics import cohen_kappa_score


def span_overlap_score(
    pred_span: Tuple[int, int],
    gold_span: Tuple[int, int],
) -> float:
    """
    Compute Intersection-over-Union (IoU) for two character-offset spans.

    Useful for partial-credit evaluation in NER / span extraction tasks.

    Parameters
    ----------
    pred_span : (start, end) character offsets (exclusive end).
    gold_span : (start, end) character offsets (exclusive end).

    Returns
    -------
    float in [0, 1].
    """
    overlap_start = max(pred_span[0], gold_span[0])
    overlap_end = min(pred_span[1], gold_span[1])
    overlap = max(0, overlap_end - overlap_start)
    union = max(pred_span[1], gold_span[1]) - min(pred_span[0], gold_span[0])
    return overlap / union if union > 0 else 0.0


def inter_llm_agreement(
    llm_outputs: Dict[str, Union[List[str], np.ndarray]],
) -> Dict[str, float]:
    """
    Compute pairwise Cohen's Kappa between all LLM model outputs.

    Parameters
    ----------
    llm_outputs : dict
        Mapping from model name to list of predicted labels. All lists must
        have the same length.

    Returns
    -------
    dict mapping "(model_a, model_b)" to kappa score.
    """
    models = list(llm_outputs.keys())
    results: Dict[str, float] = {}
    for m1, m2 in combinations(models, 2):
        kappa = cohen_kappa_score(llm_outputs[m1], llm_outputs[m2])
        results[f"{m1} vs {m2}"] = float(kappa)
    return results


def compute_soft_f1(
    pred_spans: List[Tuple[int, int, str]],
    gold_spans: List[Tuple[int, int, str]],
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute soft (partial-overlap) precision, recall, and F1 for span-based tasks.

    A predicted span is a true positive if its IoU with a gold span of the same
    type exceeds `iou_threshold`.

    Parameters
    ----------
    pred_spans : list of (start, end, label)
    gold_spans : list of (start, end, label)
    iou_threshold : float

    Returns
    -------
    dict with soft_precision, soft_recall, soft_f1.
    """
    matched_gold = set()
    tp = 0

    for ps, pe, pl in pred_spans:
        best_iou = 0.0
        best_idx = -1
        for idx, (gs, ge, gl) in enumerate(gold_spans):
            if gl != pl or idx in matched_gold:
                continue
            iou = span_overlap_score((ps, pe), (gs, ge))
            if iou > best_iou:
                best_iou = iou
                best_idx = idx
        if best_iou >= iou_threshold and best_idx >= 0:
            tp += 1
            matched_gold.add(best_idx)

    precision = tp / len(pred_spans) if pred_spans else 0.0
    recall = tp / len(gold_spans) if gold_spans else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "soft_precision": precision,
        "soft_recall": recall,
        "soft_f1": f1,
    }
