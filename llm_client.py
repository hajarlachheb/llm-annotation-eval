"""
Lightweight wrapper for calling OpenAI-compatible APIs to generate annotations.

Supports OpenAI, Azure OpenAI, Ollama, vLLM, LiteLLM, or any endpoint that
implements the OpenAI chat completions API by accepting a custom `base_url`.
"""

import json
import logging
import time
from typing import Dict, List, Optional, Tuple

import pandas as pd

log = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = """\
You are an expert text annotator. Classify the following text into EXACTLY one \
of these labels: {labels}.

Respond ONLY with valid JSON in this exact format (no markdown, no extra text):
{{"label": "<your_label>", "confidence": <0.0-1.0>, "rationale": "<brief explanation>"}}\
"""

DEFAULT_USER_PROMPT = "Text to classify:\n\n{text}"


def build_prompt(
    text: str,
    labels: List[str],
    system_template: str = DEFAULT_SYSTEM_PROMPT,
    user_template: str = DEFAULT_USER_PROMPT,
) -> List[Dict[str, str]]:
    labels_str = ", ".join(labels)
    return [
        {"role": "system", "content": system_template.format(labels=labels_str)},
        {"role": "user", "content": user_template.format(text=text)},
    ]


def parse_response(raw: str, valid_labels: List[str]) -> Dict:
    """Parse LLM JSON response, with fallback heuristics."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

    try:
        data = json.loads(raw)
        label = str(data.get("label", "")).strip().lower()
        confidence = float(data.get("confidence", 0.5))
        rationale = str(data.get("rationale", ""))
        if label not in [l.lower() for l in valid_labels]:
            for vl in valid_labels:
                if vl.lower() in raw.lower():
                    label = vl.lower()
                    break
        return {"label": label, "confidence": confidence, "rationale": rationale}
    except (json.JSONDecodeError, ValueError, TypeError):
        for vl in valid_labels:
            if vl.lower() in raw.lower():
                return {"label": vl.lower(), "confidence": 0.5, "rationale": raw[:200]}
        return {"label": valid_labels[0].lower(), "confidence": 0.1, "rationale": f"PARSE_FAILED: {raw[:200]}"}


def annotate_single(
    client,
    model: str,
    text: str,
    labels: List[str],
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    temperature: float = 0.0,
) -> Dict:
    """Annotate a single text via the OpenAI-compatible API."""
    messages = build_prompt(text, labels, system_template=system_prompt)
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=256,
    )
    raw = resp.choices[0].message.content or ""
    return parse_response(raw, labels)


def annotate_batch(
    api_key: str,
    base_url: Optional[str],
    model: str,
    texts: List[str],
    sample_ids: List[str],
    labels: List[str],
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    temperature: float = 0.0,
    progress_callback=None,
) -> pd.DataFrame:
    """
    Annotate a batch of texts and return a DataFrame matching the evaluation format.

    Parameters
    ----------
    progress_callback : callable, optional
        Called with (current_index, total) after each annotation.
    """
    from openai import OpenAI

    client_kwargs = {"api_key": api_key}
    if base_url and base_url.strip():
        client_kwargs["base_url"] = base_url.strip()
    client = OpenAI(**client_kwargs)

    records = []
    total = len(texts)
    for i, (sid, text) in enumerate(zip(sample_ids, texts)):
        try:
            result = annotate_single(client, model, text, labels, system_prompt, temperature)
        except Exception as e:
            log.warning("Failed on %s: %s", sid, e)
            result = {"label": labels[0].lower(), "confidence": 0.1, "rationale": f"API_ERROR: {e}"}
            time.sleep(1)

        records.append({
            "sample_id": sid,
            "text": text,
            "label": result["label"],
            "confidence": result["confidence"],
            "rationale": result["rationale"],
        })

        if progress_callback:
            progress_callback(i + 1, total)

    return pd.DataFrame(records)
