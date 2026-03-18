#!/usr/bin/env python3
"""
Generate synthetic sample data for testing the evaluation framework.

Creates:
  - sample_data/human_annotations.jsonl   (ground truth)
  - sample_data/llm_gpt4_annotations.jsonl  (with confidence + rationale)
  - sample_data/llm_gpt35_annotations.jsonl (with confidence + rationale)

Usage:
    python generate_sample_data.py
    python generate_sample_data.py --n 5000 --noise-gpt4 0.08 --noise-gpt35 0.18
"""

import argparse
import json
import random
from pathlib import Path

LABELS = ["positive", "negative", "neutral", "mixed"]
DOMAINS = ["finance", "healthcare", "technology", "sports", "entertainment"]

SAMPLE_TEXTS = {
    "positive": [
        "Absolutely loved the new update! Performance is incredible.",
        "Great customer service, they resolved my issue in minutes.",
        "The product exceeded all my expectations — highly recommended.",
        "Brilliant design and intuitive user experience throughout.",
        "This is hands down the best tool I've used this year.",
    ],
    "negative": [
        "Terrible experience. The app crashed repeatedly and lost my data.",
        "Customer support was unresponsive for over a week.",
        "Completely unusable — filled with bugs and missing features.",
        "Worst purchase I've made. Quality is far below what was advertised.",
        "The update broke everything. I'm switching to a competitor.",
    ],
    "neutral": [
        "The product works as described. Nothing special, nothing bad.",
        "Average performance. It does what it says on the tin.",
        "Received the item on time. Standard packaging, standard quality.",
        "It's an okay solution. Neither impressive nor disappointing.",
        "The service meets basic requirements but doesn't stand out.",
    ],
    "mixed": [
        "Good features but the interface is confusing and slow.",
        "Love the concept but execution needs serious improvement.",
        "Some things work great, others are frustratingly broken.",
        "Excellent core product hampered by poor documentation.",
        "Fast shipping but the item quality was lower than expected.",
    ],
}

RATIONALE_TEMPLATES = {
    "correct": {
        "positive": [
            "Strong positive language ('loved', 'incredible', 'exceeded') with no negative qualifiers.",
            "Enthusiastic tone and explicit recommendation indicate clear positive sentiment.",
            "Superlative expressions and satisfaction markers throughout the text.",
        ],
        "negative": [
            "Strongly negative vocabulary ('terrible', 'crashed', 'unusable') dominates the text.",
            "Explicit dissatisfaction and intent to leave indicate negative sentiment.",
            "Multiple complaint markers with no redeeming positive statements.",
        ],
        "neutral": [
            "Balanced, matter-of-fact language with no strong positive or negative markers.",
            "Descriptive tone without emotional language — neither praise nor criticism.",
            "Text describes factual experience without expressing strong opinion.",
        ],
        "mixed": [
            "Contains both positive ('good', 'love', 'excellent') and negative ('confusing', 'broken') signals.",
            "Contrasting clauses ('but', 'however') indicate mixed sentiment.",
            "Positive aspects explicitly paired with negative qualifiers.",
        ],
    },
    "wrong": [
        "The hedging language could suggest uncertainty, leading to a different classification.",
        "Ambiguous phrasing made the sentiment boundary less clear.",
        "The presence of contrasting signals caused a misread of the dominant sentiment.",
        "Subtle context cues were missed, leading to an adjacent-class prediction.",
    ],
}


def generate_text(label: str, rng: random.Random) -> str:
    base = rng.choice(SAMPLE_TEXTS[label])
    suffix = rng.choice([
        "", " Would use again.", " Not sure what to think.", " Needs work.",
        " Overall satisfied.", " Disappointed.", " Will recommend to friends.",
    ])
    return base + suffix


def flip_label(label: str, noise_rate: float, rng: random.Random) -> str:
    if rng.random() < noise_rate:
        return rng.choice([l for l in LABELS if l != label])
    return label


def generate_confidence(is_correct: bool, rng: random.Random) -> float:
    if is_correct:
        conf = rng.gauss(0.88, 0.08)
    else:
        conf = rng.gauss(0.55, 0.18)
    return round(max(0.01, min(0.99, conf)), 2)


def generate_rationale(true_label: str, pred_label: str, rng: random.Random) -> str:
    if true_label == pred_label:
        return rng.choice(RATIONALE_TEMPLATES["correct"][pred_label])
    return rng.choice(RATIONALE_TEMPLATES["wrong"])


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic evaluation data")
    parser.add_argument("--n", type=int, default=500, help="Number of samples")
    parser.add_argument("--noise-gpt4", type=float, default=0.08, help="GPT-4 error rate")
    parser.add_argument("--noise-gpt35", type=float, default=0.18, help="GPT-3.5 error rate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="sample_data")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    human_records = []
    gpt4_records = []
    gpt35_records = []

    for i in range(args.n):
        label = rng.choice(LABELS)
        text = generate_text(label, rng)
        domain = rng.choice(DOMAINS)
        sample_id = f"sample_{i:05d}"

        human_records.append({
            "sample_id": sample_id,
            "text": text,
            "label": label,
            "domain": domain,
        })

        gpt4_label = flip_label(label, args.noise_gpt4, rng)
        gpt4_conf = generate_confidence(gpt4_label == label, rng)
        gpt4_rationale = generate_rationale(label, gpt4_label, rng)
        gpt4_records.append({
            "sample_id": sample_id,
            "text": text,
            "label": gpt4_label,
            "domain": domain,
            "confidence": gpt4_conf,
            "rationale": gpt4_rationale,
        })

        gpt35_label = flip_label(label, args.noise_gpt35, rng)
        gpt35_conf = generate_confidence(gpt35_label == label, rng)
        gpt35_rationale = generate_rationale(label, gpt35_label, rng)
        gpt35_records.append({
            "sample_id": sample_id,
            "text": text,
            "label": gpt35_label,
            "domain": domain,
            "confidence": gpt35_conf,
            "rationale": gpt35_rationale,
        })

    def write_jsonl(records, filepath):
        with open(filepath, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    write_jsonl(human_records, out / "human_annotations.jsonl")
    write_jsonl(gpt4_records, out / "llm_gpt4_annotations.jsonl")
    write_jsonl(gpt35_records, out / "llm_gpt35_annotations.jsonl")

    print(f"Generated {args.n} samples in {out}/")
    print(f"  human_annotations.jsonl")
    print(f"  llm_gpt4_annotations.jsonl   (noise={args.noise_gpt4}, +confidence, +rationale)")
    print(f"  llm_gpt35_annotations.jsonl  (noise={args.noise_gpt35}, +confidence, +rationale)")


if __name__ == "__main__":
    main()
