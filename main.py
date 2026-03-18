#!/usr/bin/env python3
"""
LLM Annotation Quality Evaluation Framework
=============================================
CLI entrypoint for running evaluations against human-labelled baselines.

Usage:
    python main.py --config config/eval_config.yaml
    python main.py --config config/eval_config.yaml --output reports/
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd

from data.loaders import load_all_annotations, load_config
from data.alignment import align_datasets
from metrics.classification import compute_classification_metrics, compute_krippendorff_alpha
from metrics.advanced import inter_llm_agreement
from analysis.bootstrap import bootstrap_all_metrics
from analysis.slicing import bin_continuous_column, evaluate_by_slice
from analysis.error_analysis import find_systematic_errors, sample_error_examples
from analysis.explainability import (
    text_features_driving_errors,
    compute_shap_explanations,
    save_shap_summary_plot,
    save_shap_beeswarm_plot,
    extract_rationale_keywords,
)
from analysis.calibration import (
    calibration_curve,
    confidence_stratified_metrics,
    find_overconfident_errors,
    find_underconfident_correct,
)
from reporting.tables import build_summary_table, save_summary
from reporting.visualizations import (
    plot_confusion_matrix,
    plot_per_class_f1,
    plot_agreement_heatmap,
    plot_reliability_diagram,
    plot_feature_importance,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def _json_dump(obj, filepath):
    """Write an object to JSON, handling numpy/pandas types."""
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str, ensure_ascii=False)


def run_classification_eval(config: dict, base_dir: Path, output_dir: Path) -> None:
    """Full classification evaluation pipeline."""
    ds_cfg = config["dataset"]
    id_col = ds_cfg.get("id_column", "sample_id")
    label_col = ds_cfg.get("label_column", "label")
    text_col = ds_cfg.get("text_column", "text")
    conf_col = ds_cfg.get("confidence_column", "confidence")
    rationale_col = ds_cfg.get("rationale_column", "rationale")
    explain_cfg = config.get("explainability", {})

    # ── Load data ────────────────────────────────────────────────
    log.info("Loading annotations...")
    all_data = load_all_annotations(config, base_dir)
    human_df = all_data.pop("human")
    llm_dfs = all_data
    log.info("  Human samples: %d | LLM models: %s", len(human_df), list(llm_dfs.keys()))

    # ── Per-model evaluation ─────────────────────────────────────
    all_metrics = {}
    all_aligned = {}

    for model_name, llm_df in llm_dfs.items():
        log.info("Evaluating model: %s", model_name)

        extra_cols = [conf_col, rationale_col]
        aligned = align_datasets(
            human_df, llm_df,
            id_column=id_col,
            label_column=label_col,
            extra_llm_columns=extra_cols,
        )
        all_aligned[model_name] = aligned

        h = aligned[f"{label_col}_human"].values
        l = aligned[f"{label_col}_llm"].values
        labels = sorted(set(h) | set(l))
        h_col = f"{label_col}_human"
        l_col = f"{label_col}_llm"

        # Core metrics
        metrics = compute_classification_metrics(h, l, labels=labels)
        log.info("  F1=%.4f  Precision=%.4f  Recall=%.4f  Kappa=%.4f",
                 metrics["f1"], metrics["precision"], metrics["recall"], metrics["cohen_kappa"])

        # Bootstrap CIs
        rel_cfg = config.get("reliability", {})
        if rel_cfg.get("bootstrap", False):
            log.info("  Computing bootstrap confidence intervals (%d iterations)...",
                     rel_cfg.get("bootstrap_n", 1000))
            bootstrap = bootstrap_all_metrics(
                h, l,
                n_iterations=rel_cfg.get("bootstrap_n", 1000),
                confidence_level=rel_cfg.get("confidence_level", 0.95),
            )
            metrics["bootstrap"] = bootstrap

        all_metrics[model_name] = metrics

        # ── Visualizations ───────────────────────────────────────
        model_dir = output_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        plot_confusion_matrix(
            metrics["confusion_matrix"], labels,
            title=f"Confusion Matrix — {model_name}",
            save_path=model_dir / "confusion_matrix.png",
        )
        plot_per_class_f1(
            metrics["per_class_report"],
            title=f"Per-Class F1 — {model_name}",
            save_path=model_dir / "per_class_f1.png",
        )

        # ── Error analysis ───────────────────────────────────────
        adv_cfg = config.get("advanced", {})
        if adv_cfg.get("error_analysis", False):
            log.info("  Running error analysis...")
            error_patterns = find_systematic_errors(aligned, human_col=h_col, llm_col=l_col)
            error_patterns.to_csv(model_dir / "error_patterns.csv", index=False)

            examples = sample_error_examples(
                aligned, human_col=h_col, llm_col=l_col, text_col=text_col,
                n_per_pattern=adv_cfg.get("error_sample_size", 3),
            )
            examples_serializable = {f"{k[0]} -> {k[1]}": v for k, v in examples.items()}
            _json_dump(examples_serializable, model_dir / "error_examples.json")

        # ── Slice-based evaluation ───────────────────────────────
        slice_cfg = config.get("slicing", {})
        if slice_cfg.get("enabled", False):
            log.info("  Running slice-based evaluation...")
            if "text_length" not in aligned.columns and text_col in aligned.columns:
                aligned["text_length"] = aligned[text_col].str.len()

            for field_cfg in slice_cfg.get("fields", []):
                field = field_cfg if isinstance(field_cfg, str) else field_cfg.get("field")
                if field not in aligned.columns:
                    log.warning("    Slice field '%s' not found, skipping.", field)
                    continue

                if isinstance(field_cfg, dict) and "bins" in field_cfg:
                    bin_col = f"{field}_bin"
                    bin_continuous_column(
                        aligned, field,
                        bins=field_cfg["bins"],
                        bin_labels=field_cfg.get("bin_labels"),
                        target_column=bin_col,
                    )
                    slice_field = bin_col
                else:
                    slice_field = field

                from metrics.classification import compute_classification_metrics as _ccm
                slice_results = evaluate_by_slice(
                    aligned, slice_field, h_col, l_col,
                    metric_fn=lambda h, l: _ccm(h, l)
                )
                _json_dump(slice_results, model_dir / f"slice_{field}.json")

        # ═══════════════════════════════════════════════════════════
        # EXPLAINABILITY
        # ═══════════════════════════════════════════════════════════

        explain_dir = model_dir / "explainability"
        explain_dir.mkdir(parents=True, exist_ok=True)

        # ── Feature Attribution (TF-IDF) ─────────────────────────
        if explain_cfg.get("feature_attribution", False):
            log.info("  Running text feature attribution...")
            feat_result = text_features_driving_errors(
                aligned, text_col=text_col, human_col=h_col, llm_col=l_col,
            )
            _json_dump(feat_result, explain_dir / "feature_attribution.json")
            plot_feature_importance(
                feat_result,
                title=f"Error-Driving Features — {model_name}",
                save_path=explain_dir / "feature_importance.png",
            )
            log.info("    Error predictor CV accuracy: %.1f%% (error rate: %.1f%%)",
                     feat_result.get("cv_accuracy", 0) * 100,
                     feat_result.get("error_rate", 0) * 100)

        # ── SHAP Values ──────────────────────────────────────────
        if explain_cfg.get("shap_values", False):
            log.info("  Computing SHAP values...")
            shap_result = compute_shap_explanations(
                aligned, text_col=text_col, human_col=h_col, llm_col=l_col,
                n_background=explain_cfg.get("shap_n_background", 100),
                n_explain=explain_cfg.get("shap_n_explain", 200),
            )
            global_shap = shap_result.get("global_shap", [])
            _json_dump({
                "global_importance": global_shap,
                "expected_value": shap_result.get("expected_value"),
                "model_accuracy": shap_result.get("model_accuracy"),
                "n_explained": len(shap_result.get("explained_indices", [])),
            }, explain_dir / "shap_global.json")

            save_shap_summary_plot(
                shap_result,
                save_path=explain_dir / "shap_summary.png",
                max_display=explain_cfg.get("shap_max_display", 20),
                title=f"SHAP Feature Importance — {model_name}",
            )
            save_shap_beeswarm_plot(
                shap_result,
                save_path=explain_dir / "shap_beeswarm.png",
                max_display=explain_cfg.get("shap_max_display", 20),
            )
            if global_shap:
                log.info("    Top SHAP features: %s",
                         ", ".join(f"{f}({v:.4f})" for f, v in global_shap[:5]))

        # ── Confidence Calibration ───────────────────────────────
        has_confidence = conf_col in aligned.columns
        if explain_cfg.get("calibration", False) and has_confidence:
            log.info("  Running confidence calibration analysis...")
            cal_df, ece = calibration_curve(
                aligned, conf_col, h_col, l_col,
                n_bins=explain_cfg.get("calibration_bins", 10),
            )
            cal_df.to_csv(explain_dir / "calibration_curve.csv", index=False)
            plot_reliability_diagram(
                cal_df, ece,
                title=f"Reliability Diagram — {model_name}",
                save_path=explain_dir / "reliability_diagram.png",
            )
            log.info("    ECE = %.4f", ece)
            metrics["ece"] = ece

            strat = confidence_stratified_metrics(aligned, conf_col, h_col, l_col)
            strat.to_csv(explain_dir / "confidence_tiers.csv", index=False)

            overconf = find_overconfident_errors(
                aligned, conf_col, h_col, l_col, text_col=text_col,
            )
            if not overconf.empty:
                overconf.to_csv(explain_dir / "overconfident_errors.csv", index=False)
                log.info("    Found %d high-confidence errors", len(overconf))

            underconf = find_underconfident_correct(
                aligned, conf_col, h_col, l_col, text_col=text_col,
            )
            if not underconf.empty:
                underconf.to_csv(explain_dir / "underconfident_correct.csv", index=False)

        elif explain_cfg.get("calibration", False) and not has_confidence:
            log.warning("  Calibration enabled but '%s' column not found — skipping.", conf_col)

        # ── Rationale Analysis ───────────────────────────────────
        has_rationale = rationale_col in aligned.columns
        if explain_cfg.get("rationale_analysis", False) and has_rationale:
            log.info("  Analyzing LLM rationales...")
            kw_profiles = extract_rationale_keywords(aligned, rationale_col, l_col)
            serializable_profiles = {k: list(v) for k, v in kw_profiles.items()}
            _json_dump(serializable_profiles, explain_dir / "rationale_keywords.json")

            correct_mask = aligned[h_col] == aligned[l_col]
            rationale_stats = {
                "avg_length_correct": round(float(aligned.loc[correct_mask, rationale_col].str.len().mean()), 1),
                "avg_length_incorrect": round(float(aligned.loc[~correct_mask, rationale_col].str.len().mean()), 1),
                "n_correct": int(correct_mask.sum()),
                "n_incorrect": int((~correct_mask).sum()),
            }
            _json_dump(rationale_stats, explain_dir / "rationale_stats.json")
            log.info("    Avg rationale length — correct: %.0f chars, incorrect: %.0f chars",
                     rationale_stats["avg_length_correct"],
                     rationale_stats["avg_length_incorrect"])

        elif explain_cfg.get("rationale_analysis", False) and not has_rationale:
            log.warning("  Rationale analysis enabled but '%s' column not found — skipping.",
                        rationale_col)

    # ── Inter-LLM agreement ──────────────────────────────────────
    adv_cfg = config.get("advanced", {})
    if adv_cfg.get("inter_llm_agreement", False) and len(llm_dfs) > 1:
        log.info("Computing inter-LLM agreement...")
        merged_labels = {}
        for model_name, aligned in all_aligned.items():
            merged_labels[model_name] = aligned[f"{label_col}_llm"].tolist()
        agreement = inter_llm_agreement(merged_labels)
        log.info("  %s", agreement)
        _json_dump(agreement, output_dir / "inter_llm_agreement.json")
        plot_agreement_heatmap(agreement, save_path=output_dir / "inter_llm_agreement.png")

    # ── Krippendorff's alpha (human + all LLMs) ─────────────────
    if config.get("metrics", {}).get("krippendorff_alpha", False):
        log.info("Computing Krippendorff's Alpha (all raters)...")
        rater_labels = [human_df[label_col].tolist()]
        all_labels_set = set(human_df[label_col])
        for model_name, aligned in all_aligned.items():
            rater_labels.append(aligned[f"{label_col}_llm"].tolist())
            all_labels_set.update(aligned[f"{label_col}_llm"])

        label_to_int = {lbl: i for i, lbl in enumerate(sorted(all_labels_set))}
        matrix = np.array([[label_to_int[l] for l in rater] for rater in rater_labels], dtype=float)
        alpha = compute_krippendorff_alpha(matrix)
        log.info("  Krippendorff's Alpha = %.4f", alpha)
        all_metrics["_krippendorff_alpha"] = alpha

    # ── Summary table ────────────────────────────────────────────
    log.info("Building summary table...")
    model_metrics = {k: v for k, v in all_metrics.items() if not k.startswith("_")}
    summary = build_summary_table(model_metrics)
    save_summary(summary, output_dir, formats=("csv", "json"))
    log.info("\n%s", summary.to_string())

    # ── Save full results as JSON ────────────────────────────────
    full_results = {}
    for model, m in all_metrics.items():
        if not isinstance(m, dict):
            full_results[model] = m
            continue
        serializable = {}
        for k, v in m.items():
            if isinstance(v, np.ndarray):
                serializable[k] = v.tolist()
            else:
                serializable[k] = v
        full_results[model] = serializable

    _json_dump(full_results, output_dir / "full_results.json")
    log.info("All reports saved to %s", output_dir)


def main():
    parser = argparse.ArgumentParser(
        description="LLM Annotation Quality Evaluation Framework",
    )
    parser.add_argument(
        "--config", type=str, default="config/eval_config.yaml",
        help="Path to the YAML evaluation config.",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output directory for reports (overrides config).",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)

    base_dir = config_path.parent.parent
    output_dir = Path(args.output) if args.output else base_dir / config.get("output", {}).get("directory", "reports")
    output_dir.mkdir(parents=True, exist_ok=True)

    task_type = config.get("task_type", "classification")

    if task_type in ("classification", "sentiment"):
        run_classification_eval(config, base_dir, output_dir)
    elif task_type == "ner":
        log.error("NER evaluation pipeline — run via: python -m metrics.sequence_labeling")
        sys.exit(1)
    else:
        log.error("Unsupported task_type: %s", task_type)
        sys.exit(1)


if __name__ == "__main__":
    main()
