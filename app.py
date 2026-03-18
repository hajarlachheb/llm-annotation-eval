#!/usr/bin/env python3
"""
Gradio Dashboard for LLM Annotation Quality Evaluation.

Simple: upload your files (or use sample data), click Run, see results.

Launch:
    python app.py
    python app.py --port 7861 --share
"""

import argparse
import json
import logging
import tempfile
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gradio as gr
import numpy as np
import pandas as pd

from data.alignment import align_datasets
from data.loaders import load_annotations
from metrics.classification import compute_classification_metrics
from metrics.advanced import inter_llm_agreement
from analysis.bootstrap import bootstrap_all_metrics
from analysis.error_analysis import find_systematic_errors, sample_error_examples
from analysis.explainability import (
    text_features_driving_errors,
    compute_shap_explanations,
    save_shap_summary_plot,
    extract_rationale_keywords,
)
from analysis.calibration import (
    calibration_curve,
    confidence_stratified_metrics,
    find_overconfident_errors,
)
from reporting.visualizations import (
    plot_confusion_matrix,
    plot_per_class_f1,
    plot_agreement_heatmap,
    plot_reliability_diagram,
    plot_feature_importance,
)
from llm_client import annotate_batch, annotate_single, DEFAULT_SYSTEM_PROMPT

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

SAMPLE_DIR = Path(__file__).parent / "sample_data"


def _load_file(file_obj) -> pd.DataFrame:
    if file_obj is None:
        raise ValueError("No file uploaded.")
    return load_annotations(Path(file_obj))


def _close(fig):
    if fig is not None:
        plt.close(fig)
    return fig


def _build_log(lines: List[str]) -> str:
    return "\n".join(lines)


def _fmt(val, decimals=4):
    return f"{val:.{decimals}f}" if isinstance(val, float) else str(val)


# ═══════════════════════════════════════════════════════════════════════════════
# CORE: run evaluation and return (log_text, figures, dataframes)
# ═══════════════════════════════════════════════════════════════════════════════

def run_full_evaluation(
    human_df: pd.DataFrame,
    llm_dfs: Dict[str, pd.DataFrame],
    id_col="sample_id", label_col="label", text_col="text",
    conf_col="confidence", rationale_col="rationale",
    do_bootstrap=True, do_shap=True, do_calibration=True, do_errors=True,
    progress=None,
):
    """Run the full pipeline and return everything the UI needs in one dict."""
    out = {"log": [], "models": {}, "summary_rows": []}
    out["log"].append("=" * 60)
    out["log"].append("  LLM Annotation Quality Evaluation")
    out["log"].append("=" * 60)
    out["log"].append("")

    model_names = list(llm_dfs.keys())
    total = len(model_names)
    all_aligned = {}

    for idx, (model_name, llm_df) in enumerate(llm_dfs.items()):
        out["log"].append(f"\n{'-' * 50}")
        out["log"].append(f"  Model: {model_name}")
        out["log"].append(f"{'-' * 50}")

        extra_cols = [conf_col, rationale_col]
        aligned = align_datasets(
            human_df, llm_df, id_column=id_col, label_column=label_col,
            extra_llm_columns=extra_cols,
        )
        all_aligned[model_name] = aligned
        h = aligned[f"{label_col}_human"].values
        l = aligned[f"{label_col}_llm"].values
        labels = sorted(set(h) | set(l))
        h_col = f"{label_col}_human"
        l_col = f"{label_col}_llm"

        m = compute_classification_metrics(h, l, labels=labels)
        out["log"].append(f"\n  Samples:   {len(aligned)}")
        out["log"].append(f"  Labels:    {', '.join(labels)}")
        out["log"].append(f"  Precision: {m['precision']:.4f}")
        out["log"].append(f"  Recall:    {m['recall']:.4f}")
        out["log"].append(f"  F1:        {m['f1']:.4f}")
        out["log"].append(f"  Kappa:     {m['cohen_kappa']:.4f}")

        mr = {"metrics": m, "labels": labels, "n": len(aligned)}
        summary = {
            "Model": model_name, "Samples": len(aligned),
            "Precision": round(m["precision"], 4), "Recall": round(m["recall"], 4),
            "F1": round(m["f1"], 4), "Kappa": round(m["cohen_kappa"], 4),
        }

        mr["cm_fig"] = _close(plot_confusion_matrix(
            m["confusion_matrix"], labels, title=f"Confusion Matrix — {model_name}"))
        mr["f1_fig"] = _close(plot_per_class_f1(
            m["per_class_report"], title=f"Per-Class F1 — {model_name}"))

        if do_bootstrap:
            if progress:
                progress((idx + 0.2) / total, desc=f"{model_name}: bootstrap...")
            boot = bootstrap_all_metrics(h, l, n_iterations=500)
            mr["bootstrap"] = boot
            out["log"].append(f"\n  Bootstrap 95% CIs:")
            for k in ["f1", "precision", "recall", "cohen_kappa"]:
                out["log"].append(f"    {k:>14s}: [{boot[k]['ci_lower']:.3f}, {boot[k]['ci_upper']:.3f}]")
            summary["F1 95% CI"] = f"[{boot['f1']['ci_lower']:.3f}, {boot['f1']['ci_upper']:.3f}]"

        if do_errors:
            if progress:
                progress((idx + 0.4) / total, desc=f"{model_name}: errors...")
            errs = find_systematic_errors(aligned, human_col=h_col, llm_col=l_col)
            mr["error_df"] = errs
            examples = sample_error_examples(aligned, human_col=h_col, llm_col=l_col, text_col=text_col, n_per_pattern=5)
            mr["error_examples"] = {f"{k[0]} -> {k[1]}": v for k, v in examples.items()}
            out["log"].append(f"\n  Error patterns: {len(errs)}")

        if do_shap and text_col in aligned.columns:
            if progress:
                progress((idx + 0.6) / total, desc=f"{model_name}: SHAP...")
            feat = text_features_driving_errors(aligned, text_col, h_col, l_col)
            mr["feat_result"] = feat
            mr["feat_fig"] = _close(plot_feature_importance(
                feat, title=f"Error-Driving Features — {model_name}"))
            out["log"].append(f"\n  Feature attribution model accuracy: {feat.get('model_accuracy', 'N/A')}")
            out["log"].append(f"  Top error predictors:")
            for name, coef in feat.get("error_predictors", [])[:5]:
                out["log"].append(f"    {name:>30s}  {coef:+.4f}")

            try:
                shap_r = compute_shap_explanations(
                    aligned, text_col, h_col, l_col,
                    n_background=min(50, len(aligned)),
                    n_explain=min(100, len(aligned)),
                )
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                    save_shap_summary_plot(shap_r, save_path=f.name,
                                           title=f"SHAP Importance — {model_name}")
                    fig, ax = plt.subplots(figsize=(10, 7))
                    ax.imshow(plt.imread(f.name))
                    ax.axis("off")
                    mr["shap_fig"] = _close(fig)

                top_shap = shap_r.get("global_shap", [])[:5]
                if top_shap:
                    out["log"].append(f"  Top SHAP features:")
                    for name, val in top_shap:
                        out["log"].append(f"    {name:>30s}  {val:.6f}")
            except Exception as e:
                out["log"].append(f"  SHAP skipped: {e}")

        has_conf = conf_col in aligned.columns and aligned[conf_col].notna().any()
        if do_calibration and has_conf:
            if progress:
                progress((idx + 0.8) / total, desc=f"{model_name}: calibration...")
            try:
                cal_df, ece = calibration_curve(aligned, conf_col, h_col, l_col, n_bins=10)
                mr["ece"] = ece
                summary["ECE"] = round(ece, 4)
                mr["cal_fig"] = _close(plot_reliability_diagram(
                    cal_df, ece, title=f"Reliability Diagram — {model_name}"))
                out["log"].append(f"\n  ECE (Expected Calibration Error): {ece:.4f}")

                overconf = find_overconfident_errors(aligned, conf_col, h_col, l_col, text_col=text_col)
                if len(overconf) > 0:
                    out["log"].append(f"  Overconfident errors: {len(overconf)}")
            except Exception as e:
                out["log"].append(f"  Calibration skipped: {e}")

        has_rat = rationale_col in aligned.columns and aligned[rationale_col].notna().any()
        if has_rat:
            try:
                kw = extract_rationale_keywords(aligned, rationale_col, l_col)
                mr["rationale_kw"] = {k: list(v) for k, v in kw.items()}
                out["log"].append(f"\n  Rationale keywords by class:")
                for cls, words in mr["rationale_kw"].items():
                    top3 = ", ".join(f"{w}({c})" for w, c in words[:3])
                    out["log"].append(f"    {cls}: {top3}")
            except Exception:
                pass

        out["models"][model_name] = mr
        out["summary_rows"].append(summary)

    if len(model_names) > 1:
        merged = {mn: all_aligned[mn][f"{label_col}_llm"].tolist() for mn in model_names}
        agreement = inter_llm_agreement(merged)
        out["inter_llm"] = agreement
        out["log"].append(f"\n{'-' * 50}")
        out["log"].append("  Inter-LLM Agreement")
        out["log"].append(f"{'-' * 50}")
        for k, v in agreement.items():
            out["log"].append(f"  {k}: {v:.4f}")
        try:
            out["agreement_fig"] = _close(plot_agreement_heatmap(agreement))
        except Exception:
            pass

    out["log"].append(f"\n{'=' * 60}")
    out["log"].append("  DONE")
    out["log"].append(f"{'=' * 60}")

    if progress:
        progress(1.0, desc="Done")

    return out


# ═══════════════════════════════════════════════════════════════════════════════
# GRADIO APP — single page, simple flow
# ═══════════════════════════════════════════════════════════════════════════════

def _discover_sample_models():
    """Find available sample LLM annotation files and return model name list."""
    models = []
    for p in sorted(SAMPLE_DIR.glob("llm_*_annotations.jsonl")):
        models.append(p.stem.replace("llm_", "").replace("_annotations", ""))
    return models


SAMPLE_MODELS = _discover_sample_models()


def create_app() -> gr.Blocks:
    with gr.Blocks(title="LLM Annotation Evaluator") as app:

        gr.Markdown(
            "# LLM Annotation Evaluator\n"
            "Pick your data, choose model(s), click **Run**, see results."
        )

        # ──────────────── INPUT SECTION ────────────────────────────
        with gr.Row(equal_height=False):
            with gr.Column(scale=1):
                gr.Markdown("### Data")
                data_source = gr.Radio(
                    choices=["Built-in sample data", "Upload custom files"],
                    value="Built-in sample data",
                    label="Data source",
                )
                upload_box = gr.Column(visible=False)
                with upload_box:
                    human_file = gr.File(
                        label="Human Annotations (JSONL / CSV)",
                        file_types=[".jsonl", ".csv", ".json"],
                    )
                    llm_files = gr.File(
                        label="LLM Annotations (one or more files)",
                        file_types=[".jsonl", ".csv", ".json"],
                        file_count="multiple",
                    )

            with gr.Column(scale=1):
                gr.Markdown("### Model")
                model_pick = gr.Dropdown(
                    choices=SAMPLE_MODELS + ["All"],
                    value="All",
                    label="Evaluate model",
                    interactive=True,
                )

            with gr.Column(scale=1):
                gr.Markdown("### Options")
                chk_bootstrap = gr.Checkbox(value=True, label="Bootstrap CIs")
                chk_shap = gr.Checkbox(value=True, label="SHAP + Feature Attribution")
                chk_calibration = gr.Checkbox(value=True, label="Confidence Calibration")
                chk_errors = gr.Checkbox(value=True, label="Error Analysis")

        def on_data_source_change(source):
            is_custom = source == "Upload custom files"
            if is_custom:
                return (
                    gr.update(visible=True),
                    gr.update(choices=["All"], value="All"),
                )
            return (
                gr.update(visible=False),
                gr.update(choices=SAMPLE_MODELS + ["All"], value="All"),
            )

        data_source.change(
            on_data_source_change,
            [data_source],
            [upload_box, model_pick],
        )

        btn_run = gr.Button("Run Evaluation", variant="primary", size="lg")

        # ──────────────── RESULTS SECTION ──────────────────────────
        gr.Markdown("---")
        gr.Markdown("## Results")

        result_log = gr.Code(label="Evaluation Log", language=None, lines=20)
        result_summary = gr.DataFrame(label="Summary Metrics")

        gr.Markdown("### Charts")
        chart_model = gr.Dropdown(label="View charts for model", choices=[], interactive=True)

        with gr.Row():
            plot_cm = gr.Plot(label="Confusion Matrix")
            plot_f1 = gr.Plot(label="Per-Class F1")
        with gr.Row():
            plot_shap = gr.Plot(label="SHAP Importance")
            plot_feat = gr.Plot(label="Error-Driving Features")
        with gr.Row():
            plot_cal = gr.Plot(label="Calibration")
            plot_agree = gr.Plot(label="Inter-LLM Agreement")

        gr.Markdown("### Error Patterns")
        error_table = gr.DataFrame(label="Top Error Patterns")
        error_json = gr.Code(label="Error Examples", language="json", lines=12)

        eval_results = gr.State(value=None)

        # ──────────────── CALLBACKS ────────────────────────────────

        EMPTY_RESULT = (
            "", pd.DataFrame(), None, gr.update(),
            None, None, None, None, None, None,
            pd.DataFrame(), "",
        )

        def on_run(source, human_f, llm_fs, chosen_model,
                   do_boot, do_shap, do_cal, do_err, progress=gr.Progress()):
            try:
                if source == "Built-in sample data":
                    human_df = load_annotations(SAMPLE_DIR / "human_annotations.jsonl")
                    all_llm = {}
                    for p in sorted(SAMPLE_DIR.glob("llm_*_annotations.jsonl")):
                        name = p.stem.replace("llm_", "").replace("_annotations", "")
                        all_llm[name] = load_annotations(p)
                    if not all_llm:
                        return ("No sample files found. Run: python generate_sample_data.py",
                                *EMPTY_RESULT[1:])

                    if chosen_model and chosen_model != "All" and chosen_model in all_llm:
                        llm_dfs = {chosen_model: all_llm[chosen_model]}
                    else:
                        llm_dfs = all_llm
                else:
                    human_df = _load_file(human_f)
                    if not llm_fs:
                        return ("Upload at least one LLM annotation file.",
                                *EMPTY_RESULT[1:])
                    llm_dfs = {}
                    for f in llm_fs:
                        name = Path(f).stem.replace("llm_", "").replace("_annotations", "")
                        llm_dfs[name] = _load_file(f)

                results = run_full_evaluation(
                    human_df, llm_dfs,
                    do_bootstrap=do_boot, do_shap=do_shap,
                    do_calibration=do_cal, do_errors=do_err,
                    progress=progress,
                )

                log_text = _build_log(results["log"])
                summary_df = pd.DataFrame(results["summary_rows"])
                model_names = list(results["models"].keys())
                first = model_names[0]
                mr = results["models"][first]

                return (
                    log_text, summary_df, results,
                    gr.update(choices=model_names, value=first),
                    mr.get("cm_fig"), mr.get("f1_fig"),
                    mr.get("shap_fig"), mr.get("feat_fig"),
                    mr.get("cal_fig"), results.get("agreement_fig"),
                    mr.get("error_df", pd.DataFrame()),
                    json.dumps(mr.get("error_examples", {}), indent=2, ensure_ascii=False),
                )
            except Exception as e:
                return (f"ERROR: {e}", *EMPTY_RESULT[1:])

        btn_run.click(
            fn=on_run,
            inputs=[data_source, human_file, llm_files, model_pick,
                    chk_bootstrap, chk_shap, chk_calibration, chk_errors],
            outputs=[result_log, result_summary, eval_results, chart_model,
                     plot_cm, plot_f1, plot_shap, plot_feat, plot_cal, plot_agree,
                     error_table, error_json],
        )

        def on_chart_model_change(results, model_name):
            if not results or not model_name:
                return None, None, None, None, None, None, pd.DataFrame(), ""
            mr = results["models"].get(model_name, {})
            return (
                mr.get("cm_fig"), mr.get("f1_fig"),
                mr.get("shap_fig"), mr.get("feat_fig"),
                mr.get("cal_fig"), results.get("agreement_fig"),
                mr.get("error_df", pd.DataFrame()),
                json.dumps(mr.get("error_examples", {}), indent=2, ensure_ascii=False),
            )

        chart_model.change(
            fn=on_chart_model_change,
            inputs=[eval_results, chart_model],
            outputs=[plot_cm, plot_f1, plot_shap, plot_feat, plot_cal, plot_agree,
                     error_table, error_json],
        )

        # ──────────────── QUICK TEST (collapsible) ─────────────────
        with gr.Accordion("Quick Test -- Annotate a single text via LLM API", open=False):
            gr.Markdown("Paste one text, pick a model, see the LLM's annotation instantly.")
            with gr.Row():
                with gr.Column():
                    qt_key = gr.Textbox(label="API Key", type="password", placeholder="sk-...")
                    qt_base = gr.Textbox(label="Base URL (blank = OpenAI)", placeholder="https://api.openai.com/v1")
                    qt_model = gr.Textbox(label="Model", value="gpt-4o-mini")
                    qt_labels = gr.Textbox(label="Labels (comma-separated)", value="positive, negative, neutral, mixed")
                with gr.Column():
                    qt_text = gr.Textbox(label="Your Text", lines=4, placeholder="Paste any text here...")
                    qt_btn = gr.Button("Annotate", variant="secondary")
                    qt_result = gr.JSON(label="Result")

            def on_quick_test(key, base, model, labels_str, text):
                if not key:
                    return {"error": "Enter an API key."}
                if not text.strip():
                    return {"error": "Enter some text."}
                from openai import OpenAI
                labels = [l.strip() for l in labels_str.split(",") if l.strip()]
                kw = {"api_key": key}
                if base and base.strip():
                    kw["base_url"] = base.strip()
                try:
                    client = OpenAI(**kw)
                    return annotate_single(client, model, text, labels)
                except Exception as e:
                    return {"error": str(e)}

            qt_btn.click(on_quick_test, [qt_key, qt_base, qt_model, qt_labels, qt_text], [qt_result])

    return app


def main():
    parser = argparse.ArgumentParser(description="LLM Annotation Evaluation Dashboard")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    app = create_app()
    app.launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
