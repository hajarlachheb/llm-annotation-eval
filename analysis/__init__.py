from .bootstrap import bootstrap_metric, bootstrap_all_metrics
from .slicing import evaluate_by_slice, bin_continuous_column
from .error_analysis import find_systematic_errors, sample_error_examples
from .explainability import (
    text_features_driving_errors,
    compute_shap_explanations,
    save_shap_summary_plot,
    save_shap_beeswarm_plot,
    extract_rationale_keywords,
)
from .calibration import (
    calibration_curve,
    confidence_stratified_metrics,
    find_overconfident_errors,
    find_underconfident_correct,
)

__all__ = [
    "bootstrap_metric",
    "bootstrap_all_metrics",
    "evaluate_by_slice",
    "bin_continuous_column",
    "find_systematic_errors",
    "sample_error_examples",
    "text_features_driving_errors",
    "compute_shap_explanations",
    "save_shap_summary_plot",
    "save_shap_beeswarm_plot",
    "extract_rationale_keywords",
    "calibration_curve",
    "confidence_stratified_metrics",
    "find_overconfident_errors",
    "find_underconfident_correct",
]
