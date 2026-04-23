"""
Gemma 4 - Good AI for Humanity, Not Just Performance
=====================================================
src/evaluation/metrics.py
Custom Evaluation Metrics (see validation.py for full metrics)

This file provides the primary metrics functions imported by the pipeline.
Re-exports everything from validation.py for backward compatibility.
"""

# Re-export all metrics from the comprehensive validation module
from src.evaluation.validation import (
    compute_all_metrics,
    rmse, mae, r2_score, mape, medae, max_error,
    pearson_correlation, spearman_correlation, explained_variance,
    humanity_alignment_score,
    fairness_aware_rmse,
    calibration_error,
    score_band_accuracy,
    paired_ttest, wilcoxon_test,
)

__all__ = [
    "compute_all_metrics",
    "rmse", "mae", "r2_score", "mape", "medae", "max_error",
    "pearson_correlation", "spearman_correlation", "explained_variance",
    "humanity_alignment_score",
    "fairness_aware_rmse",
    "calibration_error",
    "score_band_accuracy",
    "paired_ttest", "wilcoxon_test",
]
