"""
Gemma 4 - Good AI for Humanity, Not Just Performance
=====================================================
src/evaluation/evaluate.py
Evaluation Pipeline

End-to-end evaluation: predictions → metrics → analysis → reporting.
"""

import json
import logging
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple

import numpy as np
import pandas as pd

from src.evaluation.metrics import compute_all_metrics

logger = logging.getLogger(__name__)


class EvaluationPipeline:
    """Full evaluation pipeline for model predictions.

    Takes predictions and produces:
        - Standard metrics (RMSE, MAE, R2)
        - Per-category breakdown
        - Error analysis
        - Visualisation data
        - JSON report

    Example
    -------
    >>> pipe = EvaluationPipeline()
    >>> report = pipe.evaluate(y_true, y_pred, df, category_col="category")
    >>> pipe.save_report("results/")
    """

    def __init__(self, score_range: Tuple[float, float] = (1.0, 5.0)):
        self.score_range = score_range
        self.report: Dict[str, Any] = {}

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        df: Optional[pd.DataFrame] = None,
        category_col: str = "category",
        difficulty_col: str = "difficulty",
        model_name: str = "model",
    ) -> Dict[str, Any]:
        """Run the full evaluation pipeline.

        Parameters
        ----------
        y_true : np.ndarray
        y_pred : np.ndarray
        df : pd.DataFrame, optional
            Original DataFrame with metadata (category, difficulty, …).
        category_col : str
        difficulty_col : str
        model_name : str

        Returns
        -------
        dict — evaluation report.
        """
        # ---- Global metrics ----
        global_metrics = compute_all_metrics(y_true, y_pred)
        self.report = {
            "model_name": model_name,
            "n_samples": len(y_true),
            "global_metrics": {k: round(v, 4) for k, v in global_metrics.items()},
        }
        logger.info("Global metrics: RMSE=%.4f, MAE=%.4f, R2=%.4f",
                     global_metrics["rmse"], global_metrics["mae"], global_metrics["r2"])

        # ---- Per-category breakdown ----
        if df is not None and category_col in df.columns:
            cat_breakdown = self._per_group_analysis(y_true, y_pred, df[category_col].values, category_col)
            self.report["category_breakdown"] = cat_breakdown

        # ---- Per-difficulty breakdown ----
        if df is not None and difficulty_col in df.columns:
            diff_breakdown = self._per_group_analysis(y_true, y_pred, df[difficulty_col].values, difficulty_col)
            self.report["difficulty_breakdown"] = diff_breakdown

        # ---- Error analysis ----
        residuals = y_true - y_pred
        self.report["error_analysis"] = {
            "mean_residual": round(float(np.mean(residuals)), 4),
            "std_residual": round(float(np.std(residuals)), 4),
            "max_positive_error": round(float(np.max(residuals)), 4),
            "max_negative_error": round(float(np.min(residuals)), 4),
            "pct_within_0.5": round(float(np.mean(np.abs(residuals) <= 0.5)) * 100, 2),
            "pct_within_1.0": round(float(np.mean(np.abs(residuals) <= 1.0)) * 100, 2),
        }

        # ---- Worst & Best predictions ----
        abs_errors = np.abs(residuals)
        worst_idx = np.argsort(abs_errors)[-5:][::-1]
        best_idx = np.argsort(abs_errors)[:5]

        worst_examples = []
        best_examples = []
        if df is not None and "prompt" in df.columns:
            y_true_arr = np.array(y_true)
            y_pred_arr = np.array(y_pred)
            for idx in worst_idx:
                worst_examples.append({
                    "prompt": str(df.iloc[idx]["prompt"])[:100],
                    "actual": round(float(y_true_arr[idx]), 2),
                    "predicted": round(float(y_pred_arr[idx]), 2),
                    "error": round(float(abs_errors[idx]), 4),
                })
            for idx in best_idx:
                best_examples.append({
                    "prompt": str(df.iloc[idx]["prompt"])[:100],
                    "actual": round(float(y_true_arr[idx]), 2),
                    "predicted": round(float(y_pred_arr[idx]), 2),
                    "error": round(float(abs_errors[idx]), 4),
                })

        self.report["worst_predictions"] = worst_examples
        self.report["best_predictions"] = best_examples

        return self.report

    def _per_group_analysis(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        group_labels: np.ndarray,
        group_name: str,
    ) -> Dict[str, Dict]:
        """Compute metrics per group (category, difficulty, …)."""
        groups = pd.Series(group_labels)
        breakdown = {}
        for g in groups.unique():
            mask = groups == g
            if mask.sum() < 2:
                continue
            m = compute_all_metrics(y_true[mask], y_pred[mask])
            breakdown[g] = {k: round(v, 4) for k, v in m.items()}
            breakdown[g]["count"] = int(mask.sum())
        return breakdown

    def save_report(self, output_dir: str, filename: str = "evaluation_report.json"):
        """Save the evaluation report to JSON."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        path = out_path / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.report, f, indent=2, ensure_ascii=False, default=str)
        logger.info("Evaluation report saved: %s", path)
        return path

    def print_summary(self):
        """Print a human-readable summary of the evaluation."""
        if not self.report:
            logger.warning("No report available. Run .evaluate() first.")
            return

        print(f"\n{'=' * 55}")
        print(f"  EVALUATION REPORT — {self.report.get('model_name', 'N/A')}")
        print(f"{'=' * 55}")
        print(f"  Samples: {self.report['n_samples']}")
        print(f"  RMSE:    {self.report['global_metrics']['rmse']:.4f}")
        print(f"  MAE:     {self.report['global_metrics']['mae']:.4f}")
        print(f"  R2:      {self.report['global_metrics']['r2']:.4f}")
        print(f"  MAPE:    {self.report['global_metrics']['mape']:.2f}%")

        ea = self.report.get("error_analysis", {})
        print(f"\n  Error Analysis:")
        print(f"    Mean residual:     {ea.get('mean_residual', 'N/A')}")
        print(f"    Std residual:      {ea.get('std_residual', 'N/A')}")
        print(f"    Within +/-0.5:     {ea.get('pct_within_0.5', 'N/A')}%")
        print(f"    Within +/-1.0:     {ea.get('pct_within_1.0', 'N/A')}%")

        if "category_breakdown" in self.report:
            print(f"\n  Category Breakdown:")
            for cat, m in self.report["category_breakdown"].items():
                print(f"    {cat:20s} | RMSE={m['rmse']:.3f} | R2={m['r2']:.3f} | n={m['count']}")

        if "difficulty_breakdown" in self.report:
            print(f"\n  Difficulty Breakdown:")
            for diff, m in self.report["difficulty_breakdown"].items():
                print(f"    {diff:20s} | RMSE={m['rmse']:.3f} | R2={m['r2']:.3f} | n={m['count']}")

        print(f"{'=' * 55}\n")
