"""
Gemma 4 - Good AI for Humanity, Not Just Performance
=====================================================
src/evaluation/metrics.py
Custom Evaluation Metrics

Comprehensive metrics for evaluating AI response quality predictions,
including humanity-specific measures beyond standard regression metrics.
"""

import numpy as np
from typing import Dict, Optional, Tuple
from scipy import stats as scipy_stats


# ---------------------------------------------------------------------------
# Standard regression metrics
# ---------------------------------------------------------------------------
def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute all evaluation metrics.

    Returns dict with: rmse, mae, r2, mape, medae, max_error,
                       pearson_r, spearman_r, explained_variance.
    """
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "medae": medae(y_true, y_pred),
        "max_error": max_error(y_true, y_pred),
        "pearson_r": pearson_correlation(y_true, y_pred),
        "spearman_r": spearman_correlation(y_true, y_pred),
        "explained_variance": explained_variance(y_true, y_pred),
    }


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R-squared (coefficient of determination)."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1 - ss_res / ss_tot)


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error."""
    mask = y_true != 0
    if not mask.any():
        return float("inf")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def medae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Median Absolute Error."""
    return float(np.median(np.abs(y_true - y_pred)))


def max_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Maximum absolute error."""
    return float(np.max(np.abs(y_true - y_pred)))


def pearson_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Pearson correlation coefficient."""
    r, _ = scipy_stats.pearsonr(y_true, y_pred)
    return float(r)


def spearman_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Spearman rank correlation."""
    r, _ = scipy_stats.spearmanr(y_true, y_pred)
    return float(r)


def explained_variance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Explained variance score."""
    return float(1 - np.var(y_true - y_pred) / np.var(y_true)) if np.var(y_true) > 0 else 0.0


# ---------------------------------------------------------------------------
# Humanity-specific metrics
# ---------------------------------------------------------------------------
def humanity_alignment_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Measures how well predictions preserve the *relative ranking* of
    humanity-aligned responses.

    A high alignment score means the model correctly identifies which
    responses are more humanity-focused (higher composite scores).

    Returns a value between 0 and 1 (1 = perfect ranking).
    """
    # Normalised discounted cumulative gain at k
    k = min(10, len(y_true))
    true_order = np.argsort(y_true)[::-1][:k]
    pred_order = np.argsort(y_pred)[::-1][:k]

    # DCG
    gains = y_true[pred_order]
    dcg = np.sum(gains / np.log2(np.arange(2, k + 2)))

    # Ideal DCG
    ideal_gains = np.sort(y_true)[::-1][:k]
    idcg = np.sum(ideal_gains / np.log2(np.arange(2, k + 2)))

    return float(dcg / idcg) if idcg > 0 else 0.0


def fairness_aware_rmse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group_labels: np.ndarray,
) -> Dict[str, float]:
    """Compute RMSE separately for each group to detect bias.

    Returns a dict mapping each group label to its RMSE.
    The *worst-group RMSE* indicates potential fairness issues.
    """
    groups = np.unique(group_labels)
    results = {}
    for g in groups:
        mask = group_labels == g
        if mask.sum() >= 2:
            results[str(g)] = rmse(y_true[mask], y_pred[mask])
    results["worst_group_rmse"] = max(results.values()) if results else float("inf")
    results["rmse_gap"] = max(results.values()) - min(results.values()) if results else 0.0
    return results


def calibration_error(y_true: np.ndarray, y_pred: np.ndarray,
                       n_bins: int = 10) -> float:
    """Expected Calibration Error (ECE).

    Measures how well predicted scores are calibrated against actual scores.
    Lower is better (0 = perfect calibration).
    """
    bin_edges = np.linspace(y_true.min(), y_true.max(), n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_true >= bin_edges[i]) & (y_true < bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        avg_true = y_true[mask].mean()
        avg_pred = y_pred[mask].mean()
        ece += mask.sum() / len(y_true) * abs(avg_true - avg_pred)
    return float(ece)


def score_band_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    bands: Optional[list] = None,
) -> Dict[str, Dict[str, float]]:
    """Compute accuracy within score bands (e.g., 1-2, 2-3, 3-4, 4-5).

    Returns per-band: mean_error, std_error, count, pct_within_0.5.
    """
    if bands is None:
        bands = [(1, 2), (2, 3), (3, 4), (4, 5)]

    results = {}
    for low, high in bands:
        mask = (y_true >= low) & (y_true < high)
        if mask.sum() == 0:
            continue
        errors = np.abs(y_true[mask] - y_pred[mask])
        key = f"{low:.0f}-{high:.0f}"
        results[key] = {
            "mean_error": round(float(errors.mean()), 4),
            "std_error": round(float(errors.std()), 4),
            "count": int(mask.sum()),
            "pct_within_0.5": round(float((errors <= 0.5).mean()) * 100, 2),
        }
    return results


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------
def paired_ttest(y_true: np.ndarray, y_pred_a: np.ndarray,
                 y_pred_b: np.ndarray) -> Dict[str, float]:
    """Paired t-test between two sets of predictions.

    Returns t-statistic and p-value. Low p-value (< 0.05) indicates
    statistically significant difference between models.
    """
    errors_a = np.abs(y_true - y_pred_a)
    errors_b = np.abs(y_true - y_pred_b)
    t_stat, p_value = scipy_stats.ttest_rel(errors_a, errors_b)
    return {"t_statistic": round(float(t_stat), 4), "p_value": round(float(p_value), 6)}


def wilcoxon_test(y_true: np.ndarray, y_pred_a: np.ndarray,
                  y_pred_b: np.ndarray) -> Dict[str, float]:
    """Wilcoxon signed-rank test (non-parametric alternative to paired t-test)."""
    errors_a = np.abs(y_true - y_pred_a)
    errors_b = np.abs(y_true - y_pred_b)
    stat, p_value = scipy_stats.wilcoxon(errors_a, errors_b)
    return {"statistic": round(float(stat), 4), "p_value": round(float(p_value), 6)}


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(42)
    n = 200
    y_true = np.random.uniform(1.0, 5.0, n)
    y_pred = y_true + np.random.normal(0, 0.3, n)
    y_pred = np.clip(y_pred, 1.0, 5.0)
    groups = np.random.choice(["A", "B", "C"], size=n)

    print("=" * 55)
    print("METRICS DEMO")
    print("=" * 55)

    all_metrics = compute_all_metrics(y_true, y_pred)
    for k, v in all_metrics.items():
        print(f"  {k:20s}: {v:.4f}")

    print(f"\n  Humanity Alignment:   {humanity_alignment_score(y_true, y_pred):.4f}")
    print(f"  Calibration Error:   {calibration_error(y_true, y_pred):.4f}")

    print(f"\n  Fairness-Aware RMSE:")
    for g, v in fairness_aware_rmse(y_true, y_pred, groups).items():
        print(f"    {g:20s}: {v:.4f}")

    print(f"\n  Score Band Accuracy:")
    for band, v in score_band_accuracy(y_true, y_pred).items():
        print(f"    [{band}]  err={v['mean_error']:.3f}  within_0.5={v['pct_within_0.5']}%  n={v['count']}")
