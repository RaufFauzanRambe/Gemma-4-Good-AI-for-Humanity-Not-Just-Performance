"""
Gemma 4 - Good AI for Humanity, Not Just Performance
=====================================================
src/training/train.py
Main Training Entry-Point

Orchestrates the full training pipeline: data loading → preprocessing →
feature engineering → model training → evaluation → artifact saving.
"""

import os
import sys
import json
import pickle
import logging
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import DataLoader
from src.data.preprocess import preprocess_pipeline
from src.data.augmentation import TextAugmenter
from src.models.baseline import BaselineRunner
from src.models.advanced_model import HumanityStackingEnsemble
from src.evaluation.metrics import compute_all_metrics
from src.evaluation.evaluate import EvaluationPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(PROJECT_ROOT / "training.log"),
    ],
)
logger = logging.getLogger(__name__)

# Output directory for artifacts
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Combine structural text features with pre-computed scores."""
    pp = preprocess_pipeline(df, include_tokens=False)
    return pp


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------
def train(
    data_dir: str = None,
    model_type: str = "stacking",
    augment: bool = False,
    test_size: float = 0.2,
    cv_folds: int = 5,
    save_artifacts: bool = True,
) -> dict:
    """Execute the full training pipeline.

    Parameters
    ----------
    data_dir : str, optional
        Override the raw data directory.
    model_type : str
        ``"baseline"``, ``"stacking"``, or ``"all"``.
    augment : bool
        Whether to apply data augmentation.
    test_size : float
        Validation split ratio.
    cv_folds : int
        Number of cross-validation folds.
    save_artifacts : bool
        Persist model + results to disk.

    Returns
    -------
    dict with keys: ``results``, ``best_model_name``, ``metrics``.
    """
    logger.info("=" * 60)
    logger.info("TRAINING PIPELINE STARTED")
    logger.info("Model type: %s | Augment: %s | CV folds: %d", model_type, augment, cv_folds)

    # ---- 1. Load data ----
    loader = DataLoader(data_dir=data_dir)
    train_df, test_df, labels_df = loader.load_all()
    merged = loader.get_merged_dataset()

    logger.info("Data loaded: train=%d, test=%d, merged=%d",
                 len(train_df), len(test_df), len(merged))

    # ---- 2. Preprocess & feature engineering ----
    df = build_feature_matrix(merged)

    # Define feature columns
    STRUCTURAL_COLS = [
        "prompt_length", "response_length", "prompt_word_count", "response_word_count",
        "length_ratio", "sentence_count", "avg_sentence_length", "has_list", "has_number",
        "paragraph_count", "unique_word_count", "lexical_diversity",
        "question_marks", "exclamation_marks",
        "has_medical_terms", "has_empathy_words",
    ]
    ANNOTATION_COLS = ["safety_score", "fairness_score", "clarity_score"]
    available_features = [c for c in STRUCTURAL_COLS + ANNOTATION_COLS if c in df.columns]

    X = df[available_features].values
    y = df["composite_score"].values

    logger.info("Feature matrix: %d samples × %d features", X.shape[0], X.shape[1])

    # ---- 3. Optional augmentation ----
    if augment:
        augmenter = TextAugmenter(seed=42)
        df_aug = augmenter.augment_dataframe(
            df, text_col="response", strategies=["synonym_replacement", "random_swap"],
            copies=1, score_noise=0.05,
        )
        X_aug = df_aug[available_features].values
        y_aug = df_aug["composite_score"].values
        X = np.vstack([X, X_aug])
        y = np.concatenate([y, y_aug])
        logger.info("After augmentation: %d samples", len(y))

    # ---- 4. Train / val split ----
    X_train, X_val, y_train, y_val = loader.stratified_split(
        df[:len(y)].assign(composite_score=y),
        test_size=test_size,
    )
    X_train = X[X_train.index.values] if len(X_train.index.intersection(range(len(X)))) > 0 else X[:int(len(X) * (1 - test_size))]
    X_val = X[X_val.index.values] if len(X_val.index.intersection(range(len(X)))) > 0 else X[int(len(X) * (1 - test_size)):]
    y_train, y_val = X_train[:, :1], X_val[:, :1]  # Placeholder — real split handled by indices

    # Simpler approach:
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    logger.info("Split: train=%d, val=%d", len(y_train), len(y_val))

    # ---- 5. Train models ----
    results_summary = {}

    if model_type in ("baseline", "all"):
        logger.info("Training baseline models...")
        runner = BaselineRunner(scale=True, cv_folds=cv_folds)
        runner.fit(X_train, y_train)
        baseline_results = runner.evaluate(X_val, y_val)
        results_summary["baseline"] = baseline_results.to_dict()

    if model_type in ("stacking", "all"):
        logger.info("Training stacking ensemble...")
        stacking = HumanityStackingEnsemble(cv_folds=cv_folds)
        cv_metrics = stacking.cross_validate(X_train, y_train)
        stacking.fit(X_train, y_train)
        y_pred = stacking.predict(X_val)
        stack_metrics = compute_all_metrics(y_val, y_pred)
        results_summary["stacking"] = {**cv_metrics, **stack_metrics}

    # ---- 6. Evaluation ----
    logger.info("Running evaluation pipeline...")
    eval_results = {}
    if model_type == "stacking":
        eval_results = {
            "y_true": y_val.tolist(),
            "y_pred": y_pred.tolist(),
            "metrics": stack_metrics,
        }

    # ---- 7. Save artifacts ----
    if save_artifacts:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Results JSON
        results_path = OUTPUT_DIR / f"training_results_{timestamp}.json"
        with open(results_path, "w") as f:
            json.dump(results_summary, f, indent=2, default=str)
        logger.info("Results saved: %s", results_path)

        # Model pickle
        if model_type == "stacking":
            model_path = OUTPUT_DIR / f"stacking_model_{timestamp}.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(stacking, f)
            logger.info("Model saved: %s", model_path)

    logger.info("=" * 60)
    logger.info("TRAINING PIPELINE COMPLETE")
    logger.info("=" * 60)

    return {
        "results": results_summary,
        "features": available_features,
        "n_train": len(y_train),
        "n_val": len(y_val),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train models for Gemma 4 project")
    parser.add_argument("--model", type=str, default="all", choices=["baseline", "stacking", "all"])
    parser.add_argument("--augment", action="store_true", help="Apply data augmentation")
    parser.add_argument("--cv", type=int, default=5, help="Cross-validation folds")
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    results = train(
        model_type=args.model,
        augment=args.augment,
        test_size=args.test_size,
        cv_folds=args.cv,
    )

    print("\nTraining complete!")
    print(f"  Models trained: {list(results['results'].keys())}")
    print(f"  Features used: {len(results['features'])}")
    print(f"  Train samples: {results['n_train']}")
    print(f"  Val samples:   {results['n_val']}")
