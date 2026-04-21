"""
Gemma 4 - Good AI for Humanity, Not Just Performance
=====================================================
src/models/advanced_model.py
Advanced Model Architectures

Implements Stacking, Blending, and custom ensemble methods
combining baseline models with TF-IDF / deep features.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple, Any
from sklearn.ensemble import (
    StackingRegressor,
    VotingRegressor,
    BaggingRegressor,
)
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import logging

logger = logging.getLogger(__name__)

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


# ---------------------------------------------------------------------------
# Stacking Ensemble
# ---------------------------------------------------------------------------
class HumanityStackingEnsemble:
    """Stacking regressor optimised for the composite quality score.

    Architecture
    -------------
    - **Base learners**: Ridge, RandomForest, GradientBoosting, SVR
    - **Meta-learner**: Ridge(alpha=0.5)
    - **Features**: Structural + TF-IDF/SVD combined

    Example
    -------
    >>> ensemble = HumanityStackingEnsemble()
    >>> ensemble.fit(X_train, y_train)
    >>> y_pred = ensemble.predict(X_val)
    """

    def __init__(self, cv_folds: int = 5, scale: bool = True, random_state: int = 42):
        self.cv_folds = cv_folds
        self.scale = scale
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = self._build_stacking()
        self._fitted = False

    def _build_stacking(self) -> StackingRegressor:
        """Construct the stacking ensemble."""
        base_estimators = [
            ("ridge", Ridge(alpha=1.0)),
            ("rf", RandomForestRegressor(
                n_estimators=200, max_depth=10,
                random_state=self.random_state, n_jobs=-1,
            )),
            ("gb", GradientBoostingRegressor(
                n_estimators=150, max_depth=5, learning_rate=0.1,
                random_state=self.random_state,
            )),
            ("svr", SVR(kernel="rbf", C=10, epsilon=0.1)),
        ]

        if HAS_XGBOOST:
            base_estimators.append((
                "xgb", XGBRegressor(
                    n_estimators=200, max_depth=6, learning_rate=0.1,
                    random_state=self.random_state, n_jobs=-1, verbosity=0,
                )
            ))

        stacking = StackingRegressor(
            estimators=base_estimators,
            final_estimator=Ridge(alpha=0.5),
            cv=self.cv_folds,
            n_jobs=-1,
        )
        return stacking

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HumanityStackingEnsemble":
        """Fit the full stacking pipeline (scale + model)."""
        if self.scale:
            X = self.scaler.fit_transform(X)
        self.model.fit(X, y)
        self._fitted = True
        logger.info("Stacking ensemble trained (%d base learners)", len(self.model.estimators))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict composite quality scores."""
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call .fit() first.")
        if self.scale:
            X = self.scaler.transform(X)
        preds = self.model.predict(X)
        return np.clip(preds, 1.0, 5.0)

    def cross_validate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Run cross-validation and return metrics."""
        X_input = self.scaler.fit_transform(X) if self.scale else X
        cv_rmse = cross_val_score(self.model, X_input, y,
                                   cv=self.cv_folds, scoring="neg_root_mean_squared_error")
        cv_r2 = cross_val_score(self.model, X_input, y,
                                 cv=self.cv_folds, scoring="r2")

        metrics = {
            "cv_rmse_mean": round(-cv_rmse.mean(), 4),
            "cv_rmse_std": round(cv_rmse.std(), 4),
            "cv_r2_mean": round(cv_r2.mean(), 4),
            "cv_r2_std": round(cv_r2.std(), 4),
        }
        logger.info("CV RMSE: %.4f ± %.4f | CV R2: %.4f ± %.4f",
                     metrics["cv_rmse_mean"], metrics["cv_rmse_std"],
                     metrics["cv_r2_mean"], metrics["cv_r2_std"])
        return metrics

    def get_feature_importance(self) -> Optional[Dict[str, np.ndarray]]:
        """Extract feature importance from tree-based base learners."""
        importance = {}
        for name, model in self.model.estimators_:
            if hasattr(model, "feature_importances_"):
                importance[name] = model.feature_importances_
        return importance if importance else None


# ---------------------------------------------------------------------------
# Blending Ensemble
# ---------------------------------------------------------------------------
class BlendingEnsemble:
    """Blending (out-of-fold) ensemble with configurable weights.

    Unlike stacking, blending uses a simple weighted average of base model
    predictions, with weights optimised on a hold-out set.
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None, scale: bool = True):
        self.weights = weights or {
            "ridge": 0.15, "rf": 0.20, "gb": 0.30, "svr": 0.15,
        }
        if HAS_XGBOOST:
            self.weights["xgb"] = 0.20
        self.scale = scale
        self.scaler = StandardScaler()
        self.models: Dict[str, Any] = {}
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BlendingEnsemble":
        """Fit all base models."""
        if self.scale:
            X = self.scaler.fit_transform(X)

        model_defs = {
            "ridge": Ridge(alpha=1.0),
            "rf": RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
            "gb": GradientBoostingRegressor(n_estimators=150, max_depth=5, learning_rate=0.1, random_state=42),
            "svr": SVR(kernel="rbf", C=10, epsilon=0.1),
        }
        if HAS_XGBOOST:
            model_defs["xgb"] = XGBRegressor(
                n_estimators=200, max_depth=6, learning_rate=0.1,
                random_state=42, n_jobs=-1, verbosity=0
            )

        for name, model_def in model_defs.items():
            if name in self.weights:
                self.models[name] = model_def.fit(X, y)
                logger.info("Blending: trained %s", name)

        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Weighted average of base model predictions."""
        if not self._fitted:
            raise RuntimeError("Model not fitted.")
        if self.scale:
            X = self.scaler.transform(X)

        weighted_sum = np.zeros(len(X))
        total_weight = 0.0
        for name, model in self.models.items():
            w = self.weights.get(name, 0.0)
            weighted_sum += w * model.predict(X)
            total_weight += w

        return np.clip(weighted_sum / total_weight, 1.0, 5.0)


# ---------------------------------------------------------------------------
# Voting Ensemble
# ---------------------------------------------------------------------------
class VotingEnsemble:
    """Scikit-learn VotingRegressor wrapper with pre-built model configs."""

    def __init__(self, scale: bool = True):
        self.scale = scale
        self.scaler = StandardScaler()
        estimators = [
            ("ridge", Ridge(alpha=1.0)),
            ("rf", RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)),
            ("gb", GradientBoostingRegressor(n_estimators=150, max_depth=5, learning_rate=0.1, random_state=42)),
            ("svr", SVR(kernel="rbf", C=10)),
        ]
        if HAS_XGBOOST:
            estimators.append(("xgb", XGBRegressor(n_estimators=200, max_depth=6,
                                                   random_state=42, n_jobs=-1, verbosity=0)))
        self.model = VotingRegressor(estimators=estimators)
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "VotingEnsemble":
        if self.scale:
            X = self.scaler.fit_transform(X)
        self.model.fit(X, y)
        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Model not fitted.")
        if self.scale:
            X = self.scaler.transform(X)
        return np.clip(self.model.predict(X), 1.0, 5.0)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(42)
    n = 200
    X = np.random.randn(n, 30)
    y = 3.8 + 0.4 * X[:, 0] - 0.3 * X[:, 3] + 0.2 * X[:, 5] + np.random.randn(n) * 0.25
    y = np.clip(y, 1.0, 5.0)
    X_train, X_val = X[:160], X[160:]
    y_train, y_val = y[:160], y[160:]

    # --- Stacking ---
    print("=" * 60)
    print("STACKING ENSEMBLE")
    stacking = HumanityStackingEnsemble()
    cv_metrics = stacking.cross_validate(X_train, y_train)
    stacking.fit(X_train, y_train)
    y_pred = stacking.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    print(f"  CV RMSE: {cv_metrics['cv_rmse_mean']} +/- {cv_metrics['cv_rmse_std']}")
    print(f"  Val RMSE: {rmse:.4f}")

    # --- Blending ---
    print("\nBLENDING ENSEMBLE")
    blending = BlendingEnsemble()
    blending.fit(X_train, y_train)
    y_pred_b = blending.predict(X_val)
    print(f"  Val RMSE: {np.sqrt(mean_squared_error(y_val, y_pred_b)):.4f}")

    # --- Voting ---
    print("\nVOTING ENSEMBLE")
    voting = VotingEnsemble()
    voting.fit(X_train, y_train)
    y_pred_v = voting.predict(X_val)
    print(f"  Val RMSE: {np.sqrt(mean_squared_error(y_val, y_pred_v)):.4f}")
