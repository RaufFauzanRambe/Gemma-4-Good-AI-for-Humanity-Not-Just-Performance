"""
Gemma 4 - Good AI for Humanity, Not Just Performance
=====================================================
src/models/baseline.py
Baseline Model Architectures

Implements a suite of classical ML baseline models for predicting
the composite quality score of AI responses.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple, Any
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
    ExtraTreesRegressor,
)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

logger = logging.getLogger(__name__)

# Try importing XGBoost
try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    logger.info("XGBoost not installed — XGBRegressor will be skipped.")


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
MODEL_REGISTRY: Dict[str, dict] = {
    "ridge": {
        "class": Ridge,
        "params": {"alpha": 1.0},
        "description": "L2-regularised linear regression (ridge). Good default for dense features.",
    },
    "lasso": {
        "class": Lasso,
        "params": {"alpha": 0.01},
        "description": "L1-regularised linear regression. Performs feature selection.",
    },
    "elastic_net": {
        "class": ElasticNet,
        "params": {"alpha": 0.01, "l1_ratio": 0.5},
        "description": "Combination of L1 and L2 regularisation.",
    },
    "random_forest": {
        "class": RandomForestRegressor,
        "params": {"n_estimators": 200, "max_depth": 10, "random_state": 42, "n_jobs": -1},
        "description": "Bagged decision trees. Robust to outliers.",
    },
    "gradient_boosting": {
        "class": GradientBoostingRegressor,
        "params": {"n_estimators": 150, "max_depth": 5, "learning_rate": 0.1, "random_state": 42},
        "description": "Sequential boosting. Often the strongest baseline.",
    },
    "ada_boost": {
        "class": AdaBoostRegressor,
        "params": {"n_estimators": 100, "random_state": 42},
        "description": "Adaptive boosting with decision stumps.",
    },
    "extra_trees": {
        "class": ExtraTreesRegressor,
        "params": {"n_estimators": 200, "max_depth": 12, "random_state": 42, "n_jobs": -1},
        "description": "Extremely randomised trees. Faster than RF with more regularisation.",
    },
    "svr_rbf": {
        "class": SVR,
        "params": {"kernel": "rbf", "C": 10, "epsilon": 0.1},
        "description": "Support Vector Regression with RBF kernel.",
    },
    "svr_linear": {
        "class": SVR,
        "params": {"kernel": "linear", "C": 1.0},
        "description": "Linear SVR. Good for high-dimensional data.",
    },
    "knn": {
        "class": KNeighborsRegressor,
        "params": {"n_neighbors": 5, "weights": "distance"},
        "description": "K-Nearest Neighbours. Instance-based learning.",
    },
    "decision_tree": {
        "class": DecisionTreeRegressor,
        "params": {"max_depth": 8, "random_state": 42},
        "description": "Single decision tree. Fast but prone to overfitting.",
    },
}

if HAS_XGBOOST:
    MODEL_REGISTRY["xgboost"] = {
        "class": XGBRegressor,
        "params": {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.1,
                    "random_state": 42, "n_jobs": -1, "verbosity": 0},
        "description": "Gradient boosting with XGBoost. Often best performance.",
    }


# ---------------------------------------------------------------------------
# BaselineRunner
# ---------------------------------------------------------------------------
class BaselineRunner:
    """Train and compare all baseline models.

    Example
    -------
    >>> runner = BaselineRunner()
    >>> runner.fit(X_train, y_train)
    >>> results = runner.evaluate(X_val, y_val)
    >>> print(results.sort_values("R2", ascending=False))
    """

    def __init__(self, scale: bool = True, cv_folds: int = 5):
        self.scale = scale
        self.cv_folds = cv_folds
        self.scaler = StandardScaler() if scale else None
        self.models: Dict[str, Any] = {}
        self.results: Optional[pd.DataFrame] = None

    def fit(self, X: np.ndarray, y: np.ndarray,
            model_names: Optional[List[str]] = None) -> "BaselineRunner":
        """Fit all (or selected) baseline models.

        Parameters
        ----------
        X : np.ndarray  shape (n_samples, n_features)
        y : np.ndarray  shape (n_samples,)
        model_names : list[str], optional
            Subset of MODEL_REGISTRY keys to train.  ``None`` = all.
        """
        X_scaled = self.scaler.fit_transform(X) if self.scaler else X

        if model_names is None:
            model_names = list(MODEL_REGISTRY.keys())

        for name in model_names:
            if name not in MODEL_REGISTRY:
                logger.warning("Unknown model '%s' — skipping.", name)
                continue
            entry = MODEL_REGISTRY[name]
            model = entry["class"](**entry["params"])
            model.fit(X_scaled, y)
            self.models[name] = model
            logger.info("Trained %s", name)

        return self

    def predict(self, X: np.ndarray, model_name: Optional[str] = None) -> np.ndarray:
        """Predict with a specific model or the best model."""
        X_scaled = self.scaler.transform(X) if self.scaler else X
        if model_name:
            return self.models[model_name].predict(X_scaled)
        best = self.results["R2"].idxmax() if self.results is not None else list(self.models.keys())[0]
        return self.models[best].predict(X_scaled)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
        """Evaluate all fitted models and return a results DataFrame.

        Metrics: RMSE, MAE, R2, CV-RMSE (mean ± std).
        """
        X_scaled = self.scaler.transform(X) if self.scaler else X
        records = []

        for name, model in self.models.items():
            y_pred = model.predict(X_scaled)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            mae = mean_absolute_error(y, y_pred)
            r2 = r2_score(y, y_pred)

            # Cross-validation on training data
            X_train_scaled = self.scaler.inverse_transform(X_scaled) if self.scaler else X_scaled
            X_train_scaled = self.scaler.fit_transform(X_train_scaled) if self.scaler else X_train_scaled
            cv = cross_val_score(model, X_train_scaled, y, cv=self.cv_folds,
                                 scoring="neg_root_mean_squared_error")

            records.append({
                "Model": name,
                "RMSE": round(rmse, 4),
                "MAE": round(mae, 4),
                "R2": round(r2, 4),
                "CV_RMSE_Mean": round(-cv.mean(), 4),
                "CV_RMSE_Std": round(cv.std(), 4),
            })
            logger.info("%-20s | RMSE=%.4f | MAE=%.4f | R2=%.4f | CV=%.4f±%.4f",
                         name, rmse, mae, r2, -cv.mean(), cv.std())

        self.results = pd.DataFrame(records).set_index("Model")
        return self.results

    def cross_validate(self, X: np.ndarray, y: np.ndarray,
                       model_names: Optional[List[str]] = None) -> pd.DataFrame:
        """Run cross-validation only (no train/val split)."""
        X_scaled = self.scaler.fit_transform(X) if self.scaler else X
        if model_names is None:
            model_names = list(self.models.keys())

        records = []
        for name in model_names:
            if name not in self.models:
                continue
            cv = cross_val_score(self.models[name], X_scaled, y,
                                 cv=self.cv_folds, scoring="neg_root_mean_squared_error")
            records.append({
                "Model": name,
                "CV_RMSE_Mean": round(-cv.mean(), 4),
                "CV_RMSE_Std": round(cv.std(), 4),
                "CV_Fold_Scores": [round(-s, 4) for s in cv],
            })
        return pd.DataFrame(records).set_index("Model")

    def get_best_model(self, metric: str = "R2") -> Tuple[str, Any]:
        """Return (name, model) of the best-performing model."""
        if self.results is None:
            raise RuntimeError("Call .evaluate() first.")
        best_name = self.results[metric].idxmax()
        return best_name, self.models[best_name]

    @staticmethod
    def list_models() -> pd.DataFrame:
        """Return a summary of all registered models."""
        rows = []
        for name, info in MODEL_REGISTRY.items():
            rows.append({"Name": name, "Class": info["class"].__name__,
                          "Description": info["description"]})
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Quick demo with synthetic data
    np.random.seed(42)
    n = 200
    X = np.random.randn(n, 20)
    y = 3.5 + 0.5 * X[:, 0] + 0.3 * X[:, 1] - 0.2 * X[:, 2] + np.random.randn(n) * 0.3
    y = np.clip(y, 1.0, 5.0)

    X_train, X_val = X[:160], X[160:]
    y_train, y_val = y[:160], y[160:]

    runner = BaselineRunner(scale=True, cv_folds=5)
    runner.fit(X_train, y_train)
    results = runner.evaluate(X_val, y_val)

    print("\n" + "=" * 65)
    print("BASELINE MODEL RESULTS")
    print("=" * 65)
    print(results.sort_values("R2", ascending=False).to_string())

    best_name, best_model = runner.get_best_model()
    print(f"\nBest model: {best_name}")
