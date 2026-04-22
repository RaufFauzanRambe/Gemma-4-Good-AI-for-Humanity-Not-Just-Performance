"""
Gemma 4 - Good AI for Humanity, Not Just Performance
=====================================================
src/training/scheduler.py
Learning Rate & Hyperparameter Schedulers

Implements various scheduling strategies for model training, including
cosine annealing, step decay, warm-up, and hyperparameter grid search.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LR Schedules
# ---------------------------------------------------------------------------
class LRScheduler:
    """Learning rate schedule for iterative training.

    Supports multiple scheduling strategies that return a learning rate
    for a given epoch / step.

    Example
    -------
    >>> scheduler = LRScheduler("cosine", base_lr=0.1, total_steps=100)
    >>> for step in range(100):
    ...     lr = scheduler.get_lr(step)
    """

    def __init__(
        self,
        strategy: str = "cosine",
        base_lr: float = 0.01,
        min_lr: float = 1e-6,
        total_steps: int = 100,
        warmup_steps: int = 0,
        step_size: int = 30,
        gamma: float = 0.1,
    ):
        self.strategy = strategy
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.step_size = step_size
        self.gamma = gamma
        self._history: List[float] = []

        valid = ["constant", "step", "exponential", "cosine", "linear", "warmup_cosine"]
        if strategy not in valid:
            raise ValueError(f"Unknown strategy '{strategy}'. Choose from {valid}")

        logger.info("LR scheduler: %s, base_lr=%.6f, total_steps=%d",
                     strategy, base_lr, total_steps)

    def get_lr(self, step: int) -> float:
        """Compute the learning rate for a given step."""
        # Warm-up phase
        if step < self.warmup_steps and self.warmup_steps > 0:
            return self.base_lr * (step + 1) / self.warmup_steps

        lr = self.base_lr

        if self.strategy == "constant":
            pass  # lr stays at base_lr

        elif self.strategy == "step":
            lr = self.base_lr * (self.gamma ** (step // self.step_size))

        elif self.strategy == "exponential":
            lr = self.base_lr * (self.gamma ** (step / self.step_size))

        elif self.strategy == "cosine":
            progress = min(step / self.total_steps, 1.0)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
                1 + np.cos(np.pi * progress)
            )

        elif self.strategy == "linear":
            progress = min(step / self.total_steps, 1.0)
            lr = self.base_lr * (1 - progress) + self.min_lr * progress

        elif self.strategy == "warmup_cosine":
            progress = min(step / self.total_steps, 1.0)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
                1 + np.cos(np.pi * progress)
            )

        lr = max(lr, self.min_lr)
        self._history.append(lr)
        return lr

    def get_schedule(self) -> List[float]:
        """Return the full schedule (all LR values for every step)."""
        return [self.get_lr(s) for s in range(self.total_steps)]

    @staticmethod
    def compare_strategies(
        base_lr: float = 0.01,
        total_steps: int = 100,
        strategies: Optional[List[str]] = None,
    ) -> Dict[str, List[float]]:
        """Compare multiple LR schedules side by side."""
        if strategies is None:
            strategies = ["constant", "step", "cosine", "exponential", "linear"]

        results = {}
        for name in strategies:
            scheduler = LRScheduler(name, base_lr=base_lr, total_steps=total_steps)
            results[name] = scheduler.get_schedule()
        return results


# ---------------------------------------------------------------------------
# Hyperparameter search
# ---------------------------------------------------------------------------
class HyperparameterGrid:
    """Simple grid search over hyperparameter combinations.

    Example
    -------
    >>> grid = HyperparameterGrid({
    ...     "n_estimators": [100, 200],
    ...     "max_depth": [5, 10],
    ...     "learning_rate": [0.01, 0.1],
    ... })
    >>> for params in grid:
    ...     print(params)
    """

    def __init__(self, param_grid: Dict[str, List]):
        self.param_grid = param_grid
        self._combinations = self._build_combinations()

    def _build_combinations(self) -> List[Dict[str, Any]]:
        """Build all combinations of parameter values."""
        import itertools
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        combinations = []
        for combo in itertools.product(*values):
            combinations.append(dict(zip(keys, combo)))
        return combinations

    def __iter__(self):
        return iter(self._combinations)

    def __len__(self):
        return len(self._combinations)

    def search(
        self,
        model_class,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        metric: str = "rmse",
        verbose: bool = True,
    ) -> Tuple[Dict[str, Any], float]:
        """Run grid search and return best params + score.

        Parameters
        ----------
        model_class : class
            Sklearn-compatible model class.
        metric : str
            ``"rmse"``, ``"mae"``, or ``"r2"``.
        """
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        metric_fn = {"rmse": mean_squared_error, "mae": mean_absolute_error, "r2": r2_score}
        best_score = np.inf if metric != "r2" else -np.inf
        best_params = None
        results = []

        for i, params in enumerate(self._combinations):
            model = model_class(**params)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)

            if metric == "rmse":
                score = np.sqrt(mean_squared_error(y_val, y_pred))
            elif metric == "mae":
                score = mean_absolute_error(y_val, y_pred)
            else:
                score = r2_score(y_val, y_pred)

            results.append({"params": params, metric: score})

            is_better = (score < best_score) if metric != "r2" else (score > best_score)
            if is_better:
                best_score = score
                best_params = params

            if verbose and (i % 5 == 0 or is_better):
                logger.info("[%d/%d] %s=%.4f | %s", i + 1, len(self), metric, score, params)

        logger.info("Grid search complete. Best %s=%.4f | Params=%s", metric, best_score, best_params)
        return best_params, best_score


# ---------------------------------------------------------------------------
# Recommended HP grids for common models
# ---------------------------------------------------------------------------
GRIDS = {
    "random_forest": {
        "n_estimators": [100, 200, 300],
        "max_depth": [8, 10, 15, None],
        "min_samples_split": [2, 5, 10],
    },
    "gradient_boosting": {
        "n_estimators": [100, 150, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.8, 1.0],
    },
    "ridge": {
        "alpha": [0.01, 0.1, 1.0, 10.0],
    },
    "svr": {
        "C": [1, 10, 50],
        "epsilon": [0.01, 0.1, 0.2],
        "kernel": ["rbf", "linear"],
    },
}


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Compare LR schedules
    schedules = LRScheduler.compare_strategies(base_lr=0.01, total_steps=100)

    print("=" * 60)
    print("LR SCHEDULE COMPARISON")
    print("=" * 60)
    for name, values in schedules.items():
        print(f"  {name:20s}: start={values[0]:.6f}, end={values[-1]:.6f}")

    # Hyperparameter grid demo
    print(f"\n{'=' * 60}")
    print("HYPERPARAMETER GRID DEMO")
    print("=" * 60)
    grid = HyperparameterGrid({"n_estimators": [100, 200], "max_depth": [5, 10]})
    print(f"Grid size: {len(grid)} combinations")
    for params in grid:
        print(f"  {params}")
