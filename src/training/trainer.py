"""
Gemma 4 - Good AI for Humanity, Not Just Performance
=====================================================
src/training/trainer.py
Trainer Class — Managed Training Loop

Provides a high-level Trainer class with callbacks, early stopping,
checkpointing, and experiment logging for reproducible training runs.
"""

import time
import json
import logging
from pathlib import Path
from typing import Optional, Dict, List, Callable, Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Callback types
# ---------------------------------------------------------------------------
class EarlyStopping:
    """Stop training when a metric stops improving.

    Parameters
    ----------
    monitor : str
        Metric name to watch (e.g., ``"val_rmse"``).
    patience : int
        Number of epochs with no improvement before stopping.
    mode : str
        ``"min"`` (lower is better) or ``"max"`` (higher is better).
    min_delta : float
        Minimum change to qualify as an improvement.
    """

    def __init__(self, monitor: str = "val_rmse", patience: int = 5,
                 mode: str = "min", min_delta: float = 1e-4):
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best_score = np.inf if mode == "min" else -np.inf
        self.counter = 0
        self.best_epoch = 0
        self.stopped_epoch = 0

    def step(self, current_score: int, epoch: int) -> bool:
        """Return True if training should stop."""
        improved = False
        if self.mode == "min":
            improved = current_score < (self.best_score - self.min_delta)
        else:
            improved = current_score > (self.best_score + self.min_delta)

        if improved:
            self.best_score = current_score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.stopped_epoch = epoch
            logger.info("Early stopping at epoch %d (best=%d, %s=%.4f)",
                         epoch, self.best_epoch, self.monitor, self.best_score)
            return True
        return False


class CheckpointManager:
    """Save model checkpoints during training."""

    def __init__(self, save_dir: str, prefix: str = "ckpt", keep_best: int = 3):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.keep_best = keep_best
        self.history: List[Dict[str, Any]] = []

    def save(self, model: Any, epoch: int, metrics: Dict[str, float]):
        """Save a model checkpoint."""
        import pickle
        path = self.save_dir / f"{self.prefix}_epoch{epoch:03d}.pkl"
        with open(path, "wb") as f:
            pickle.dump({"model": model, "epoch": epoch, "metrics": metrics}, f)
        self.history.append({"epoch": epoch, "metrics": metrics, "path": str(path)})
        logger.info("Checkpoint saved: %s", path)

        # Keep only the best N checkpoints
        if len(self.history) > self.keep_best:
            worst = max(self.history, key=lambda x: x["metrics"].get("val_rmse", float("inf")))
            worst_path = Path(worst["path"])
            if worst_path.exists():
                worst_path.unlink()
            self.history.remove(worst)


# ---------------------------------------------------------------------------
# Trainer class
# ---------------------------------------------------------------------------
class Trainer:
    """Managed training loop with logging, callbacks, and experiment tracking.

    Example
    -------
    >>> trainer = Trainer(experiment_name="baseline_run")
    >>> trainer.fit(model, X_train, y_train, X_val, y_val, epochs=100)
    >>> trainer.plot_history()
    """

    def __init__(
        self,
        experiment_name: str = "experiment",
        log_dir: Optional[str] = None,
        early_stopping_patience: int = 10,
        checkpoint_dir: Optional[str] = None,
    ):
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir) if log_dir else Path("logs") / experiment_name
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.early_stopping = EarlyStopping(patience=early_stopping_patience)
        self.checkpoint = CheckpointManager(checkpoint_dir or str(self.log_dir / "checkpoints"))
        self.history: List[Dict[str, float]] = []
        self.best_model = None

    def fit(
        self,
        model,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: Optional[int] = None,
        callbacks: Optional[List[Callable]] = None,
        verbose: bool = True,
    ) -> "Trainer":
        """Run the training loop.

        For sklearn-style models (``model.fit(X, y)``), this is a single-epoch
        wrapper. For iterative models, increment epochs internally.
        """
        from sklearn.metrics import mean_squared_error

        start_time = time.time()

        for epoch in range(1, epochs + 1):
            epoch_start = time.time()

            # Fit model (sklearn-style: single fit)
            model.fit(X_train, y_train)

            # Compute metrics
            y_train_pred = model.predict(X_train)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))

            record = {"epoch": epoch, "train_rmse": round(train_rmse, 4)}

            if X_val is not None and y_val is not None:
                y_val_pred = model.predict(X_val)
                val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
                record["val_rmse"] = round(val_rmse, 4)

                # Early stopping check
                if self.early_stopping.step(val_rmse, epoch):
                    if verbose:
                        logger.info("Stopping at epoch %d", epoch)
                    break

                # Checkpoint
                if epoch % 10 == 0 or val_rmse == self.early_stopping.best_score:
                    self.checkpoint.save(model, epoch, record)

            record["elapsed_s"] = round(time.time() - epoch_start, 2)
            self.history.append(record)

            if verbose and (epoch % 10 == 0 or epoch == 1):
                val_str = f" | val_rmse={record.get('val_rmse', 'N/A')}" if "val_rmse" in record else ""
                logger.info("Epoch %d/%d | train_rmse=%.4f%s | %.2fs",
                             epoch, epochs, train_rmse, val_str, record["elapsed_s"])

        # Store best model
        self.best_model = model
        total_time = time.time() - start_time
        logger.info("Training complete: %d epochs in %.1fs", len(self.history), total_time)

        # Save history
        self._save_history()
        return self

    def get_history(self) -> pd.DataFrame:
        """Return training history as a DataFrame."""
        return pd.DataFrame(self.history)

    def plot_history(self, save_path: Optional[str] = None):
        """Plot training curves (requires matplotlib)."""
        import matplotlib.pyplot as plt

        df = self.get_history()
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # RMSE over epochs
        axes[0].plot(df["epoch"], df["train_rmse"], label="Train RMSE", color="#3498db")
        if "val_rmse" in df.columns:
            axes[0].plot(df["epoch"], df["val_rmse"], label="Val RMSE", color="#e74c3c")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("RMSE")
        axes[0].set_title("Training Curves")
        axes[0].legend(loc="best")

        # Elapsed time per epoch
        if "elapsed_s" in df.columns:
            axes[1].bar(df["epoch"], df["elapsed_s"], color="#2ecc71", alpha=0.7)
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("Time (s)")
            axes[1].set_title("Epoch Duration")

        plt.suptitle(f"Experiment: {self.experiment_name}", fontsize=14, fontweight="bold")
        plt.tight_layout()

        path = save_path or str(self.log_dir / "training_curves.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Training curves saved: %s", path)

    def _save_history(self):
        path = self.log_dir / "history.json"
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
        logger.info("History saved: %s", path)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from sklearn.ensemble import GradientBoostingRegressor

    np.random.seed(42)
    X = np.random.randn(200, 20)
    y = 3.5 + 0.5 * X[:, 0] - 0.3 * X[:, 1] + np.random.randn(200) * 0.3
    y = np.clip(y, 1.0, 5.0)
    X_train, X_val = X[:160], X[160:]
    y_train, y_val = y[:160], y[160:]

    model = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)

    trainer = Trainer(
        experiment_name="demo_run",
        log_dir="logs/demo",
        early_stopping_patience=5,
    )
    trainer.fit(model, X_train, y_train, X_val, y_val, epochs=50, verbose=True)

    print("\nHistory:")
    print(trainer.get_history().to_string(index=False))
