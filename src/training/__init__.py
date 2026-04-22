"""
Gemma 4 - Good AI for Humanity, Not Just Performance
=====================================================
src/training/__init__.py
Training package init.
"""

from src.training.train import train
from src.training.trainer import Trainer, EarlyStopping, CheckpointManager
from src.training.scheduler import LRScheduler, HyperparameterGrid, GRIDS

__all__ = [
    "train",
    "Trainer", "EarlyStopping", "CheckpointManager",
    "LRScheduler", "HyperparameterGrid", "GRIDS",
]
