"""
Gemma 4 - Good AI for Humanity, Not Just Performance
=====================================================
src/data/loader.py
Data Loading Utilities

Handles loading of CSV, JSON, TXT files from the data/ directory.
Supports train/test split, label merging, and data validation.
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict, List, Union
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
EXTERNAL_DATA_DIR = PROJECT_ROOT / "data" / "external"


# ---------------------------------------------------------------------------
# Core loader
# ---------------------------------------------------------------------------
class DataLoader:
    """Centralised data loader for the project.

    Example
    -------
    >>> loader = DataLoader()
    >>> train, test, labels = loader.load_all()
    >>> print(train.shape)
    """

    # Score weights for composite target
    HUMANITY_WEIGHT: float = 0.35
    PERFORMANCE_WEIGHT: float = 0.25
    HELPFULNESS_WEIGHT: float = 0.40

    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = Path(data_dir) if data_dir else RAW_DATA_DIR
        self._train: Optional[pd.DataFrame] = None
        self._test: Optional[pd.DataFrame] = None
        self._labels: Optional[pd.DataFrame] = None
        self._prompts: Optional[dict] = None
        self._responses: Optional[dict] = None
        self._merged: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def load_all(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load train, test, and labels CSVs in one call."""
        self._train = self._load_csv("train.csv")
        self._test = self._load_csv("test.csv")
        self._labels = self._load_csv("labels.csv")
        self._compute_composite_score(self._train)
        self._compute_composite_score(self._test)
        logger.info("All CSV files loaded — train=%d, test=%d, labels=%d",
                     len(self._train), len(self._test), len(self._labels))
        return self._train, self._test, self._labels

    def load_json(self, filename: str) -> dict:
        """Load a JSON file from the data directory."""
        path = self.data_dir / filename
        if not path.exists():
            logger.warning("JSON file not found: %s", path)
            return {}
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info("Loaded JSON: %s (%d top-level keys)", filename, len(data))
        return data

    def load_corpus(self) -> str:
        """Load the corpus.txt file."""
        path = self.data_dir / "corpus.txt"
        if not path.exists():
            logger.warning("Corpus not found: %s", path)
            return ""
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        logger.info("Loaded corpus: %d characters", len(text))
        return text

    def merge_labels(self, df: pd.DataFrame,
                     label_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """Merge annotation labels into a dataframe on 'id'.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain an 'id' column.
        label_cols : list[str], optional
            Columns to pull from labels.csv.  Defaults to all score columns.

        Returns
        -------
        pd.DataFrame with label columns attached.
        """
        if self._labels is None:
            self._labels = self._load_csv("labels.csv")

        if label_cols is None:
            label_cols = ["safety_score", "fairness_score", "clarity_score", "annotator_id"]

        cols_to_merge = ["id"] + [c for c in label_cols if c in self._labels.columns]
        merged = df.merge(self._labels[cols_to_merge], on="id", how="left")
        logger.info("Merged %d label columns into DataFrame", len(cols_to_merge) - 1)
        return merged

    def get_merged_dataset(self) -> pd.DataFrame:
        """Return train DataFrame merged with labels and composite score."""
        if self._merged is None:
            if self._train is None:
                self.load_all()
            self._merged = self.merge_labels(self._train)
            self._compute_composite_score(self._merged)
        return self._merged

    def load_sample_data(self) -> pd.DataFrame:
        """Load the sample_data.csv for quick testing."""
        return self._load_csv("sample_data.csv")

    def load_prompts(self) -> dict:
        """Load prompts.json."""
        if self._prompts is None:
            self._prompts = self.load_json("prompts.json")
        return self._prompts

    def load_responses(self) -> dict:
        """Load responses.json."""
        if self._responses is None:
            self._responses = self.load_json("responses.json")
        return self._responses

    def filter_by_category(self, df: pd.DataFrame,
                           categories: Union[str, List[str]]) -> pd.DataFrame:
        """Filter DataFrame by one or more categories."""
        if isinstance(categories, str):
            categories = [categories]
        mask = df["category"].isin(categories)
        return df[mask].copy()

    def stratified_split(self, df: pd.DataFrame,
                         target_col: str = "composite_score",
                         test_size: float = 0.2,
                         random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create a stratified train/validation split based on quantile bins."""
        n_bins = min(5, df[target_col].nunique())
        df["_strat_bin"] = pd.qcut(df[target_col], q=n_bins, labels=False, duplicates="drop")
        train = df.groupby("_strat_bin", group_keys=False).sample(
            frac=1 - test_size, random_state=random_state
        )
        val = df.drop(train.index)
        df.drop(columns="_strat_bin", inplace=True)
        train = train.drop(columns="_strat_bin")
        val = val.drop(columns="_strat_bin")
        logger.info("Stratified split: train=%d, val=%d", len(train), len(val))
        return train, val

    def summary(self) -> Dict[str, int]:
        """Return a summary dict of loaded data."""
        return {
            "train_rows": len(self._train) if self._train is not None else 0,
            "test_rows": len(self._test) if self._test is not None else 0,
            "label_rows": len(self._labels) if self._labels is not None else 0,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _load_csv(self, filename: str) -> pd.DataFrame:
        path = self.data_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")
        df = pd.read_csv(path)
        logger.info("Loaded %s: %d rows × %d columns", filename, df.shape[0], df.shape[1])
        return df

    def _compute_composite_score(self, df: pd.DataFrame) -> None:
        """Add composite_score column in-place (weighted average)."""
        required = ["humanity_score", "performance_score", "helpfulness_score"]
        if all(c in df.columns for c in required):
            df["composite_score"] = (
                df["humanity_score"] * self.HUMANITY_WEIGHT
                + df["performance_score"] * self.PERFORMANCE_WEIGHT
                + df["helpfulness_score"] * self.HELPFULNESS_WEIGHT
            )
            logger.info("Computed composite_score for %d rows", len(df))


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    loader = DataLoader()
    train, test, labels = loader.load_all()
    prompts = loader.load_prompts()
    responses = loader.load_responses()
    corpus = loader.load_corpus()

    print(f"\n{'='*50}")
    print("DATA LOADER SUMMARY")
    print(f"{'='*50}")
    for k, v in loader.summary().items():
        print(f"  {k}: {v}")
    print(f"  prompts.json keys: {len(prompts)}")
    print(f"  responses.json keys: {len(responses)}")
    print(f"  corpus.txt chars: {len(corpus)}")

    # Quick preview
    print(f"\nTrain columns: {list(train.columns)}")
    print(train.head(2).to_string(index=False))
