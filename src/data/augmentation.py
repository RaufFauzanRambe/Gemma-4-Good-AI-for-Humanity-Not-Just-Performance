"""
Gemma 4 - Good AI for Humanity, Not Just Performance
=====================================================
src/data/augmentation.py
Data Augmentation Techniques

Provides augmentation strategies for text data focused on preserving
semantic meaning and humanity-aligned quality labels.
"""

import re
import random
import pandas as pd
import numpy as np
from typing import List, Optional, Tuple, Dict, Callable
from collections import Counter
import logging

logger = logging.getLogger(__name__)

# Try to import NLTK (optional dependency)
try:
    import nltk
    from nltk.corpus import wordnet
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logger.info("NLTK not available — synonym augmentation will use fallback.")


# ---------------------------------------------------------------------------
# Synonym dictionary (fallback when NLTK is unavailable)
# ---------------------------------------------------------------------------
FALLBACK_SYNONYMS = {
    "help": ["assist", "support", "aid"],
    "improve": ["enhance", "boost", "elevate"],
    "important": ["crucial", "essential", "vital", "critical"],
    "use": ["utilise", "employ", "leverage"],
    "create": ["develop", "build", "establish"],
    "provide": ["offer", "deliver", "supply"],
    "reduce": ["decrease", "lower", "minimise"],
    "increase": ["boost", "enhance", "raise"],
    "show": ["demonstrate", "illustrate", "reveal"],
    "good": ["beneficial", "positive", "valuable"],
    "make": ["create", "build", "develop"],
    "need": ["require", "demand"],
    "people": ["individuals", "persons", "communities"],
    "children": ["young people", "youth"],
    "community": ["communities", "society", "population"],
    "problem": ["challenge", "issue", "concern"],
    "method": ["approach", "strategy", "technique"],
    "result": ["outcome", "effect", "impact"],
}


# ---------------------------------------------------------------------------
# Augmenter class
# ---------------------------------------------------------------------------
class TextAugmenter:
    """Text augmentation with multiple strategies.

    Strategies
    ----------
    - **synonym_replacement**: Replace words with synonyms (NLTK or fallback).
    - **random_deletion**: Randomly remove words with probability *p*.
    - **random_swap**: Swap two adjacent words.
    - **back_translation_simulation**: Paraphrase by shuffling clause order.
    - **prompt_paraphrase**: Rephrase prompts while preserving intent.
    """

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)

    # ------------------------------------------------------------------
    # Public augmentation methods
    # ------------------------------------------------------------------
    def synonym_replacement(
        self,
        text: str,
        n_replacements: int = 2,
    ) -> str:
        """Replace *n_replacements* random words with synonyms."""
        words = text.split()
        if len(words) <= 1:
            return text

        replaceable = [i for i, w in enumerate(words) if len(w) > 3 and w.isalpha()]
        self.rng.shuffle(replaceable)
        replaced = 0

        for idx in replaceable:
            if replaced >= n_replacements:
                break
            syn = self._get_synonym(words[idx])
            if syn and syn.lower() != words[idx].lower():
                words[idx] = syn
                replaced += 1

        return " ".join(words)

    def random_deletion(self, text: str, p: float = 0.1) -> str:
        """Randomly delete each word with probability *p*."""
        if p <= 0:
            return text
        words = text.split()
        if len(words) <= 3:
            return text

        remaining = [w for w in words if self.rng.random() > p]
        return " ".join(remaining) if remaining else words[-1]

    def random_swap(self, text: str, n_swaps: int = 1) -> str:
        """Randomly swap *n_swaps* pairs of adjacent words."""
        words = text.split()
        if len(words) < 2:
            return text

        for _ in range(min(n_swaps, len(words) // 2)):
            i = self.rng.randint(0, len(words) - 2)
            words[i], words[i + 1] = words[i + 1], words[i]

        return " ".join(words)

    def shuffle_sentences(self, text: str) -> str:
        """Shuffle the order of sentences (preserves intra-sentence structure)."""
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        if len(sentences) <= 1:
            return text
        self.rng.shuffle(sentences)
        return " ".join(sentences)

    def back_translation_simulation(self, text: str) -> str:
        """Simulate back-translation by applying synonym + swap."""
        text = self.synonym_replacement(text, n_replacements=3)
        text = self.random_swap(text, n_swaps=1)
        return text

    def prompt_paraphrase(self, prompt: str) -> str:
        """Paraphrase a question-style prompt while preserving its intent.

        Uses template transformations that are safe for the domain.
        """
        templates = [
            "How can we address the following: {base}?",
            "What are the best approaches to: {base}?",
            "Could you explain strategies for: {base}?",
            "In what ways might we tackle: {base}?",
            "Please discuss methods related to: {base}",
        ]

        # Remove trailing question mark for template injection
        base = re.sub(r"\?$", "", prompt.strip()).strip()
        if base.startswith("How") or base.startswith("What") or base.startswith("Explain"):
            base = re.sub(r"^(How|What|Explain|Describe|Discuss|Write)\s+", "", base)
            # Capitalise
            base = base[0].upper() + base[1:] if base else prompt

        paraphrase = self.rng.choice(templates).format(base=base)
        return paraphrase

    # ------------------------------------------------------------------
    # DataFrame-level augmentation
    # ------------------------------------------------------------------
    def augment_dataframe(
        self,
        df: pd.DataFrame,
        text_col: str = "response",
        prompt_col: str = "prompt",
        strategies: Optional[List[str]] = None,
        copies: int = 1,
        score_noise: float = 0.0,
        seed: Optional[int] = None,
    ) -> pd.DataFrame:
        """Create augmented copies of a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
        text_col : str
            Column containing the text to augment.
        prompt_col : str
            Column containing the prompt (for prompt paraphrasing).
        strategies : list[str], optional
            Which augmentation strategies to apply.  One per augmented copy.
            Defaults to ``["synonym_replacement", "random_deletion", "random_swap"]``.
        copies : int
            Number of augmented copies per original row.
        score_noise : float
            Gaussian noise std to add to numeric scores (humanity, performance, …).
        seed : int, optional
            Override seed for reproducibility.

        Returns
        -------
        pd.DataFrame concatenating originals + augmented rows.
        """
        if strategies is None:
            strategies = ["synonym_replacement", "random_deletion", "random_swap"]

        augmented_rows = []
        strategy_map: Dict[str, Callable] = {
            "synonym_replacement": lambda t: self.synonym_replacement(t, n_replacements=2),
            "random_deletion": lambda t: self.random_deletion(t, p=0.1),
            "random_swap": lambda t: self.random_swap(t, n_swaps=1),
            "shuffle_sentences": self.shuffle_sentences,
            "back_translation_simulation": self.back_translation_simulation,
        }

        score_cols = ["humanity_score", "performance_score", "helpfulness_score",
                       "safety_score", "fairness_score", "clarity_score", "composite_score"]

        for _, row in df.iterrows():
            for i in range(min(copies, len(strategies))):
                strategy_name = strategies[i % len(strategies)]
                aug_func = strategy_map.get(strategy_name)
                if aug_func is None:
                    continue

                new_row = row.copy()
                original_text = str(row[text_col])

                # Apply augmentation
                try:
                    new_row[text_col] = aug_func(original_text)
                except Exception:
                    new_row[text_col] = original_text

                # Optionally paraphrase prompt
                if prompt_col in row.index and prompt_col in strategy_map:
                    new_row[prompt_col] = self.prompt_paraphrase(str(row[prompt_col]))

                # Add noise to scores
                if score_noise > 0:
                    for sc in score_cols:
                        if sc in new_row.index and pd.notna(new_row[sc]):
                            noise = self.np_rng.normal(0, score_noise)
                            new_row[sc] = np.clip(new_row[sc] + noise, 1.0, 5.0)

                new_row["augmentation_strategy"] = strategy_name
                augmented_rows.append(new_row)

        aug_df = pd.DataFrame(augmented_rows)
        result = pd.concat([df, aug_df], ignore_index=True)
        logger.info("Augmented %d rows → %d rows (strategies: %s)",
                     len(df), len(result), strategies)
        return result

    def balance_categories(
        self,
        df: pd.DataFrame,
        category_col: str = "category",
        target_col: str = "response",
        min_samples: int = 10,
    ) -> pd.DataFrame:
        """Oversample under-represented categories using augmentation.

        Categories with fewer than *min_samples* rows are oversampled
        up to *min_samples* using synonym replacement.
        """
        counts = df[category_col].value_counts()
        balanced_frames = [df]

        for cat, count in counts.items():
            if count < min_samples:
                deficit = min_samples - count
                subset = df[df[category_col] == cat]
                oversample = subset.sample(
                    n=deficit, replace=True, random_state=42
                ).copy()
                oversample[target_col] = oversample[target_col].apply(
                    lambda x: self.synonym_replacement(str(x), n_replacements=2)
                )
                oversample["augmentation_strategy"] = "category_balance"
                balanced_frames.append(oversample)
                logger.info("Oversampled '%s': %d → %d (+%d)", cat, count, min_samples, deficit)

        return pd.concat(balanced_frames, ignore_index=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _get_synonym(self, word: str) -> Optional[str]:
        """Get a synonym for a word. Falls back to built-in dict."""
        if NLTK_AVAILABLE:
            try:
                synsets = wordnet.synsets(word)
                if synsets:
                    candidates = []
                    for syn in synsets:
                        for lemma in syn.lemmas():
                            candidate = lemma.name().replace("_", " ")
                            if candidate.lower() != word.lower() and len(candidate.split()) == 1:
                                candidates.append(candidate)
                    if candidates:
                        return self.rng.choice(candidates)
            except Exception:
                pass

        # Fallback dictionary
        return self.rng.choice(FALLBACK_SYNONYMS[word.lower()]) if word.lower() in FALLBACK_SYNONYMS else None


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    augmenter = TextAugmenter(seed=42)

    # Demo texts
    demos = [
        "AI can help improve access to clean water in rural communities through predictive analytics.",
        "How can technology support elderly people living alone?",
        "Machine learning models can detect early signs of disease from medical images.",
    ]

    print("=" * 60)
    print("DATA AUGMENTATION DEMO")
    print("=" * 60)

    for text in demos:
        print(f"\nOriginal:        {text}")
        print(f"Synonym:         {augmenter.synonym_replacement(text, 2)}")
        print(f"Deletion (p=0.1):{augmenter.random_deletion(text, 0.1)}")
        print(f"Swap:            {augmenter.random_swap(text, 1)}")
        print(f"Shuffle sent:    {augmenter.shuffle_sentences(text)}")
        print(f"Prompt para:     {augmenter.prompt_paraphrase(text)}")
