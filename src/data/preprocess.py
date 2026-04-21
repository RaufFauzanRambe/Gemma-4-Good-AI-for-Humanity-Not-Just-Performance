"""
Gemma 4 - Good AI for Humanity, Not Just Performance
=====================================================
src/data/preprocess.py
Text Preprocessing Pipeline

Cleans, normalises, and tokenises text data for model training.
Includes domain-specific handling for healthcare, ethical, and social-good content.
"""

import re
import unicodedata
import pandas as pd
import numpy as np
from typing import Optional, List, Tuple, Dict
from collections import Counter
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Domain-specific stop-word list
# ---------------------------------------------------------------------------
ENGLISH_STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "shall", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "before", "after", "above", "below", "between", "and", "but", "or",
    "nor", "not", "so", "yet", "both", "either", "neither", "each",
    "every", "all", "any", "few", "more", "most", "other", "some",
    "such", "no", "only", "own", "same", "than", "too", "very",
    "just", "because", "if", "when", "where", "how", "what", "which",
    "who", "whom", "this", "that", "these", "those", "it", "its",
    "we", "they", "their", "them", "our", "us", "you", "your",
    "he", "she", "his", "her", "also", "about", "up", "out",
    "while", "like", "make", "way", "many", "much", "well",
    "need", "however", "therefore", "furthermore", "addition",
    "example", "include", "including",
}

# Extra domain stop-words for humanity/ethics context
DOMAIN_STOP_WORDS = {
    "ai", "artificial", "intelligence", "technology", "system",
    "use", "using", "used", "one", "two", "three",
    "first", "second", "third", "new", "based",
}

ALL_STOP_WORDS = ENGLISH_STOP_WORDS | DOMAIN_STOP_WORDS


# ---------------------------------------------------------------------------
# Contraction mapping
# ---------------------------------------------------------------------------
CONTRACTIONS = {
    "n't": " not", "'re": " are", "'ve": " have", "'ll": " will",
    "'d": " would", "'m": " am", "'s": " is",
}


# ---------------------------------------------------------------------------
# Medical term patterns for healthcare domain
# ---------------------------------------------------------------------------
MEDICAL_PATTERNS = re.compile(
    r"\b(diagnos|treatment|symptom|therapy|medication|clinical|patient|"
    r"disease|disorder|health|mental|psycholog|surgic|prescription|"
    r"vaccine|immun|patholog|rehab|prevent|screen|chronic|acute)\w*\b",
    re.IGNORECASE,
)

# Empathy keywords
EMPATHY_PATTERNS = re.compile(
    r"\b(understand|support|care|compassion|important|community|"
    r"together|hope|empath|vulnerable|inclusive|accessible|"
    r"affordable|equitable|empower|dignit|respect|wellbeing)\w*\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Preprocessor class
# ---------------------------------------------------------------------------
class TextPreprocessor:
    """Full-featured text preprocessor for the project.

    Steps (configurable):
        1. Unicode normalisation
        2. HTML/entity removal
        3. Contraction expansion
        4. Lower-casing
        5. Punctuation stripping (preserves sentence-ending)
        6. Stop-word removal
        7. Tokenisation
    """

    def __init__(
        self,
        remove_stopwords: bool = True,
        remove_numbers: bool = False,
        min_word_length: int = 2,
        lemmatize: bool = False,
        extra_stop_words: Optional[set] = None,
    ):
        self.remove_stopwords = remove_stopwords
        self.remove_numbers = remove_numbers
        self.min_word_length = min_word_length
        self.lemmatize = lemmatize
        self._stop_words = ALL_STOP_WORDS.copy()
        if extra_stop_words:
            self._stop_words.update(extra_stop_words)

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------
    def clean(self, text: str) -> str:
        """Run the full cleaning pipeline on a single string."""
        if pd.isna(text) or not text:
            return ""
        text = self._normalize_unicode(text)
        text = self._remove_html(text)
        text = self._expand_contractions(text)
        text = text.lower()
        if self.remove_numbers:
            text = re.sub(r"\d+", " ", text)
        text = self._clean_punctuation(text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def tokenize(self, text: str) -> List[str]:
        """Clean and return a list of tokens."""
        cleaned = self.clean(text)
        tokens = cleaned.split()
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self._stop_words]
        tokens = [t for t in tokens if len(t) >= self.min_word_length]
        return tokens

    def preprocess_dataframe(
        self,
        df: pd.DataFrame,
        text_columns: List[str],
        tokenized_columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Apply cleaning/tokenisation to multiple columns of a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
        text_columns : list[str]
            Columns to clean (stored as ``<col>_clean``).
        tokenized_columns : list[str], optional
            Subset of *text_columns* that also needs tokenisation
            (stored as ``<col>_tokens``).

        Returns
        -------
        pd.DataFrame with new ``_clean`` and ``_tokens`` columns added.
        """
        df = df.copy()
        for col in text_columns:
            df[f"{col}_clean"] = df[col].apply(self.clean)
            logger.info("Cleaned column '%s' → '%s_clean'", col, col)

        if tokenized_columns:
            for col in tokenized_columns:
                if col in text_columns:
                    df[f"{col}_tokens"] = df[col].apply(self.tokenize)
                    logger.info("Tokenized column '%s' → '%s_tokens'", col, col)
        return df

    def compute_text_features(self, df: pd.DataFrame,
                              response_col: str = "response",
                              prompt_col: str = "prompt") -> pd.DataFrame:
        """Compute structural / statistical text features.

        Adds the following columns:
            - prompt_length, response_length
            - prompt_word_count, response_word_count
            - length_ratio
            - sentence_count, avg_sentence_length
            - unique_word_count, lexical_diversity
            - has_list, has_number
            - question_marks, exclamation_marks
            - has_medical_terms (bool)
            - has_empathy_words (bool)
        """
        df = df.copy()

        # Lengths
        df["prompt_length"] = df[prompt_col].str.len().fillna(0).astype(int)
        df["response_length"] = df[response_col].str.len().fillna(0).astype(int)

        # Word counts
        df["prompt_word_count"] = df[prompt_col].str.split().str.len().fillna(0).astype(int)
        df["response_word_count"] = df[response_col].str.split().str.len().fillna(0).astype(int)
        df["length_ratio"] = (
            df["response_word_count"] / df["prompt_word_count"].replace(0, 1)
        )

        # Sentence info
        df["sentence_count"] = df[response_col].apply(
            lambda x: max(1, len(re.split(r"[.!?]+", str(x)))) if pd.notna(x) else 1
        )
        df["avg_sentence_length"] = (
            df["response_word_count"] / df["sentence_count"].replace(0, 1)
        )

        # Lexical diversity
        df["unique_word_count"] = df[response_col].apply(
            lambda x: len(set(str(x).lower().split())) if pd.notna(x) else 0
        )
        df["lexical_diversity"] = (
            df["unique_word_count"] / df["response_word_count"].replace(0, 1)
        )

        # Punctuation
        df["question_marks"] = df[response_col].str.count(r"\?").fillna(0).astype(int)
        df["exclamation_marks"] = df[response_col].str.count(r"!").fillna(0).astype(int)

        # Structural patterns
        df["has_list"] = df[response_col].apply(
            lambda x: int(bool(re.search(r"(\d\)|-\s|\*)", str(x)))) if response_col in df.columns else 0
        df["has_number"] = df[response_col].apply(
            lambda x: int(bool(re.search(r"\d", str(x)))) if pd.notna(x) else 0
        )
        df["paragraph_count"] = df[response_col].apply(
            lambda x: max(1, str(x).count("\n\n") + 1) if pd.notna(x) else 1
        )

        # Domain features
        df["has_medical_terms"] = df[response_col].apply(
            lambda x: int(bool(MEDICAL_PATTERNS.search(str(x)))) if pd.notna(x) else 0
        )
        df["has_empathy_words"] = df[response_col].apply(
            lambda x: int(bool(EMPATHY_PATTERNS.search(str(x)))) if pd.notna(x) else 0
        )

        logger.info("Computed %d text features", 16)
        return df

    @staticmethod
    def extract_keywords(texts: List[str],
                         top_n: int = 30,
                         stop_words: Optional[set] = None) -> List[Tuple[str, int]]:
        """Extract the most frequent keywords from a list of texts."""
        if stop_words is None:
            stop_words = ALL_STOP_WORDS
        words: List[str] = []
        for text in texts:
            tokens = re.findall(r"\b[a-zA-Z]{4,}\b", str(text).lower())
            words.extend([w for w in tokens if w not in stop_words])
        return Counter(words).most_common(top_n)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_unicode(text: str) -> str:
        return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")

    @staticmethod
    def _remove_html(text: str) -> str:
        return re.sub(r"<[^>]+>", " ", text)

    @staticmethod
    def _expand_contractions(text: str) -> str:
        for contraction, expansion in CONTRACTIONS.items():
            text = text.replace(contraction, expansion)
        return text

    @staticmethod
    def _clean_punctuation(text: str) -> str:
        # Keep sentence-ending punctuation, remove the rest
        text = re.sub(r"[^\w\s.!?']", " ", text)
        return text


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------
def preprocess_pipeline(
    df: pd.DataFrame,
    prompt_col: str = "prompt",
    response_col: str = "response",
    include_tokens: bool = True,
) -> pd.DataFrame:
    """One-call preprocessing: clean + features + tokens."""
    pp = TextPreprocessor()
    text_cols = [prompt_col, response_col]
    token_cols = text_cols if include_tokens else None
    df = pp.preprocess_dataframe(df, text_columns=text_cols, tokenized_columns=token_cols)
    df = pp.compute_text_features(df, response_col=response_col, prompt_col=prompt_col)
    return df


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

    from src.data.loader import DataLoader

    loader = DataLoader()
    train, _, _ = loader.load_all()

    df = preprocess_pipeline(train)
    print(f"\nPreprocessed DataFrame: {df.shape}")
    feature_cols = [c for c in df.columns if c not in train.columns]
    print(f"New columns added ({len(feature_cols)}):")
    for col in sorted(feature_cols):
        print(f"  {col}")
