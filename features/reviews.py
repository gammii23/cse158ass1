"""Feature engineering for review text data.

This module provides text preprocessing utilities including tokenization,
TF-IDF vectorizers, and optional embedding loaders with dependency injection.
"""

import re
from typing import Iterable, Optional, Protocol

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from constants import CategoryColumns
from utils.logging import get_logger
from utils.random_seed import get_random_state

logger = get_logger("features.reviews")


class TextFeaturizer(Protocol):
    """Protocol for text featurizers with fit/transform interface."""

    def fit(self, texts: Iterable[str]) -> "TextFeaturizer":
        """Fit featurizer on training texts."""
        ...

    def transform(self, texts: Iterable[str]) -> np.ndarray:
        """Transform texts to feature vectors."""
        ...


def clean_text(text: str) -> str:
    """Clean and normalize text.

    Args:
        text: Raw text string.

    Returns:
        Cleaned text (lowercased, stripped, normalized whitespace).
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = text.strip()
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)
    return text


class TfidfFeaturizer:
    """TF-IDF vectorizer with dependency injection support."""

    def __init__(
        self,
        max_features: int = 200000,
        ngram_range: tuple[int, int] = (1, 2),
        min_df: int = 2,
        max_df: float = 0.95,
        random_seed: Optional[int] = None,
    ) -> None:
        """Initialize TF-IDF featurizer.

        Args:
            max_features: Maximum number of features.
            ngram_range: Range of n-grams to extract.
            min_df: Minimum document frequency.
            max_df: Maximum document frequency.
            random_seed: Random seed for reproducibility.
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.random_seed = random_seed
        self.random_state = get_random_state(random_seed)
        self.vectorizer: Optional[TfidfVectorizer] = None
        self._is_fitted = False

    def fit(self, texts: Iterable[str]) -> "TfidfFeaturizer":
        """Fit TF-IDF vectorizer on training texts.

        Args:
            texts: Iterable of text strings.

        Returns:
            Self for method chaining.
        """
        logger.info("Fitting TF-IDF vectorizer")
        cleaned_texts = [clean_text(text) for text in texts]

        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_df=self.max_df,
            lowercase=True,
            stop_words="english",
        )
        self.vectorizer.fit(cleaned_texts)
        self._is_fitted = True
        logger.info(
            f"Fitted TF-IDF with {len(self.vectorizer.vocabulary_)} features"
        )
        return self

    def transform(self, texts: Iterable[str]) -> np.ndarray:
        """Transform texts to TF-IDF feature vectors.

        Args:
            texts: Iterable of text strings.

        Returns:
            Dense feature matrix (n_samples, n_features).
        """
        if not self._is_fitted or self.vectorizer is None:
            raise ValueError("Must call fit() before transform")

        cleaned_texts = [clean_text(text) for text in texts]
        sparse_matrix = self.vectorizer.transform(cleaned_texts)
        # Convert to dense array (models expect dense)
        # For very large matrices, this might fail - consider reducing max_features
        return sparse_matrix.toarray()


class EmbeddingFeaturizer:
    """Sentence transformer featurizer (optional, behind common interface)."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        random_seed: Optional[int] = None,
    ) -> None:
        """Initialize embedding featurizer.

        Args:
            model_name: Name of SentenceTransformer model.
            random_seed: Random seed (not used but kept for interface consistency).
        """
        self.model_name = model_name
        self.random_seed = random_seed
        self.model: Optional[Any] = None
        self._is_fitted = False

    def fit(self, texts: Iterable[str]) -> "EmbeddingFeaturizer":
        """Fit embedding model (lazy load on first use).

        Args:
            texts: Iterable of text strings (not used, kept for interface).

        Returns:
            Self for method chaining.
        """
        logger.info(f"Loading SentenceTransformer model: {self.model_name}")
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(self.model_name)
            self._is_fitted = True
            logger.info("Model loaded successfully")
        except ImportError:
            logger.warning(
                "sentence-transformers not available, embeddings disabled"
            )
            self._is_fitted = False
        return self

    def transform(self, texts: Iterable[str]) -> np.ndarray:
        """Transform texts to embedding vectors.

        Args:
            texts: Iterable of text strings.

        Returns:
            Dense feature matrix (n_samples, embedding_dim).
        """
        if not self._is_fitted or self.model is None:
            raise ValueError("Must call fit() before transform")

        text_list = list(texts)
        cleaned_texts = [clean_text(text) for text in text_list]
        embeddings = self.model.encode(cleaned_texts, show_progress_bar=False)
        return np.array(embeddings)


def build_review_features(
    df: pd.DataFrame,
    featurizer: TextFeaturizer,
    text_column: str = CategoryColumns.REVIEW_TEXT,
) -> np.ndarray:
    """Build features from review texts using provided featurizer.

    Args:
        df: DataFrame containing review texts.
        featurizer: Fitted text featurizer instance.
        text_column: Name of column containing review text.

    Returns:
        Feature matrix (n_samples, n_features).
    """
    texts = df[text_column].fillna("")
    return featurizer.transform(texts)

