"""Unit tests for review text features."""

import numpy as np
import pytest

from features.reviews import TfidfFeaturizer, clean_text


def test_clean_text():
    """Test text cleaning normalizes whitespace and lowercases."""
    assert clean_text("  Hello   World  ") == "hello world"
    assert clean_text("HELLO") == "hello"
    assert clean_text("") == ""
    assert clean_text("test\n\n\nmultiple\nlines") == "test multiple lines"


def test_tfidf_fit_transform():
    """Test TF-IDF vectorizer builds vocabulary correctly."""
    texts = [
        "This is a test document",
        "Another test document",
        "Yet another document",
    ]
    featurizer = TfidfFeaturizer(max_features=100, random_seed=42)
    featurizer.fit(texts)

    features = featurizer.transform(texts)
    assert features.shape[0] == len(texts)
    assert features.shape[1] > 0


def test_tfidf_deterministic():
    """Test TF-IDF produces deterministic features."""
    texts = ["test document", "another document"]
    featurizer1 = TfidfFeaturizer(random_seed=42)
    featurizer1.fit(texts)
    features1 = featurizer1.transform(texts)

    featurizer2 = TfidfFeaturizer(random_seed=42)
    featurizer2.fit(texts)
    features2 = featurizer2.transform(texts)

    np.testing.assert_array_equal(features1, features2)


def test_embedding_featurizer():
    """Test embedding featurizer loads model if available."""
    try:
        from features.reviews import EmbeddingFeaturizer

        featurizer = EmbeddingFeaturizer()
        featurizer.fit(["test"])
        # If model loads, should be able to transform
        if featurizer._is_fitted and featurizer.model is not None:
            features = featurizer.transform(["test document"])
            assert features.shape[0] == 1
            assert features.shape[1] > 0
    except ImportError:
        pytest.skip("sentence-transformers not available")


