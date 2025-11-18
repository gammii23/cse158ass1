"""Unit tests for category classifier model."""

import numpy as np
import pytest

from features.reviews import TfidfFeaturizer
from models.category_classifier import CategoryClassifier


@pytest.fixture
def sample_texts():
    """Create sample review texts."""
    return [
        "This is a fantasy book",
        "A mystery thriller novel",
        "Children's story book",
        "Comic graphic novel",
        "Young adult romance",
    ]


@pytest.fixture
def sample_labels():
    """Create sample category labels."""
    return np.array([0, 1, 2, 3, 4])


def test_fit_predict(sample_texts, sample_labels):
    """Test fit and predict return valid category integers."""
    featurizer = TfidfFeaturizer(max_features=100, random_seed=42)
    featurizer.fit(sample_texts)

    classifier = CategoryClassifier(
        featurizer=featurizer, random_seed=42
    )
    classifier.fit(sample_texts, sample_labels)

    predictions = classifier.predict(sample_texts)
    assert len(predictions) == len(sample_texts)
    assert set(predictions) <= {0, 1, 2, 3, 4}


def test_di_featurizer(sample_texts, sample_labels):
    """Test dependency injection accepts different featurizers."""
    featurizer1 = TfidfFeaturizer(max_features=50, random_seed=42)
    featurizer1.fit(sample_texts)

    classifier1 = CategoryClassifier(
        featurizer=featurizer1, random_seed=42
    )
    classifier1.fit(sample_texts, sample_labels)

    # Should work with different featurizer
    featurizer2 = TfidfFeaturizer(max_features=100, random_seed=42)
    featurizer2.fit(sample_texts)

    classifier2 = CategoryClassifier(
        featurizer=featurizer2, random_seed=42
    )
    classifier2.fit(sample_texts, sample_labels)

    # Both should produce valid predictions
    pred1 = classifier1.predict(sample_texts)
    pred2 = classifier2.predict(sample_texts)
    assert len(pred1) == len(pred2)


def test_class_weights(sample_texts, sample_labels):
    """Test classifier handles imbalanced classes."""
    # Create imbalanced labels
    imbalanced_labels = np.array([0, 0, 0, 1, 2])

    featurizer = TfidfFeaturizer(max_features=100, random_seed=42)
    featurizer.fit(sample_texts)

    classifier = CategoryClassifier(
        featurizer=featurizer,
        class_weight="balanced",
        random_seed=42,
    )
    classifier.fit(sample_texts, imbalanced_labels)

    predictions = classifier.predict(sample_texts)
    assert len(predictions) == len(sample_texts)


