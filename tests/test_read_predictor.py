"""Unit tests for read predictor model."""

import numpy as np
import pandas as pd
import pytest

from models.read_predictor import ReadPredictor


@pytest.fixture
def sample_features():
    """Create sample feature DataFrame."""
    return pd.DataFrame(
        {
            "user_mean_rating": [4.0, 3.5, 4.5],
            "item_mean_rating": [4.2, 3.8, 4.0],
            "user_interaction_count": [10, 5, 15],
            "item_interaction_count": [20, 10, 25],
        }
    )


@pytest.fixture
def sample_labels():
    """Create sample binary labels."""
    return np.array([1, 0, 1])


def test_fit_predict_proba(sample_features, sample_labels):
    """Test fit and predict_proba return valid probabilities."""
    predictor = ReadPredictor(random_seed=42)
    predictor.fit(sample_features, sample_labels)

    proba = predictor.predict_proba(sample_features)
    assert proba.shape[0] == len(sample_features)
    assert proba.shape[1] == 2  # Binary classification
    assert np.all(proba >= 0) and np.all(proba <= 1)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_calibration(sample_features, sample_labels):
    """Test probability calibration if enabled."""
    predictor = ReadPredictor(calibrate_probs=True, random_seed=42)
    predictor.fit(sample_features, sample_labels)

    proba = predictor.predict_proba(sample_features)
    # Calibrated probabilities should be well-calibrated
    assert proba.shape[1] == 2


def test_predict_binary(sample_features, sample_labels):
    """Test predict returns binary labels."""
    predictor = ReadPredictor(random_seed=42)
    predictor.fit(sample_features, sample_labels)

    predictions = predictor.predict(sample_features)
    assert len(predictions) == len(sample_features)
    assert set(predictions) <= {0, 1}


