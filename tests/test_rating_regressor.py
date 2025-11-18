"""Unit tests for rating regressor model."""

import numpy as np
import pandas as pd
import pytest

from models.rating_regressor import RatingRegressor


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
def sample_ratings():
    """Create sample ratings."""
    return np.array([5, 3, 4])


@pytest.fixture
def sample_user_ids():
    """Create sample user IDs."""
    return pd.Series(["u1", "u2", "u3"])


@pytest.fixture
def sample_item_ids():
    """Create sample item IDs."""
    return pd.Series(["i1", "i2", "i3"])


def test_fit_predict(sample_features, sample_ratings):
    """Test fit and predict return ratings in [1, 5]."""
    regressor = RatingRegressor(random_seed=42)
    regressor.fit(sample_features, sample_ratings)

    predictions = regressor.predict(sample_features)
    assert len(predictions) == len(sample_features)
    assert np.all(predictions >= 1.0) and np.all(predictions <= 5.0)


def test_bias_terms(
    sample_features, sample_ratings, sample_user_ids, sample_item_ids
):
    """Test user/item biases are computed."""
    regressor = RatingRegressor(use_bias=True, random_seed=42)
    regressor.fit(
        sample_features,
        sample_ratings,
        user_ids=sample_user_ids,
        item_ids=sample_item_ids,
    )

    assert regressor.global_mean > 0
    assert len(regressor.user_biases) > 0
    assert len(regressor.item_biases) > 0


def test_clipping(sample_features, sample_ratings):
    """Test predictions are clipped to valid range."""
    regressor = RatingRegressor(random_seed=42)
    regressor.fit(sample_features, sample_ratings)

    predictions = regressor.predict(sample_features)
    # Even if model predicts outside range, should be clipped
    assert np.all(predictions >= 1.0) and np.all(predictions <= 5.0)


