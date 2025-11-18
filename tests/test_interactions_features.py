"""Unit tests for interaction features."""

import numpy as np
import pandas as pd
import pytest

from features.interactions import InteractionFeatureBuilder


@pytest.fixture
def sample_interactions():
    """Create sample interactions DataFrame."""
    return pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2", "u2", "u3"],
            "item_id": ["i1", "i2", "i1", "i3", "i2"],
            "rating": [5, 4, 3, 5, 4],
        }
    )


def test_feature_builder_fit(sample_interactions):
    """Test feature builder computes statistics correctly."""
    builder = InteractionFeatureBuilder(random_seed=42)
    builder.fit(sample_interactions)

    assert builder.global_mean > 0
    assert len(builder.user_means) > 0
    assert len(builder.item_means) > 0
    assert len(builder.user_to_idx) > 0
    assert len(builder.item_to_idx) > 0


def test_feature_builder_deterministic(sample_interactions):
    """Test feature builder produces deterministic outputs."""
    builder1 = InteractionFeatureBuilder(random_seed=42)
    builder1.fit(sample_interactions)

    builder2 = InteractionFeatureBuilder(random_seed=42)
    builder2.fit(sample_interactions)

    assert builder1.global_mean == builder2.global_mean
    assert builder1.user_means == builder2.user_means
    assert builder1.item_means == builder2.item_means


def test_user_features(sample_interactions):
    """Test user feature transformation."""
    builder = InteractionFeatureBuilder(random_seed=42)
    builder.fit(sample_interactions)

    user_ids = pd.Series(["u1", "u2"])
    features = builder.transform_user_features(user_ids)

    assert "user_mean_rating" in features.columns
    assert "user_interaction_count" in features.columns
    assert "user_bias" in features.columns
    assert len(features) == 2


def test_item_features(sample_interactions):
    """Test item feature transformation."""
    builder = InteractionFeatureBuilder(random_seed=42)
    builder.fit(sample_interactions)

    item_ids = pd.Series(["i1", "i2"])
    features = builder.transform_item_features(item_ids)

    assert "item_mean_rating" in features.columns
    assert "item_interaction_count" in features.columns
    assert "item_bias" in features.columns
    assert "item_popularity" in features.columns


def test_sparse_matrix_shape(sample_interactions):
    """Test sparse matrix dimensions."""
    builder = InteractionFeatureBuilder(random_seed=42)
    builder.fit(sample_interactions)

    matrix, user_map, item_map = builder.build_sparse_matrix(
        sample_interactions
    )

    assert matrix.shape[0] == len(user_map)
    assert matrix.shape[1] == len(item_map)


