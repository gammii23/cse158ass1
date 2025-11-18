"""Feature engineering for user-item interactions.

This module provides shared user/book feature builders including
frequency statistics, popularity metrics, collaborative statistics,
and matrix factorization inputs.
"""

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from constants import InteractionColumns
from utils.logging import get_logger
from utils.random_seed import get_random_state

logger = get_logger("features.interactions")


class InteractionFeatureBuilder:
    """Builder for interaction-based features with fit/transform API."""

    def __init__(self, random_seed: Optional[int] = None) -> None:
        """Initialize feature builder.

        Args:
            random_seed: Random seed for reproducibility.
        """
        self.random_seed = random_seed
        self.random_state = get_random_state(random_seed)
        self.global_mean: float = 0.0
        self.user_means: Dict[str, float] = {}
        self.item_means: Dict[str, float] = {}
        self.user_counts: Dict[str, int] = {}
        self.item_counts: Dict[str, int] = {}
        self.user_to_idx: Dict[str, int] = {}
        self.item_to_idx: Dict[str, int] = {}
        self.idx_to_user: Dict[int, str] = {}
        self.idx_to_item: Dict[int, str] = {}
        self._is_fitted = False

    def fit(self, df: pd.DataFrame) -> "InteractionFeatureBuilder":
        """Fit feature builder on training data.

        Args:
            df: DataFrame with user_id, item_id, rating columns.

        Returns:
            Self for method chaining.
        """
        logger.info("Fitting interaction feature builder")
        user_id_col = InteractionColumns.USER_ID
        item_id_col = InteractionColumns.ITEM_ID
        rating_col = InteractionColumns.RATING

        # Global statistics
        self.global_mean = df[rating_col].mean()

        # User statistics
        user_stats = df.groupby(user_id_col)[rating_col].agg(["mean", "count"])
        self.user_means = user_stats["mean"].to_dict()
        self.user_counts = user_stats["count"].to_dict()

        # Item statistics
        item_stats = df.groupby(item_id_col)[rating_col].agg(["mean", "count"])
        self.item_means = item_stats["mean"].to_dict()
        self.item_counts = item_stats["count"].to_dict()

        # Create ID mappings for matrix factorization
        unique_users = df[user_id_col].unique()
        unique_items = df[item_id_col].unique()
        self.user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
        self.item_to_idx = {item: idx for idx, item in enumerate(unique_items)}
        self.idx_to_user = {idx: user for user, idx in self.user_to_idx.items()}
        self.idx_to_item = {idx: item for item, idx in self.item_to_idx.items()}

        self._is_fitted = True
        logger.info(
            f"Fitted on {len(unique_users)} users and {len(unique_items)} items"
        )
        return self

    def transform_user_features(
        self, user_ids: pd.Series
    ) -> pd.DataFrame:
        """Transform user IDs to feature vectors.

        Args:
            user_ids: Series of user IDs.

        Returns:
            DataFrame with user features: mean_rating, interaction_count.
        """
        if not self._is_fitted:
            raise ValueError("Must call fit() before transform")

        features = pd.DataFrame(index=user_ids.index)
        features["user_mean_rating"] = user_ids.map(
            self.user_means
        ).fillna(self.global_mean)
        features["user_interaction_count"] = user_ids.map(
            self.user_counts
        ).fillna(0)
        features["user_bias"] = (
            features["user_mean_rating"] - self.global_mean
        )

        return features

    def transform_item_features(
        self, item_ids: pd.Series
    ) -> pd.DataFrame:
        """Transform item IDs to feature vectors.

        Args:
            item_ids: Series of item IDs.

        Returns:
            DataFrame with item features: mean_rating, interaction_count.
        """
        if not self._is_fitted:
            raise ValueError("Must call fit() before transform")

        features = pd.DataFrame(index=item_ids.index)
        features["item_mean_rating"] = item_ids.map(
            self.item_means
        ).fillna(self.global_mean)
        features["item_interaction_count"] = item_ids.map(
            self.item_counts
        ).fillna(0)
        features["item_bias"] = (
            features["item_mean_rating"] - self.global_mean
        )
        features["item_popularity"] = (
            features["item_interaction_count"] / max(self.item_counts.values())
            if self.item_counts
            else 0.0
        )

        return features

    def build_interaction_features(
        self, df: pd.DataFrame
    ) -> pd.DataFrame:
        """Build combined features for user-item pairs.

        Args:
            df: DataFrame with user_id and item_id columns.

        Returns:
            DataFrame with combined interaction features.
        """
        if not self._is_fitted:
            raise ValueError("Must call fit() before building features")

        user_features = self.transform_user_features(
            df[InteractionColumns.USER_ID]
        )
        item_features = self.transform_item_features(
            df[InteractionColumns.ITEM_ID]
        )

        # Combine features
        features = pd.concat([user_features, item_features], axis=1)
        features["global_mean"] = self.global_mean

        return features

    def build_sparse_matrix(
        self, df: pd.DataFrame
    ) -> Tuple[csr_matrix, Dict[str, int], Dict[str, int]]:
        """Build sparse user-item matrix for matrix factorization.

        Args:
            df: DataFrame with user_id, item_id, rating columns.

        Returns:
            Tuple of (sparse matrix, user_to_idx mapping, item_to_idx mapping).
        """
        if not self._is_fitted:
            raise ValueError("Must call fit() before building matrix")

        user_ids = df[InteractionColumns.USER_ID].map(self.user_to_idx)
        item_ids = df[InteractionColumns.ITEM_ID].map(self.item_to_idx)
        ratings = df[InteractionColumns.RATING].values

        n_users = len(self.user_to_idx)
        n_items = len(self.item_to_idx)

        matrix = csr_matrix(
            (ratings, (user_ids, item_ids)), shape=(n_users, n_items)
        )

        return matrix, self.user_to_idx, self.item_to_idx


