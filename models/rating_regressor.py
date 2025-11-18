"""Rating regression model with bias terms and gradient boosting.

This module combines baseline bias terms with gradient boosted trees
or factorization models for rating prediction.
"""

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge

from constants import InteractionColumns
from utils.logging import get_logger
from utils.random_seed import get_random_state

logger = get_logger("models.rating_regressor")


class RatingRegressor:
    """Regressor for rating prediction task."""

    def __init__(
        self,
        use_bias: bool = True,
        use_xgboost: bool = True,
        use_ridge: bool = False,
        xgb_n_estimators: int = 100,
        xgb_max_depth: int = 6,
        xgb_learning_rate: float = 0.1,
        ridge_alpha: float = 1.0,
        random_seed: Optional[int] = None,
    ) -> None:
        """Initialize rating regressor.

        Args:
            use_bias: Whether to use user/item bias terms.
            use_xgboost: Whether to use XGBoost (fallback to sklearn GBR).
            use_ridge: Whether to use Ridge regression.
            xgb_n_estimators: Number of boosting rounds.
            xgb_max_depth: Maximum tree depth.
            xgb_learning_rate: Learning rate.
            ridge_alpha: Ridge regularization strength.
            random_seed: Random seed for reproducibility.
        """
        self.use_bias = use_bias
        self.use_xgboost = use_xgboost
        self.use_ridge = use_ridge
        self.xgb_n_estimators = xgb_n_estimators
        self.xgb_max_depth = xgb_max_depth
        self.xgb_learning_rate = xgb_learning_rate
        self.ridge_alpha = ridge_alpha
        self.random_seed = random_seed
        self.random_state = get_random_state(random_seed)

        self.global_mean: float = 0.0
        self.user_biases: dict[str, float] = {}
        self.item_biases: dict[str, float] = {}
        self.model: Optional[GradientBoostingRegressor | Ridge] = None
        self._is_fitted = False

    def fit(
        self,
        X_features: pd.DataFrame,
        y: np.ndarray,
        user_ids: Optional[pd.Series] = None,
        item_ids: Optional[pd.Series] = None,
    ) -> "RatingRegressor":
        """Fit rating regressor on training data.

        Args:
            X_features: Feature matrix (user/item features).
            y: Rating targets (1-5 scale).
            user_ids: Optional user IDs for bias computation.
            item_ids: Optional item IDs for bias computation.

        Returns:
            Self for method chaining.
        """
        logger.info("Fitting rating regressor")

        self.global_mean = float(np.mean(y))

        # Compute biases if enabled
        if self.use_bias and user_ids is not None and item_ids is not None:
            df = pd.DataFrame(
                {
                    InteractionColumns.USER_ID: user_ids,
                    InteractionColumns.ITEM_ID: item_ids,
                    InteractionColumns.RATING: y,
                }
            )

            # User biases
            user_means = (
                df.groupby(InteractionColumns.USER_ID)[
                    InteractionColumns.RATING
                ].mean()
                - self.global_mean
            )
            self.user_biases = user_means.to_dict()

            # Item biases
            item_means = (
                df.groupby(InteractionColumns.ITEM_ID)[
                    InteractionColumns.RATING
                ].mean()
                - self.global_mean
            )
            self.item_biases = item_means.to_dict()

            logger.info(
                f"Computed biases: {len(self.user_biases)} users, "
                f"{len(self.item_biases)} items"
            )

        # Fit model
        if self.use_xgboost:
            try:
                import xgboost as xgb

                self.model = xgb.XGBRegressor(
                    n_estimators=self.xgb_n_estimators,
                    max_depth=self.xgb_max_depth,
                    learning_rate=self.xgb_learning_rate,
                    random_state=self.random_state,
                )
                logger.info("Using XGBoost regressor")
            except ImportError:
                logger.warning("XGBoost not available, using sklearn GBR")
                self.model = GradientBoostingRegressor(
                    n_estimators=self.xgb_n_estimators,
                    max_depth=self.xgb_max_depth,
                    learning_rate=self.xgb_learning_rate,
                    random_state=self.random_state,
                )
        elif self.use_ridge:
            self.model = Ridge(alpha=self.ridge_alpha, random_state=self.random_state)
            logger.info("Using Ridge regressor")
        else:
            self.model = GradientBoostingRegressor(
                n_estimators=self.xgb_n_estimators,
                max_depth=self.xgb_max_depth,
                learning_rate=self.xgb_learning_rate,
                random_state=self.random_state,
            )
            logger.info("Using sklearn GradientBoostingRegressor")

        self.model.fit(X_features, y)
        self._is_fitted = True
        logger.info("Rating regressor fitted successfully")
        return self

    def predict(
        self,
        X_features: pd.DataFrame,
        user_ids: Optional[pd.Series] = None,
        item_ids: Optional[pd.Series] = None,
    ) -> np.ndarray:
        """Predict ratings.

        Args:
            X_features: Feature matrix for prediction.
            user_ids: Optional user IDs for bias adjustment.
            item_ids: Optional item IDs for bias adjustment.

        Returns:
            Predicted ratings (clipped to [1, 5] range).
        """
        if not self._is_fitted or self.model is None:
            raise ValueError("Must call fit() before predict")

        predictions = self.model.predict(X_features)

        # Add biases if enabled
        if self.use_bias and user_ids is not None and item_ids is not None:
            user_bias = user_ids.map(self.user_biases).fillna(0.0).values
            item_bias = item_ids.map(self.item_biases).fillna(0.0).values
            predictions = predictions + user_bias + item_bias

        # Clip to valid rating range
        predictions = np.clip(predictions, 1.0, 5.0)

        return predictions


