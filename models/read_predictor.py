"""Read predictor using implicit matrix factorization and popularity priors."""

from typing import Any, Optional

import numpy as np
import pandas as pd
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import bm25_weight
from scipy.special import expit

from utils.logging import get_logger
logger = get_logger("models.read_predictor")


class ReadPredictor:
    """Predictor for read prediction task."""

    def __init__(
        self,
        use_implicit: bool = True,
        n_factors: int = 50,
        n_iterations: int = 15,
        regularization: float = 0.1,
        popularity_weight: float = 0.35,
        random_seed: Optional[int] = None,
    ) -> None:
        """Initialize read predictor.

        Args:
            use_implicit: Whether to use implicit matrix factorization.
            n_factors: Number of latent factors for ALS.
            n_iterations: Number of ALS training iterations.
            regularization: Regularization strength for ALS.
            popularity_weight: Mixing weight for popularity prior.
            random_seed: Random seed for reproducibility.
        """

        self.use_implicit = use_implicit
        self.n_factors = n_factors
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.popularity_weight = popularity_weight
        self.random_seed = random_seed
        self.implicit_model: Optional[AlternatingLeastSquares] = None
        self.user_to_idx: dict[str, int] = {}
        self.item_to_idx: dict[str, int] = {}
        self.item_popularity: pd.Series | None = None
        self._is_fitted: bool = False

    def fit(
        self,
        interaction_matrix: Any,
        user_to_idx: dict[str, int],
        item_to_idx: dict[str, int],
        item_popularity: pd.Series,
    ) -> "ReadPredictor":
        """Fit implicit matrix factorization model.

        Args:
            interaction_matrix: User-item sparse matrix of implicit feedback.
            user_to_idx: Mapping from user IDs to matrix row indices.
            item_to_idx: Mapping from item IDs to matrix column indices.
            item_popularity: Series mapping item IDs to popularity values.

        Returns:
            Self for method chaining.
        """

        logger.info("Fitting read predictor with implicit ALS")

        self.user_to_idx = user_to_idx
        self.item_to_idx = item_to_idx
        self.item_popularity = item_popularity

        if self.use_implicit:
            logger.info(
                "Applying BM25 weighting and training AlternatingLeastSquares"
            )
            weighted = bm25_weight(interaction_matrix).tocsr()
            self.implicit_model = AlternatingLeastSquares(
                factors=self.n_factors,
                iterations=self.n_iterations,
                regularization=self.regularization,
                random_state=self.random_seed,
            )
            self.implicit_model.fit(weighted)
        else:
            logger.warning(
                "Implicit modeling disabled; predictions will use popularity only"
            )

        self._is_fitted = True
        logger.info("Read predictor fitted successfully")
        return self

    def predict_proba_pairs(
        self,
        user_ids: pd.Series,
        item_ids: pd.Series,
    ) -> np.ndarray:
        """Predict read probabilities for user-item pairs.

        Args:
            user_ids: Series of user IDs.
            item_ids: Series of item IDs.

        Returns:
            Probability estimates in [0, 1].
        """

        if not self._is_fitted:
            raise ValueError("Must call fit() before predicting")

        popularity_scores = (
            item_ids.map(self.item_popularity).fillna(0.0).to_numpy()
            if self.item_popularity is not None
            else np.zeros(len(item_ids))
        )

        implicit_scores = np.zeros(len(user_ids))

        if self.implicit_model is not None and self.user_to_idx and self.item_to_idx:
            user_idx = user_ids.map(self.user_to_idx)
            item_idx = item_ids.map(self.item_to_idx)
            valid_mask = user_idx.notna() & item_idx.notna()
            if valid_mask.any():
                u_factors = self.implicit_model.user_factors[
                    user_idx[valid_mask].astype(int)
                ]
                i_factors = self.implicit_model.item_factors[
                    item_idx[valid_mask].astype(int)
                ]
                implicit_scores[valid_mask.to_numpy()] = np.sum(
                    u_factors * i_factors, axis=1
                )

        implicit_probs = expit(implicit_scores)

        if popularity_scores.max() > 0:
            popularity_norm = popularity_scores / popularity_scores.max()
        else:
            popularity_norm = popularity_scores

        blended = (
            (1 - self.popularity_weight) * implicit_probs
            + self.popularity_weight * popularity_norm
        )

        return np.clip(blended, 0.0, 1.0)

