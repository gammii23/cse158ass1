"""Read prediction model with collaborative filtering and popularity prior.

This module encapsulates hybrid models (matrix factorization + popularity)
for predicting whether a user will read a book.
"""

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from constants import InteractionColumns
from utils.logging import get_logger
from utils.random_seed import get_random_state

logger = get_logger("models.read_predictor")


class ReadPredictor:
    """Predictor for read prediction task."""

    def __init__(
        self,
        use_implicit: bool = True,
        use_lightfm: bool = False,
        lr_C: float = 1.0,
        lr_class_weight: Optional[str] = "balanced",
        calibrate_probs: bool = True,
        random_seed: Optional[int] = None,
    ) -> None:
        """Initialize read predictor.

        Args:
            use_implicit: Whether to use implicit matrix factorization.
            use_lightfm: Whether to use LightFM (not implemented yet).
            lr_C: Regularization strength for logistic regression.
            lr_class_weight: Class weight strategy for logistic regression.
            calibrate_probs: Whether to calibrate probability outputs.
            random_seed: Random seed for reproducibility.
        """
        self.use_implicit = use_implicit
        self.use_lightfm = use_lightfm
        self.lr_C = lr_C
        self.lr_class_weight = lr_class_weight
        self.calibrate_probs = calibrate_probs
        self.random_seed = random_seed
        self.random_state = get_random_state(random_seed)
        self.model: Optional[Pipeline] = None
        self.implicit_model: Optional[Any] = None
        self.user_to_idx: Optional[dict] = None
        self.item_to_idx: Optional[dict] = None
        self._is_fitted = False

    def fit(
        self,
        X_features: pd.DataFrame,
        y: np.ndarray,
        user_ids: Optional[pd.Series] = None,
        item_ids: Optional[pd.Series] = None,
    ) -> "ReadPredictor":
        """Fit read predictor on training data.

        Args:
            X_features: Feature matrix (user/item features).
            y: Binary labels (1 = read, 0 = not read).
            user_ids: Optional user IDs for implicit MF.
            item_ids: Optional item IDs for implicit MF.

        Returns:
            Self for method chaining.
        """
        logger.info("Fitting read predictor")

        # Baseline: logistic regression on collaborative features
        base_model = LogisticRegression(
            C=self.lr_C,
            class_weight=self.lr_class_weight,
            random_state=self.random_state,
            max_iter=1000,
        )

        if self.calibrate_probs:
            self.model = CalibratedClassifierCV(
                base_model, method="isotonic", cv=3
            )
        else:
            self.model = base_model

        self.model.fit(X_features, y)

        # Optional: fit implicit MF model
        if self.use_implicit and user_ids is not None and item_ids is not None:
            try:
                import implicit

                logger.info("Fitting implicit matrix factorization model")
                # Build sparse matrix (simplified - would need full implementation)
                # For now, just store mappings
                self.user_to_idx = {
                    uid: idx for idx, uid in enumerate(user_ids.unique())
                }
                self.item_to_idx = {
                    iid: idx for idx, iid in enumerate(item_ids.unique())
                }
                # TODO: Implement full implicit MF integration
                logger.info("Implicit MF mappings stored (full integration TODO)")
            except ImportError:
                logger.warning("implicit library not available, skipping MF")

        self._is_fitted = True
        logger.info("Read predictor fitted successfully")
        return self

    def predict_proba(self, X_features: pd.DataFrame) -> np.ndarray:
        """Predict read probabilities.

        Args:
            X_features: Feature matrix for prediction.

        Returns:
            Array of shape (n_samples, 2) with probabilities for [0, 1].
        """
        if not self._is_fitted or self.model is None:
            raise ValueError("Must call fit() before predict_proba")

        proba = self.model.predict_proba(X_features)
        return proba

    def predict(self, X_features: pd.DataFrame) -> np.ndarray:
        """Predict binary read labels.

        Args:
            X_features: Feature matrix for prediction.

        Returns:
            Binary predictions (0 or 1).
        """
        proba = self.predict_proba(X_features)
        return (proba[:, 1] > 0.5).astype(int)


