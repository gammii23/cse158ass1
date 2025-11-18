"""Category classification model with dependency-injected featurizer.

This module wraps text classification models (logistic regression on TF-IDF
or lightweight transformers) with support for featurizer substitution
for testing (Liskov substitution principle).
"""

from typing import Optional, Protocol

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from utils.logging import get_logger
from utils.random_seed import get_random_state

logger = get_logger("models.category_classifier")


class TextFeaturizer(Protocol):
    """Protocol for text featurizers."""

    def fit(self, texts: list[str]) -> "TextFeaturizer":
        """Fit featurizer."""
        ...

    def transform(self, texts: list[str]) -> np.ndarray:
        """Transform texts to features."""
        ...


class CategoryClassifier:
    """Multi-class classifier for genre prediction."""

    def __init__(
        self,
        featurizer: TextFeaturizer,
        C: float = 1.0,
        class_weight: Optional[str] = "balanced",
        use_svm: bool = False,
        random_seed: Optional[int] = None,
    ) -> None:
        """Initialize category classifier.

        Args:
            featurizer: Text featurizer instance (dependency injection).
            C: Regularization strength.
            class_weight: Class weight strategy.
            use_svm: Whether to use SVM instead of logistic regression.
            random_seed: Random seed for reproducibility.
        """
        self.featurizer = featurizer
        self.C = C
        self.class_weight = class_weight
        self.use_svm = use_svm
        self.random_seed = random_seed
        self.random_state = get_random_state(random_seed)
        self.model: Optional[LogisticRegression | LinearSVC] = None
        self._is_fitted = False

    def fit(
        self, X_text: list[str], y: np.ndarray
    ) -> "CategoryClassifier":
        """Fit classifier on training texts and labels.

        Args:
            X_text: List of review texts.
            y: Category labels (integers 0-4).

        Returns:
            Self for method chaining.
        """
        logger.info("Fitting category classifier")

        # Transform texts to features
        X_features = self.featurizer.transform(X_text)

        # Choose model
        if self.use_svm:
            self.model = LinearSVC(
                C=self.C,
                class_weight=self.class_weight,
                random_state=self.random_state,
                max_iter=10000,
            )
        else:
            self.model = LogisticRegression(
                C=self.C,
                class_weight=self.class_weight,
                random_state=self.random_state,
                max_iter=1000,
                multi_class="multinomial",
                solver="lbfgs",
            )

        self.model.fit(X_features, y)
        self._is_fitted = True
        logger.info("Category classifier fitted successfully")
        return self

    def predict(self, X_text: list[str]) -> np.ndarray:
        """Predict category labels.

        Args:
            X_text: List of review texts.

        Returns:
            Predicted category labels (integers 0-4).
        """
        if not self._is_fitted or self.model is None:
            raise ValueError("Must call fit() before predict")

        X_features = self.featurizer.transform(X_text)
        return self.model.predict(X_features)

    def predict_proba(self, X_text: list[str]) -> np.ndarray:
        """Predict category probabilities.

        Args:
            X_text: List of review texts.

        Returns:
            Probability matrix (n_samples, n_classes).
        """
        if not self._is_fitted or self.model is None:
            raise ValueError("Must call fit() before predict_proba")

        X_features = self.featurizer.transform(X_text)
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X_features)
        else:
            # SVM doesn't have predict_proba by default
            # Return one-hot encoding of predictions
            predictions = self.model.predict(X_features)
            n_classes = len(self.model.classes_)
            proba = np.zeros((len(predictions), n_classes))
            proba[np.arange(len(predictions)), predictions] = 1.0
            return proba


