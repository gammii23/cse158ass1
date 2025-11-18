"""Cross-validation utilities for stratified, time-aware, and random splits.

This module provides splitter classes for different validation strategies
to ensure proper train/validation/test separation.
"""

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    TimeSeriesSplit,
)

from utils.logging import get_logger
from utils.random_seed import get_random_state

logger = get_logger("utils.validation")


class StratifiedKFoldSplitter:
    """Stratified K-fold splitter for balanced class distribution."""

    def __init__(
        self, n_splits: int = 5, shuffle: bool = True, random_seed: Optional[int] = None
    ) -> None:
        """Initialize stratified K-fold splitter.

        Args:
            n_splits: Number of folds.
            shuffle: Whether to shuffle before splitting.
            random_seed: Random seed for reproducibility.
        """
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_seed = random_seed
        self.random_state = get_random_state(random_seed)

    def split(
        self, X: pd.DataFrame, y: np.ndarray
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Generate stratified train/validation splits.

        Args:
            X: Feature matrix (not used, kept for interface).
            y: Class labels for stratification.

        Returns:
            List of (train_indices, val_indices) tuples.
        """
        skf = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=self.shuffle,
            random_state=self.random_state,
        )
        splits = list(skf.split(X, y))
        logger.info(f"Generated {self.n_splits} stratified splits")
        return splits


class TimeAwareSplitter:
    """Time-aware splitter for temporal data."""

    def __init__(self, n_splits: int = 5) -> None:
        """Initialize time-aware splitter.

        Args:
            n_splits: Number of splits.
        """
        self.n_splits = n_splits

    def split(
        self, timestamps: pd.Series
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Generate time-aware train/validation splits.

        Args:
            timestamps: Series of timestamps (sorted).

        Returns:
            List of (train_indices, val_indices) tuples.
        """
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        sorted_indices = timestamps.sort_values().index.values
        splits = list(tscv.split(sorted_indices))
        logger.info(f"Generated {self.n_splits} time-aware splits")
        return splits


class RandomSplitter:
    """Random splitter with seed control."""

    def __init__(
        self,
        n_splits: int = 5,
        test_size: float = 0.2,
        shuffle: bool = True,
        random_seed: Optional[int] = None,
    ) -> None:
        """Initialize random splitter.

        Args:
            n_splits: Number of splits (for K-fold).
            test_size: Test set size fraction (for single split).
            shuffle: Whether to shuffle before splitting.
            random_seed: Random seed for reproducibility.
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.shuffle = shuffle
        self.random_seed = random_seed
        self.random_state = get_random_state(random_seed)

    def split(
        self, X: pd.DataFrame, y: Optional[np.ndarray] = None
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Generate random train/validation splits.

        Args:
            X: Feature matrix.
            y: Optional labels (not used, kept for interface).

        Returns:
            List of (train_indices, val_indices) tuples.
        """
        kf = KFold(
            n_splits=self.n_splits,
            shuffle=self.shuffle,
            random_state=self.random_state,
        )
        splits = list(kf.split(X))
        logger.info(f"Generated {self.n_splits} random splits")
        return splits


def create_splits(
    X: pd.DataFrame,
    y: Optional[np.ndarray] = None,
    splitter_type: str = "random",
    n_splits: int = 5,
    timestamps: Optional[pd.Series] = None,
    random_seed: Optional[int] = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create train/validation splits based on splitter type.

    Args:
        X: Feature matrix.
        y: Optional labels for stratification.
        splitter_type: Type of splitter ('stratified', 'time_aware', 'random').
        n_splits: Number of splits.
        timestamps: Optional timestamps for time-aware splitting.
        random_seed: Random seed for reproducibility.

    Returns:
        List of (train_indices, val_indices) tuples.
    """
    if splitter_type == "stratified" and y is not None:
        splitter = StratifiedKFoldSplitter(
            n_splits=n_splits, random_seed=random_seed
        )
        return splitter.split(X, y)
    elif splitter_type == "time_aware" and timestamps is not None:
        splitter = TimeAwareSplitter(n_splits=n_splits)
        return splitter.split(timestamps)
    else:
        splitter = RandomSplitter(n_splits=n_splits, random_seed=random_seed)
        return splitter.split(X, y)


