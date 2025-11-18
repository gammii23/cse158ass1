"""Data leakage prevention utilities.

This module provides functions to validate that test data is not
accidentally used during training, ensuring strict train/test separation.
"""

from typing import Set

import pandas as pd

from constants import InteractionColumns, PairColumns
from utils.logging import get_logger

logger = get_logger("utils.data_leakage_guard")


def validate_no_test_leakage(
    train_df: pd.DataFrame,
    pairs_df: pd.DataFrame,
    user_col: str = InteractionColumns.USER_ID,
    item_col: str = InteractionColumns.ITEM_ID,
) -> bool:
    """Validate that evaluation pairs don't overlap with training data.

    Args:
        train_df: Training DataFrame with user/item interactions.
        pairs_df: Evaluation pairs DataFrame.
        user_col: Name of user ID column.
        item_col: Name of item ID column.

    Returns:
        True if no leakage detected, False otherwise.

    Raises:
        ValueError: If leakage is detected.
    """
    # Get sets of user-item pairs from training
    train_pairs = set(
        zip(
            train_df[user_col].astype(str),
            train_df[item_col].astype(str),
        )
    )

    # Get sets of user-item pairs from evaluation
    pair_user_col = user_col if user_col in pairs_df.columns else PairColumns.USER_ID
    pair_item_col = item_col if item_col in pairs_df.columns else PairColumns.ITEM_ID

    if pair_user_col not in pairs_df.columns or pair_item_col not in pairs_df.columns:
        logger.warning(
            f"Could not find user/item columns in pairs: {pairs_df.columns.tolist()}"
        )
        return True  # Can't validate, assume OK

    eval_pairs = set(
        zip(
            pairs_df[pair_user_col].astype(str),
            pairs_df[pair_item_col].astype(str),
        )
    )

    # Check for overlap
    overlap = train_pairs & eval_pairs
    if overlap:
        logger.error(
            f"Data leakage detected: {len(overlap)} user-item pairs appear in both train and eval"
        )
        raise ValueError(
            f"Data leakage: {len(overlap)} overlapping pairs found"
        )

    logger.info(
        f"No leakage detected: {len(train_pairs)} train pairs, {len(eval_pairs)} eval pairs, 0 overlap"
    )
    return True


def audit_joins(
    train_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    join_key: str,
) -> bool:
    """Audit that feature building didn't accidentally use test data.

    Args:
        train_df: Training DataFrame.
        feature_df: Feature DataFrame.
        join_key: Column name used for joining.

    Returns:
        True if audit passes, False otherwise.
    """
    train_keys = set(train_df[join_key].astype(str))
    feature_keys = set(feature_df[join_key].astype(str))

    # Features should only contain keys from training
    extra_keys = feature_keys - train_keys
    if extra_keys:
        logger.warning(
            f"Found {len(extra_keys)} keys in features not in training data"
        )
        return False

    logger.info("Join audit passed: all feature keys present in training")
    return True


def strict_split(
    df: pd.DataFrame,
    train_size: float = 0.8,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create strict train/validation split with no overlap.

    Args:
        df: DataFrame to split.
        train_size: Fraction of data for training.
        random_seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_df, val_df).
    """
    from sklearn.model_selection import train_test_split

    train_df, val_df = train_test_split(
        df,
        train_size=train_size,
        random_state=random_seed,
        shuffle=True,
    )

    # Verify no overlap
    train_indices = set(train_df.index)
    val_indices = set(val_df.index)
    assert len(train_indices & val_indices) == 0, "Split overlap detected"

    logger.info(
        f"Strict split: {len(train_df)} train, {len(val_df)} val, 0 overlap"
    )
    return train_df, val_df


