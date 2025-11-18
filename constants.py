"""Constants and enumerations for assignment1.

This module centralizes all magic strings, file names, column names,
and genre labels to prevent duplication and ensure consistency.
"""

from enum import Enum


class FileNames:
    """Dataset and output file names."""

    # Training data
    TRAIN_INTERACTIONS = "train_Interactions.csv.gz"
    TRAIN_CATEGORY = "train_Category.json.gz"
    TEST_CATEGORY = "test_Category.json.gz"

    # Evaluation pairs
    PAIRS_READ = "pairs_Read.csv"
    PAIRS_CATEGORY = "pairs_Category.csv"
    PAIRS_RATING = "pairs_Rating.csv"

    # Output predictions
    PREDICTIONS_READ = "predictions_Read.csv"
    PREDICTIONS_CATEGORY = "predictions_Category.csv"
    PREDICTIONS_RATING = "predictions_Rating.csv"

    # Reference
    BASELINES = "baselines.py"
    WRITEUP = "writeup.txt"


class InteractionColumns:
    """Column names for interactions dataset."""

    USER_ID = "user_id"
    ITEM_ID = "item_id"
    RATING = "rating"
    TIMESTAMP = "timestamp"  # if present


class CategoryColumns:
    """Column names for category datasets."""

    USER_ID = "user_id"
    ITEM_ID = "item_id"
    REVIEW_TEXT = "review_text"
    CATEGORY = "category"  # in train only


class PairColumns:
    """Column names for pair files."""

    USER_ID = "user_id"
    ITEM_ID = "item_id"


class PredictionColumns:
    """Column names for prediction outputs."""

    USER_ID = "user_id"
    ITEM_ID = "item_id"
    PREDICTION = "prediction"


class GenreLabels(Enum):
    """Genre labels for category classification."""

    # Values will be determined from data exploration
    # Placeholder enum structure
    UNKNOWN = "unknown"

    @classmethod
    def from_data(cls, unique_labels: list[str]) -> list[str]:
        """Get genre labels from actual data.

        Args:
            unique_labels: List of unique category labels from dataset.

        Returns:
            Sorted list of genre labels.
        """
        return sorted(set(unique_labels))


class TaskType(Enum):
    """Task types for CLI routing."""

    READ = "read"
    CATEGORY = "category"
    RATING = "rating"


