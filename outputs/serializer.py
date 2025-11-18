"""Output serialization with schema validation.

This module provides a single responsibility for writing CSV predictions
with schema validation to prevent duplication and ensure correctness.
"""

from pathlib import Path
from typing import Optional

import pandas as pd

from constants import FileNames, PredictionColumns
from utils.logging import get_logger

logger = get_logger("outputs.serializer")


def validate_schema(
    df: pd.DataFrame, task: str, required_columns: Optional[list[str]] = None
) -> bool:
    """Validate DataFrame schema for submission.

    Args:
        df: DataFrame to validate.
        task: Task name ('read', 'category', 'rating').
        required_columns: Optional list of required columns.

    Returns:
        True if schema is valid.

    Raises:
        ValueError: If schema validation fails.
    """
    if required_columns is None:
        if task == "read":
            required_columns = [
                PredictionColumns.USER_ID,
                PredictionColumns.ITEM_ID,
                "prediction",
            ]
        elif task == "category":
            required_columns = ["userID", "reviewID", "prediction"]
        elif task == "rating":
            required_columns = [
                PredictionColumns.USER_ID,
                PredictionColumns.ITEM_ID,
                "prediction",
            ]
        else:
            raise ValueError(f"Unknown task: {task}")

    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Missing required columns for {task}: {missing_cols}"
        )

    # Check for NaN values
    if df[required_columns].isnull().any().any():
        raise ValueError(f"Found NaN values in required columns for {task}")

    # Validate prediction ranges
    if task == "read":
        if not df["prediction"].between(0, 1, inclusive="both").all():
            logger.warning(
                "Read predictions outside [0,1] range (will be clipped)"
            )
    elif task == "category":
        if not df["prediction"].between(0, 4, inclusive="both").all():
            raise ValueError("Category predictions must be integers 0-4")
    elif task == "rating":
        if not df["prediction"].between(1, 5, inclusive="both").all():
            logger.warning(
                "Rating predictions outside [1,5] range (will be clipped)"
            )

    return True


def write_predictions(
    df: pd.DataFrame,
    task: str,
    output_dir: str = "outputs",
    overwrite: bool = True,
) -> Path:
    """Write predictions to CSV with schema validation.

    Args:
        df: DataFrame with predictions.
        task: Task name ('read', 'category', 'rating').
        output_dir: Output directory.
        overwrite: Whether to overwrite existing files.

    Returns:
        Path to written file.

    Raises:
        ValueError: If schema validation fails or file exists and overwrite=False.
    """
    # Map task to filename
    filename_map = {
        "read": FileNames.PREDICTIONS_READ,
        "category": FileNames.PREDICTIONS_CATEGORY,
        "rating": FileNames.PREDICTIONS_RATING,
    }

    if task not in filename_map:
        raise ValueError(f"Unknown task: {task}")

    filename = filename_map[task]
    output_path = Path(output_dir) / filename

    # Check if file exists
    if output_path.exists() and not overwrite:
        raise ValueError(f"File exists and overwrite=False: {output_path}")

    # Validate schema
    validate_schema(df, task)

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Write CSV with correct header format
    logger.info(f"Writing predictions to {output_path}")
    
    # Rename columns to match expected format
    if task == "read":
        df = df.rename(columns={
            "user_id": "userID",
            "userID": "userID",  # Already correct
            "item_id": "bookID",
            "itemID": "bookID",
            "bookID": "bookID",  # Already correct
        })
        # Convert probabilities to binary 0/1
        df["prediction"] = (df["prediction"] > 0.5).astype(int)
    elif task == "rating":
        df = df.rename(columns={
            "user_id": "userID",
            "userID": "userID",  # Already correct
            "item_id": "bookID",
            "itemID": "bookID",
            "bookID": "bookID",  # Already correct
        })
    elif task == "category":
        # Category already has correct headers (userID, reviewID, prediction)
        # Ensure prediction is integer
        df["prediction"] = df["prediction"].astype(int)
    
    # Select only required columns in correct order
    if task == "read":
        df = df[["userID", "bookID", "prediction"]]
    elif task == "category":
        df = df[["userID", "reviewID", "prediction"]]
    elif task == "rating":
        df = df[["userID", "bookID", "prediction"]]
    
    df.to_csv(output_path, index=False)

    logger.info(
        f"Successfully wrote {len(df)} predictions for {task} task"
    )
    return output_path


