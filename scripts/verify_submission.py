"""Verify submission file formats against baseline outputs.

This script validates that prediction files match the expected
format, headers, column order, and data types.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

from constants import FileNames
from utils.logging import configure_logging, get_logger

logger = get_logger("scripts.verify_submission")


def verify_file_format(
    prediction_file: Path,
    pairs_file: Path,
    task: str,
    baseline_file: Optional[Path] = None,
) -> bool:
    """Verify a prediction file format.

    Args:
        prediction_file: Path to prediction CSV.
        pairs_file: Path to pairs CSV.
        task: Task name ('read', 'category', 'rating').
        baseline_file: Optional path to baseline output for comparison.

    Returns:
        True if format is valid, False otherwise.
    """
    logger.info(f"Verifying {prediction_file} for {task} task")

    # Check file exists
    if not prediction_file.exists():
        logger.error(f"Prediction file not found: {prediction_file}")
        return False

    # Load prediction file
    try:
        pred_df = pd.read_csv(prediction_file)
    except Exception as e:
        logger.error(f"Failed to load prediction file: {e}")
        return False

    # Load pairs file
    try:
        pairs_df = pd.read_csv(pairs_file)
    except Exception as e:
        logger.error(f"Failed to load pairs file: {e}")
        return False

    # Verify row count matches pairs
    if len(pred_df) != len(pairs_df):
        logger.error(
            f"Row count mismatch: predictions={len(pred_df)}, pairs={len(pairs_df)}"
        )
        return False

    # Verify required columns based on task
    if task == "read":
        required_cols = ["user_id", "item_id", "prediction"]
        # Also accept userID/itemID format
        if "userID" in pred_df.columns:
            required_cols = ["userID", "itemID", "prediction"]
    elif task == "category":
        required_cols = ["userID", "reviewID", "prediction"]
    elif task == "rating":
        required_cols = ["user_id", "item_id", "prediction"]
        if "userID" in pred_df.columns:
            required_cols = ["userID", "itemID", "prediction"]
    else:
        logger.error(f"Unknown task: {task}")
        return False

    missing_cols = set(required_cols) - set(pred_df.columns)
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        return False

    # Verify no NaN values
    if pred_df[required_cols].isnull().any().any():
        logger.error("Found NaN values in required columns")
        return False

    # Verify prediction value ranges
    pred_col = "prediction"
    if task == "read":
        if not pred_df[pred_col].between(0, 1, inclusive="both").all():
            logger.warning(
                "Read predictions outside [0,1] range (will be clipped)"
            )
    elif task == "category":
        if not pred_df[pred_col].between(0, 4, inclusive="both").all():
            logger.error("Category predictions must be integers 0-4")
            return False
        if not pred_df[pred_col].dtype in ["int64", "int32"]:
            logger.warning("Category predictions should be integers")
    elif task == "rating":
        if not pred_df[pred_col].between(1, 5, inclusive="both").all():
            logger.warning(
                "Rating predictions outside [1,5] range (will be clipped)"
            )

    # Compare with baseline if provided
    if baseline_file and baseline_file.exists():
        try:
            baseline_df = pd.read_csv(baseline_file)
            # Compare headers
            if list(pred_df.columns) != list(baseline_df.columns):
                logger.warning(
                    f"Column order differs from baseline: {list(pred_df.columns)} vs {list(baseline_df.columns)}"
                )
            # Compare first few rows
            if len(pred_df) > 0 and len(baseline_df) > 0:
                logger.info(
                    f"First row comparison:\nPred: {pred_df.iloc[0].to_dict()}\nBaseline: {baseline_df.iloc[0].to_dict()}"
                )
        except Exception as e:
            logger.warning(f"Could not compare with baseline: {e}")

    logger.info(f"Format verification passed for {prediction_file}")
    return True


def main() -> int:
    """Main entry point for submission verification.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    parser = argparse.ArgumentParser(
        description="Verify submission file formats"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory containing prediction files",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="assignment1",
        help="Directory containing pairs files",
    )
    parser.add_argument(
        "--baseline-dir",
        type=str,
        default=None,
        help="Directory containing baseline outputs (optional)",
    )

    args = parser.parse_args()

    configure_logging()

    output_dir = Path(args.output_dir)
    data_dir = Path(args.data_dir)
    baseline_dir = Path(args.baseline_dir) if args.baseline_dir else None

    tasks = [
        ("read", FileNames.PREDICTIONS_READ, FileNames.PAIRS_READ),
        (
            "category",
            FileNames.PREDICTIONS_CATEGORY,
            FileNames.PAIRS_CATEGORY,
        ),
        ("rating", FileNames.PREDICTIONS_RATING, FileNames.PAIRS_RATING),
    ]

    all_valid = True
    for task, pred_file, pairs_file in tasks:
        pred_path = output_dir / pred_file
        pairs_path = data_dir / pairs_file
        baseline_path = (
            baseline_dir / pred_file if baseline_dir else None
        )

        is_valid = verify_file_format(
            pred_path, pairs_path, task, baseline_path
        )
        if not is_valid:
            all_valid = False

    if all_valid:
        logger.info("All submission files verified successfully")
        return 0
    else:
        logger.error("Some submission files failed verification")
        return 1


if __name__ == "__main__":
    sys.exit(main())

