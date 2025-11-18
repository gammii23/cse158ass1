"""Spot-check prediction files for correctness.

This script samples random rows from prediction files and
verifies value ranges, data types, and prints samples for manual inspection.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from constants import FileNames
from utils.logging import configure_logging, get_logger

logger = get_logger("scripts.spot_check")


def spot_check_file(
    prediction_file: Path, task: str, n_samples: int = 10
) -> bool:
    """Spot-check a prediction file.

    Args:
        prediction_file: Path to prediction CSV.
        task: Task name ('read', 'category', 'rating').
        n_samples: Number of rows to sample.

    Returns:
        True if checks pass, False otherwise.
    """
    logger.info(f"Spot-checking {prediction_file} for {task} task")

    if not prediction_file.exists():
        logger.error(f"File not found: {prediction_file}")
        return False

    try:
        df = pd.read_csv(prediction_file)
    except Exception as e:
        logger.error(f"Failed to load file: {e}")
        return False

    # Sample random rows
    n_samples = min(n_samples, len(df))
    sampled = df.sample(n=n_samples, random_state=42)

    logger.info(f"\n=== Spot Check Results for {task} ===")
    logger.info(f"Total rows: {len(df)}")
    logger.info(f"Sampled rows: {n_samples}\n")

    # Check for NaN
    if df.isnull().any().any():
        logger.warning("Found NaN values in file")
        nan_cols = df.columns[df.isnull().any()].tolist()
        logger.warning(f"Columns with NaN: {nan_cols}")
    else:
        logger.info("✓ No NaN values found")

    # Check value ranges
    pred_col = "prediction"
    if pred_col not in df.columns:
        logger.error(f"Missing 'prediction' column")
        return False

    if task == "read":
        valid_range = df[pred_col].between(0, 1, inclusive="both").all()
        if valid_range:
            logger.info("✓ All predictions in [0, 1] range")
        else:
            logger.warning(
                f"Found {(~df[pred_col].between(0, 1, inclusive='both')).sum()} predictions outside [0, 1]"
            )
        logger.info(
            f"Prediction stats: min={df[pred_col].min():.4f}, max={df[pred_col].max():.4f}, mean={df[pred_col].mean():.4f}"
        )

    elif task == "category":
        valid_range = df[pred_col].between(0, 4, inclusive="both").all()
        if valid_range:
            logger.info("✓ All predictions in [0, 4] range")
        else:
            logger.error(
                f"Found {(~df[pred_col].between(0, 4, inclusive='both')).sum()} predictions outside [0, 4]"
            )
            return False
        if df[pred_col].dtype in ["int64", "int32"]:
            logger.info("✓ Predictions are integers")
        else:
            logger.warning("Predictions are not integers")
        logger.info(
            f"Prediction distribution:\n{df[pred_col].value_counts().sort_index()}"
        )

    elif task == "rating":
        valid_range = df[pred_col].between(1, 5, inclusive="both").all()
        if valid_range:
            logger.info("✓ All predictions in [1, 5] range")
        else:
            logger.warning(
                f"Found {(~df[pred_col].between(1, 5, inclusive='both')).sum()} predictions outside [1, 5]"
            )
        logger.info(
            f"Prediction stats: min={df[pred_col].min():.4f}, max={df[pred_col].max():.4f}, mean={df[pred_col].mean():.4f}"
        )

    # Print sample rows
    logger.info(f"\n=== Sample Rows ({n_samples}) ===")
    for idx, row in sampled.iterrows():
        logger.info(f"Row {idx}: {row.to_dict()}")

    logger.info("\n=== Spot Check Complete ===\n")
    return True


def main() -> int:
    """Main entry point for spot checking.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    parser = argparse.ArgumentParser(
        description="Spot-check prediction files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory containing prediction files",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=10,
        help="Number of rows to sample",
    )

    args = parser.parse_args()

    configure_logging()

    output_dir = Path(args.output_dir)

    tasks = [
        ("read", FileNames.PREDICTIONS_READ),
        ("category", FileNames.PREDICTIONS_CATEGORY),
        ("rating", FileNames.PREDICTIONS_RATING),
    ]

    all_valid = True
    for task, pred_file in tasks:
        pred_path = output_dir / pred_file
        is_valid = spot_check_file(pred_path, task, args.n_samples)
        if not is_valid:
            all_valid = False

    if all_valid:
        logger.info("All spot checks passed")
        return 0
    else:
        logger.error("Some spot checks failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())


