"""Validate submission artifacts before handoff.

This script performs comprehensive validation of all submission files:
file sizes, row counts, schemas, and sample correctness.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from constants import FileNames
from utils.logging import configure_logging, get_logger

logger = get_logger("scripts.validate_submission")


def validate_file_size(file_path: Path, min_size_kb: int = 1) -> bool:
    """Validate file size is reasonable.

    Args:
        file_path: Path to file.
        min_size_kb: Minimum size in KB.

    Returns:
        True if size is valid.
    """
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        return False

    size_kb = file_path.stat().st_size / 1024
    if size_kb < min_size_kb:
        logger.error(f"File too small: {size_kb:.2f} KB < {min_size_kb} KB")
        return False

    if size_kb > 100 * 1024:  # 100 MB
        logger.warning(f"File very large: {size_kb:.2f} KB")

    logger.info(f"File size OK: {size_kb:.2f} KB")
    return True


def validate_row_count(
    prediction_file: Path, pairs_file: Path
) -> bool:
    """Validate row counts match.

    Args:
        prediction_file: Path to prediction CSV.
        pairs_file: Path to pairs CSV.

    Returns:
        True if counts match.
    """
    try:
        pred_df = pd.read_csv(prediction_file)
        pairs_df = pd.read_csv(pairs_file)
    except Exception as e:
        logger.error(f"Failed to load files: {e}")
        return False

    if len(pred_df) != len(pairs_df):
        logger.error(
            f"Row count mismatch: predictions={len(pred_df)}, pairs={len(pairs_df)}"
        )
        return False

    logger.info(f"Row count matches: {len(pred_df)} rows")
    return True


def validate_schema(prediction_file: Path, task: str) -> bool:
    """Validate prediction file schema.

    Args:
        prediction_file: Path to prediction CSV.
        task: Task name.

    Returns:
        True if schema is valid.
    """
    try:
        df = pd.read_csv(prediction_file)
    except Exception as e:
        logger.error(f"Failed to load file: {e}")
        return False

    # Check required columns
    if task == "read":
        required = ["user_id", "item_id", "prediction"]
        if "userID" in df.columns:
            required = ["userID", "itemID", "prediction"]
    elif task == "category":
        required = ["userID", "reviewID", "prediction"]
    elif task == "rating":
        required = ["user_id", "item_id", "prediction"]
        if "userID" in df.columns:
            required = ["userID", "itemID", "prediction"]
    else:
        logger.error(f"Unknown task: {task}")
        return False

    missing = set(required) - set(df.columns)
    if missing:
        logger.error(f"Missing columns: {missing}")
        return False

    # Check no NaN
    if df[required].isnull().any().any():
        logger.error("Found NaN values in required columns")
        return False

    logger.info(f"Schema valid: {required}")
    return True


def spot_check_samples(
    prediction_file: Path, task: str, n_samples: int = 20
) -> bool:
    """Spot-check sample rows.

    Args:
        prediction_file: Path to prediction CSV.
        task: Task name.
        n_samples: Number of samples to check.

    Returns:
        True if samples are valid.
    """
    try:
        df = pd.read_csv(prediction_file)
    except Exception as e:
        logger.error(f"Failed to load file: {e}")
        return False

    n_samples = min(n_samples, len(df))
    sampled = df.sample(n=n_samples, random_state=42)

    pred_col = "prediction"
    if pred_col not in df.columns:
        logger.error("Missing 'prediction' column")
        return False

    # Check value ranges
    if task == "read":
        valid = sampled[pred_col].between(0, 1, inclusive="both").all()
    elif task == "category":
        valid = sampled[pred_col].between(0, 4, inclusive="both").all()
        if not sampled[pred_col].dtype in ["int64", "int32"]:
            logger.warning("Category predictions should be integers")
    elif task == "rating":
        valid = sampled[pred_col].between(1, 5, inclusive="both").all()
    else:
        logger.error(f"Unknown task: {task}")
        return False

    if not valid:
        logger.error(f"Found invalid predictions in sample")
        return False

    logger.info(f"Spot check passed: {n_samples} samples valid")
    return True


def validate_all_artifacts(
    output_dir: str = "outputs",
    data_dir: str = "assignment1",
) -> bool:
    """Validate all submission artifacts.

    Args:
        output_dir: Directory containing prediction files.
        data_dir: Directory containing pairs files.

    Returns:
        True if all validations pass.
    """
    output_path = Path(output_dir)
    data_path = Path(data_dir)

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
        logger.info(f"\n=== Validating {task} task ===")
        pred_path = output_path / pred_file
        pairs_path = data_path / pairs_file

        # File size
        if not validate_file_size(pred_path):
            all_valid = False
            continue

        # Row count
        if not validate_row_count(pred_path, pairs_path):
            all_valid = False
            continue

        # Schema
        if not validate_schema(pred_path, task):
            all_valid = False
            continue

        # Spot check
        if not spot_check_samples(pred_path, task):
            all_valid = False
            continue

        logger.info(f"✓ {task} task validation passed")

    return all_valid


def main() -> int:
    """Main entry point for submission validation.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    parser = argparse.ArgumentParser(
        description="Validate submission artifacts"
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

    args = parser.parse_args()

    configure_logging()

    is_valid = validate_all_artifacts(args.output_dir, args.data_dir)

    if is_valid:
        logger.info("\n✓ All submission artifacts validated successfully")
        return 0
    else:
        logger.error("\n✗ Some validations failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())


