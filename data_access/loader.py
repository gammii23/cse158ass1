"""Data loading utilities with schema validation and caching.

This module provides pure data ingestion functions that return typed
pandas DataFrames or typed review objects, handling compression,
schema validation, and missing-value policies.
"""

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from constants import (
    CategoryColumns,
    FileNames,
    InteractionColumns,
    PairColumns,
)
from utils.logging import get_logger

logger = get_logger("data_access.loader")


def _compute_checksum(file_path: Path) -> str:
    """Compute MD5 checksum of a file.

    Args:
        file_path: Path to file.

    Returns:
        Hex digest of file checksum.
    """
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def load_interactions(
    data_dir: str = "assignment1",
    filename: str = FileNames.TRAIN_INTERACTIONS,
    use_cache: bool = True,
    cache_dir: str = "cache",
) -> pd.DataFrame:
    """Load interactions dataset from compressed CSV.

    Args:
        data_dir: Directory containing data files.
        filename: Name of interactions file.
        use_cache: Whether to use parquet cache if available.
        cache_dir: Directory for cached parquet files.

    Returns:
        DataFrame with columns: user_id, item_id, rating.

    Raises:
        FileNotFoundError: If data file doesn't exist.
        ValueError: If schema validation fails.
    """
    file_path = Path(data_dir) / filename
    if not file_path.exists():
        raise FileNotFoundError(f"Interactions file not found: {file_path}")

    cache_path = Path(cache_dir) / f"{filename}.parquet"
    checksum_file = Path(cache_dir) / f"{filename}.checksum"

    # Check cache
    if use_cache and cache_path.exists() and checksum_file.exists():
        current_checksum = _compute_checksum(file_path)
        cached_checksum = checksum_file.read_text().strip()
        if current_checksum == cached_checksum:
            logger.info(f"Loading cached interactions from {cache_path}")
            df = pd.read_parquet(cache_path)
            return df

    # Load from CSV (skip header row)
    logger.info(f"Loading interactions from {file_path}")
    df = pd.read_csv(
        file_path,
        compression="gzip",
        header=0,  # Use first row as header
        dtype={
            "userID": str,
            "bookID": str,
            "rating": int,
        },
    )
    # Standardize column names
    df = df.rename(columns={
        "userID": InteractionColumns.USER_ID,
        "bookID": InteractionColumns.ITEM_ID,
        "rating": InteractionColumns.RATING,
    })

    # Schema validation
    required_cols = [
        InteractionColumns.USER_ID,
        InteractionColumns.ITEM_ID,
        InteractionColumns.RATING,
    ]
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Missing value policy: drop rows with any NaN
    original_len = len(df)
    df = df.dropna()
    if len(df) < original_len:
        logger.warning(
            f"Dropped {original_len - len(df)} rows with missing values"
        )

    # Validate rating range (assuming 1-5 scale)
    invalid_ratings = df[
        ~df[InteractionColumns.RATING].between(1, 5, inclusive="both")
    ]
    if len(invalid_ratings) > 0:
        logger.warning(
            f"Found {len(invalid_ratings)} ratings outside [1,5] range"
        )

    # Cache if enabled
    if use_cache:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)
        checksum_file.write_text(_compute_checksum(file_path))
        logger.info(f"Cached interactions to {cache_path}")

    return df


def load_category_data(
    data_dir: str = "assignment1",
    filename: str = FileNames.TRAIN_CATEGORY,
    use_cache: bool = True,
    cache_dir: str = "cache",
) -> pd.DataFrame:
    """Load category dataset from compressed JSON lines.

    Args:
        data_dir: Directory containing data files.
        filename: Name of category file (train or test).
        use_cache: Whether to use parquet cache if available.
        cache_dir: Directory for cached parquet files.

    Returns:
        DataFrame with columns: user_id, review_id, review_text, category
        (category only present in train data).

    Raises:
        FileNotFoundError: If data file doesn't exist.
        ValueError: If schema validation fails.
    """
    file_path = Path(data_dir) / filename
    if not file_path.exists():
        raise FileNotFoundError(f"Category file not found: {file_path}")

    cache_path = Path(cache_dir) / f"{filename}.parquet"
    checksum_file = Path(cache_dir) / f"{filename}.checksum"

    # Check cache
    if use_cache and cache_path.exists() and checksum_file.exists():
        current_checksum = _compute_checksum(file_path)
        cached_checksum = checksum_file.read_text().strip()
        if current_checksum == cached_checksum:
            logger.info(f"Loading cached category data from {cache_path}")
            df = pd.read_parquet(cache_path)
            return df

    # Load from JSON lines (using eval as per baselines.py)
    logger.info(f"Loading category data from {file_path}")
    records: List[Dict[str, Any]] = []
    with gzip.open(file_path, "rt", encoding="utf-8") as f:
        for line in f:
            try:
                # Use eval() as per baselines.py format
                record = eval(line.strip())
                records.append(record)
            except (SyntaxError, ValueError) as e:
                logger.warning(f"Skipping invalid line: {e}")
                continue

    df = pd.DataFrame(records)

    # Standardize column names (map genre to category)
    column_mapping = {
        "user_id": CategoryColumns.USER_ID,
        "review_id": "review_id",
        "review_text": CategoryColumns.REVIEW_TEXT,
        "genre": CategoryColumns.CATEGORY,  # Map genre to category
    }
    df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})

    # Schema validation
    required_cols = [
        CategoryColumns.USER_ID,
        "review_id",
        CategoryColumns.REVIEW_TEXT,
    ]
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Missing value policy: drop rows with missing review_text
    original_len = len(df)
    df = df.dropna(subset=[CategoryColumns.REVIEW_TEXT])
    if len(df) < original_len:
        logger.warning(
            f"Dropped {original_len - len(df)} rows with missing review_text"
        )

    # Cache if enabled
    if use_cache:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)
        checksum_file.write_text(_compute_checksum(file_path))
        logger.info(f"Cached category data to {cache_path}")

    return df


def load_pairs(
    data_dir: str = "assignment1",
    filename: str = FileNames.PAIRS_READ,
) -> pd.DataFrame:
    """Load evaluation pairs from CSV.

    Args:
        data_dir: Directory containing data files.
        filename: Name of pairs file.

    Returns:
        DataFrame with columns: user_id, item_id (or review_id for category).

    Raises:
        FileNotFoundError: If pairs file doesn't exist.
        ValueError: If schema validation fails.
    """
    file_path = Path(data_dir) / filename
    if not file_path.exists():
        raise FileNotFoundError(f"Pairs file not found: {file_path}")

    logger.info(f"Loading pairs from {file_path}")
    df = pd.read_csv(file_path)

    # Standardize column names (handle both userID/user_id, bookID/itemID, etc.)
    column_mapping = {
        "userID": PairColumns.USER_ID,
        "user_id": PairColumns.USER_ID,
        "bookID": PairColumns.ITEM_ID,  # Rating/Read pairs use bookID
        "itemID": PairColumns.ITEM_ID,
        "item_id": PairColumns.ITEM_ID,
        "reviewID": "review_id",
        "review_id": "review_id",
    }
    df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})

    # Schema validation
    if PairColumns.USER_ID not in df.columns:
        raise ValueError(f"Missing required column: {PairColumns.USER_ID}")

    # Missing value policy: drop rows with NaN in required columns (not prediction column)
    required_cols = [PairColumns.USER_ID]
    if PairColumns.ITEM_ID in df.columns:
        required_cols.append(PairColumns.ITEM_ID)
    elif "review_id" in df.columns:
        required_cols.append("review_id")
    
    original_len = len(df)
    df = df.dropna(subset=required_cols)
    if len(df) < original_len:
        logger.warning(
            f"Dropped {original_len - len(df)} rows with missing values in required columns"
        )

    return df

