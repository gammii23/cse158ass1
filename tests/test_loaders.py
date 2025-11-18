"""Unit tests for data loaders."""

import gzip
import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from data_access.loader import (
    load_category_data,
    load_interactions,
    load_pairs,
)


@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_interactions_file(temp_dir):
    """Create sample interactions CSV.gz file."""
    file_path = temp_dir / "train_Interactions.csv.gz"
    data = [
        ["user1", "item1", "5"],
        ["user1", "item2", "4"],
        ["user2", "item1", "3"],
    ]
    with gzip.open(file_path, "wt") as f:
        for row in data:
            f.write(",".join(row) + "\n")
    return file_path


@pytest.fixture
def sample_category_file(temp_dir):
    """Create sample category JSON.gz file."""
    file_path = temp_dir / "train_Category.json.gz"
    records = [
        {
            "user_id": "user1",
            "review_id": "rev1",
            "review_text": "Great book!",
            "category": "fantasy_paranormal",
        },
        {
            "user_id": "user2",
            "review_id": "rev2",
            "review_text": "Interesting mystery",
            "category": "mystery_thriller_crime",
        },
    ]
    with gzip.open(file_path, "wt", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    return file_path


@pytest.fixture
def sample_pairs_file(temp_dir):
    """Create sample pairs CSV file."""
    file_path = temp_dir / "pairs_Read.csv"
    df = pd.DataFrame(
        {
            "userID": ["user1", "user2"],
            "itemID": ["item1", "item2"],
        }
    )
    df.to_csv(file_path, index=False)
    return file_path


def test_load_interactions_success(sample_interactions_file, temp_dir):
    """Test loading valid interactions file."""
    df = load_interactions(
        data_dir=str(temp_dir),
        filename=sample_interactions_file.name,
        use_cache=False,
    )
    assert len(df) == 3
    assert "user_id" in df.columns
    assert "item_id" in df.columns
    assert "rating" in df.columns
    assert df["rating"].dtype == "int64"


def test_load_interactions_missing_file(temp_dir):
    """Test loading non-existent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_interactions(
            data_dir=str(temp_dir),
            filename="nonexistent.csv.gz",
            use_cache=False,
        )


def test_load_interactions_caching(sample_interactions_file, temp_dir):
    """Test caching functionality."""
    cache_dir = temp_dir / "cache"
    cache_dir.mkdir()

    # First load - should create cache
    df1 = load_interactions(
        data_dir=str(temp_dir),
        filename=sample_interactions_file.name,
        use_cache=True,
        cache_dir=str(cache_dir),
    )

    # Second load - should use cache
    df2 = load_interactions(
        data_dir=str(temp_dir),
        filename=sample_interactions_file.name,
        use_cache=True,
        cache_dir=str(cache_dir),
    )

    assert len(df1) == len(df2)
    pd.testing.assert_frame_equal(df1, df2)


def test_load_category_json_lines(sample_category_file, temp_dir):
    """Test loading valid category JSON lines file."""
    df = load_category_data(
        data_dir=str(temp_dir),
        filename=sample_category_file.name,
        use_cache=False,
    )
    assert len(df) == 2
    assert "user_id" in df.columns
    assert "review_text" in df.columns
    assert "category" in df.columns


def test_load_category_malformed_json(temp_dir):
    """Test handling malformed JSON lines."""
    file_path = temp_dir / "test_Category.json.gz"
    with gzip.open(file_path, "wt", encoding="utf-8") as f:
        f.write('{"valid": "json"}\n')
        f.write("invalid json line\n")
        f.write('{"another": "valid"}\n')

    df = load_category_data(
        data_dir=str(temp_dir),
        filename=file_path.name,
        use_cache=False,
    )
    # Should load 2 valid records, skip 1 invalid
    assert len(df) == 2


def test_load_pairs_schema_validation(sample_pairs_file, temp_dir):
    """Test pairs file schema validation."""
    df = load_pairs(
        data_dir=str(temp_dir), filename=sample_pairs_file.name
    )
    assert len(df) == 2
    assert "user_id" in df.columns or "userID" in df.columns


