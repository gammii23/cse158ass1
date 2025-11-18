"""Integration tests for end-to-end pipelines."""

import tempfile
from pathlib import Path

import gzip
import json
import pandas as pd
import pytest

from config.settings import Settings
from pipelines.category_workflow import run_category_workflow
from pipelines.read_workflow import run_read_workflow
from pipelines.rating_workflow import run_rating_workflow


@pytest.fixture
def temp_data_dir():
    """Create temporary directory with sample data files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir)
        data_dir.mkdir(exist_ok=True)

        # Create sample interactions file
        interactions_file = data_dir / "train_Interactions.csv.gz"
        with gzip.open(interactions_file, "wt") as f:
            f.write("user_id,item_id,rating\n")
            for i in range(100):
                f.write(f"user{i%10},item{i%20},{3+(i%3)}\n")

        # Create sample category files
        train_cat_file = data_dir / "train_Category.json.gz"
        with gzip.open(train_cat_file, "wt", encoding="utf-8") as f:
            for i in range(50):
                record = {
                    "user_id": f"user{i%10}",
                    "review_id": f"rev{i}",
                    "review_text": f"Review text {i}",
                    "category": ["fantasy_paranormal", "mystery_thriller_crime", "children"][i % 3],
                }
                f.write(json.dumps(record) + "\n")

        test_cat_file = data_dir / "test_Category.json.gz"
        with gzip.open(test_cat_file, "wt", encoding="utf-8") as f:
            for i in range(20):
                record = {
                    "user_id": f"user{i%10}",
                    "review_id": f"test_rev{i}",
                    "review_text": f"Test review {i}",
                }
                f.write(json.dumps(record) + "\n")

        # Create sample pairs files
        pairs_read = data_dir / "pairs_Read.csv"
        pd.DataFrame({
            "userID": ["user1", "user2"],
            "itemID": ["item1", "item2"],
        }).to_csv(pairs_read, index=False)

        pairs_category = data_dir / "pairs_Category.csv"
        pd.DataFrame({
            "userID": ["user1", "user2"],
            "reviewID": ["test_rev0", "test_rev1"],
        }).to_csv(pairs_category, index=False)

        pairs_rating = data_dir / "pairs_Rating.csv"
        pd.DataFrame({
            "userID": ["user1", "user2"],
            "itemID": ["item1", "item2"],
        }).to_csv(pairs_rating, index=False)

        yield data_dir


def test_read_pipeline_end_to_end(temp_data_dir):
    """Test read pipeline end-to-end with sample data."""
    settings = Settings.from_env()
    settings.experiment.enable_caching = False

    result = run_read_workflow(settings, data_dir=str(temp_data_dir))

    assert result["success"] is True
    assert "data" in result
    assert len(result["data"]) == 2
    assert "prediction" in result["data"].columns


def test_category_pipeline_end_to_end(temp_data_dir):
    """Test category pipeline end-to-end with sample data."""
    settings = Settings.from_env()
    settings.experiment.enable_caching = False
    settings.category.max_features = 1000  # Reduce for test speed

    result = run_category_workflow(settings, data_dir=str(temp_data_dir))

    assert result["success"] is True
    assert "data" in result
    assert len(result["data"]) == 2
    assert "prediction" in result["data"].columns


def test_rating_pipeline_end_to_end(temp_data_dir):
    """Test rating pipeline end-to-end with sample data."""
    settings = Settings.from_env()
    settings.experiment.enable_caching = False
    settings.rating.xgb_n_estimators = 10  # Reduce for test speed

    result = run_rating_workflow(settings, data_dir=str(temp_data_dir))

    assert result["success"] is True
    assert "data" in result
    assert len(result["data"]) == 2
    assert "prediction" in result["data"].columns


