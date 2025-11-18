"""Pipeline orchestrator for category classification workflow."""

import pandas as pd

from config.settings import CategoryConfig, Settings
from constants import FileNames, PairColumns
from data_access.loader import load_category_data, load_pairs
from features.reviews import TfidfFeaturizer
from models.category_classifier import CategoryClassifier
from pipelines.types import SubmissionResult
from utils.logging import get_logger
from utils.random_seed import set_global_seed

logger = get_logger("pipelines.category_workflow")


def run_category_workflow(
    settings: Settings,
    data_dir: str = "assignment1",
) -> SubmissionResult:
    """Execute category classification pipeline end-to-end.

    Args:
        settings: Configuration settings.
        data_dir: Directory containing data files.

    Returns:
        SubmissionResult with success status and predictions DataFrame.
    """
    try:
        logger.info("Starting category classification workflow")
        set_global_seed(settings.experiment.random_seed)

        # Load data
        logger.info("Loading training category data")
        train_df = load_category_data(
            data_dir=data_dir,
            filename=FileNames.TRAIN_CATEGORY,
            use_cache=settings.experiment.enable_caching,
            cache_dir=settings.paths.cache_dir,
        )

        logger.info("Loading test category data")
        test_df = load_category_data(
            data_dir=data_dir,
            filename=FileNames.TEST_CATEGORY,
            use_cache=settings.experiment.enable_caching,
            cache_dir=settings.paths.cache_dir,
        )

        logger.info("Loading evaluation pairs")
        pairs_df = load_pairs(
            data_dir=data_dir, filename=FileNames.PAIRS_CATEGORY
        )

        # Map pairs to test reviews
        pairs_df = pairs_df.merge(
            test_df[["review_id", "review_text"]],
            on="review_id",
            how="left",
        )

        # Build text features
        logger.info("Building text features")
        featurizer = TfidfFeaturizer(
            max_features=settings.category.max_features,
            ngram_range=settings.category.ngram_range,
            min_df=settings.category.min_df,
            max_df=settings.category.max_df,
            random_seed=settings.experiment.random_seed,
        )
        featurizer.fit(train_df["review_text"].tolist())

        # Train classifier
        logger.info("Training category classifier")
        # Map category labels to integers
        unique_categories = sorted(train_df["category"].unique())
        category_to_int = {cat: idx for idx, cat in enumerate(unique_categories)}
        train_labels = train_df["category"].map(category_to_int).values

        classifier = CategoryClassifier(
            featurizer=featurizer,
            C=settings.category.C,
            class_weight=settings.category.class_weight,
            use_svm=settings.category.use_svm,
            random_seed=settings.experiment.random_seed,
        )
        classifier.fit(train_df["review_text"].tolist(), train_labels)

        # Predict on pairs
        logger.info("Generating predictions for pairs")
        pair_texts = pairs_df["review_text"].fillna("").tolist()
        predictions = classifier.predict(pair_texts)

        # Build submission DataFrame
        submission_df = pd.DataFrame(
            {
                "userID": pairs_df[PairColumns.USER_ID],
                "reviewID": pairs_df["review_id"],
                "prediction": predictions,
            }
        )

        logger.info(
            f"Category workflow completed: {len(submission_df)} predictions"
        )
        return {"success": True, "data": submission_df}

    except Exception as e:
        logger.error(f"Category workflow failed: {e}", exc_info=True)
        return {"success": False, "error": str(e)}


