"""Pipeline orchestrator for read prediction workflow."""

import numpy as np
import pandas as pd

from config.settings import Settings
from constants import FileNames, InteractionColumns, PairColumns
from data_access.loader import load_interactions, load_pairs
from features.interactions import InteractionFeatureBuilder
from models.read_predictor import ReadPredictor
from pipelines.types import SubmissionResult
from utils.logging import get_logger
from utils.random_seed import set_global_seed

logger = get_logger("pipelines.read_workflow")


def run_read_workflow(
    settings: Settings,
    data_dir: str = "assignment1",
) -> SubmissionResult:
    """Execute read prediction pipeline end-to-end.

    Args:
        settings: Configuration settings.
        data_dir: Directory containing data files.

    Returns:
        SubmissionResult with success status and predictions DataFrame.
    """
    try:
        logger.info("Starting read prediction workflow")
        set_global_seed(settings.experiment.random_seed)

        # Load data
        logger.info("Loading interactions data")
        train_df = load_interactions(
            data_dir=data_dir,
            filename=FileNames.TRAIN_INTERACTIONS,
            use_cache=settings.experiment.enable_caching,
            cache_dir=settings.paths.cache_dir,
        )

        logger.info("Loading evaluation pairs")
        pairs_df = load_pairs(
            data_dir=data_dir, filename=FileNames.PAIRS_READ
        )

        # Build features and implicit matrix
        logger.info("Building interaction features and implicit matrix")
        feature_builder = InteractionFeatureBuilder(
            random_seed=settings.experiment.random_seed
        )
        feature_builder.fit(train_df)

        interaction_matrix, user_to_idx, item_to_idx = feature_builder.build_sparse_matrix(
            train_df
        )

        # Popularity prior (fraction of total reads)
        item_counts = train_df[InteractionColumns.ITEM_ID].value_counts()
        total_reads = len(train_df)
        item_popularity = item_counts / total_reads

        # Train implicit read predictor
        predictor = ReadPredictor(
            use_implicit=settings.read.use_implicit,
            n_factors=settings.read.n_factors,
            n_iterations=settings.read.n_iterations,
            regularization=settings.read.als_regularization,
            popularity_weight=settings.read.popularity_weight,
            random_seed=settings.experiment.random_seed,
        )
        predictor.fit(
            interaction_matrix,
            user_to_idx=user_to_idx,
            item_to_idx=item_to_idx,
            item_popularity=item_popularity,
        )

        # Generate blended implicit + popularity predictions
        predictions = predictor.predict_proba_pairs(
            pairs_df[PairColumns.USER_ID], pairs_df[PairColumns.ITEM_ID]
        )

        # Build submission DataFrame
        submission_df = pd.DataFrame(
            {
                PairColumns.USER_ID: pairs_df[PairColumns.USER_ID],
                PairColumns.ITEM_ID: pairs_df[PairColumns.ITEM_ID],
                "prediction": predictions,
            }
        )

        logger.info(
            f"Read workflow completed: {len(submission_df)} predictions"
        )
        return {"success": True, "data": submission_df}

    except Exception as e:
        logger.error(f"Read workflow failed: {e}", exc_info=True)
        return {"success": False, "error": str(e)}

