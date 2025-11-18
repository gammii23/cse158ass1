"""Pipeline orchestrator for read prediction workflow."""

from typing import Optional

import numpy as np
import pandas as pd

from config.settings import ReadConfig, Settings
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

        # Build features
        logger.info("Building interaction features")
        feature_builder = InteractionFeatureBuilder(
            random_seed=settings.experiment.random_seed
        )
        feature_builder.fit(train_df)

        # For read prediction, use popularity-based approach
        # Since all training interactions are "read", we predict based on item popularity
        logger.info("Computing popularity-based read predictions")
        
        # Compute item popularity (fraction of total reads)
        item_counts = train_df[InteractionColumns.ITEM_ID].value_counts()
        total_reads = len(train_df)
        item_popularity = item_counts / total_reads
        
        # For pairs, predict based on item popularity
        # Use popularity score as probability (normalized to [0, 1])
        pair_item_ids = pairs_df[PairColumns.ITEM_ID]
        predictions = pair_item_ids.map(item_popularity).fillna(0.0).values
        
        # Normalize to ensure [0, 1] range
        if predictions.max() > 0:
            predictions = predictions / predictions.max()
        
        # Clip to [0, 1]
        predictions = np.clip(predictions, 0.0, 1.0)

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

