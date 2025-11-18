"""Pipeline orchestrator for rating regression workflow."""

import pandas as pd

from config.settings import RatingConfig, Settings
from constants import FileNames, InteractionColumns, PairColumns
from data_access.loader import load_interactions, load_pairs
from features.interactions import InteractionFeatureBuilder
from models.rating_regressor import RatingRegressor
from pipelines.types import SubmissionResult
from utils.logging import get_logger
from utils.random_seed import set_global_seed

logger = get_logger("pipelines.rating_workflow")


def run_rating_workflow(
    settings: Settings,
    data_dir: str = "assignment1",
) -> SubmissionResult:
    """Execute rating regression pipeline end-to-end.

    Args:
        settings: Configuration settings.
        data_dir: Directory containing data files.

    Returns:
        SubmissionResult with success status and predictions DataFrame.
    """
    try:
        logger.info("Starting rating regression workflow")
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
            data_dir=data_dir, filename=FileNames.PAIRS_RATING
        )

        # Build features
        logger.info("Building interaction features")
        feature_builder = InteractionFeatureBuilder(
            random_seed=settings.experiment.random_seed
        )
        feature_builder.fit(train_df)

        train_features = feature_builder.build_interaction_features(train_df)
        train_labels = train_df[InteractionColumns.RATING].values

        # Train model
        logger.info("Training rating regressor")
        regressor = RatingRegressor(
            use_bias=settings.rating.use_bias,
            use_xgboost=settings.rating.use_xgboost,
            use_ridge=settings.rating.use_ridge,
            xgb_n_estimators=settings.rating.xgb_n_estimators,
            xgb_max_depth=settings.rating.xgb_max_depth,
            xgb_learning_rate=settings.rating.xgb_learning_rate,
            ridge_alpha=settings.rating.ridge_alpha,
            random_seed=settings.experiment.random_seed,
        )
        regressor.fit(
            train_features,
            train_labels,
            user_ids=train_df[InteractionColumns.USER_ID],
            item_ids=train_df[InteractionColumns.ITEM_ID],
        )

        # Predict on pairs
        logger.info("Generating predictions for pairs")
        pair_features = feature_builder.build_interaction_features(pairs_df)
        predictions = regressor.predict(
            pair_features,
            user_ids=pairs_df[PairColumns.USER_ID],
            item_ids=pairs_df[PairColumns.ITEM_ID],
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
            f"Rating workflow completed: {len(submission_df)} predictions"
        )
        return {"success": True, "data": submission_df}

    except Exception as e:
        logger.error(f"Rating workflow failed: {e}", exc_info=True)
        return {"success": False, "error": str(e)}

