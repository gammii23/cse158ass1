"""Main CLI entry point for assignment1.

This script provides command-line interface to execute read, category,
and rating prediction workflows and generate submission CSVs.
"""

import argparse
import sys
from pathlib import Path

from config.settings import Settings
from outputs.serializer import write_predictions
from pipelines.category_workflow import run_category_workflow
from pipelines.read_workflow import run_read_workflow
from pipelines.rating_workflow import run_rating_workflow
from utils.logging import configure_logging, get_logger

logger = get_logger("assignment1")


def main() -> int:
    """Main CLI entry point.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    parser = argparse.ArgumentParser(
        description="Assignment 1: Read, Category, and Rating Predictions"
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["read", "category", "rating", "all"],
        required=True,
        help="Task to execute",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="assignment1",
        help="Directory containing data files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory for output predictions",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable caching",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )

    args = parser.parse_args()

    # Configure logging
    log_level_map = {
        "DEBUG": 10,
        "INFO": 20,
        "WARNING": 30,
        "ERROR": 40,
    }
    configure_logging(level=log_level_map[args.log_level])

    # Load settings
    settings = Settings.from_env()
    if args.seed is not None:
        settings.experiment.random_seed = args.seed
    if args.no_cache:
        settings.experiment.enable_caching = False
    settings.paths.output_dir = args.output_dir

    tasks_to_run = []
    if args.task == "all":
        tasks_to_run = ["read", "category", "rating"]
    else:
        tasks_to_run = [args.task]

    success_count = 0
    for task in tasks_to_run:
        logger.info(f"Executing {task} task")
        try:
            if task == "read":
                result = run_read_workflow(settings, data_dir=args.data_dir)
            elif task == "category":
                result = run_category_workflow(
                    settings, data_dir=args.data_dir
                )
            elif task == "rating":
                result = run_rating_workflow(
                    settings, data_dir=args.data_dir
                )
            else:
                logger.error(f"Unknown task: {task}")
                continue

            if result["success"]:
                # Write predictions
                write_predictions(
                    result["data"],
                    task=task,
                    output_dir=args.output_dir,
                )
                logger.info(f"{task} task completed successfully")
                success_count += 1
            else:
                logger.error(
                    f"{task} task failed: {result.get('error', 'Unknown error')}"
                )

        except Exception as e:
            logger.error(f"{task} task failed with exception: {e}", exc_info=True)

    if success_count == len(tasks_to_run):
        logger.info("All tasks completed successfully")
        return 0
    else:
        logger.error(
            f"Only {success_count}/{len(tasks_to_run)} tasks succeeded"
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())


