"""Performance profiling script for pipeline components.

This script measures runtime and memory usage for each pipeline
component to identify bottlenecks.
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import Settings
from pipelines.category_workflow import run_category_workflow
from pipelines.read_workflow import run_read_workflow
from pipelines.rating_workflow import run_rating_workflow
from utils.logging import configure_logging, get_logger

logger = get_logger("scripts.profile_performance")


def profile_task(
    task: str, settings: Settings, data_dir: str = "assignment1"
) -> dict:
    """Profile a single task pipeline.

    Args:
        task: Task name ('read', 'category', 'rating').
        settings: Configuration settings.
        data_dir: Directory containing data files.

    Returns:
        Dictionary with timing and memory metrics.
    """
    logger.info(f"Profiling {task} task")

    start_time = time.time()

    try:
        if task == "read":
            result = run_read_workflow(settings, data_dir=data_dir)
        elif task == "category":
            result = run_category_workflow(settings, data_dir=data_dir)
        elif task == "rating":
            result = run_rating_workflow(settings, data_dir=data_dir)
        else:
            raise ValueError(f"Unknown task: {task}")

        end_time = time.time()
        elapsed_time = end_time - start_time

        metrics = {
            "task": task,
            "success": result["success"],
            "elapsed_time_seconds": elapsed_time,
            "elapsed_time_minutes": elapsed_time / 60,
        }

        if result["success"] and "data" in result:
            metrics["num_predictions"] = len(result["data"])

        logger.info(
            f"{task} task completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)"
        )

        return metrics

    except Exception as e:
        logger.error(f"Profiling {task} failed: {e}", exc_info=True)
        return {
            "task": task,
            "success": False,
            "error": str(e),
        }


def main() -> int:
    """Main entry point for performance profiling.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    parser = argparse.ArgumentParser(
        description="Profile pipeline performance"
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["read", "category", "rating", "all"],
        default="all",
        help="Task to profile",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="assignment1",
        help="Directory containing data files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/performance_log.csv",
        help="Output CSV file for metrics",
    )

    args = parser.parse_args()

    configure_logging()

    settings = Settings.from_env()

    tasks_to_profile = (
        ["read", "category", "rating"]
        if args.task == "all"
        else [args.task]
    )

    all_metrics = []
    for task in tasks_to_profile:
        metrics = profile_task(task, settings, args.data_dir)
        all_metrics.append(metrics)

    # Save to CSV
    df = pd.DataFrame(all_metrics)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    logger.info(f"Performance metrics saved to {output_path}")

    # Print summary
    logger.info("\n=== Performance Summary ===")
    for metrics in all_metrics:
        if metrics.get("success"):
            logger.info(
                f"{metrics['task']}: {metrics.get('elapsed_time_minutes', 0):.2f} minutes"
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())


