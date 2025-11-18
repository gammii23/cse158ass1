"""Experiment logging utilities for tracking CV results and hyperparameters.

This module provides a centralized way to log experiments with timestamps,
hyperparameters, CV scores, and notes to a CSV file.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from utils.logging import get_logger

logger = get_logger("utils.experiment_logger")


class ExperimentLogger:
    """Logger for experiment results and hyperparameters."""

    def __init__(self, results_dir: str = "results") -> None:
        """Initialize experiment logger.

        Args:
            results_dir: Directory to store experiment logs.
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.results_dir / "experiments.csv"

        # Initialize CSV with headers if it doesn't exist
        if not self.log_file.exists():
            self._initialize_log_file()

    def _initialize_log_file(self) -> None:
        """Create CSV file with headers."""
        columns = [
            "timestamp",
            "task",
            "model_type",
            "hyperparams_json",
            "cv_scores_json",
            "best_score",
            "seed",
            "notes",
        ]
        df = pd.DataFrame(columns=columns)
        df.to_csv(self.log_file, index=False)
        logger.info(f"Initialized experiment log: {self.log_file}")

    def log_experiment(
        self,
        task: str,
        model_type: str,
        hyperparams: Dict[str, Any],
        cv_scores: List[float],
        best_score: Optional[float] = None,
        seed: Optional[int] = None,
        notes: Optional[str] = None,
    ) -> None:
        """Log an experiment to CSV.

        Args:
            task: Task name ('read', 'category', 'rating').
            model_type: Model type identifier.
            hyperparams: Dictionary of hyperparameters.
            cv_scores: List of CV fold scores.
            best_score: Best score across folds (optional).
            seed: Random seed used (optional).
            notes: Additional notes (optional).
        """
        if best_score is None:
            best_score = np.mean(cv_scores) if cv_scores else None

        timestamp = datetime.now().isoformat()
        hyperparams_json = json.dumps(hyperparams, sort_keys=True)
        cv_scores_json = json.dumps(cv_scores, sort_keys=True)

        new_row = {
            "timestamp": timestamp,
            "task": task,
            "model_type": model_type,
            "hyperparams_json": hyperparams_json,
            "cv_scores_json": cv_scores_json,
            "best_score": best_score,
            "seed": seed,
            "notes": notes,
        }

        # Append to CSV
        df = pd.DataFrame([new_row])
        df.to_csv(
            self.log_file, mode="a", header=False, index=False
        )

        logger.info(
            f"Logged experiment: {task}/{model_type} (best_score={best_score:.4f})"
        )

    def get_best_experiment(
        self, task: str, metric: str = "best_score"
    ) -> Optional[Dict[str, Any]]:
        """Get best experiment for a task.

        Args:
            task: Task name.
            metric: Metric to optimize ('best_score' or specific metric).

        Returns:
            Dictionary of best experiment details, or None if no experiments.
        """
        if not self.log_file.exists():
            return None

        df = pd.read_csv(self.log_file)
        task_df = df[df["task"] == task]

        if len(task_df) == 0:
            return None

        # Find best score
        best_idx = task_df[metric].idxmax()
        best_row = task_df.loc[best_idx]

        return {
            "timestamp": best_row["timestamp"],
            "task": best_row["task"],
            "model_type": best_row["model_type"],
            "hyperparams": json.loads(best_row["hyperparams_json"]),
            "cv_scores": json.loads(best_row["cv_scores_json"]),
            "best_score": best_row["best_score"],
            "seed": best_row["seed"],
            "notes": best_row["notes"],
        }

