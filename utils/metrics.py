"""Evaluation metrics for assignment1 tasks.

This module provides unified metric computation interfaces for
balanced accuracy (read), accuracy (category), and MSE (rating).
"""

from typing import Dict, Optional

import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error

from utils.logging import get_logger

logger = get_logger("utils.metrics")


def balanced_accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute balanced accuracy for read prediction.

    Balanced accuracy is the average of recall obtained on each class.
    For binary classification with balanced classes, this equals accuracy.

    Args:
        y_true: True binary labels (0 or 1).
        y_pred: Predicted binary labels (0 or 1).

    Returns:
        Balanced accuracy score [0, 1].
    """
    # For binary classification, balanced accuracy = average of per-class recall
    # Which simplifies to regular accuracy when classes are balanced
    # But we compute it properly for imbalanced cases
    classes = np.unique(y_true)
    if len(classes) != 2:
        logger.warning(
            f"Expected binary classification, got {len(classes)} classes. Using accuracy."
        )
        return float(accuracy_score(y_true, y_pred))

    recalls = []
    for cls in classes:
        mask = y_true == cls
        if mask.sum() > 0:
            recall = (y_pred[mask] == cls).sum() / mask.sum()
            recalls.append(recall)
        else:
            recalls.append(0.0)

    balanced_acc = np.mean(recalls)
    return float(balanced_acc)


def accuracy_score_wrapper(
    y_true: np.ndarray, y_pred: np.ndarray
) -> float:
    """Wrapper for accuracy score for category classification.

    Args:
        y_true: True category labels (0-4).
        y_pred: Predicted category labels (0-4).

    Returns:
        Accuracy score [0, 1].
    """
    return float(accuracy_score(y_true, y_pred))


def mse_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute mean squared error for rating regression.

    Args:
        y_true: True ratings (1-5).
        y_pred: Predicted ratings (1-5).

    Returns:
        MSE score (lower is better).
    """
    return float(mean_squared_error(y_true, y_pred))


def compute_metrics(
    task: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute metrics for a given task.

    Args:
        task: Task name ('read', 'category', 'rating').
        y_true: True labels/ratings.
        y_pred: Predicted labels/ratings.
        y_proba: Optional probability predictions (for read task).

    Returns:
        Dictionary of metric names to scores.
    """
    metrics: Dict[str, float] = {}

    if task == "read":
        metrics["balanced_accuracy"] = balanced_accuracy_score(y_true, y_pred)
        metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
        if y_proba is not None:
            # Additional metrics for probabilities
            metrics["log_loss"] = float(
                -np.mean(
                    y_true * np.log(y_proba + 1e-15)
                    + (1 - y_true) * np.log(1 - y_proba + 1e-15)
                )
            )

    elif task == "category":
        metrics["accuracy"] = accuracy_score_wrapper(y_true, y_pred)

    elif task == "rating":
        metrics["mse"] = mse_score(y_true, y_pred)
        metrics["rmse"] = float(np.sqrt(metrics["mse"]))
        metrics["mae"] = float(np.mean(np.abs(y_true - y_pred)))

    else:
        raise ValueError(f"Unknown task: {task}")

    return metrics


