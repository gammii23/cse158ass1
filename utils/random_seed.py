"""Global random seed utility for reproducibility.

This module provides functions to set random seeds across all
libraries used in the project (numpy, random, sklearn, etc.)
to ensure deterministic behavior.
"""

import random
from typing import Optional

import numpy as np
from sklearn.utils import check_random_state


def set_global_seed(seed: int = 42) -> None:
    """Set random seed for all libraries used in the project.

    Sets seeds for:
    - Python's random module
    - NumPy
    - Scikit-learn (via check_random_state)

    Args:
        seed: Random seed value (default: 42).
    """
    random.seed(seed)
    np.random.seed(seed)
    # sklearn uses numpy's random state, so this covers it
    check_random_state(seed)


def get_random_state(seed: Optional[int] = None) -> np.random.RandomState:
    """Get a numpy RandomState instance for deterministic operations.

    Args:
        seed: Random seed. If None, uses numpy's current state.

    Returns:
        RandomState instance.
    """
    return check_random_state(seed)


