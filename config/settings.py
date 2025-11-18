"""Configuration settings with dependency injection support.

This module provides dataclasses for paths, hyperparameters, and
other configuration that can be overridden via environment variables
or CLI arguments.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class PathsConfig:
    """Configuration for file paths."""

    data_dir: str = field(default_factory=lambda: os.getenv("DATA_DIR", "data"))
    output_dir: str = field(
        default_factory=lambda: os.getenv("OUTPUT_DIR", "outputs")
    )
    cache_dir: str = field(
        default_factory=lambda: os.getenv("CACHE_DIR", "cache")
    )
    results_dir: str = field(
        default_factory=lambda: os.getenv("RESULTS_DIR", "results")
    )

    def __post_init__(self) -> None:
        """Create directories if they don't exist."""
        for dir_path in [
            self.data_dir,
            self.output_dir,
            self.cache_dir,
            self.results_dir,
        ]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)


@dataclass
class ReadConfig:
    """Hyperparameters for read prediction task."""

    # Model selection
    use_lightfm: bool = False
    use_implicit: bool = True

    # Logistic regression params
    lr_C: float = 1.0
    lr_class_weight: Optional[str] = "balanced"

    # Matrix factorization params
    n_factors: int = 50
    n_iterations: int = 10
    learning_rate: float = 0.01
    als_regularization: float = 0.1
    popularity_weight: float = 0.35

    # Calibration
    calibrate_probs: bool = True


@dataclass
class CategoryConfig:
    """Hyperparameters for category classification task."""

    # Text processing
    max_features: int = 5000  # Reduced to avoid memory issues
    ngram_range: tuple[int, int] = (1, 2)
    min_df: int = 2
    max_df: float = 0.95

    # Model params
    C: float = 1.0
    class_weight: Optional[str] = "balanced"
    use_svm: bool = False

    # Embeddings (optional)
    use_embeddings: bool = False
    embedding_model: str = "all-MiniLM-L6-v2"


@dataclass
class RatingConfig:
    """Hyperparameters for rating regression task."""

    # Baseline bias
    use_bias: bool = True

    # Model selection
    use_xgboost: bool = True
    use_ridge: bool = False

    # XGBoost params
    xgb_n_estimators: int = 100
    xgb_max_depth: int = 6
    xgb_learning_rate: float = 0.1

    # Ridge params
    ridge_alpha: float = 1.0

    # Stacking
    use_stacking: bool = False


@dataclass
class ExperimentConfig:
    """Configuration for experiments and validation."""

    random_seed: int = field(
        default_factory=lambda: int(os.getenv("RANDOM_SEED", "42"))
    )
    n_folds: int = 5
    test_size: float = 0.2
    enable_caching: bool = field(
        default_factory=lambda: os.getenv("ENABLE_CACHE", "true").lower()
        == "true"
    )


@dataclass
class Settings:
    """Main settings container with all configurations."""

    paths: PathsConfig = field(default_factory=PathsConfig)
    read: ReadConfig = field(default_factory=ReadConfig)
    category: CategoryConfig = field(default_factory=CategoryConfig)
    rating: RatingConfig = field(default_factory=RatingConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)

    @classmethod
    def from_env(cls) -> "Settings":
        """Create settings from environment variables.

        Returns:
            Settings instance with values from environment.
        """
        return cls()

