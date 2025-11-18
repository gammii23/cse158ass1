"""Assignment 1: Read, Category, and Rating Predictions - Consolidated Version."""

import argparse
import gzip
import hashlib
import logging
import os
import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

import numpy as np
import pandas as pd
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import bm25_weight
from scipy.special import expit
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.utils import check_random_state

# ============================================================================
# LOGGING & UTILITIES
# ============================================================================

def configure_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure logging."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger = logging.getLogger("assignment1")
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger

def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get logger instance."""
    if name:
        return logging.getLogger(f"assignment1.{name}")
    return logging.getLogger("assignment1")

def set_global_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    check_random_state(seed)

def get_random_state(seed: Optional[int] = None):
    """Get numpy RandomState instance."""
    return check_random_state(seed)

logger = get_logger()

# ============================================================================
# CONSTANTS
# ============================================================================

TRAIN_INTERACTIONS = "train_Interactions.csv.gz"
TRAIN_CATEGORY = "train_Category.json.gz"
TEST_CATEGORY = "test_Category.json.gz"
PAIRS_READ = "pairs_Read.csv"
PAIRS_CATEGORY = "pairs_Category.csv"
PAIRS_RATING = "pairs_Rating.csv"
PREDICTIONS_READ = "predictions_Read.csv"
PREDICTIONS_CATEGORY = "predictions_Category.csv"
PREDICTIONS_RATING = "predictions_Rating.csv"

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class Settings:
    """Configuration settings."""
    data_dir: str = "assignment1"
    output_dir: str = "outputs"
    cache_dir: str = "cache"
    random_seed: int = 42
    enable_caching: bool = True
    
    # Read prediction
    read_n_factors: int = 50
    read_n_iterations: int = 15
    read_regularization: float = 0.1
    read_popularity_weight: float = 0.35
    
    # Category classification
    category_max_features: int = 5000
    category_C: float = 1.0
    category_class_weight: Optional[str] = "balanced"
    
    # Rating regression
    rating_use_bias: bool = True
    rating_use_xgboost: bool = True
    rating_n_estimators: int = 100
    rating_max_depth: int = 6
    rating_learning_rate: float = 0.1

# ============================================================================
# DATA LOADING
# ============================================================================

def _compute_checksum(file_path: Path) -> str:
    """Compute MD5 checksum."""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def load_interactions(data_dir: str = "assignment1", filename: str = TRAIN_INTERACTIONS,
                     use_cache: bool = True, cache_dir: str = "cache") -> pd.DataFrame:
    """Load interactions from CSV.gz."""
    file_path = Path(data_dir) / filename
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    cache_path = Path(cache_dir) / f"{filename}.parquet"
    checksum_file = Path(cache_dir) / f"{filename}.checksum"
    
    if use_cache and cache_path.exists() and checksum_file.exists():
        current_checksum = _compute_checksum(file_path)
        cached_checksum = checksum_file.read_text().strip()
        if current_checksum == cached_checksum:
            logger.info(f"Loading cached interactions from {cache_path}")
            return pd.read_parquet(cache_path)
    
    logger.info(f"Loading interactions from {file_path}")
    df = pd.read_csv(file_path, compression="gzip", header=0,
                     dtype={"userID": str, "bookID": str, "rating": int})
    df = df.rename(columns={"userID": "user_id", "bookID": "item_id", "rating": "rating"})
    df = df.dropna()
    
    if use_cache:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)
        checksum_file.write_text(_compute_checksum(file_path))
    
    return df

def load_category_data(data_dir: str = "assignment1", filename: str = TRAIN_CATEGORY,
                      use_cache: bool = True, cache_dir: str = "cache") -> pd.DataFrame:
    """Load category data from JSON.gz."""
    file_path = Path(data_dir) / filename
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    cache_path = Path(cache_dir) / f"{filename}.parquet"
    checksum_file = Path(cache_dir) / f"{filename}.checksum"
    
    if use_cache and cache_path.exists() and checksum_file.exists():
        current_checksum = _compute_checksum(file_path)
        cached_checksum = checksum_file.read_text().strip()
        if current_checksum == cached_checksum:
            logger.info(f"Loading cached category data from {cache_path}")
            return pd.read_parquet(cache_path)
    
    logger.info(f"Loading category data from {file_path}")
    records: List[Dict[str, Any]] = []
    with gzip.open(file_path, "rt", encoding="utf-8") as f:
        for line in f:
            try:
                record = eval(line.strip())
                records.append(record)
            except (SyntaxError, ValueError):
                continue
    
    df = pd.DataFrame(records)
    df = df.rename(columns={
        "user_id": "user_id",
        "review_id": "review_id",
        "review_text": "review_text",
        "genre": "category",
    })
    df = df.dropna(subset=["review_text"])
    
    if use_cache:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path, index=False)
        checksum_file.write_text(_compute_checksum(file_path))
    
    return df

def load_pairs(data_dir: str = "assignment1", filename: str = PAIRS_READ) -> pd.DataFrame:
    """Load evaluation pairs."""
    file_path = Path(data_dir) / filename
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    logger.info(f"Loading pairs from {file_path}")
    df = pd.read_csv(file_path)
    df = df.rename(columns={
        "userID": "user_id",
        "user_id": "user_id",
        "bookID": "item_id",
        "itemID": "item_id",
        "item_id": "item_id",
        "reviewID": "review_id",
        "review_id": "review_id",
    })
    
    required_cols = ["user_id"]
    if "item_id" in df.columns:
        required_cols.append("item_id")
    elif "review_id" in df.columns:
        required_cols.append("review_id")
    
    df = df.dropna(subset=required_cols)
    return df

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

class InteractionFeatureBuilder:
    """Builder for interaction features."""
    
    def __init__(self, random_seed: Optional[int] = None):
        self.random_seed = random_seed
        self.global_mean: float = 0.0
        self.user_means: Dict[str, float] = {}
        self.item_means: Dict[str, float] = {}
        self.user_counts: Dict[str, int] = {}
        self.item_counts: Dict[str, int] = {}
        self.user_to_idx: Dict[str, int] = {}
        self.item_to_idx: Dict[str, int] = {}
        self._is_fitted = False
    
    def fit(self, df: pd.DataFrame) -> "InteractionFeatureBuilder":
        """Fit feature builder."""
        self.global_mean = df["rating"].mean()
        user_stats = df.groupby("user_id")["rating"].agg(["mean", "count"])
        self.user_means = user_stats["mean"].to_dict()
        self.user_counts = user_stats["count"].to_dict()
        item_stats = df.groupby("item_id")["rating"].agg(["mean", "count"])
        self.item_means = item_stats["mean"].to_dict()
        self.item_counts = item_stats["count"].to_dict()
        
        unique_users = df["user_id"].unique()
        unique_items = df["item_id"].unique()
        self.user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
        self.item_to_idx = {item: idx for idx, item in enumerate(unique_items)}
        self._is_fitted = True
        return self
    
    def build_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build interaction features."""
        if not self._is_fitted:
            raise ValueError("Must call fit() first")
        
        features = pd.DataFrame(index=df.index)
        features["user_mean_rating"] = df["user_id"].map(self.user_means).fillna(self.global_mean)
        features["user_interaction_count"] = df["user_id"].map(self.user_counts).fillna(0)
        features["user_bias"] = features["user_mean_rating"] - self.global_mean
        features["item_mean_rating"] = df["item_id"].map(self.item_means).fillna(self.global_mean)
        features["item_interaction_count"] = df["item_id"].map(self.item_counts).fillna(0)
        features["item_bias"] = features["item_mean_rating"] - self.global_mean
        features["item_popularity"] = features["item_interaction_count"] / max(self.item_counts.values()) if self.item_counts else 0.0
        features["global_mean"] = self.global_mean
        return features
    
    def build_sparse_matrix(self, df: pd.DataFrame) -> tuple[csr_matrix, Dict[str, int], Dict[str, int]]:
        """Build sparse user-item matrix."""
        if not self._is_fitted:
            raise ValueError("Must call fit() first")
        
        user_ids = df["user_id"].map(self.user_to_idx)
        item_ids = df["item_id"].map(self.item_to_idx)
        ratings = df["rating"].values
        n_users = len(self.user_to_idx)
        n_items = len(self.item_to_idx)
        matrix = csr_matrix((ratings, (user_ids, item_ids)), shape=(n_users, n_items))
        return matrix, self.user_to_idx, self.item_to_idx

def clean_text(text: str) -> str:
    """Clean text."""
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text

class TfidfFeaturizer:
    """TF-IDF featurizer."""
    
    def __init__(self, max_features: int = 5000, ngram_range: tuple[int, int] = (1, 2),
                 min_df: int = 2, max_df: float = 0.95, random_seed: Optional[int] = None):
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.vectorizer: Optional[TfidfVectorizer] = None
        self._is_fitted = False
    
    def fit(self, texts: List[str]) -> "TfidfFeaturizer":
        """Fit TF-IDF."""
        cleaned_texts = [clean_text(text) for text in texts]
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_df=self.max_df,
            lowercase=True,
            stop_words="english",
        )
        self.vectorizer.fit(cleaned_texts)
        self._is_fitted = True
        logger.info(f"Fitted TF-IDF with {len(self.vectorizer.vocabulary_)} features")
        return self
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """Transform texts to features."""
        if not self._is_fitted or self.vectorizer is None:
            raise ValueError("Must call fit() first")
        cleaned_texts = [clean_text(text) for text in texts]
        return self.vectorizer.transform(cleaned_texts).toarray()

# ============================================================================
# MODELS
# ============================================================================

class ReadPredictor:
    """Read prediction model."""
    
    def __init__(self, use_implicit: bool = True, n_factors: int = 50,
                 n_iterations: int = 15, regularization: float = 0.1,
                 popularity_weight: float = 0.35, random_seed: Optional[int] = None):
        self.use_implicit = use_implicit
        self.n_factors = n_factors
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.popularity_weight = popularity_weight
        self.random_seed = random_seed
        self.implicit_model: Optional[AlternatingLeastSquares] = None
        self.user_to_idx: Dict[str, int] = {}
        self.item_to_idx: Dict[str, int] = {}
        self.item_popularity: Optional[pd.Series] = None
        self._is_fitted = False
    
    def fit(self, interaction_matrix: Any, user_to_idx: Dict[str, int],
            item_to_idx: Dict[str, int], item_popularity: pd.Series) -> "ReadPredictor":
        """Fit model."""
        self.user_to_idx = user_to_idx
        self.item_to_idx = item_to_idx
        self.item_popularity = item_popularity
        
        if self.use_implicit:
            weighted = bm25_weight(interaction_matrix).tocsr()
            self.implicit_model = AlternatingLeastSquares(
                factors=self.n_factors,
                iterations=self.n_iterations,
                regularization=self.regularization,
                random_state=self.random_seed,
            )
            self.implicit_model.fit(weighted)
        
        self._is_fitted = True
        return self
    
    def predict_proba_pairs(self, user_ids: pd.Series, item_ids: pd.Series) -> np.ndarray:
        """Predict probabilities."""
        if not self._is_fitted:
            raise ValueError("Must call fit() first")
        
        popularity_scores = (
            item_ids.map(self.item_popularity).fillna(0.0).to_numpy()
            if self.item_popularity is not None
            else np.zeros(len(item_ids))
        )
        
        implicit_scores = np.zeros(len(user_ids))
        if self.implicit_model is not None and self.user_to_idx and self.item_to_idx:
            user_idx = user_ids.map(self.user_to_idx)
            item_idx = item_ids.map(self.item_to_idx)
            valid_mask = user_idx.notna() & item_idx.notna()
            if valid_mask.any():
                u_factors = self.implicit_model.user_factors[user_idx[valid_mask].astype(int)]
                i_factors = self.implicit_model.item_factors[item_idx[valid_mask].astype(int)]
                implicit_scores[valid_mask.to_numpy()] = np.sum(u_factors * i_factors, axis=1)
        
        implicit_probs = expit(implicit_scores)
        if popularity_scores.max() > 0:
            popularity_norm = popularity_scores / popularity_scores.max()
        else:
            popularity_norm = popularity_scores
        
        blended = (1 - self.popularity_weight) * implicit_probs + self.popularity_weight * popularity_norm
        return np.clip(blended, 0.0, 1.0)

class CategoryClassifier:
    """Category classification model."""
    
    def __init__(self, featurizer: TfidfFeaturizer, C: float = 1.0,
                 class_weight: Optional[str] = "balanced", random_seed: Optional[int] = None):
        self.featurizer = featurizer
        self.C = C
        self.class_weight = class_weight
        self.random_seed = random_seed
        self.random_state = get_random_state(random_seed)
        self.model: Optional[LogisticRegression] = None
        self._is_fitted = False
    
    def fit(self, X_text: List[str], y: np.ndarray) -> "CategoryClassifier":
        """Fit classifier."""
        X_features = self.featurizer.transform(X_text)
        self.model = LogisticRegression(
            C=self.C,
            class_weight=self.class_weight,
            random_state=self.random_state,
            max_iter=1000,
            multi_class="multinomial",
            solver="lbfgs",
        )
        self.model.fit(X_features, y)
        self._is_fitted = True
        logger.info("Category classifier fitted successfully")
        return self
    
    def predict(self, X_text: List[str]) -> np.ndarray:
        """Predict categories."""
        if not self._is_fitted or self.model is None:
            raise ValueError("Must call fit() first")
        X_features = self.featurizer.transform(X_text)
        return self.model.predict(X_features)

class RatingRegressor:
    """Rating regression model."""
    
    def __init__(self, use_bias: bool = True, use_xgboost: bool = True,
                 n_estimators: int = 100, max_depth: int = 6,
                 learning_rate: float = 0.1, random_seed: Optional[int] = None):
        self.use_bias = use_bias
        self.use_xgboost = use_xgboost
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_seed = random_seed
        self.random_state = get_random_state(random_seed)
        self.global_mean: float = 0.0
        self.user_biases: Dict[str, float] = {}
        self.item_biases: Dict[str, float] = {}
        self.model: Optional[Any] = None
        self._is_fitted = False
    
    def fit(self, X_features: pd.DataFrame, y: np.ndarray,
            user_ids: Optional[pd.Series] = None, item_ids: Optional[pd.Series] = None) -> "RatingRegressor":
        """Fit regressor."""
        self.global_mean = float(np.mean(y))
        
        if self.use_bias and user_ids is not None and item_ids is not None:
            df = pd.DataFrame({"user_id": user_ids, "item_id": item_ids, "rating": y})
            user_means = df.groupby("user_id")["rating"].mean() - self.global_mean
            self.user_biases = user_means.to_dict()
            item_means = df.groupby("item_id")["rating"].mean() - self.global_mean
            self.item_biases = item_means.to_dict()
            logger.info(f"Computed biases: {len(self.user_biases)} users, {len(self.item_biases)} items")
        
        if self.use_xgboost:
            try:
                import xgboost as xgb
                self.model = xgb.XGBRegressor(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    learning_rate=self.learning_rate,
                    random_state=self.random_state,
                )
                logger.info("Using XGBoost regressor")
            except ImportError:
                self.model = GradientBoostingRegressor(
                    n_estimators=self.n_estimators,
                    max_depth=self.max_depth,
                    learning_rate=self.learning_rate,
                    random_state=self.random_state,
                )
        else:
            self.model = GradientBoostingRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=self.random_state,
            )
        
        self.model.fit(X_features, y)
        self._is_fitted = True
        logger.info("Rating regressor fitted successfully")
        return self
    
    def predict(self, X_features: pd.DataFrame, user_ids: Optional[pd.Series] = None,
                item_ids: Optional[pd.Series] = None) -> np.ndarray:
        """Predict ratings."""
        if not self._is_fitted or self.model is None:
            raise ValueError("Must call fit() first")
        
        predictions = self.model.predict(X_features)
        
        if self.use_bias and user_ids is not None and item_ids is not None:
            user_bias = user_ids.map(self.user_biases).fillna(0.0).values
            item_bias = item_ids.map(self.item_biases).fillna(0.0).values
            predictions = predictions + user_bias + item_bias
        
        return np.clip(predictions, 1.0, 5.0)

# ============================================================================
# WORKFLOWS
# ============================================================================

def run_read_workflow(settings: Settings, data_dir: str = "assignment1") -> Dict[str, Any]:
    """Run read prediction workflow."""
    try:
        logger.info("Starting read prediction workflow")
        set_global_seed(settings.random_seed)
        
        train_df = load_interactions(data_dir=data_dir, filename=TRAIN_INTERACTIONS,
                                    use_cache=settings.enable_caching, cache_dir=settings.cache_dir)
        pairs_df = load_pairs(data_dir=data_dir, filename=PAIRS_READ)
        
        feature_builder = InteractionFeatureBuilder(random_seed=settings.random_seed)
        feature_builder.fit(train_df)
        interaction_matrix, user_to_idx, item_to_idx = feature_builder.build_sparse_matrix(train_df)
        
        item_counts = train_df["item_id"].value_counts()
        total_reads = len(train_df)
        item_popularity = item_counts / total_reads
        
        predictor = ReadPredictor(
            use_implicit=True,
            n_factors=settings.read_n_factors,
            n_iterations=settings.read_n_iterations,
            regularization=settings.read_regularization,
            popularity_weight=settings.read_popularity_weight,
            random_seed=settings.random_seed,
        )
        predictor.fit(interaction_matrix, user_to_idx, item_to_idx, item_popularity)
        
        predictions = predictor.predict_proba_pairs(pairs_df["user_id"], pairs_df["item_id"])
        
        submission_df = pd.DataFrame({
            "user_id": pairs_df["user_id"],
            "item_id": pairs_df["item_id"],
            "prediction": predictions,
        })
        
        logger.info(f"Read workflow completed: {len(submission_df)} predictions")
        return {"success": True, "data": submission_df}
    except Exception as e:
        logger.error(f"Read workflow failed: {e}", exc_info=True)
        return {"success": False, "error": str(e)}

def run_category_workflow(settings: Settings, data_dir: str = "assignment1") -> Dict[str, Any]:
    """Run category classification workflow."""
    try:
        logger.info("Starting category classification workflow")
        set_global_seed(settings.random_seed)
        
        train_df = load_category_data(data_dir=data_dir, filename=TRAIN_CATEGORY,
                                     use_cache=settings.enable_caching, cache_dir=settings.cache_dir)
        test_df = load_category_data(data_dir=data_dir, filename=TEST_CATEGORY,
                                    use_cache=settings.enable_caching, cache_dir=settings.cache_dir)
        pairs_df = load_pairs(data_dir=data_dir, filename=PAIRS_CATEGORY)
        
        pairs_df = pairs_df.merge(test_df[["review_id", "review_text"]], on="review_id", how="left")
        
        featurizer = TfidfFeaturizer(
            max_features=settings.category_max_features,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
        )
        featurizer.fit(train_df["review_text"].tolist())
        
        unique_categories = sorted(train_df["category"].unique())
        category_to_int = {cat: idx for idx, cat in enumerate(unique_categories)}
        train_labels = train_df["category"].map(category_to_int).values
        
        classifier = CategoryClassifier(
            featurizer=featurizer,
            C=settings.category_C,
            class_weight=settings.category_class_weight,
            random_seed=settings.random_seed,
        )
        classifier.fit(train_df["review_text"].tolist(), train_labels)
        
        pair_texts = pairs_df["review_text"].fillna("").tolist()
        predictions = classifier.predict(pair_texts)
        
        submission_df = pd.DataFrame({
            "userID": pairs_df["user_id"],
            "reviewID": pairs_df["review_id"],
            "prediction": predictions,
        })
        
        logger.info(f"Category workflow completed: {len(submission_df)} predictions")
        return {"success": True, "data": submission_df}
    except Exception as e:
        logger.error(f"Category workflow failed: {e}", exc_info=True)
        return {"success": False, "error": str(e)}

def run_rating_workflow(settings: Settings, data_dir: str = "assignment1") -> Dict[str, Any]:
    """Run rating regression workflow."""
    try:
        logger.info("Starting rating regression workflow")
        set_global_seed(settings.random_seed)
        
        train_df = load_interactions(data_dir=data_dir, filename=TRAIN_INTERACTIONS,
                                    use_cache=settings.enable_caching, cache_dir=settings.cache_dir)
        pairs_df = load_pairs(data_dir=data_dir, filename=PAIRS_RATING)
        
        feature_builder = InteractionFeatureBuilder(random_seed=settings.random_seed)
        feature_builder.fit(train_df)
        
        train_features = feature_builder.build_interaction_features(train_df)
        train_labels = train_df["rating"].values
        
        regressor = RatingRegressor(
            use_bias=settings.rating_use_bias,
            use_xgboost=settings.rating_use_xgboost,
            n_estimators=settings.rating_n_estimators,
            max_depth=settings.rating_max_depth,
            learning_rate=settings.rating_learning_rate,
            random_seed=settings.random_seed,
        )
        regressor.fit(train_features, train_labels,
                     user_ids=train_df["user_id"], item_ids=train_df["item_id"])
        
        pair_features = feature_builder.build_interaction_features(pairs_df)
        predictions = regressor.predict(pair_features,
                                       user_ids=pairs_df["user_id"], item_ids=pairs_df["item_id"])
        
        submission_df = pd.DataFrame({
            "user_id": pairs_df["user_id"],
            "item_id": pairs_df["item_id"],
            "prediction": predictions,
        })
        
        logger.info(f"Rating workflow completed: {len(submission_df)} predictions")
        return {"success": True, "data": submission_df}
    except Exception as e:
        logger.error(f"Rating workflow failed: {e}", exc_info=True)
        return {"success": False, "error": str(e)}

# ============================================================================
# OUTPUT SERIALIZATION
# ============================================================================

def write_predictions(df: pd.DataFrame, task: str, output_dir: str = "outputs") -> Path:
    """Write predictions to CSV."""
    filename_map = {
        "read": PREDICTIONS_READ,
        "category": PREDICTIONS_CATEGORY,
        "rating": PREDICTIONS_RATING,
    }
    
    if task not in filename_map:
        raise ValueError(f"Unknown task: {task}")
    
    filename = filename_map[task]
    output_path = Path(output_dir) / filename
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Rename columns and format predictions
    if task == "read":
        df = df.rename(columns={"user_id": "userID", "item_id": "bookID"})
        df["prediction"] = (df["prediction"] > 0.5).astype(int)
        df = df[["userID", "bookID", "prediction"]]
    elif task == "rating":
        df = df.rename(columns={"user_id": "userID", "item_id": "bookID"})
        df = df[["userID", "bookID", "prediction"]]
    elif task == "category":
        df["prediction"] = df["prediction"].astype(int)
        df = df[["userID", "reviewID", "prediction"]]
    
    df.to_csv(output_path, index=False)
    logger.info(f"Successfully wrote {len(df)} predictions for {task} task")
    return output_path

# ============================================================================
# MAIN CLI
# ============================================================================

def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Assignment 1: Read, Category, and Rating Predictions")
    parser.add_argument("--task", type=str, choices=["read", "category", "rating", "all"], required=True)
    parser.add_argument("--data-dir", type=str, default="assignment1", help="Directory containing data files")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Directory for output predictions")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--no-cache", action="store_true", help="Disable caching")
    parser.add_argument("--log-level", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    
    args = parser.parse_args()
    
    log_level_map = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40}
    configure_logging(level=log_level_map[args.log_level])
    
    settings = Settings()
    if args.seed is not None:
        settings.random_seed = args.seed
    if args.no_cache:
        settings.enable_caching = False
    settings.data_dir = args.data_dir
    settings.output_dir = args.output_dir
    
    tasks_to_run = ["read", "category", "rating"] if args.task == "all" else [args.task]
    
    success_count = 0
    for task in tasks_to_run:
        logger.info(f"Executing {task} task")
        try:
            if task == "read":
                result = run_read_workflow(settings, data_dir=args.data_dir)
            elif task == "category":
                result = run_category_workflow(settings, data_dir=args.data_dir)
            elif task == "rating":
                result = run_rating_workflow(settings, data_dir=args.data_dir)
            else:
                logger.error(f"Unknown task: {task}")
                continue
            
            if result["success"]:
                write_predictions(result["data"], task=task, output_dir=args.output_dir)
                logger.info(f"{task} task completed successfully")
                success_count += 1
            else:
                logger.error(f"{task} task failed: {result.get('error', 'Unknown error')}")
        except Exception as e:
            logger.error(f"{task} task failed with exception: {e}", exc_info=True)
    
    if success_count == len(tasks_to_run):
        logger.info("All tasks completed successfully")
        return 0
    else:
        logger.error(f"Only {success_count}/{len(tasks_to_run)} tasks succeeded")
        return 1

if __name__ == "__main__":
    sys.exit(main())
