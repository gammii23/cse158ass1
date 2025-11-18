# Assignment 1 File List (Implementation Order)

## Status: ✅ **PROJECT COMPLETE** (All Sections 1-15 Implemented)
**All core architecture, testing, validation, documentation, and packaging files created**

---

## Section 1: Environment Bootstrap

1. **requirements.txt**
   - Frozen dependencies from virtual environment
   - All packages pinned for reproducibility

2. **utils/__init__.py**
   - Package initialization

3. **utils/logging.py**
   - Logging wrapper with stdout formatter
   - Sentry placeholder integration
   - Centralized logger configuration

4. **utils/random_seed.py**
   - Global random seed utility
   - Sets seeds for random, numpy, sklearn
   - Reproducibility support

---

## Section 2: Project Scaffolding

5. **constants.py**
   - Enumerations for file names, column names, genre labels
   - Prevents magic strings throughout codebase
   - TaskType enum for CLI routing

6. **config/__init__.py**
   - Package initialization

7. **config/settings.py**
   - Dataclasses for paths, hyperparameters
   - Dependency injection support
   - Environment variable overrides
   - Settings container: PathsConfig, ReadConfig, CategoryConfig, RatingConfig, ExperimentConfig

8. **data_access/__init__.py**
   - Package initialization

9. **features/__init__.py**
   - Package initialization

10. **models/__init__.py**
    - Package initialization

11. **pipelines/__init__.py**
    - Package initialization

12. **outputs/__init__.py**
    - Package initialization

13. **tests/__init__.py**
    - Package initialization (ready for test implementation)

---

## Section 3: Data Ingestion & Auditing

14. **data_access/loader.py**
    - Pure data ingestion functions
    - Load interactions from CSV.gz with schema validation
    - Load category data from JSON.gz lines
    - Load evaluation pairs from CSV
    - Parquet caching with checksum validation
    - Missing value policies
    - Standardized column name mapping

---

## Section 4: Feature Engineering

15. **features/interactions.py**
    - InteractionFeatureBuilder class
    - User/item frequency statistics
    - Global means, user/book biases
    - Popularity/recency features
    - Matrix factorization inputs (sparse matrices, ID mappings)
    - Deterministic fit/transform API

16. **features/reviews.py**
    - Text cleaning utilities
    - TfidfFeaturizer class with DI support
    - EmbeddingFeaturizer class (SentenceTransformer adapter)
    - TextFeaturizer protocol for Liskov substitution
    - Deterministic fit/transform API

---

## Section 5: Models

17. **models/read_predictor.py**
    - ReadPredictor class
    - Logistic regression baseline with class balancing
    - Probability calibration (isotonic)
    - Optional implicit matrix factorization support
    - fit() and predict_proba() interface

18. **models/category_classifier.py**
    - CategoryClassifier class
    - Dependency-injected featurizer (Liskov substitution)
    - Multinomial logistic regression or LinearSVC
    - Class weighting support
    - fit() and predict() interface

19. **models/rating_regressor.py**
    - RatingRegressor class
    - User/item bias terms
    - XGBoost or Ridge regression
    - Gradient boosting fallback
    - Rating clipping to [1, 5]
    - fit() and predict() interface

---

## Section 6: Pipelines (Return SubmissionResult)

20. **pipelines/types.py**
    - SubmissionResult TypedDict definition
    - Standardized pipeline return contract
    - { success: bool, data?: DataFrame, error?: str }

21. **pipelines/read_workflow.py**
    - run_read_workflow() orchestrator
    - Loads interactions and pairs
    - Builds features, trains model
    - Generates predictions for pairs
    - Returns SubmissionResult

22. **pipelines/category_workflow.py**
    - run_category_workflow() orchestrator
    - Loads train/test category data and pairs
    - Fits featurizer + classifier
    - Predicts classes for pairs
    - Returns SubmissionResult

23. **pipelines/rating_workflow.py**
    - run_rating_workflow() orchestrator
    - Loads interactions and pairs
    - Trains regressor with biases
    - Predicts ratings for pairs
    - Returns SubmissionResult

---

## Section 7: Output Serialization

24. **outputs/serializer.py**
    - write_predictions() function
    - Schema validation per task
    - Validates columns, dtypes, NaN values
    - Validates prediction ranges
    - Safe overwrite rules
    - Directory creation

---

## Section 9: CLI Integration

25. **assignment1.py**
    - Main CLI entry point
    - Argument parser: --task, --data-dir, --output-dir, --seed, --no-cache, --log-level
    - Routes to respective workflows
    - Exception handling
    - Calls serializer on success
    - Exit codes for success/failure

---

## Section 8: Validation Strategy & Experiment Tracking

29. **utils/validation.py**
    - Cross-validation infrastructure
    - StratifiedKFoldSplitter for balanced class distribution
    - TimeAwareSplitter for temporal data
    - RandomSplitter with seed control
    - create_splits() unified interface

30. **utils/metrics.py**
    - Evaluation metrics for all tasks
    - balanced_accuracy_score() for read prediction
    - accuracy_score_wrapper() for category
    - mse_score() for rating regression
    - compute_metrics() unified interface

31. **utils/experiment_logger.py**
    - ExperimentLogger class
    - Logs experiments to results/experiments.csv
    - Tracks hyperparameters, CV scores, timestamps
    - get_best_experiment() for hyperparameter selection

32. **scripts/ablation_studies.py**
    - Feature ablation experiments
    - Hyperparameter grid search
    - Model comparison utilities
    - Logs results to experiments.csv

---

## Section 9: CLI Integration

(Already listed above as item 25)

---

## Section 10: Submission Generation & Verification

33. **scripts/verify_submission.py**
    - Format validation against baseline outputs
    - Header and column order comparison
    - Row count verification
    - Schema validation per task
    - Value range checks

34. **scripts/spot_check.py**
    - Random sampling of prediction files
    - Value range verification
    - Data type checks
    - Sample row printing for manual inspection

---

## Section 11: Testing & Quality Assurance

35. **pytest.ini**
    - Test discovery configuration
    - Coverage settings (80% target)
    - Test markers (unit, integration, slow)

36. **.mypy.ini**
    - Type checking configuration
    - Strict mode settings
    - Ignore missing imports for optional deps

37. **.flake8**
    - Linting configuration
    - Max line length: 100
    - Exclude patterns for cache/venv

38. **tests/test_loaders.py**
    - Data loader unit tests
    - Missing file handling
    - Schema validation tests
    - Caching functionality tests
    - Malformed data handling

39. **tests/test_interactions_features.py**
    - InteractionFeatureBuilder tests
    - Deterministic output verification
    - User/item feature transformation tests
    - Sparse matrix shape validation

40. **tests/test_reviews_features.py**
    - Text cleaning utilities tests
    - TF-IDF vectorizer tests
    - Embedding featurizer tests (if available)
    - Deterministic feature generation

41. **tests/test_read_predictor.py**
    - ReadPredictor fit/predict_proba tests
    - Probability calibration tests
    - Binary prediction tests

42. **tests/test_category_classifier.py**
    - CategoryClassifier fit/predict tests
    - Dependency injection tests
    - Class weight handling tests

43. **tests/test_rating_regressor.py**
    - RatingRegressor fit/predict tests
    - Bias term computation tests
    - Rating clipping validation

44. **tests/test_pipelines.py**
    - Pipeline workflow tests
    - SubmissionResult contract tests
    - Error handling verification

45. **tests/test_integration.py**
    - End-to-end pipeline tests
    - Sample data integration tests
    - Output schema validation

---

## Section 12: Performance & Caching

46. **scripts/profile_performance.py**
    - Runtime profiling for each task
    - Memory usage tracking
    - Performance metrics logging to CSV
    - Time budget enforcement

---

## Section 13: Documentation

47. **writeup.txt**
    - Complete writeup template
    - All required sections: data, features, models, validation, results
    - Key decisions and trade-offs
    - Reproduction instructions

48. **scripts/generate_writeup_sections.py**
    - Auto-fills metrics from experiments.csv
    - Generates results section markdown
    - Updates writeup with best CV scores

---

## Section 14: Risk Controls

49. **utils/data_leakage_guard.py**
    - validate_no_test_leakage() function
    - audit_joins() for feature validation
    - strict_split() for train/val separation
    - Prevents data leakage in pipelines

---

## Section 15: Final Packaging & Handoff

50. **scripts/validate_submission.py**
    - Comprehensive artifact validation
    - File size checks
    - Row count verification
    - Schema validation
    - Spot-check sampling

51. **SUBMISSION_CHECKLIST.md**
    - Required files checklist
    - Verification steps
    - Pre-submission commands
    - Submission instructions

---

## Documentation & Planning Files

52. **master plan.md**
    - Detailed ordered task list
    - All 15 sections with subtasks

53. **docs/assignment1-detailed-plan.md**
    - Original detailed execution plan
    - Requirements, architecture, workflow, testing, timeline, risks

54. **reports/misc-reports-2025-11-18.md**
    - Task execution log
    - Batched report entries

55. **filelist.md**
    - This file: comprehensive file inventory

---

## Generated Files (Created at Runtime)

These files are generated when running the pipelines:

- `results/experiments.csv` - Experiment logs (generated by ExperimentLogger)
- `results/performance_log.csv` - Performance metrics (generated by profile_performance.py)
- `outputs/predictions_Read.csv` - Read predictions (generated by CLI)
- `outputs/predictions_Category.csv` - Category predictions (generated by CLI)
- `outputs/predictions_Rating.csv` - Rating predictions (generated by CLI)
- `cache/*.parquet` - Cached preprocessed data (generated by loaders)

---

## Execution Order Summary

1. **Setup**: requirements.txt, utils/ (logging, random_seed)
2. **Structure**: constants.py, config/, package __init__.py files
3. **Data**: data_access/loader.py
4. **Features**: features/interactions.py, features/reviews.py
5. **Models**: models/read_predictor.py, category_classifier.py, rating_regressor.py
6. **Pipelines**: pipelines/types.py, read_workflow.py, category_workflow.py, rating_workflow.py
7. **Output**: outputs/serializer.py
8. **CLI**: assignment1.py
9. **Validation**: utils/validation.py, utils/metrics.py, utils/experiment_logger.py
10. **Testing**: pytest.ini, .mypy.ini, .flake8, tests/*.py
11. **Scripts**: scripts/ablation_studies.py, scripts/verify_submission.py, scripts/spot_check.py, scripts/profile_performance.py, scripts/generate_writeup_sections.py, scripts/validate_submission.py
12. **Documentation**: writeup.txt, SUBMISSION_CHECKLIST.md
13. **Risk Controls**: utils/data_leakage_guard.py

---

## Quick Start Commands

```bash
# Activate virtual environment
.venv\Scripts\activate  # Windows
# or
source .venv/bin/activate  # Linux/Mac

# Run individual tasks
python assignment1.py --task read
python assignment1.py --task category
python assignment1.py --task rating

# Run all tasks
python assignment1.py --task all

# With custom settings
python assignment1.py --task read --seed 42 --output-dir outputs --log-level DEBUG
```

---

## Next Steps (Execution & Testing)

All code is implemented. Next steps are to execute and test:

1. **Run Tests**: Execute test suite to verify functionality
   ```bash
   pytest tests/ --cov=. --cov-report=html
   ```

2. **Run Static Checks**: Verify code quality
   ```bash
   mypy . --config-file .mypy.ini
   flake8 . --config .flake8
   ```

3. **Generate Predictions**: Run pipelines to create submission files
   ```bash
   python assignment1.py --task all --seed 42
   ```

4. **Validate Submissions**: Verify prediction files
   ```bash
   python scripts/validate_submission.py
   python scripts/spot_check.py
   ```

5. **Run Experiments**: Execute ablation studies and log results
   ```bash
   python scripts/ablation_studies.py --task all
   ```

6. **Update Writeup**: Generate results section from experiments
   ```bash
   python scripts/generate_writeup_sections.py
   ```

7. **Final Validation**: Complete submission checklist
   ```bash
   python scripts/validate_submission.py
   # Review SUBMISSION_CHECKLIST.md
   ```

## Total File Count

- **55 source files** created (Python modules, configs, tests, scripts, docs)
- **5 generated files** (created at runtime: predictions, experiments, cache)
- **Total: 60 files** in project

