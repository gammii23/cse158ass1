# Assignment 1 Detailed Execution Plan

## 1. Requirements & Deliverables Synthesis
- **Submission package**: `writeup.txt`, `assignment1.py`, plus `predictions_Read.csv`, `predictions_Category.csv`, `predictions_Rating.csv`; each script section must map directly to the related Gradescope task so manual graders can trace logic.
- **Datasets in scope**:
  - `train_Interactions.csv.gz`: 200k implicit/explicit ratings for read and rating predictions.
  - `train_Category.json.gz` and `test_Category.json.gz`: review texts and metadata for multi-class genre prediction.
  - `pairs_Read.csv`, `pairs_Category.csv`, `pairs_Rating.csv`: evaluation pairs for generating leaderboard submissions.
  - `baselines.py`: reference for file formats and sanity checks.
- **Evaluation metrics**:
  - Read prediction → balanced accuracy (simple fraction correct on 50/50 split).
  - Category prediction → multi-class accuracy across five genres.
  - Rating prediction → mean-squared error (MSE) on hidden test targets.
- **Grading levers**: outperform provided baselines on hidden test data, maintain strong leaderboard ranks, and document strategies succinctly in `writeup.txt`.

## 2. Architecture & Module Boundaries (SOLID/DRY Enforcement)
- **Core layout**:
  - `data_access/loader.py`: pure data ingestion functions returning typed `pandas` DataFrames or typed review objects; handles compression, schema validation, and missing-value policies.
  - `features/`:
    - `interactions.py`: shared user/book feature builders (counts, TF-IDF over tags if available, collaborative statistics).
    - `reviews.py`: text preprocessing utilities (tokenization, TF-IDF vectorizers, embedding loaders) with dependency-injected vectorizers.
  - `models/`:
    - `read_predictor.py`: encapsulates hybrid model (matrix-factorization + popularity prior); exposes `fit`, `predict_proba`.
    - `category_classifier.py`: wraps fine-tuned text model (e.g., logistic regression on TF-IDF or lightweight transformer); accepts injected featurizer for testing substitution (Liskov).
    - `rating_regressor.py`: combines baseline bias terms with gradient boosted trees or factorization model; returns MSE-ready predictions.
  - `pipelines/`:
    - `read_workflow.py`, `category_workflow.py`, `rating_workflow.py`: orchestrators composing data loaders, feature builders, and models; each returns `{ success: bool; data?: SubmissionFrame; error?: str }`.
  - `outputs/serializer.py`: single responsibility for writing CSV predictions with schema validation (prevents duplication).
- **Dependency inversion**: inject configuration (paths, hyperparameters) via `config/settings.py` and pass services down through constructors rather than hardcoding.
- **Shared constants & enums**:
  - `constants.py`: enumerations for file names, column names, genre labels to prevent magic strings.
- **Testing hooks**: each public function documented with JSDoc-style comments and exposes deterministic behaviour for unit tests.

## 3. Step-by-Step Implementation Workflow
1. **Environment bootstrap**
   - Create virtual environment, lock dependencies (`pandas`, `numpy`, `scikit-learn`, `lightfm` or `implicit`, `xgboost`, `nltk`, `scipy`).
   - Implement logging wrapper (stdout + Sentry placeholder) for consistent error tracing.
2. **Data ingestion & auditing**
   - Implement loaders with schema assertions and exploratory notebooks to profile rating distributions, sparsity, review lengths.
   - Generate cached parquet versions for faster iteration; record checksum to detect stale caches.
3. **Feature engineering**
   - Read task: compute user/book frequency stats, temporal splits if needed, and matrix factorization inputs; design negative sampling strategy for additional implicit data augmentation.
   - Category task: build text pipeline (cleaning, stop-word removal, lemmatization), evaluate TF-IDF vs. pretrained embeddings (e.g., `SentenceTransformer` distilled for speed), and create vote/rating numeric features.
   - Rating task: derive user/book bias, interaction counts, similar user clusters, and optionally metadata embeddings.
4. **Model development**
   - Read predictor: start with logistic regression on collaborative features, iterate to factorization machine or LightFM; calibrate probability outputs.
   - Category classifier: train multinomial logistic regression baseline, upgrade to fine-tuned lightweight transformer if compute allows; incorporate class-weighting based on training distribution.
   - Rating regressor: baseline global mean + user/book offsets, then blend with gradient boosted model using cross-validation; optionally integrate matrix factorization predictions via stacking.
5. **Validation strategy**
   - Implement time-aware or stratified K-fold splits; ensure each pipeline supports custom split objects.
   - Track metrics in centralized `results/experiments.csv` with timestamp, params, validation scores, and leaderboard submission IDs.
6. **Submission generation**
   - Build CLI entry points `python assignment1.py --task read|category|rating` to execute corresponding workflow and produce CSV outputs with schema checks.
   - Verify predictions match leaderboard formatting by diffing against baseline headers.
7. **Documentation & packaging**
   - Auto-generate sections of `writeup.txt` from experiment logs; manually polish narrative describing final models, features, hyperparameters, and lessons learned.

## 4. Testing & Quality Assurance Strategy
- **Unit tests** (`tests/` directory):
  - Validate data loaders handle missing columns and compression edge cases.
  - Assert feature builders produce deterministic outputs for fixed seeds.
  - Mock model interfaces to confirm pipelines propagate success/error objects without throwing.
- **Integration tests**:
  - Smoke test each workflow end-to-end on sampled subset (e.g., 1k interactions) verifying output schema and metric computation.
  - Regression tests to compare new predictions vs. cached baseline metrics to detect degradation.
- **Performance checks**:
  - Measure runtime and memory footprint; enforce time budget per task (e.g., <10 minutes on lab machine) using profiler logs.
  - Add optional caching layer toggled via environment variable for repeated experiments.
- **Report validation**:
  - Lint `writeup.txt` for forbidden formatting (must be plain text).
  - Ensure `assignment1.py` passes static typing (`mypy`) and linting (`flake8`), hitting 80% coverage with `pytest --cov`.

## 5. Timeline, Milestones, and Risk Mitigation
- **Week 1 (Days 1-3)**: Environment setup, baseline reproduction, data profiling. Deliverable: cached datasets, initial metrics.
- **Week 1 (Days 4-7)**: Feature engineering prototypes for all tasks; unit tests for loaders/features. Risk: sparse data → Mitigate by clipping to active users/books.
- **Week 2 (Days 8-10)**: Train refined models (read + category). Conduct cross-validation, log experiments. Risk: overfitting text models → employ early stopping and dropout.
- **Week 2 (Days 11-14)**: Develop rating regressor ensemble, run hyperparameter sweeps. Risk: long training time → parallelize using randomized search with resource caps.
- **Week 3 (Days 15-17)**: Integrate pipelines, finalize CLI, run integration tests. Prepare draft predictions and submit to leaderboard for sanity.
- **Week 3 (Days 18-20)**: Documentation polish, finalize `writeup.txt`, lock artifacts, and perform final leaderboard submissions with safe margin before deadline.
- **Buffer & contingencies**:
  - Allocate final weekend for debugging submission mismatches.
  - Maintain rollback checkpoints for models; persist top-performing weights and random seeds.

## 6. Risk Register & Mitigations
- **Data leakage**: Guard by strict separation of train/validation/test pairs; enforce filtering utilities and audit SQL-like joins.
- **Leaderboard overfitting**: Limit submissions per day, rely on offline metrics; use hold-out validation replicating leaderboard distribution.
- **Compute limits**: Pre-compute embeddings, down-sample for prototyping, and document scaling strategy if running on shared lab machines.
- **Collaboration compliance**: Maintain private repository and avoid sharing intermediate predictions to uphold individual work requirement.

This plan provides the actionable blueprint to deliver production-quality, leaderboard-ready solutions while satisfying all assignment constraints.

