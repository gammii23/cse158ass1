# Assignment 1 Ordered Task List (Detailed)

## 1. Environment Bootstrap
1. Create a Python virtual environment (e.g., `.venv`) and activate it.
2. Pin and install dependencies: `pandas`, `numpy`, `scikit-learn`, `scipy`, `xgboost`, `nltk`, `lightfm` or `implicit`, `sentence-transformers` (optional), `pytest`, `pytest-cov`, `mypy`, `flake8`.
3. Freeze `requirements.txt` and verify reinstall works (`pip install -r requirements.txt`).
4. Configure logging wrapper (stdout formatter + Sentry placeholder function).
5. Set global random seed utility for reproducibility.

## 2. Project Scaffolding (SOLID/DRY)
1. Create packages with `__init__.py`: `data_access/`, `features/`, `models/`, `pipelines/`, `outputs/`, `config/`, `tests/`.
2. Add `constants.py` with enums for filenames, columns, genre labels.
3. Add `config/settings.py` with dataclasses (paths, hyperparams), reading env/CLI overrides.
4. Create `results/` (experiments log), `outputs/` (predictions), `cache/` (optional parquet).

## 3. Data Ingestion & Auditing
1. Implement `data_access/loader.py`:
   - Load `train_Interactions.csv.gz` -> DataFrame; schema assertions; dtypes.
   - Load `train_Category.json.gz` and `test_Category.json.gz` -> text/metadata frames.
   - Load `pairs_Read.csv`, `pairs_Category.csv`, `pairs_Rating.csv` with schema checks.
   - Centralize missing-value policy; standardize column names (per `constants.py`).
2. Add optional parquet caching and checksum to detect staleness.
3. Quick EDA scripts/notebooks (optional): distributions, sparsity, review lengths.

## 4. Feature Engineering
1. `features/interactions.py`:
   - User/item frequency stats, global means, user/book biases.
   - Popularity/recency features; interaction counts; optional temporal bins.
   - Matrix factorization inputs (user/item id maps; sparse matrices).
2. `features/reviews.py`:
   - Text cleaning (lower, strip, normalize whitespace); simple tokenizer.
   - TF-IDF vectorizers (word-level 1–2 n-grams; char-level optional) with DI.
   - Optional embeddings adapter (SentenceTransformer) behind common interface.
3. Create deterministic `fit/transform` APIs with stored state and seed control.

## 5. Models
1. `models/read_predictor.py`:
   - Baseline logistic regression on collaborative features; class weight balance.
   - Optional LightFM/implicit model; calibrated probabilities (Platt/Isotonic).
   - Expose `fit`, `predict_proba`.
2. `models/category_classifier.py`:
   - DI: accept featurizer; default TF-IDF + multinomial logistic regression (or linear SVM).
   - Class weights; C/regularization hyperparams.
   - Expose `fit`, `predict`.
3. `models/rating_regressor.py`:
   - Baseline: global mean + user/book biases.
   - Add Ridge/ElasticNet/GBR on engineered features; optional MF predictions for stacking.
   - Expose `fit`, `predict`.

## 6. Pipelines (Return SubmissionResult)
1. `pipelines/read_workflow.py`:
   - Load interactions and `pairs_Read`.
   - Build features; train model; predict probabilities for pairs.
   - Return `{ success, data=SubmissionFrame[user_id,item_id,prediction], error }`.
2. `pipelines/category_workflow.py`:
   - Load train/test category data and `pairs_Category`.
   - Fit featurizer + classifier; predict classes for pairs.
   - Return submission frame with required columns.
3. `pipelines/rating_workflow.py`:
   - Load interactions and `pairs_Rating`.
   - Train regressor; predict ratings for pairs.
   - Return submission frame.

## 7. Output Serialization
1. Implement `outputs/serializer.py`:
   - Validate schema/columns per task; enforce dtypes and no NaNs.
   - Write `predictions_Read.csv`, `predictions_Category.csv`, `predictions_Rating.csv`.
   - Safe overwrite rules and directory creation.

## 8. Validation Strategy & Experiment Tracking
1. Implement stratified (read/category) and time-aware or random K-fold splits.
2. Cross-validate each model; record metrics: balanced accuracy, accuracy, MSE.
3. Log experiments to `results/experiments.csv` (timestamp, params, scores, seed).
4. Run ablations to measure feature impact; keep best params.

## 9. CLI Integration
1. In `assignment1.py`, add CLI: `--task read|category|rating` plus common flags (seed, cache, paths).
2. Route to respective pipeline; catch exceptions; print summary metrics.
3. On success, call serializer to emit correct CSV.

## 10. Submission Generation & Verification
1. Generate all three predictions files via CLI.
2. Diff headers/order against `baselines.py` outputs to validate format.
3. Spot-check sample rows; ensure no missing predictions and value ranges sane.

## 11. Testing & Quality Assurance
1. Unit tests (`tests/`):
   - Loaders: missing columns/compression edge cases handled.
   - Features: deterministic outputs given fixed seed.
   - Models: `fit`/`predict(_proba)` contracts; small synthetic data.
   - Pipelines: propagate `SubmissionResult` success/error without exceptions.
2. Integration tests: subset (≈1k rows) end-to-end for each pipeline; validate output schema and metric computation.
3. Static checks: `mypy` (no `Any` in public APIs), `flake8`; coverage ≥ 80% with `pytest --cov`.

## 12. Performance & Caching
1. Profile runtime/memory; enforce time budget per task (<10 minutes target).
2. Toggle parquet/text cache via env/flag.
3. Optimize hot paths (vectorizer max_features, sparse ops, batch predictions).

## 13. Documentation
1. Draft `writeup.txt` structure: data, features, models, validation, results, decisions, reproduction steps.
2. Auto-fill key metrics from `results/experiments.csv`; polish narrative.

## 14. Risk Controls (Continuous)
1. Data leakage guard: strict split utilities; no leaking test into train; audit joins.
2. Leaderboard overfitting: limit submissions; rely on offline metrics; use hold-out close to leaderboard distribution.
3. Compute limits: precompute embeddings, downsample for prototyping, document scaling.

## 15. Final Packaging & Handoff
1. Re-run CLI to regenerate final `predictions_*.csv` with locked seeds/params.
2. Validate artifacts: file sizes, row counts, schema, and sample correctness.
3. Finalize `writeup.txt`; include rerun instructions and environment details.
4. Prepare submission bundle: `assignment1.py`, `writeup.txt`, `predictions_*.csv`.

