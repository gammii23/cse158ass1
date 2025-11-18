# Assignment 1 Submission Checklist

## Required Files

- [ ] `assignment1.py` - Main script with CLI
- [ ] `writeup.txt` - Documentation (plain text format)
- [ ] `predictions_Read.csv` - Read prediction outputs
- [ ] `predictions_Category.csv` - Category prediction outputs
- [ ] `predictions_Rating.csv` - Rating prediction outputs
- [ ] `requirements.txt` - Python dependencies (optional but recommended)

## Verification Steps

### 1. File Existence
- [ ] All required files present
- [ ] No temporary files included
- [ ] No cache files included

### 2. File Formats
- [ ] `writeup.txt` is plain text (no markdown formatting)
- [ ] Prediction CSVs have correct headers
- [ ] Prediction CSVs match pairs file row counts

### 3. Content Validation
Run validation script:
```bash
python scripts/validate_submission.py
```

- [ ] File sizes reasonable (not empty, not huge)
- [ ] Row counts match pairs files exactly
- [ ] Schemas correct (required columns present)
- [ ] No NaN values in predictions
- [ ] Value ranges valid:
  - Read: [0, 1] probabilities
  - Category: integers [0, 4]
  - Rating: floats [1.0, 5.0]

### 4. Code Verification
- [ ] `assignment1.py` runs without errors
- [ ] Can regenerate predictions with fixed seed
- [ ] Code is readable and well-documented

### 5. Documentation
- [ ] `writeup.txt` includes:
  - Data overview
  - Feature engineering description
  - Model descriptions
  - Validation strategy
  - Results (CV scores, leaderboard if available)
  - Key decisions and trade-offs
  - Reproduction instructions

## Pre-Submission Commands

```bash
# 1. Regenerate final predictions with locked seed
python assignment1.py --task all --seed 42

# 2. Validate submission files
python scripts/validate_submission.py

# 3. Spot-check predictions
python scripts/spot_check.py

# 4. Verify writeup format
# Check that writeup.txt is plain text, no markdown

# 5. Final check: ensure all files present
ls -la assignment1.py writeup.txt predictions_*.csv
```

## Submission Instructions

1. Create submission bundle with required files only
2. Verify file sizes are reasonable
3. Test that predictions can be regenerated
4. Double-check writeup.txt format (plain text)
5. Submit to Gradescope

## Notes

- Do NOT include: cache/, results/, .venv/, __pycache__/, .git/
- Do NOT include: test files, scripts/, docs/, reports/
- Keep submission minimal: only required files

