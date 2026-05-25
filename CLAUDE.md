# Credit Risk: Loan Default Predictor — Claude Instructions

## Project Overview
Binary classification project predicting loan defaults using LendingClub data (2007–2018).
- **Target**: `default` (1 = Charged Off/Default/Late 31-120d, 0 = Fully Paid)
- **Default rate**: ~21.2%
- **Clean dataset**: `data/processed/loans_cleaned.parquet` (1.37M rows, 84 features)

## Directory Layout
```
├── api/            FastAPI inference service
├── app/            Streamlit or frontend app
├── claude/         Claude Code commands and rules
│   ├── commands/
│   └── rules/
├── configs/        YAML configs (model hyperparams, feature lists)
├── data/
│   ├── raw/        Original gzipped CSVs (never modify)
│   ├── interim/    Intermediate outputs (feature engineering)
│   └── processed/  Final cleaned parquet files
├── doc/            Project documentation
├── mlruns/         MLflow experiment tracking
├── notebook/       Exploratory Jupyter notebooks
├── src/            Production Python source code
└── tests/          Pytest unit and integration tests
```

## Tech Stack
- **Python 3.14**
- **Data**: pandas 3.x, numpy 2.x, pyarrow (parquet)
- **ML**: scikit-learn 1.8, (xgboost, lightgbm — install as needed)
- **Experiment tracking**: MLflow
- **Explainability**: SHAP
- **API**: FastAPI + uvicorn
- **Viz**: matplotlib, seaborn, plotly
- **Config**: python-dotenv, pydantic v2

## Code Conventions
- All source modules go in `src/`; notebooks are exploratory only — never import from `notebook/`
- Load cleaned data from `data/processed/loans_cleaned.parquet` (not raw CSVs)
- Use `pathlib.Path` for all file paths; derive base path from `__file__`
- Pandas 3.x / Copy-on-Write: avoid `inplace=True` on chained assignments; use `df.fillna(values)` pattern
- Use `float32` instead of `float64` for feature matrices to reduce memory
- Log all experiments to MLflow (`mlruns/` directory)

## Data Notes
- Raw files: `data/raw/archive/accepted_2007_to_2018Q4.csv.gz` (~1.3 GB unzipped ~5 GB)
- Do NOT commit raw data or processed parquet files to git
- Leakage columns (post-origination payments/recoveries) are already removed in cleaning
- Joint-application secondary-applicant fields are dropped (>95% missing)

## Environment
- Copy `.env.example` to `.env` and fill in values before running
- Never commit `.env` to git
- Load env vars with `python-dotenv` at module entry points

## Running the Pipeline
```bash
# 1. Clean raw data
python src/data_cleaning.py

# 2. (Future) Feature engineering
python src/feature_engineering.py

# 3. (Future) Train model
python src/train.py

# 4. (Future) Serve API
uvicorn api.main:app --reload
```

## Testing
```bash
pytest tests/ -v
```
