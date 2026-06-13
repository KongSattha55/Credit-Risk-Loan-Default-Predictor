# Credit Risk: Loan Default Predictor

Binary classification model that predicts whether a LendingClub loan will default, enabling credit risk assessment at origination time.

## Problem Statement

Given loan application attributes available **at origination** (borrower profile, credit history, loan terms), predict the probability that a loan will default (Charged Off, Default, or Late 31–120 days).

- **Dataset**: LendingClub accepted loans, 2007–2018 (2.26M loans, 151 features)
- **Target**: `default` — binary (1 = default, 0 = fully paid)
- **Class balance**: ~21% default rate

## Project Structure

```
├── api/                    FastAPI inference service
├── app/                    Frontend/dashboard app
├── configs/                Model hyperparameters and feature configs
├── data/
│   ├── raw/                Original source data (not committed)
│   ├── interim/            Intermediate feature-engineered data
│   └── processed/          Cleaned, model-ready parquet files
├── doc/                    Additional documentation
├── mlruns/                 MLflow experiment tracking runs
├── notebook/               Exploratory analysis notebooks
├── src/                    Production source code
│   └── data_cleaning.py    Raw → cleaned data pipeline
└── tests/                  Unit and integration tests
```

## Quick Start

### 1. Clone and install dependencies

```bash
git clone <repo-url>
cd "Credit Risk : Loan Default Predictor"
pip install -r requirements.txt
```

### 2. Set up environment variables

```bash
cp .env.example .env
# Edit .env and fill in any required values
```

### 3. Place raw data

Download the LendingClub dataset and place the `.csv.gz` files in:
```
data/raw/archive/
├── accepted_2007_to_2018Q4.csv.gz
└── rejected_2007_to_2018Q4.csv.gz
```

### 4. Run data cleaning

```bash
python src/data_cleaning.py
# Output: data/processed/loans_cleaned.parquet
```

### 5. (Optional) Feature engineering

```bash
python src/feature_engineering.py
# Output: data/interim/loans_features.parquet
```

### 6. Train model (via config)

```bash
python src/train.py --model lightgbm        # default
python src/train.py --model logistic_regression
python src/train.py --model random_forest
python src/train.py --model xgboost
```

Or run Optuna-based LightGBM tuning directly:

```bash
python src/tune_lightgbm.py --n-trials 50
```

### 7. Start the API

```bash
uvicorn api.main:app --reload
# Docs: http://localhost:8000/docs
```

### 8. Launch the dashboard

```bash
streamlit run app/main.py
```

### 9. Run tests

```bash
pytest tests/ -v
```

## Data Cleaning Summary

| Step | Detail |
|------|--------|
| Filter ambiguous statuses | Kept Fully Paid, Charged Off, Default, Late 31-120d |
| Binary target | 1 = default, 0 = fully paid |
| Drop high-missing columns | 44 columns with >50% missing (hardship, settlement, joint fields) |
| Drop data-leakage columns | Post-origination payment/recovery fields |
| Feature parsing | `term`, `emp_length`, `int_rate`, `revol_util`, date fields |
| Outlier capping | DTI capped at 100, revol_util at 100, annual_inc at 99th pct |
| Imputation | Numeric → median, Categorical → "Unknown" |
| **Output** | **1,369,566 rows × 84 features, 0 nulls** |

## Key Features (post-cleaning)

| Feature | Description |
|---------|-------------|
| `loan_amnt` | Requested loan amount |
| `int_rate` | Interest rate (%) |
| `term` | Loan term in months (36 or 60) |
| `grade` / `sub_grade` | LendingClub credit grade |
| `emp_length` | Employment length (0–10 years) |
| `annual_inc` | Annual income |
| `dti` | Debt-to-income ratio |
| `fico_range_low/high` | FICO score range |
| `purpose` | Loan purpose (debt consolidation, credit card, etc.) |
| `revol_util` | Revolving line utilization rate |
| `cr_history_months` | Months since earliest credit line |
| `loan_age_months` | Months since loan issued (to reference date) |

## Experiment Tracking

MLflow is used for all training experiments:

```bash
mlflow ui
# Dashboard: http://localhost:5000
```

## Tech Stack

- **Python 3.14**
- **Data**: pandas 3.x, numpy 2.x, pyarrow
- **ML**: scikit-learn, XGBoost, LightGBM
- **Explainability**: SHAP
- **Tracking**: MLflow
- **API**: FastAPI + uvicorn
- **Visualization**: matplotlib, seaborn, plotly

## License

For educational and personal use only. LendingClub data is subject to its own terms of use.
