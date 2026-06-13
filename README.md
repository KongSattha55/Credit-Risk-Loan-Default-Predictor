# Credit Risk / Loan Default Predictor

Binary classification project for predicting LendingClub loan default risk at origination time. The model uses borrower, credit, and loan application fields available when a loan is issued, then returns a probability of default and a risk label for review or downstream lending decisions.

## Current Status

| Area | Current implementation |
|---|---|
| Dataset | LendingClub accepted loans, 2007-2018 |
| Target | `default` where 1 = default / charged off / late 31-120 days, 0 = fully paid |
| Core model | LightGBM |
| Split strategy | Time-based using `issue_d` |
| Leakage prevention | Centralized in `src/leakage.py` |
| Explainability | Global and local SHAP artifacts plus dashboard explanations |
| Serving | FastAPI endpoint and Streamlit dashboard using calibrated sigmoid PDs |
| Business analytics | Expected loss, approval/rejection trade-offs, risk-band distribution |
| Model comparison | Six-model comparison on the same time-based split |
| Tracking | MLflow local runs and artifacts |
| Tests | 210 passing, with generated-artifact checks skipped when outputs are absent |

Latest leakage-free LightGBM artifact:

| Metric | Validation | Test |
|---|---:|---:|
| ROC-AUC | 0.722606 | 0.710493 |
| Average precision / PR-AUC | 0.451971 | 0.448317 |
| F1 | Not recorded in current metadata | 0.496237 |
| Brier score | Not recorded in current metadata | 0.211191 |
| Threshold | 0.4877 | 0.4877 |

Model metadata:

- Model: `mlruns/artifacts/lightgbm_model.pkl`
- Serving probability model: `models/lightgbm_calibrated_sigmoid.pkl`
- Metadata: `mlruns/artifacts/lightgbm_metadata.json`
- Features: 98 leakage-free model features
- Split strategy: `time_based_issue_d`

## Problem Statement

Predict whether a loan will default using only information available at origination time. This is a binary classification problem designed for credit-risk review, underwriting support, portfolio monitoring, and interview-ready ML demonstration.

Why this matters:

- Random splits can overstate performance in credit data.
- Post-origination fields can leak the answer.
- Business users need probability estimates and understandable risk drivers, not only a model score.

<<<<<<< HEAD
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
=======
## Dataset
>>>>>>> 851b49f (Update project files)

Raw data should be placed in:

```text
data/raw/archive/
├── accepted_2007_to_2018Q4.csv.gz
└── rejected_2007_to_2018Q4.csv.gz
```

Current processed data:

| Dataset | Shape / detail |
|---|---|
| Raw accepted loans | 2,260,701 rows x 151 columns |
| Cleaned accepted loans | 1,369,566 rows x 85 columns |
| Feature-engineered data | 1,369,566 rows x 100 columns |
| Final model features | 98 features |

## Time-Based Split

The split logic is centralized in `src/splits.py` and uses `issue_d`.

| Split | Rule | Rows | Default rate |
|---|---|---:|---:|
| Train | `issue_d < 2016-01-01` | 831,051 | 18.62% |
| Validation | `2016-01-01 <= issue_d < 2017-01-01` | 297,651 | 24.46% |
| Test | `issue_d >= 2017-01-01` | 240,864 | 26.27% |

This better mimics deployment because the model is trained on older loans and evaluated on future loans.

## Data Leakage Policy

Models must only use origination-time features. Post-origination fields are excluded through `src/leakage.py`.

Examples of excluded leakage fields:

```text
loan_status, last_pymnt_d, last_pymnt_amnt, next_pymnt_d,
last_credit_pull_d, total_pymnt, total_pymnt_inv,
total_rec_prncp, total_rec_int, total_rec_late_fee,
recoveries, collection_recovery_fee, out_prncp, out_prncp_inv,
settlement_status, settlement_date, settlement_amount,
settlement_percentage, settlement_term, debt_settlement_flag,
hardship_flag, loan_age_months
```

`issue_d` is used for splitting only and is not included directly as a model feature.

## Project Structure

```text
├── api/                    FastAPI inference service
├── app/                    Streamlit dashboard
├── artifacts/              SHAP and calibration plots/results
├── configs/                Model and split configuration
├── data/                   Raw, processed, and interim data
├── mlruns/                 MLflow runs and model artifacts
├── models/                 Calibration model artifacts
├── notebook/               EDA and LightGBM modeling notebooks
├── src/                    Data, training, calibration, SHAP, split, leakage code
├── tests/                  Unit and integration tests
├── doc.md                  Full project documentation
└── README.md               Quick-start and project summary
```

Important source files:

| File | Purpose |
|---|---|
| `src/data_cleaning.py` | Raw accepted-loan data cleaning |
| `src/feature_engineering.py` | Feature engineering and train-only encodings |
| `src/leakage.py` | Central leakage exclusion list |
| `src/splits.py` | Time-based split logic |
| `src/train.py` | General model training |
| `src/tune_lightgbm.py` | Optuna LightGBM tuning |
| `src/calibrate.py` | Probability calibration workflow |
| `src/business_risk.py` | Expected loss, risk bands, and threshold business analytics |
| `src/model_comparison.py` | Six-model comparison workflow and summary artifacts |
| `src/explain_shap.py` | SHAP artifact generation |
| `src/explain_utils.py` | Business-friendly explanation helpers |
| `api/main.py` | FastAPI prediction API |
| `app/main.py` | Streamlit dashboard |

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Optional environment setup:

```bash
cp .env.example .env
```

### 2. Clean and Feature-Engineer Data

```bash
python src/data_cleaning.py
python src/feature_engineering.py
```

Outputs:

```text
data/processed/loans_cleaned.parquet
data/interim/loans_features.parquet
mlruns/artifacts/encoding_maps.json
```

### 3. Train LightGBM

```bash
python src/train.py --model lightgbm
```

Other model options:

```bash
python src/train.py --model logistic_regression
python src/train.py --model random_forest
python src/train.py --model xgboost
```

Optional tuning:

```bash
python src/tune_lightgbm.py --n-trials 50
```

### 4. Generate SHAP Explanations

```bash
python src/explain_shap.py --sample 2000
```

Generated artifacts:

```text
artifacts/shap_summary_beeswarm.png
artifacts/shap_summary_bar.png
artifacts/shap_top_features.csv
artifacts/shap_local_example_waterfall.png
artifacts/shap_local_example.json
```

Latest SHAP run used the 2017+ test set sample and logged artifacts to MLflow run:

```text
5924a9f4fd9449a995006cd3a3b59995
```

Top global SHAP features:

| Rank | Feature | Meaning |
|---:|---|---|
| 1 | `sub_grade_enc` | Weaker LendingClub sub-grades indicate higher risk |
| 2 | `int_rate_x_term` | High rate combined with longer term indicates repayment uncertainty |
| 3 | `acc_open_past_24mths` | Recent credit growth can raise risk |
| 4 | `term_60` | Longer repayment horizon increases uncertainty |
| 5 | `fico_dti_score` | Stronger credit score relative to debt burden lowers risk |

### 5. Run API

```bash
uvicorn api.main:app --reload
```

Open:

```text
http://localhost:8000/docs
```

The API currently loads:

- `models/lightgbm_calibrated_sigmoid.pkl` for calibrated default probabilities
- `mlruns/artifacts/lightgbm_model.pkl` only where raw LightGBM explanations are needed
- `mlruns/artifacts/lightgbm_metadata.json`
- `mlruns/artifacts/encoding_maps.json`

### 6. Run Dashboard

```bash
streamlit run app/main.py
```

The dashboard includes:

- single-loan prediction,
- predicted probability of default,
- risk label,
- factors increasing risk,
- factors reducing risk,
- local explanation bar chart,
- threshold analysis,
- business risk analytics,
- batch CSV scoring.

### 7. Run Tests

```bash
pytest tests/ -v
```

Latest local result:

```text
210 passed
```

GitHub Actions also runs the test suite on pushes and pull requests.

Targeted checks:

```bash
pytest tests/test_leakage.py -v
pytest tests/test_splits.py -v
pytest tests/test_explainability.py -v
pytest tests/test_api.py -v
pytest tests/test_calibration.py -v
pytest tests/test_business_risk.py -v
pytest tests/test_model_comparison.py -v
```

## Model Comparison

Phase 10 compares six models on the same `issue_d` time-based split with leakage-free features:

1. FICO/grade logistic baseline
2. Logistic Regression
3. Random Forest
4. XGBoost
5. LightGBM
6. Calibrated LightGBM sigmoid

Artifacts:

```text
artifacts/model_comparison.csv
artifacts/model_comparison.json
```

Latest comparison used the full validation/test windows and a capped 200,000-row training sample for comparison runtime.

| Model | ROC-AUC | PR-AUC | Brier | Log Loss | F1 |
|---|---:|---:|---:|---:|---:|
| LightGBM | 0.706106 | 0.442494 | 0.210386 | 0.606193 | 0.494609 |
| XGBoost | 0.704391 | 0.441801 | 0.177859 | 0.535000 | 0.492217 |
| Calibrated LightGBM sigmoid | 0.701052 | 0.435000 | 0.175399 | 0.526919 | 0.490993 |
| Random Forest | 0.697311 | 0.430738 | 0.211647 | 0.608582 | 0.488426 |
| Logistic Regression | 0.690643 | 0.415379 | 0.216593 | 0.627836 | 0.481552 |
| FICO/grade logistic baseline | 0.669895 | 0.392886 | 0.221947 | 0.635531 | 0.472395 |

Summary:

- Best ranking model: LightGBM.
- Best calibrated probability model: Calibrated LightGBM sigmoid.
- Most interpretable model: FICO/grade logistic baseline.
- Production recommendation: keep Calibrated LightGBM sigmoid as the production probability model.

## Business Risk Analytics

Business analytics use calibrated PD values from `models/lightgbm_calibrated_sigmoid.pkl`.

Core formulas and rules:

- Expected Loss = `PD x LGD x EAD`
- Default LGD = `0.45`
- EAD defaults to `loan_amnt`
- Risk bands:
  - Low: PD < 0.10
  - Medium: 0.10 <= PD < 0.20
  - High: 0.20 <= PD < 0.35
  - Very High: PD >= 0.35

The Streamlit dashboard includes a **Business Risk Analytics** tab with approval rate, rejection rate, default rate among approved loans, defaults caught, expected loss, threshold table, and risk-band distribution.

## SHAP Explainability

The project supports both global and local explainability.

| Level | Output | Use |
|---|---|---|
| Global | `shap_summary_beeswarm.png`, `shap_summary_bar.png`, `shap_top_features.csv` | Identify portfolio-level model drivers |
| Local | `shap_local_example_waterfall.png`, `shap_local_example.json`, dashboard explanation panel | Explain one loan prediction |

Dashboard wording is business-friendly. It avoids technical SHAP language in the main prediction view and uses:

- Factors increasing risk
- Factors reducing risk
- Plain-English explanation

## Calibration

Calibration work is available in `src/calibrate.py`, with artifacts in:

```text
models/lightgbm_raw.pkl
models/lightgbm_calibrated_sigmoid.pkl
models/lightgbm_calibrated_isotonic.pkl
artifacts/calibration_curve.png
artifacts/calibration_results.json
```

Current calibration results file reports:

| Method | ROC-AUC | PR-AUC | Brier | Log loss |
|---|---:|---:|---:|---:|
| Raw | 0.708342 | 0.443849 | 0.184572 | 0.552882 |
| Sigmoid | 0.708342 | 0.443849 | 0.173664 | 0.521444 |
| Isotonic | 0.708263 | 0.440396 | 0.173687 | 0.521320 |

Production probability outputs now use `models/lightgbm_calibrated_sigmoid.pkl`. The API and dashboard import `src.calibration_classes` before unpickling, call `predict_proba(...)[..., 1]`, and keep raw LightGBM only for SHAP explainability. `artifacts/calibration_results.json` reports `leakage_in_model: []`.

## MLflow

Start the local MLflow UI:

```bash
mlflow ui
```

Then open:

```text
http://localhost:5000
```

Tracked/logged items include:

- model parameters,
- split strategy,
- validation and test metrics,
- model artifacts,
- SHAP artifacts,
- calibration artifacts where applicable.

## Reproducible Workflow

Full current workflow:

```bash
pip install -r requirements.txt

python src/data_cleaning.py
python src/feature_engineering.py
python src/train.py --model lightgbm

python src/explain_shap.py --sample 2000
python src/calibrate.py

uvicorn api.main:app --reload
streamlit run app/main.py

pytest tests/ -v
```

## Tech Stack

- Python 3.14
- pandas, NumPy, pyarrow
- scikit-learn, LightGBM, XGBoost
- SHAP
- MLflow
- FastAPI, Uvicorn, Pydantic
- Streamlit, Plotly, Matplotlib, Seaborn
- Optuna
- pytest

## Documentation

For the full project write-up, see:

```text
doc.md
```

That document includes detailed notes for project review, interview preparation, future maintenance, SHAP explainability, calibration, API/dashboard behavior, tests, and recommended improvements.

## License

For educational and personal use only. LendingClub data is subject to its own terms of use.
