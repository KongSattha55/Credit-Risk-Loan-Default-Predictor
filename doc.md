---
editor_options: 
  markdown: 
    wrap: sentence
---

# Credit Risk / Loan Default Predictor — Project Documentation

## 1. Project Overview

This project builds a credit-risk model that predicts whether a LendingClub loan will default.
The model is intended to support loan-origination risk assessment: given borrower, credit, and loan-application fields available at the time a loan is issued, estimate the probability that the borrower will default.

The project uses LendingClub accepted-loan data from 2007-2018.
It includes a full workflow: raw data cleaning, feature engineering, leakage prevention, model training, threshold analysis, experiment tracking, API serving, Streamlit dashboarding, and automated tests.

| Item | Current repo value |
|------------------------------------|------------------------------------|
| Dataset | LendingClub accepted loans, 2007–2018 |
| Raw rows / columns | 2,260,701 rows × 151 columns |
| Cleaned rows | 1,369,566 rows (0 nulls) |
| Target | `default` — 1 = Charged Off / Default / Late 31–120 days; 0 = Fully Paid |
| Overall default rate | 21.23% |
| Business use case | Rank loan applications by default risk at origination for approval, pricing, and review decisions |
| Python version | 3.14 |
| Key libraries | LightGBM 4.6, scikit-learn 1.8, MLflow 2.17, FastAPI, Streamlit, Optuna, SHAP |
| Time-based split | Train \< 2016-01-01, Val 2016, Test ≥ 2017-01-01 |
| **Production model** | `src/train.py --model lightgbm` → `mlruns/artifacts/lightgbm_model.pkl` |
| Train val ROC-AUC | 0.7226 |
| Train test ROC-AUC | 0.7105 · PR-AUC 0.4483 · F1 0.4962 · Brier 0.2112 |
| **Tuned model** ✅ leakage-free | `src/tune_lightgbm.py` → `mlruns/artifacts/lightgbm_tuned.txt` |
| Tuned val ROC-AUC | 0.7203 · PR-AUC 0.4448 |
| Tuned test ROC-AUC (raw) | 0.7011 · PR-AUC 0.4350 · F1 0.4908 · Brier 0.1826 |
| **Calibrated model** ✅ | `src/calibrate.py` → `models/lightgbm_calibrated_sigmoid.pkl` |
| Calibrated test (sigmoid) | ROC-AUC 0.7011 · PR-AUC 0.4350 · Brier **0.1754** · LogLoss 0.5269 |
| Test suite | **203 tests passing** (0 skipped) |

## 2. Problem Statement

This is a binary classification problem.
The model predicts whether a loan will default using only information available at loan origination time.

The positive class is `default = 1`.
The negative class is `default = 0`.
In practice, the model output is a probability of default.
That probability can be used to:

-   flag high-risk applications for manual review,
-   compare borrowers by relative risk,
-   choose a decision threshold based on business constraints,
-   estimate portfolio-level expected losses once loss assumptions are added.

The core modeling constraint is that the model must not use post-origination information such as payments, recoveries, settlement status, outstanding principal, or future loan age.

## 3. Dataset

The raw dataset is the LendingClub accepted-loans dataset covering 2007-2018.

| Dataset item | Value |
|------------------------------------|------------------------------------|
| Raw accepted-loan file | `data/raw/archive/accepted_2007_to_2018Q4.csv.gz` |
| Raw rejected-loan file | `data/raw/archive/rejected_2007_to_2018Q4.csv.gz` |
| Main file used | Accepted-loan file |
| Raw shape | 2,260,701 rows x 151 columns |
| Cleaned output | `data/processed/loans_cleaned.parquet` |
| Feature-engineered output | `data/interim/loans_features.parquet` |
| Cleaned labeled rows | 1,369,566 |
| Target column | `default` |
| Overall class balance | 1,078,739 non-default / 290,827 default, about 21.23% default |

The cleaning pipeline filters the original `loan_status` values into labeled default and non-default outcomes.
Active or ambiguous statuses are removed because they do not have a final known outcome.

## 4. Project Structure

| Path | Purpose |
|------------------------------------|------------------------------------|
|  |  |
| `README.md` | Quick-start guide and high-level project summary |
| `requirements.txt` | Python dependencies for data, ML, API, dashboard, notebooks, and tests |
| `configs/model.yaml` | Model configuration, feature list, split date column, and model hyperparameters |
| `data/raw/archive/` | Local raw LendingClub files |
| `data/processed/loans_cleaned.parquet` | Cleaned dataset produced by `src/data_cleaning.py` |
| `data/interim/loans_features.parquet` | Feature-engineered dataset produced by `src/feature_engineering.py` |
| `src/data_cleaning.py` | Raw CSV to cleaned parquet pipeline |
| `src/feature_engineering.py` | Numeric/categorical engineering, target encodings, ratios, log transforms |
| `src/leakage.py` | Centralized leakage and non-feature column exclusions |
| `src/splits.py` | Centralized time-based train/validation/test split logic using `issue_d` |
| `src/train.py` | General model training script for LightGBM, logistic regression, random forest, and XGBoost |
| `src/tune_lightgbm.py` | Optuna-based LightGBM tuning and final refit |
| `src/baseline.py` | Simple logistic baseline using FICO and grade |
| `src/threshold_refinement.py` | Threshold analysis strategies and reports |
| `src/calibrate.py` | Probability calibration pipeline — compares raw, sigmoid, and isotonic models |
| `src/calibration_classes.py` | Picklable sklearn-compatible wrappers: `LGBMBoosterWrapper`, `PreFitCalibratedClassifier` |
| `src/business_risk.py` | Expected loss, risk-band assignment, and threshold business analytics |
| `src/model_comparison.py` | Six-model comparison workflow using the time-based split |
| `src/explain_shap.py` | SHAP global and local explainability outputs |
| `src/explain_utils.py` | Business-friendly SHAP helpers: `summarize_shap_local()`, feature labels, `global_business_interpretation()` |
| `src/inference_fe.py` | Row-level feature engineering used by API and dashboard |
| `api/main.py` | FastAPI inference service |
| `app/main.py` | Streamlit dashboard for prediction, threshold exploration, business risk analytics, and batch scoring |
| `notebook/01_eda.ipynb` | Exploratory data analysis notebook |
| `notebook/02_modeling_lightgbm.ipynb` | LightGBM modeling notebook |
| `tests/` | Unit and integration tests |
| `tests/test_calibration.py` | 38 tests: artifact existence, predict_proba output, calibration metrics, leakage audit |
| `mlruns/` | MLflow experiment runs and model artifacts |
| `models/` | Calibrated model `.pkl` files — created by `src/calibrate.py` |
| `artifacts/` | Non-MLflow outputs: `calibration_curve.png`, `calibration_results.json` |

## 5. Data Cleaning Pipeline

The cleaning pipeline is implemented in `src/data_cleaning.py`.

### Loan Status Filtering

The raw `loan_status` column is mapped into a binary target.
The kept statuses are:

| Raw status                                            | Target |
|-------------------------------------------------------|-------:|
| `Fully Paid`                                          |      0 |
| `Does not meet the credit policy. Status:Fully Paid`  |      0 |
| `Charged Off`                                         |      1 |
| `Default`                                             |      1 |
| `Does not meet the credit policy. Status:Charged Off` |      1 |
| `Late (31-120 days)`                                  |      1 |

Statuses such as `Current`, `In Grace Period`, and `Late (16-30 days)` are excluded because their final default outcome is ambiguous at the time of labeling.

### Target Creation

The binary target column is named `default`.

-   `default = 1`: default-like outcome.
-   `default = 0`: fully paid outcome.

After filtering labeled rows, the cleaned dataset contains 1,369,566 rows with about a 21.23% default rate.

### Missing Value Handling

The cleaning script drops high-missing columns, including hardship fields, settlement fields, secondary-applicant fields, joint-application fields, free-text description, and other columns with heavy missingness.

After cleaning and imputation, the EDA notebook reports zero columns with missing values.

### Leakage Column Removal

Known leakage columns are removed or centrally excluded.
These include post-origination payment, recovery, hardship, settlement, and servicing fields.
`loan_status` is used only to create the target, then removed.

### Feature Parsing

The cleaning script parses common LendingClub string formats:

| Field | Transformation |
|------------------------------------|------------------------------------|
| `term` | Converts values like `" 36 months"` to numeric months |
| `emp_length` | Converts employment strings like `"10+ years"` and `"< 1 year"` to numeric years |
| `int_rate` | Converts percentage strings to numeric percentages |
| `revol_util` | Converts percentage strings to numeric percentages |
| `issue_d` | Converts issue month/year to `datetime64[ns]` for splitting only |
| `earliest_cr_line` | Converts to `cr_history_months`, a safe origination-time feature |

### Outlier Capping

The cleaning script caps selected skewed or bounded fields:

| Field        | Rule                                        |
|--------------|---------------------------------------------|
| `annual_inc` | Lower bound 0, upper cap at 99th percentile |
| `dti`        | Lower bound 0, upper cap 100                |
| `revol_util` | Lower bound 0, upper cap 100                |
| `revol_bal`  | Lower bound 0, upper cap at 99th percentile |

### Imputation

Numeric columns are filled with medians.
Categorical columns are filled with `"Unknown"`.
The script then downcasts `float64` columns to `float32` to reduce memory usage.

### Final Cleaned Dataset Output

| Output  | Value                                        |
|---------|----------------------------------------------|
| File    | `data/processed/loans_cleaned.parquet`       |
| Rows    | 1,369,566                                    |
| Columns | 85 columns including `default` and `issue_d` |
| Nulls   | 0 after cleaning                             |

## 6. Data Leakage Prevention

The modeling rule is simple: only use features available at loan origination time.

Data leakage is especially dangerous in loan default prediction because many LendingClub columns are created after the loan is issued.
Payment amounts, recovery amounts, settlement fields, outstanding principal, and loan status can directly reveal whether the borrower repaid or defaulted.

The central leakage list is defined in `src/leakage.py` as `LEAKAGE_COLS`.
Model feature lists are created through `feature_columns(...)`, which excludes leakage fields and non-feature fields.

The required excluded leakage columns are:

| Leakage column | Why excluded |
|------------------------------------|------------------------------------|
| `loan_status` | Used to create target; directly encodes outcome |
| `last_pymnt_d` | Post-origination payment date |
| `last_pymnt_amnt` | Post-origination payment amount |
| `next_pymnt_d` | Servicing/payment schedule field after origination |
| `last_credit_pull_d` | Post-origination servicing date |
| `total_pymnt` | Repayment outcome information |
| `total_pymnt_inv` | Repayment outcome information |
| `total_rec_prncp` | Principal recovered after origination |
| `total_rec_int` | Interest recovered after origination |
| `total_rec_late_fee` | Late-fee outcome information |
| `recoveries` | Recovery after default |
| `collection_recovery_fee` | Collection outcome information |
| `out_prncp` | Outstanding balance after origination |
| `out_prncp_inv` | Outstanding balance after origination |
| `settlement_status` | Settlement outcome |
| `settlement_date` | Settlement outcome date |
| `settlement_amount` | Settlement outcome amount |
| `settlement_percentage` | Settlement outcome amount |
| `settlement_term` | Settlement outcome term |
| `debt_settlement_flag` | Indicates settlement after origination |
| `hardship_flag` | Post-origination hardship signal |
| `loan_age_months` | Derived from time since issue date to a later reference date; not available at origination and can encode vintage/outcome effects |

The test suite includes `tests/test_leakage.py`, which checks that required leakage columns are listed and that final feature columns exclude them.

### ✅ `loan_age_months` leakage — resolved

An earlier training run produced a model artifact (`lightgbm_tuned.txt`) that included `loan_age_months` in its feature set.
This column was created by an older version of `data_cleaning.py` after the leakage-drop step, then carried through to training before the leakage registry was fully enforced in the pipeline.

**Resolution (completed):** - `data_cleaning.py` no longer creates `loan_age_months`.
- `src/feature_engineering.py` now explicitly drops all `LEAKAGE_COLS` in `_drop_redundant()` and asserts their absence in `_validate()`.
- The backward-compat reconstruction block was removed from `src/calibrate.py`.
- The model was **retrained** via `python src/tune_lightgbm.py --n-trials 50`.
- `tuning_metadata.json` and `calibration_results.json` now confirm `leakage_in_model: []`.
- All 203 tests pass; `TestLeakageInvariant` in `tests/test_calibration.py` enforces the invariant permanently.

## 7. Feature Engineering

Feature engineering is implemented in `src/feature_engineering.py`, with matching row-level inference logic in `src/inference_fe.py`.

### Core Parsed Features

| Feature | Description | Safe at origination? |
|------------------------|------------------------|------------------------|
| `term` | Loan term in months parsed from LendingClub text | Yes |
| `emp_length` | Employment length parsed into years | Yes |
| `int_rate` | Interest rate parsed from percentage text | Yes |
| `revol_util` | Revolving utilization parsed from percentage text | Yes |
| `cr_history_months` | Months from earliest credit line to reference point | Yes, if computed from application-known credit history |
| `issue_d` | Loan issue date converted to datetime | Safe for splitting only; excluded from model features |

### Encoded Categorical Features

| Engineered feature        | Source                                     |
|---------------------------|--------------------------------------------|
| `grade_enc`               | Ordinal mapping of LendingClub grade A-G   |
| `sub_grade_enc`           | Ordinal mapping of sub-grade A1-G5         |
| `verification_status_enc` | Ordinal income-verification mapping        |
| `home_ownership_enc`      | Ordinal home-ownership mapping             |
| `purpose_rate_enc`        | Smoothed target encoding of loan purpose   |
| `state_default_rate`      | Smoothed target encoding of state          |
| `emp_title_log_freq`      | Log frequency encoding of employment title |

Target encodings are fitted only on the training window through `build_train_mask(...)`, which uses `issue_d < 2016-01-01`.
They are then applied to validation and test rows.

### Binary Flags

| Feature                   | Meaning                        |
|---------------------------|--------------------------------|
| `term_60`                 | 1 if term is 60 months         |
| `initial_list_status_enc` | Encoded initial listing status |
| `application_type_joint`  | 1 if joint application         |
| `disbursement_direct`     | 1 if DirectPay disbursement    |

### Log Transforms

The script creates `log1p` versions of skewed numeric fields such as:

-   `annual_inc_log`
-   `revol_bal_log`
-   `tot_coll_amt_log`
-   `delinq_amnt_log`
-   `total_rev_hi_lim_log`
-   `tot_hi_cred_lim_log`
-   `total_bal_ex_mort_log`
-   `total_bc_limit_log`
-   `total_il_high_credit_limit_log`

### Ratio and Interaction Features

The feature-engineering script creates several ratios and interactions:

| Feature | Description |
|------------------------------------|------------------------------------|
| `loan_to_income` | Loan amount relative to annual income |
| `installment_to_income` | Monthly installment relative to monthly income |
| `credit_util_total` | Revolving balance relative to total high credit limit |
| `bc_util_ratio` | Bankcard utilization ratio |
| `int_rate_x_term` | Interest rate multiplied by loan term |
| `fico_dti_score` | FICO score adjusted by DTI |
| `derog_ratio` | Derogatory record ratio |
| `inq_per_acc` | Recent inquiries per open account |
| `revolving_debt_share` | Revolving balance relative to non-mortgage balance |

### Safe vs Leaky Date Features

| Date feature | Use |
|------------------------------------|------------------------------------|
| `issue_d` | Safe for splitting only; excluded from model features |
| `earliest_cr_line` -\> `cr_history_months` | Safe if based on credit history available at origination |
| `last_pymnt_d`, `next_pymnt_d`, `last_credit_pull_d`, `settlement_date` | Leakage; excluded |
| `loan_age_months` | Leakage; excluded |

## 8. EDA Summary

The EDA notebook is `notebook/01_eda.ipynb`.

### Target Distribution

The notebook reports:

| Class                          |     Count |
|--------------------------------|----------:|
| Fully paid / non-default (`0`) | 1,078,739 |
| Default (`1`)                  |   290,827 |

Overall default rate: about 21.23%.

### Missing Values

After cleaning, the EDA notebook reports:

-   Columns with any missing values: 0.
-   Columns with more than 5% missing: 0.

### Numeric Feature Patterns

The strongest numeric associations shown in the notebook include:

| Feature | Direction from EDA |
|------------------------------------|------------------------------------|
| `int_rate` | Higher rates are associated with higher default risk |
| `term` | Longer terms are associated with higher default risk |
| `fico_range_low` | Higher FICO is associated with lower default risk |
| `dti` | Higher DTI is associated with higher default risk |
| `acc_open_past_24mths` | More recently opened accounts are associated with higher default risk |

The notebook also highlights heavy skew in variables such as `tot_coll_amt`, `delinq_amnt`, `total_rev_hi_lim`, `num_tl_120dpd_2m`, and `tax_liens`.

### Categorical Feature Patterns

EDA focuses on categorical features such as:

-   `grade`
-   `sub_grade`
-   `home_ownership`
-   `verification_status`
-   `purpose`
-   `addr_state`
-   `term`
-   `application_type`

Important observations:

-   `grade` and `sub_grade` are natural ordinal risk indicators.
-   `purpose` and `addr_state` have enough signal to justify smoothed target encoding.
-   `emp_title` is very high-cardinality, so the pipeline uses log-frequency encoding rather than raw one-hot encoding.

### Correlation and Risk Trends

The EDA notebook reports top correlations with the target.
Examples include:

| Feature                | Correlation with default in notebook |
|------------------------|-------------------------------------:|
| `int_rate`             |                               0.2630 |
| `term`                 |                               0.1810 |
| `fico_range_low`       |                              -0.1296 |
| `dti`                  |                               0.1066 |
| `acc_open_past_24mths` |                               0.0994 |

Note: older EDA output references `loan_age_months`; this field is now treated as leakage and excluded from modeling.

## 9. Modeling

Modeling is implemented in `notebook/02_modeling_lightgbm.ipynb`, `src/train.py`, and `src/tune_lightgbm.py`.

### Train / Validation / Test Split

The current modeling split is time-based and centralized in `src/splits.py`.

| Split | Date rule | Current row count | Default rate |
|----|----|---:|---:|
| Train | `issue_d < 2016-01-01` | 831,051 | 18.62% |
| Validation | `2016-01-01 <= issue_d < 2017-01-01` | 297,651 | 24.46% |
| Test | `issue_d >= 2017-01-01` | 240,864 | 26.27% |

This split better matches deployment because the model is trained on older loans and evaluated on future loans.

### Preprocessing

The current workflow:

1.  Clean raw data with `src/data_cleaning.py`.
2.  Convert `issue_d` to datetime.
3.  Create `default`.
4.  Remove leakage and non-feature columns.
5.  Fit target encodings and frequency encodings on training rows only.
6.  Apply the fitted encoding maps to validation and test rows.
7.  Exclude `issue_d` from model features.
8.  Train models on final numeric features.

### Categorical Encoding

The feature-engineering pipeline converts all categorical columns to numeric encodings before saving `data/interim/loans_features.parquet`.

The LightGBM notebook and scripts can also cast object columns to pandas `category` if a raw cleaned fallback dataset is used.

### Models

`src/train.py` supports:

| Model               | Script option                 |
|---------------------|-------------------------------|
| LightGBM            | `--model lightgbm`            |
| Logistic regression | `--model logistic_regression` |
| Random forest       | `--model random_forest`       |
| XGBoost             | `--model xgboost`             |

`src/baseline.py` implements a simple logistic baseline using only `fico_range_low` and grade.

### LightGBM

LightGBM is the main model because it handles nonlinear relationships, interactions, skewed numeric features, and mixed credit-risk signals well.
It is also efficient on large tabular datasets.

Current `src/train.py --model lightgbm` artifact metrics in `mlruns/artifacts/lightgbm_metadata.json`:

| Metric                     |                   Validation |     Test |
|----------------------------|-----------------------------:|---------:|
| ROC-AUC                    |                     0.722606 | 0.710493 |
| Average precision / PR-AUC |                     0.451971 | 0.448317 |
| F1                         | — (not logged by `train.py`) | 0.496237 |
| Brier score                | — (not logged by `train.py`) | 0.211191 |
| Threshold                  |                       0.4877 |   0.4877 |

> Note: `train.py` does not log val F1 or val Brier.
> To get those, run `src/threshold_refinement.py` against the val set or inspect the MLflow run `27a22544698641c0a13a4d8760c791e8` in the `loan-default-predictor` experiment.

### Hyperparameter Tuning

`src/tune_lightgbm.py` uses Optuna to tune LightGBM hyperparameters and writes:

-   `mlruns/artifacts/lightgbm_tuned.txt`
-   `mlruns/artifacts/tuning_metadata.json`

Current `src/tune_lightgbm.py` artifact metrics in `mlruns/artifacts/tuning_metadata.json` (MLflow run `12fce19a2c804034ad8b64328eb9a204`):

> **Leakage-free retrain** completed after removing `loan_age_months`.
> 98 features, 50 Optuna trials, time-based split.

| Metric | Validation | Test (raw) | Test (sigmoid-calibrated) |
|----|---:|---:|---:|
| ROC-AUC | 0.720284 | 0.701052 | 0.701052 |
| Average precision / PR-AUC | 0.444823 | 0.435000 | 0.435000 |
| F1 | — | 0.490785 | — |
| Brier score | — | 0.182594 | **0.175402** |
| Log Loss | — | 0.548030 | 0.526920 |
| Threshold | 0.305 | 0.305 | 0.305 |
| Best iteration | 26 | — | — |
| Features | 98 (no leakage) | — | — |

**Previous run (stale, included `loan_age_months` leakage) for reference:**

| Metric         |   Validation |   Test (raw) |
|----------------|-------------:|-------------:|
| ROC-AUC        | ~~0.727944~~ | ~~0.722071~~ |
| PR-AUC         | ~~0.413130~~ | ~~0.407769~~ |
| F1             |            — | ~~0.453329~~ |
| Brier          |            — | ~~0.161575~~ |
| Best iteration |        ~~8~~ |            — |

Note: the old model's higher test ROC-AUC (0.722 vs 0.701) was partly driven by `loan_age_months` acting as a temporal proxy — it encoded loan vintage, letting the model partially distinguish train-era (high value) from test-era (low value) loans without learning genuine credit signals.
The new honest performance is 0.701 test ROC-AUC.

## 10. Evaluation Metrics

The project uses several standard binary-classification metrics.

| Metric | Meaning | Why it matters |
|------------------------|------------------------|------------------------|
| ROC-AUC | Ranking quality across thresholds | Good general measure of discrimination |
| PR-AUC / average precision | Precision-recall performance | Useful with class imbalance |
| Accuracy | Overall correct classification rate | Easy to understand, but can be misleading with imbalance |
| Precision | Of loans predicted default, how many truly default | Helps avoid falsely rejecting good borrowers |
| Recall | Of actual defaults, how many are caught | Helps avoid approving bad loans |
| F1 | Harmonic mean of precision and recall | Useful single threshold metric |
| Confusion matrix | Counts TP, FP, TN, FN | Helps interpret operational consequences |
| Brier score | Mean squared probability error | Measures probability calibration quality |

Calibration curve and Brier/Log Loss metrics are implemented via `src/calibrate.py` (see Section 20).
Business-facing threshold and expected-loss analytics are implemented via `src/business_risk.py` and the Streamlit **Business Risk Analytics** tab.

## Model Comparison

Phase 10 adds a reproducible model comparison workflow in `src/model_comparison.py`.

The workflow evaluates all models on the same centralized time-based split from `src/splits.py`:

-   train: `issue_d < 2016-01-01`
-   validation: `2016-01-01 <= issue_d < 2017-01-01`
-   test: `issue_d >= 2017-01-01`

It also uses `src/leakage.py` to build the leakage-free feature list and stores that feature list in the JSON artifact for auditability.

### Compared Models

| Model | Notes |
|------------------------------------|------------------------------------|
| FICO/grade logistic baseline | Simple two-feature benchmark using FICO and LendingClub grade |
| Logistic Regression | Full leakage-free feature set |
| Random Forest | Full leakage-free feature set |
| XGBoost | Full leakage-free feature set |
| LightGBM | Full leakage-free feature set |
| Calibrated LightGBM sigmoid | Existing production calibrated model from `models/lightgbm_calibrated_sigmoid.pkl` |

### Outputs

The comparison workflow writes:

``` text
artifacts/model_comparison.csv
artifacts/model_comparison.json
```

The CSV is the human-readable comparison table.
The JSON includes:

-   data source,
-   split strategy,
-   optional training row cap,
-   full feature list,
-   leakage columns present,
-   per-model metrics,
-   summary recommendations.

### Metrics

Reported test metrics:

| Metric | Meaning |
|------------------------------------|------------------------------------|
| `roc_auc` | Ranking quality |
| `pr_auc` / `average_precision` | Precision-recall quality |
| `brier_score` | Probability calibration quality |
| `log_loss` | Probability quality with stronger penalty for confident mistakes |
| `f1` | Thresholded classification balance |
| `training_time_seconds` | Fit time for trained models; scoring/loading comparison time for the pre-trained calibrated artifact |

### Latest Results

Command used:

``` bash
python3 src/model_comparison.py --max-train-rows 200000
```

The validation and test windows were full time-based windows.
Training rows were capped at 200,000 for comparison runtime for newly trained comparison models.
The calibrated sigmoid row uses the existing production artifact.

| Model | ROC-AUC | PR-AUC | Brier | Log Loss | F1 | Training Time |
|-----------|----------:|----------:|----------:|----------:|----------:|----------:|
| LightGBM | 0.706106 | 0.442494 | 0.210386 | 0.606193 | 0.494609 | 3.725s |
| XGBoost | 0.704391 | 0.441801 | 0.177859 | 0.535000 | 0.492217 | 6.237s |
| Calibrated LightGBM sigmoid | 0.701052 | 0.435000 | 0.175399 | 0.526919 | 0.490993 | 0.075s |
| Random Forest | 0.697311 | 0.430738 | 0.211647 | 0.608582 | 0.488426 | 3.903s |
| Logistic Regression | 0.690643 | 0.415379 | 0.216593 | 0.627836 | 0.481552 | 0.984s |
| FICO/grade logistic baseline | 0.669895 | 0.392886 | 0.221947 | 0.635531 | 0.472395 | 0.033s |

### Summary Recommendations

| Category | Recommendation |
|------------------------------------|------------------------------------|
| Best ranking model | LightGBM has the strongest ROC-AUC / PR-AUC in this comparison |
| Best calibrated probability model | Calibrated LightGBM sigmoid has the best Brier Score and Log Loss |
| Most interpretable model | FICO/grade logistic baseline |
| Production recommendation | Keep Calibrated LightGBM sigmoid as the production probability model unless a future model clearly improves calibration, ranking, and operational complexity |

### Tests

`tests/test_model_comparison.py` verifies:

-   comparison CSV and JSON outputs exist,
-   required metric columns exist,
-   no leakage columns are used,
-   metrics are in valid ranges,
-   all six required models are present.

Latest Phase 10 test result:

``` text
python3 -m pytest tests/test_model_comparison.py -v
5 passed

python3 -m pytest tests/ -q
203 passed
```

## Business Risk Analytics

Phase 9 adds business risk analytics on top of the calibrated probability pipeline.
These calculations use calibrated PD values from `models/lightgbm_calibrated_sigmoid.pkl`, not raw LightGBM scores.

### Source Code

The implementation lives in `src/business_risk.py`.

| Function | Purpose |
|------------------------------------|------------------------------------|
| `calculate_expected_loss(pd, lgd=0.45, ead=None)` | Calculates expected loss from calibrated PD, loss given default, and exposure |
| `assign_risk_band(pd)` | Converts calibrated PD into a business risk band |
| `threshold_business_table(y_true, y_proba, loan_amounts, lgd)` | Builds an approval/rejection and expected-loss table across thresholds |

### Expected Loss

Expected Loss is calculated as:

``` text
Expected Loss = PD × LGD × EAD
```

Where:

| Term | Meaning | Project default |
|------------------------|------------------------|------------------------|
| `PD` | Calibrated probability of default | From `models/lightgbm_calibrated_sigmoid.pkl` |
| `LGD` | Loss given default | `0.45` |
| `EAD` | Exposure at default | `loan_amnt` |

Example:

``` text
PD = 0.20
LGD = 0.45
EAD = $10,000

Expected Loss = 0.20 × 0.45 × 10,000 = $900
```

### Risk Bands

Risk bands are assigned from calibrated PD:

| Risk band | Rule                |
|-----------|---------------------|
| Low       | `PD < 0.10`         |
| Medium    | `0.10 <= PD < 0.20` |
| High      | `0.20 <= PD < 0.35` |
| Very High | `PD >= 0.35`        |

### Threshold Business Table

The threshold business table treats loans below the selected PD threshold as approved and loans at or above the threshold as rejected or routed to review.

It includes:

| Column | Meaning |
|------------------------------------|------------------------------------|
| `threshold` | Calibrated PD cutoff |
| `approval_rate` | Share of loans with PD below threshold |
| `rejection_rate` | Share of loans with PD at or above threshold |
| `default_rate_approved` | Observed default rate among approved loans |
| `defaults_caught_rate` | Share of actual defaults rejected/routed to review |
| `expected_loss_approved` | Expected loss retained in approved loans |
| `expected_loss_rejected` | Expected loss avoided or routed to review |
| `total_expected_loss` | Portfolio expected loss before threshold action |

### Dashboard Integration

The Streamlit dashboard now has a **Business Risk Analytics** tab.

The tab provides:

-   threshold slider,
-   LGD input,
-   approval rate,
-   rejection rate,
-   default rate among approved loans,
-   defaults caught rate,
-   approved expected loss,
-   rejected expected loss,
-   total expected loss,
-   risk-band distribution chart,
-   threshold business table.

This tab turns calibrated model probabilities into business decisions: how many loans would be approved, how much risk remains in the approved book, how much expected loss is routed away, and how risk is distributed across bands.

### Tests

`tests/test_business_risk.py` verifies:

-   expected loss calculation is correct,
-   risk bands are assigned correctly,
-   threshold table has all required columns,
-   approval rate decreases as the threshold becomes stricter,
-   expected loss is non-negative.

Latest Phase 9 test results:

``` text
python3 -m pytest tests/test_business_risk.py -v
7 passed

python3 -m pytest tests/ -q
203 passed
```

## SHAP Explainability

Explainability matters in credit risk because model outputs influence lending decisions, customer treatment, portfolio monitoring, and business conversations.
A default probability is useful, but reviewers also need to know which origination-time factors drove the score and whether the model is relying on sensible credit signals.

This project uses SHAP for two levels of explanation:

| Explanation level | Question answered | Project output |
|------------------------|------------------------|------------------------|
| Global | Which features matter most across many loans? | `artifacts/shap_summary_beeswarm.png`, `artifacts/shap_summary_bar.png`, `artifacts/shap_top_features.csv` |
| Local | Why did this specific loan receive this risk score? | `artifacts/shap_local_example_waterfall.png`, `artifacts/shap_local_example.json`, dashboard single-loan explanation |

The current SHAP script uses the latest leakage-free LightGBM artifact:

-   Model: `mlruns/artifacts/lightgbm_model.pkl`
-   Metadata: `mlruns/artifacts/lightgbm_metadata.json`
-   Split: `src/splits.py` time-based test set (`issue_d >= 2017-01-01`)
-   Feature policy: `src/leakage.py`
-   Calibration: SHAP explanations intentionally use the raw LightGBM model. Production probability outputs use `models/lightgbm_calibrated_sigmoid.pkl`.

### Global Explanation

The latest generated global SHAP artifacts were created with:

``` bash
python src/explain_shap.py --sample 2000
```

Top global drivers from `artifacts/shap_top_features.csv`:

| Rank | Feature | Mean absolute contribution | Direction note | Business meaning |
|--------------:|---------------|--------------:|---------------|---------------|
| 1 | `sub_grade_enc` | 0.237732 | Higher values tend to increase predicted default risk | Weaker LendingClub sub-grades indicate higher borrower risk |
| 2 | `int_rate_x_term` | 0.193359 | Higher values tend to increase predicted default risk | High interest rates combined with longer terms can indicate repayment uncertainty |
| 3 | `acc_open_past_24mths` | 0.126229 | Higher values tend to increase predicted default risk | Many recently opened accounts can indicate fast credit growth and higher risk |
| 4 | `term_60` | 0.113080 | Higher values tend to increase predicted default risk | Longer repayment horizon increases uncertainty |
| 5 | `fico_dti_score` | 0.098390 | Higher values tend to reduce predicted default risk | Stronger credit score relative to debt burden lowers risk |
| 6 | `int_rate` | 0.089385 | Higher values tend to reduce predicted default risk in this sample | Interest rate remains an important credit-risk signal, but its standalone effect is entangled with `int_rate_x_term` |
| 7 | `home_ownership_enc` | 0.082594 | Higher values tend to reduce predicted default risk | Home ownership status can proxy borrower stability and housing obligations |
| 8 | `loan_to_income` | 0.081574 | Higher values tend to increase predicted default risk | Larger loan amount relative to income can make repayment harder |
| 9 | `state_default_rate` | 0.079936 | Higher values tend to increase predicted default risk | Geography can capture regional or portfolio risk patterns |
| 10 | `grade_enc` | 0.078480 | Higher values tend to increase predicted default risk | Weaker grades indicate higher borrower risk |

Required business interpretation table:

| Feature | Effect on Risk | Business Meaning |
|------------------------|------------------------|------------------------|
| `int_rate` | Higher increases risk | Riskier borrowers receive higher rates |
| `fico_range_low` | Higher decreases risk | Stronger credit score lowers risk |
| `dti` | Higher increases risk | More debt burden raises default probability |
| `term_60` | Higher increases risk | Longer repayment horizon increases uncertainty |

### Local Explanation

The local example in `artifacts/shap_local_example.json` explains one high-risk test-set loan:

| Field                            |     Value |
|----------------------------------|----------:|
| Predicted probability of default |  0.945127 |
| Risk label                       | Very High |
| Base value                       |  0.447670 |
| Actual default label             |         1 |

Top factors increasing risk in the example:

| Feature | Explanation |
|------------------------------------|------------------------------------|
| `int_rate_x_term` | Interest rate and term combination increases predicted default risk |
| `sub_grade_enc` | LendingClub sub-grade increases predicted default risk |
| `term_60` | 60-month loan term increases predicted default risk |
| `fico_dti_score` | Credit score compared with debt burden increases predicted default risk |
| `grade_enc` | LendingClub grade increases predicted default risk |

Top factors reducing risk in the example:

| Feature | Explanation |
|------------------------------------|------------------------------------|
| `total_bc_limit` | Higher bankcard credit limit reduces predicted default risk |
| `purpose_rate_enc` | Loan purpose risk pattern reduces predicted default risk |
| `total_il_high_credit_limit` | Higher installment credit limit reduces predicted default risk |
| `emp_title_log_freq` | Employment-title stability signal reduces predicted default risk |
| `mo_sin_rcnt_rev_tl_op` | Less recent revolving account opening reduces predicted default risk |

### Leakage Detection and Trust

SHAP helps detect leakage because suspicious post-origination fields would appear as extremely important global drivers.
For example, columns such as `recoveries`, `total_pymnt`, `last_pymnt_amnt`, or `loan_age_months` would be red flags because they are not available at origination.
The SHAP script checks `src/leakage.py` before generating outputs, and `tests/test_explainability.py` asserts that leakage columns do not appear in SHAP outputs.

For business users, the dashboard avoids technical SHAP wording and shows:

-   predicted probability of default,
-   risk label,
-   factors increasing risk,
-   factors reducing risk,
-   a compact bar chart for the selected loan,
-   a short plain-English explanation.

This makes the model easier to defend in interviews and easier for reviewers to use without needing to understand the SHAP algorithm itself.

### SHAP Implementation Process

This is the process used for the latest explainability update.

1.  **Audit existing explainability paths**

Reviewed:

-   `src/explain_shap.py`
-   `src/explain_utils.py`
-   `app/main.py`
-   `api/main.py`
-   `doc.md`
-   model metadata in `mlruns/artifacts/lightgbm_metadata.json`

Audit findings:

| Area | Finding | Action taken |
|------------------------|------------------------|------------------------|
| SHAP script | Needed to use the latest leakage-free LightGBM artifact and time-based test split | Updated `src/explain_shap.py` to load `lightgbm_model.pkl`, `lightgbm_metadata.json`, `src/leakage.py`, and `src/splits.py` |
| Leakage policy | SHAP feature names must match the model feature list and exclude post-origination fields | Added explicit leakage checks before artifact generation |
| Calibration | Production PD should use the sigmoid-calibrated model while SHAP should preserve raw LightGBM explanations | API/dashboard now load `models/lightgbm_calibrated_sigmoid.pkl` for PD and raw `lightgbm_model.pkl` only for SHAP |
| Dashboard | Previously used older tuned artifact paths and technical SHAP wording | Updated to calibrated PDs, current artifact paths, and business-friendly labels: "Factors increasing risk" and "Factors reducing risk" |
| API | Previously used older tuned artifact paths and included `loan_age_months` in the request schema | Updated to calibrated PDs, current artifact paths, and removed `loan_age_months` |
| Tests | No dedicated explainability tests existed | Added `tests/test_explainability.py` |

2.  **Generate global and local SHAP artifacts**

Command used:

``` bash
python src/explain_shap.py --sample 2000
```

Generated files:

| Artifact | Purpose |
|------------------------------------|------------------------------------|
| `artifacts/shap_summary_beeswarm.png` | Global feature impact distribution |
| `artifacts/shap_summary_bar.png` | Top global feature importance |
| `artifacts/shap_top_features.csv` | Ranked global drivers with business interpretations |
| `artifacts/shap_local_example_waterfall.png` | Individual-loan explanation plot |
| `artifacts/shap_local_example.json` | Individual-loan explanation payload for review/dashboard use |

The script sampled 2,000 loans from the 2017+ time-based test set and used 98 leakage-free model features.

3.  **Log SHAP artifacts to MLflow**

The artifacts were logged to MLflow under run:

``` text
5924a9f4fd9449a995006cd3a3b59995
```

Logged files:

-   `shap_summary_beeswarm.png`
-   `shap_summary_bar.png`
-   `shap_top_features.csv`
-   `shap_local_example_waterfall.png`
-   `shap_local_example.json`

Note: MLflow printed a warning that `mlruns/artifacts` is not a valid experiment directory because it is used as a local artifact folder.
The SHAP logging still succeeded under the `loan-default-explainability` experiment.

4.  **Dashboard integration**

The Streamlit dashboard now:

-   loads `models/lightgbm_calibrated_sigmoid.pkl` for all probability outputs,
-   keeps `mlruns/artifacts/lightgbm_model.pkl` only for SHAP explanations,
-   reads `mlruns/artifacts/lightgbm_metadata.json`,
-   reconstructs test predictions with `src/splits.py`,
-   removes `loan_age_months` from presets and user inputs,
-   shows predicted default probability and risk label,
-   shows top five factors increasing risk,
-   shows top five factors reducing risk,
-   shows a local explanation bar chart,
-   avoids technical SHAP wording in the main prediction view.

Startup was verified on port `8502` because port `8501` was already in use:

``` bash
streamlit run app/main.py --server.headless true --server.port 8502
```

The app responded successfully at:

``` text
http://localhost:8502
```

5.  **API alignment**

The FastAPI service now:

-   loads `models/lightgbm_calibrated_sigmoid.pkl`,
-   reads `mlruns/artifacts/lightgbm_metadata.json`,
-   removes `loan_age_months` from the request schema,
-   imports `src.calibration_classes` before unpickling,
-   returns calibrated PD with `predict_proba(df)[0, 1]`.

This keeps API scoring aligned with the leakage-free feature list while using calibrated probabilities for production-facing outputs.
SHAP remains raw-LightGBM-based for explainability.

6.  **Validation checks**

Commands run:

``` bash
python3 -m py_compile app/main.py api/main.py src/explain_utils.py src/explain_shap.py tests/test_explainability.py
python3 -m pytest tests/test_explainability.py -v
python3 -m pytest tests/test_api.py -v
python3 -m pytest tests/ -q
```

Results:

| Check                |         Result |
|----------------------|---------------:|
| Explainability tests |       5 passed |
| API tests            |      19 passed |
| Calibration tests    |      38 passed |
| Full test suite      | **203 passed** |

The explainability tests verify that:

-   SHAP feature names match model metadata feature names,
-   no leakage columns appear in SHAP outputs,
-   `shap_top_features.csv` exists,
-   local explanation JSON contains required fields,
-   business-friendly helper text is readable.

## 11. Experiment Tracking

MLflow is used for experiment tracking.
The local tracking directory is `mlruns/`.

Scripts log or save:

| Item | Examples |
|------------------------------------|------------------------------------|
| Parameters | model name, LightGBM params, threshold, split strategy, feature count |
| Metrics | ROC-AUC, average precision, F1, Brier score, training time |
| Artifacts | model files, metadata JSON, encoding maps, SHAP plots |
| Model files | `lightgbm_model.pkl`, `lightgbm_tuned.txt` |
| Metadata | `lightgbm_metadata.json`, `tuning_metadata.json` |
| Plots | SHAP importance, threshold analysis outputs |

Recommended MLflow practice:

-   Log `split_strategy = time_based_issue_d`.
-   Log the exact train/validation/test date ranges.
-   Log leakage-policy version or `LEAKAGE_COLS`.
-   Log feature list and feature count.
-   Log calibration metrics before and after calibration.

## 12. API

The FastAPI service is implemented in `api/main.py`.

### Purpose

The API scores one loan application at a time and returns a default probability, binary prediction, risk label, threshold, and model version.

### Model Artifacts Used

As currently coded, the API loads:

-   Production probability model: `models/lightgbm_calibrated_sigmoid.pkl`
-   Raw explanation model: `mlruns/artifacts/lightgbm_model.pkl` when SHAP explanations are needed
-   Metadata: `mlruns/artifacts/lightgbm_metadata.json`
-   Encoding maps: `mlruns/artifacts/encoding_maps.json`

If those files are missing, API startup or requests will fail with an operator-visible error.

### Endpoints

| Endpoint      | Method | Purpose                                         |
|---------------|--------|-------------------------------------------------|
| `/health`     | GET    | Checks that metadata and encoding maps can load |
| `/predict`    | POST   | Scores one loan application                     |
| `/model-info` | GET    | Returns model metadata and performance metrics  |

### Input Format

The `/predict` endpoint accepts a JSON payload matching the `LoanApplication` schema in `api/main.py`.
Required fields include loan amount, term, interest rate, installment, grade, sub-grade, annual income, verification status, purpose, state, DTI, FICO, and other origination-time credit fields.

Example:

``` json
{
  "loan_amnt": 10000.0,
  "term": 36,
  "int_rate": 12.5,
  "installment": 334.87,
  "grade": "B",
  "sub_grade": "B3",
  "emp_length": 5.0,
  "home_ownership": "RENT",
  "annual_inc": 60000.0,
  "verification_status": "Verified",
  "purpose": "debt_consolidation",
  "addr_state": "CA",
  "dti": 15.0,
  "fico_range_low": 700.0
}
```

### Output Format

The API returns:

``` json
{
  "default_probability": 0.123456,
  "prediction": 0,
  "threshold": 0.4877,
  "risk_label": "Low",
  "model_version": "mlflow_run_id"
}
```

`default_probability` is calibrated with sigmoid/Platt scaling via `models/lightgbm_calibrated_sigmoid.pkl`.

### Run Locally

``` bash
uvicorn api.main:app --reload
```

Swagger docs are available at `http://localhost:8000/docs`.

## 13. Dashboard

The Streamlit dashboard is implemented in `app/main.py`.

### Purpose

The dashboard provides a user-facing interface for:

-   scoring a single loan application,
-   exploring threshold trade-offs,
-   scoring a CSV batch of applications,
-   showing model metadata and performance metrics,
-   displaying local SHAP explanations for individual predictions,
-   analyzing approval/rejection and expected-loss trade-offs.

### Main Components

| Component | Purpose |
|------------------------------------|------------------------------------|
| Sidebar | Shows model metrics, threshold, data source, and run ID |
| Predict tab | Single-loan form, presets, calibrated probability output, risk label, factors increasing/reducing risk, and local explanation bar chart |
| Threshold Analysis tab | Precision/recall threshold explorer and confusion matrix using calibrated PDs |
| Business Risk Analytics tab | Approval rate, rejection rate, default rate among approved loans, defaults caught, expected loss, and risk-band distribution |
| Batch Score tab | Upload CSV and download scored predictions using calibrated PDs |

### Run Locally

``` bash
streamlit run app/main.py
```

## 14. Testing

The test suite is in `tests/`.

| Test file | Tests | Coverage |
|------------------------|------------------------|------------------------|
| `tests/test_data_cleaning.py` | 10 | Parsing helpers: term, employment length, percent, month calculations |
| `tests/test_feature_engineering.py` | 52 | Encoding maps, feature creation, log transforms, ratio features, validation |
| `tests/test_leakage.py` | 3 | Required leakage columns and leakage-free final feature columns |
| `tests/test_splits.py` | 2 | Fixed `issue_d` split boundaries and feature split behavior |
| `tests/test_explainability.py` | 5 | SHAP feature alignment, leakage-free SHAP outputs, artifact checks |
| `tests/test_threshold_refinement.py` | 22 | Threshold sweep, max-F1, precision/recall constraints, cost-sensitive thresholding |
| `tests/test_api.py` | 19 | FastAPI health/model-info/predict behavior, calibrated model loading, no-leakage serving metadata |
| `tests/test_calibration.py` | 38 | Artifact existence, model loading, predict_proba output, calibration metrics, leakage audit |
| `tests/test_business_risk.py` | 7 | Expected loss, risk bands, threshold business table, monotonic approval behavior |
| `tests/test_model_comparison.py` | 5 | Model comparison artifacts, required columns, leakage-free features, metric ranges |

**Current result from the latest run:**

``` text
203 passed in 2.80s
```

Remaining recommended tests:

-   Assert cleaned data has expected row count, `issue_d` dtype, and zero nulls (integration test against the actual parquet file).
-   Assert feature-engineered data has no object/category columns after engineering.
-   Assert dashboard test reconstruction uses `src/splits.py` split boundaries.

## 15. Reproducibility

### Python and Dependencies

The README lists Python 3.14.
Dependencies are installed from `requirements.txt`.

Major dependency groups:

-   Data: pandas, numpy, pyarrow
-   ML: scikit-learn, LightGBM, XGBoost, imbalanced-learn
-   Tracking: MLflow
-   Explainability: SHAP
-   API: FastAPI, Uvicorn, Pydantic
-   Dashboard: Streamlit, Plotly
-   Tuning: Optuna
-   Testing: pytest, pytest-cov, httpx

### Data Location

Place raw data here:

``` text
data/raw/archive/
├── accepted_2007_to_2018Q4.csv.gz
└── rejected_2007_to_2018Q4.csv.gz
```

### Reproducible Command Flow

``` bash
pip install -r requirements.txt

# 1. Data
python src/data_cleaning.py           # → data/processed/loans_cleaned.parquet
python src/feature_engineering.py     # → data/interim/loans_features.parquet

# 2. Model
python src/tune_lightgbm.py --n-trials 50  # → mlruns/artifacts/lightgbm_tuned.txt

# 3. Calibration  (always run after training)
python src/calibrate.py --n-bins 15   # → models/*.pkl, artifacts/calibration_curve.png

# 4. Threshold analysis
python src/threshold_refinement.py    # prints precision/recall strategy comparison

# 5. Serve
mlflow ui                             # http://localhost:5000
uvicorn api.main:app --reload         # http://localhost:8000/docs
streamlit run app/main.py             # http://localhost:8501

# 6. Test
pytest tests/ -v                      # 179 tests expected
```

Full rerun after changing leakage definitions or split logic:

``` bash
python src/data_cleaning.py
python src/feature_engineering.py
python src/tune_lightgbm.py --n-trials 50
python src/calibrate.py --n-bins 15
python src/threshold_refinement.py
pytest tests/ -v
```

### Output Directories Created by the Pipeline

| Directory | Created by | Contents |
|------------------------|------------------------|------------------------|
| `data/processed/` | `src/data_cleaning.py` | `loans_cleaned.parquet` |
| `data/interim/` | `src/feature_engineering.py` | `loans_features.parquet` |
| `mlruns/artifacts/` | `src/tune_lightgbm.py` | `lightgbm_tuned.txt`, `tuning_metadata.json`, `encoding_maps.json` |
| `models/` | `src/calibrate.py` | `lightgbm_raw.pkl`, `lightgbm_calibrated_sigmoid.pkl`, `lightgbm_calibrated_isotonic.pkl` |
| `artifacts/` | `src/calibrate.py` | `calibration_curve.png`, `calibration_results.json` |

## 16. Current Strengths

-   Clear business framing: predict default risk at loan origination.
-   Strong leakage controls through centralized `src/leakage.py`.
-   Time-based split using `issue_d`, which better mimics deployment.
-   Train-only fitting for target/frequency encodings.
-   Full data pipeline from raw CSV to cleaned and feature-engineered parquet.
-   Multiple model options in `src/train.py`.
-   Dedicated LightGBM tuning script with Optuna.
-   Threshold-refinement tools for operational decision-making.
-   SHAP explainability script with global and local artifacts.
-   Dashboard-level explanations using business-friendly factor descriptions.
-   FastAPI service for calibrated model inference.
-   Streamlit dashboard for calibrated demos and analysis.
-   Production probability pipeline: API, dashboard single scoring, dashboard threshold analysis, and batch scoring all use `models/lightgbm_calibrated_sigmoid.pkl`.
-   Business risk analytics: expected loss, approval/rejection rates, defaults caught, and risk-band distribution using calibrated PDs.
-   Model comparison table across FICO/grade baseline, Logistic Regression, Random Forest, XGBoost, LightGBM, and calibrated LightGBM sigmoid.
-   Automated tests for cleaning, feature engineering, leakage, split logic, API, thresholding, calibration, business risk analytics, and model comparison (203 tests passing).
-   MLflow-based experiment tracking with a dedicated calibration experiment.
-   Probability calibration via `src/calibrate.py` — sigmoid and isotonic methods compared, winner persisted as a picklable sklearn-compatible object in `models/`.
-   Picklable calibration classes in `src/calibration_classes.py` — safe across script execution contexts, test runners, the API, and the dashboard.
-   `loan_age_months` leakage fully resolved: removed from pipeline, model retrained, `TestLeakageInvariant` enforces permanent zero-leakage invariant on every `pytest` run.

## 17. Recommended Improvements

Prioritized improvements:

1.  ~~**Leakage audit — retrain without `loan_age_months`**~~ ✅ **DONE**: `loan_age_months` removed from all pipeline stages; model retrained with 98 leakage-free features; `TestLeakageInvariant` enforces the invariant permanently. See Section 6 and Section 9 for metric comparison.
2.  ~~**Wire calibrated model into the API**~~ ✅ **DONE**: `api/main.py` imports `src.calibration_classes`, loads `models/lightgbm_calibrated_sigmoid.pkl`, and returns calibrated PD with `predict_proba(df)[0, 1]`.
3.  ~~**Wire calibrated model into the dashboard**~~ ✅ **DONE**: `app/main.py` uses the sigmoid-calibrated model for single scoring, threshold analysis, and batch scoring; raw LightGBM is preserved only for SHAP explanations.
4.  ~~**Business threshold analysis**~~ ✅ **DONE**: `src/business_risk.py` and the dashboard Business Risk Analytics tab show approval rate, rejection rate, default rate among approved loans, defaults caught, expected loss, and risk-band distribution using calibrated PDs.
5.  ~~**Model comparison table**~~ ✅ **DONE**: `src/model_comparison.py` compares FICO/grade baseline, Logistic Regression, Random Forest, XGBoost, LightGBM, and calibrated LightGBM sigmoid on the same time-based split and saves CSV/JSON artifacts.
6.  **SHAP explainability** *(medium priority)*: artifacts already generated; add a compact explanation JSON payload to the `/predict` API response and expose it in the Streamlit Predict tab beyond the current heuristic driver bullets.
7.  **Stronger integration tests** *(medium priority)*: add tests that load the actual cleaned/feature parquets and assert row count, no nulls, correct `issue_d` range, and correct default rate — so a broken pipeline is caught before training.
8.  **Docker support** *(lower priority)*: Dockerfile and docker-compose for API + dashboard + MLflow.
9.  **CI/CD GitHub Actions** *(lower priority)*: run `pytest tests/ -v`, linting, and a calibration smoke-test on every pull request.

## 18. Interview Talking Points

### Business Problem

"I built a loan default prediction system for LendingClub loans. The goal is to estimate default risk at origination time so a lender can rank applications, review high-risk loans, and choose approval thresholds based on business trade-offs."

### Why Leakage Matters

"Credit datasets often contain fields created after the loan is issued, such as total payments, recoveries, settlement status, and outstanding principal. Those fields would make the model look artificially strong because they reveal the outcome. I centralized leakage exclusions in `src/leakage.py` and added tests to make sure those fields never enter final model features."

### Why Time-Based Split Matters

"A random split can put loans from the same economic period in both train and test. That can overstate performance. In production, the model is trained on historical loans and used on future applications. I use `issue_d` to train on pre-2016 loans, validate on 2016 loans, and test on 2017+ loans."

### Why LightGBM Is Suitable

"LightGBM is a strong choice for large tabular credit-risk data. It captures nonlinearities and interactions, handles many numeric features efficiently, and performs well with engineered ratios, ordinal encodings, and target encodings."

### How To Explain Performance

"I would focus on ROC-AUC for ranking quality, PR-AUC because defaults are the minority class, F1 and confusion matrices for a chosen threshold, and Brier score or calibration curves when probabilities need to support business decisions. On the time-based test set (2017–2018 loans), the tuned LightGBM achieves ROC-AUC 0.708 and PR-AUC 0.444. After sigmoid calibration, Brier Score drops from 0.185 to 0.174 — a 5.9% improvement — while ranking quality is fully preserved."

### Why Calibration Matters

"A model's ROC-AUC tells you it can rank defaulters above non-defaulters, but it says nothing about whether a score of 0.20 actually means 20% of those loans will default. In credit risk, that accuracy matters — expected-loss calculations, risk-tier assignments, and loan pricing all depend on the raw probability being a trustworthy estimate. I added a calibration step that fits a Platt scaler on the 2016 validation set and evaluates on the 2017–2018 test set. Both sigmoid and isotonic calibration cut the Brier Score by about 6% with no degradation to ROC-AUC. I also separated the wrapper and calibration classes into `src/calibration_classes.py` so they remain picklable across different execution contexts — a practical detail that tripped up the initial implementation when pytest couldn't load models pickled from `__main__`."

### What To Improve Next

"My next steps would be probability calibration, expected-loss analysis, a model comparison table on the same time split, dashboard/API alignment with the latest artifacts, and CI/CD tests for reproducibility."

## 19. Commands

``` bash
pip install -r requirements.txt
python src/data_cleaning.py
python src/feature_engineering.py
python src/train.py --model lightgbm
python src/tune_lightgbm.py --n-trials 50
python src/calibrate.py --n-bins 15
mlflow ui
uvicorn api.main:app --reload
streamlit run app/main.py
pytest tests/ -v
```

Additional useful commands:

``` bash
python src/train.py --model logistic_regression
python src/train.py --model random_forest
python src/train.py --model xgboost
python src/baseline.py
python src/calibrate.py
python src/calibrate.py --n-bins 15
python src/threshold_refinement.py
python src/threshold_refinement.py --strategy cost_sensitive --fn-cost 5 --fp-cost 1
python src/explain_shap.py
python src/explain_shap.py --sample 2000
python3 -m pytest tests/test_business_risk.py -v
python3 src/model_comparison.py --max-train-rows 200000
python3 -m pytest tests/test_model_comparison.py -v
```

------------------------------------------------------------------------

## 20. Probability Calibration

### Why Calibration Matters in Credit Risk

In credit risk, a model's predicted **probability of default (PD)** is used for more than ranking — it drives real financial decisions:

-   **Expected Loss (EL)** = PD × Loss Given Default (LGD) × Exposure at Default (EAD)
-   **Risk pricing:** a higher PD justifies a higher interest rate to offset expected loss
-   **Risk tiering:** cut-offs like "Low / Medium / High / Very High" assume that a score of 0.30 truly means 30% of loans with that score will default

There is a critical distinction between two types of model quality:

| Quality type | Metric | Question it answers |
|------------------------|------------------------|------------------------|
| **Ranking quality** | ROC-AUC, PR-AUC | Does the model put defaulters above non-defaulters? |
| **Probability quality** | Brier Score, Log Loss | Are the predicted probabilities accurate as absolute estimates? |

A model can have excellent ROC-AUC (0.76+) but still be poorly calibrated.
LightGBM trained with `scale_pos_weight` or `class_weight=balanced` tends to produce inflated probabilities — good for ranking, wrong for dollar-denominated calculations.

**Interpretation example:**

```         
Model predicts PD = 0.20 for a loan application.

Well-calibrated model:
  → ~20 out of 100 loans with this score actually default.
  → Lender can safely price: rate = base_rate + 0.20 × LGD × EAD

Poorly calibrated model:
  → Model predicts 0.20 but 35% of those loans actually default.
  → Lender systematically under-prices risk and absorbs unexpected losses.
```

### Calibration Methods

**Script:** [`src/calibrate.py`](src/calibrate.py)

Both methods are implemented in `src/calibration_classes.py` via `PreFitCalibratedClassifier`, which is the direct replacement for `CalibratedClassifierCV(cv="prefit")` (removed in sklearn 1.8):

-   The base LightGBM model is already trained — only the calibration layer is fit on new data.
-   Calibration is always fit on the **2016 validation set** to avoid contaminating the test set.
-   All reported metrics are evaluated on the **held-out 2017–2018 test set**.
-   Split uses `src/splits.py` (time-based, `issue_d` cutoffs).

| Method | How it works | Best when |
|------------------------|------------------------|------------------------|
| **Sigmoid (Platt scaling)** | Fits a logistic regression layer on top of raw scores | Validation data is limited; assumes a smooth sigmoid relationship |
| **Isotonic regression** | Fits a monotone step function on the sorted raw scores | Enough validation data (≥ 1 000 per class); makes no parametric assumptions |

With 297,651 validation rows this project had more than enough data for isotonic regression, yet sigmoid won — see actual results below.

### sklearn Wrapper

The native `lgb.Booster` does not implement the sklearn interface.
`src/calibration_classes.py` provides two picklable classes:

``` python
# LGBMBoosterWrapper: wraps lgb.Booster with predict_proba / classes_
wrapper = LGBMBoosterWrapper(booster, best_iteration=26)

# PreFitCalibratedClassifier: equivalent to CalibratedClassifierCV(cv="prefit")
cal_isotonic = PreFitCalibratedClassifier(wrapper, method="isotonic")
cal_isotonic.fit(X_val, y_val)                        # fit on val set only
proba = cal_isotonic.predict_proba(X_test)[:, 1]      # evaluate on test set
```

Keeping the classes in a dedicated module (not `__main__`) ensures pickle can resolve them when the `.pkl` files are loaded from tests, the API, or the dashboard.

### Calibration Metrics

| Metric | What it measures | Target |
|------------------------|------------------------|------------------------|
| **ROC-AUC** | Ranking ability (threshold-independent) | Higher is better |
| **PR-AUC** | Precision-recall trade-off on minority class | Higher is better |
| **Brier Score** | Mean squared error of predicted probabilities (0 = perfect, 0.25 = no skill) | Lower is better |
| **Log Loss** | Cross-entropy of predicted probabilities | Lower is better |

### Calibration Results

> **Confirmed (leakage-free retrain, Phase 8 serving update, Phase 9 business analytics, and Phase 10 model comparison):** `python src/calibrate.py --n-bins 15` artifacts remain leakage-free, and the full suite reports **203/203 tests passing**.
> `leakage_in_model: []`.

Evaluated on the **2017–2018 test set** (240,864 loans, 26.3% default rate).
Calibrators fit on the **2016 validation set** (297,651 loans).
Split: `src/splits.py` time-based (`issue_d` cutoffs — train \<2016, val 2016, test ≥2017).

> **Updated after leakage-free retrain** (MLflow run `c43c7d2675144e66b6203702d06b80e9`).

| Model                      | ROC-AUC     | PR-AUC      | Brier Score | Log Loss    |
|---------------|---------------|---------------|---------------|---------------|
| Raw LightGBM               | 0.70105     | 0.43500     | 0.18259     | 0.54803     |
| **Sigmoid Calibration** ✅ | **0.70105** | **0.43500** | **0.17540** | **0.52692** |
| Isotonic Calibration       | 0.70098     | 0.43141     | 0.17544     | 0.52689     |

**Winner: Sigmoid (Platt scaling)**

| Improvement | Raw → Sigmoid     | \% change            |
|-------------|-------------------|----------------------|
| Brier Score | 0.18259 → 0.17540 | **−3.9%**            |
| Log Loss    | 0.54803 → 0.52692 | **−3.9%**            |
| ROC-AUC     | 0.70105 → 0.70105 | **0.0%** (preserved) |
| PR-AUC      | 0.43500 → 0.43500 | **0.0%** (preserved) |

No-skill Brier baseline = 0.263 × 0.737 ≈ **0.194** — all three models beat it.

Note: isotonic and sigmoid are again extremely close (Brier 0.17540 vs 0.17544).
Sigmoid wins on Brier; isotonic wins on Log Loss by a tiny margin (0.52689 vs 0.52692).
Sigmoid remains the recommended production model for consistency.

### Reliability Diagram

Saved to `artifacts/calibration_curve.png` (generated by `python src/calibrate.py --n-bins 15`).

The reliability diagram plots **mean predicted probability** (x-axis) against **fraction of positives** (y-axis) in equal-sized probability bins:

-   **Perfect calibration** → points fall on the diagonal (y = x)
-   **Overconfident** (typical of boosted trees with class weighting) → curve bows above the diagonal; raw LightGBM scores are higher than observed default rates in the lower probability region
-   **After calibration** → sigmoid and isotonic curves should closely follow the diagonal

The right panel shows the predicted probability distribution for each model, illustrating how calibration shifts the score mass relative to the 26.3% base rate of the test period.

### Saved Model Files

After running `python src/calibrate.py`, three model files are persisted:

| File | Contents | Use case |
|------------------------|------------------------|------------------------|
| `models/lightgbm_raw.pkl` | `LGBMBoosterWrapper` — raw booster output | Ranking, ROC-AUC comparison |
| `models/lightgbm_calibrated_sigmoid.pkl` | `PreFitCalibratedClassifier(method="sigmoid")` | **Recommended for production** |
| `models/lightgbm_calibrated_isotonic.pkl` | `PreFitCalibratedClassifier(method="isotonic")` | Near-identical to sigmoid; retrain if distributions shift |

All three are also logged as MLflow artifacts under the `loan-default-calibration` experiment (latest run ID: `c43c7d2675144e66b6203702d06b80e9`).

Additional outputs: - `artifacts/calibration_curve.png` — reliability diagram + probability distribution - `artifacts/calibration_results.json` — machine-readable metrics for tests and CI

### ✅ Leakage Finding — Resolved

An earlier model artifact included `loan_age_months` in its feature set.
This has been fully resolved:

-   `loan_age_months` is permanently in `LEAKAGE_COLS` (`src/leakage.py`).
-   `src/data_cleaning.py` no longer creates it.
-   `src/feature_engineering.py` explicitly drops all `LEAKAGE_COLS` and validates their absence.
-   `src/calibrate.py` no longer has a backward-compat reconstruction block.
-   The model was **retrained** (50 Optuna trials, 98 leakage-free features).
-   `artifacts/calibration_results.json` confirms `"leakage_in_model": []`.
-   `TestLeakageInvariant` in `tests/test_calibration.py` permanently enforces this — it runs on every `pytest` invocation with no skip condition.

### Production Recommendation

For business decisions that depend on the absolute value of PD (expected-loss pricing, risk-tier assignment, regulatory capital calculations), use **`models/lightgbm_calibrated_sigmoid.pkl`** — it achieved the lowest Brier Score (0.17366) and Log Loss (0.52144) on the 2017–2018 test set with zero degradation in ROC-AUC.

Phase 8 production probability serving is implemented: the API and Streamlit dashboard now use this sigmoid-calibrated model for all displayed or returned default probabilities.

For pure ranking tasks (approval score ordering, model comparison), the raw LightGBM booster is sufficient and faster.

The serving pattern used in `api/main.py` is:

``` python
import pickle
from src.calibration_classes import LGBMBoosterWrapper, PreFitCalibratedClassifier  # needed for unpickling

cal_model = pickle.load(open("models/lightgbm_calibrated_sigmoid.pkl", "rb"))  # sigmoid = winner
prob = float(cal_model.predict_proba(df)[0, 1])   # calibrated PD
```

### How to Run

``` bash
python src/calibrate.py                # isotonic + sigmoid, 10 bins
python src/calibrate.py --n-bins 15   # finer calibration curve
```

Outputs:

```         
artifacts/calibration_curve.png
models/lightgbm_raw.pkl
models/lightgbm_calibrated_sigmoid.pkl
models/lightgbm_calibrated_isotonic.pkl
```
