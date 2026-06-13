"""
src/tune_lightgbm.py
────────────────────
LightGBM Hyperparameter Tuning with Optuna
Step 1 of the modeling pipeline.

Usage:
    python src/tune_lightgbm.py                   # 50 trials (default)
    python src/tune_lightgbm.py --n-trials 100    # custom trial count
    python src/tune_lightgbm.py --n-trials 20 --timeout 1800  # 30-min cap

Outputs:
    • MLflow run: "lgbm_tuning" experiment → best params & all trial metrics
    • MLflow run: "lgbm_best_retrained" → final model trained on train+val
    • mlruns/artifacts/lightgbm_tuned.txt     (model in LightGBM native format)
    • mlruns/artifacts/tuning_metadata.json   (best params, threshold, metrics)
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import optuna
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── Paths ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERIM_PATH = PROJECT_ROOT / "data" / "interim" / "loans_features.parquet"
CLEANED_PATH = PROJECT_ROOT / "data" / "processed" / "loans_cleaned.parquet"
MLFLOW_URI   = str(PROJECT_ROOT / "mlruns")
ARTIFACTS_DIR = PROJECT_ROOT / "mlruns" / "artifacts"

# ── Constants ─────────────────────────────────────────────────────────────
import sys as _sys
_sys.path.insert(0, str(PROJECT_ROOT))
from src.leakage import TARGET as _TARGET, feature_columns  # noqa: E402
from src.splits import (  # noqa: E402
    SPLIT_STRATEGY,
    ensure_issue_datetime,
    print_split_summary,
    split_xy_by_issue_date,
)

TARGET       = _TARGET
RANDOM_STATE = 42
EXPERIMENT   = "loan-default-lightgbm"


# ══════════════════════════════════════════════════════════════════════════
# 1.  Data Loading & Splitting
# ══════════════════════════════════════════════════════════════════════════

def load_data() -> tuple[pd.DataFrame, pd.Series, pd.Series | None, str]:
    """Load cleaned / feature-engineered data.
    Returns (X, y, issue_d_or_None, data_source_label)."""
    if INTERIM_PATH.exists():
        df = pd.read_parquet(INTERIM_PATH)
        source = "interim (feature-engineered)"
    elif CLEANED_PATH.exists():
        df = pd.read_parquet(CLEANED_PATH)
        source = "processed (raw cleaned — LightGBM native categoricals)"
    else:
        raise FileNotFoundError(
            f"No data file found at:\n  {INTERIM_PATH}\n  {CLEANED_PATH}\n"
            "Run data_cleaning.py first."
        )

    # issue_d is kept as a datetime for the fixed time-based split.
    # It's not a feature — it must not land in X.
    if "issue_d" not in df.columns:
        raise ValueError(
            "Time-based split requires issue_d. Re-run src/data_cleaning.py "
            "and src/feature_engineering.py so issue_d is preserved."
        )
    issue_d = ensure_issue_datetime(df["issue_d"])
    feature_cols = feature_columns(df.columns, extra_drop=["issue_d"])

    X = df[feature_cols].copy()
    y = df[TARGET].astype("int32")

    # Cast string/object columns to pandas Categorical for LightGBM
    for col in X.select_dtypes(include=["object"]).columns:
        X[col] = X[col].astype("category")

    return X, y, issue_d, source


def make_splits(
    X: pd.DataFrame,
    y: pd.Series,
    issue_d: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
           pd.Series,     pd.Series,     pd.Series]:
    """Split data into fixed origination-date windows."""
    return split_xy_by_issue_date(X, y, issue_d)


# ══════════════════════════════════════════════════════════════════════════
# 2.  LightGBM Datasets
# ══════════════════════════════════════════════════════════════════════════

def make_lgb_datasets(
    X_train, y_train, X_val, y_val
) -> tuple[lgb.Dataset, lgb.Dataset, list[str]]:
    cat_cols = X_train.select_dtypes(include=["category"]).columns.tolist()
    lgb_train = lgb.Dataset(
        X_train, label=y_train,
        categorical_feature=cat_cols or "auto",
        free_raw_data=False,
    )
    lgb_val = lgb.Dataset(
        X_val, label=y_val,
        categorical_feature=cat_cols or "auto",
        reference=lgb_train,
        free_raw_data=False,
    )
    return lgb_train, lgb_val, cat_cols


# ══════════════════════════════════════════════════════════════════════════
# 3.  Threshold Optimisation
# ══════════════════════════════════════════════════════════════════════════

from src.threshold_refinement import ThresholdRefiner, print_report


def best_f1_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> tuple[float, float]:
    """
    Sweep thresholds on val set via ThresholdRefiner; return (best_threshold, best_f1).
    Uses max-F1 strategy (strategy-neutral default).
    """
    refiner = ThresholdRefiner(y_true, y_proba, n_thresholds=500)
    result  = refiner.max_f1()
    return result["threshold"], result["f1"]


# ══════════════════════════════════════════════════════════════════════════
# 4.  Optuna Objective
# ══════════════════════════════════════════════════════════════════════════

def build_objective(
    lgb_train: lgb.Dataset,
    lgb_val: lgb.Dataset,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    scale_pos_weight: float,
):
    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective":          "binary",
            "metric":             "auc",
            "verbosity":          -1,
            "boosting_type":      "gbdt",
            "random_state":       RANDOM_STATE,
            "n_jobs":             -1,
            # ── Tree structure ────────────────────────────────────────────
            "num_leaves":         trial.suggest_int("num_leaves", 31, 255),
            "max_depth":          trial.suggest_int("max_depth", 4, 12),
            "min_child_samples":  trial.suggest_int("min_child_samples", 10, 200),
            # ── Boosting ─────────────────────────────────────────────────
            "learning_rate":      trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            "subsample":          trial.suggest_float("subsample", 0.5, 1.0),
            "subsample_freq":     1,
            "colsample_bytree":   trial.suggest_float("colsample_bytree", 0.5, 1.0),
            # ── Regularisation ───────────────────────────────────────────
            "reg_alpha":          trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":         trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "min_split_gain":     trial.suggest_float("min_split_gain", 0.0, 1.0),
            # ── Class imbalance ──────────────────────────────────────────
            "scale_pos_weight":   trial.suggest_float(
                                      "scale_pos_weight",
                                      max(1.0, scale_pos_weight * 0.5),
                                      scale_pos_weight * 2.0,
                                  ),
        }

        callbacks = [
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=-1),
        ]

        model = lgb.train(
            params,
            train_set=lgb_train,
            num_boost_round=1000,
            valid_sets=[lgb_val],
            valid_names=["val"],
            callbacks=callbacks,
        )

        val_proba = model.predict(X_val, num_iteration=model.best_iteration)
        auc = roc_auc_score(y_val, val_proba)

        # Store extra attributes for later retrieval
        trial.set_user_attr("best_iteration", model.best_iteration)
        trial.set_user_attr("val_avg_precision", float(average_precision_score(y_val, val_proba)))

        return auc

    return objective


# ══════════════════════════════════════════════════════════════════════════
# 5.  Final Model — retrain on train + val
# ══════════════════════════════════════════════════════════════════════════

def train_final_model(
    best_params: dict,
    best_iteration: int,
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    cat_cols: list[str],
    data_source: str,
    mlflow_run_id_study: str,
    split_strategy: str = SPLIT_STRATEGY,
) -> None:
    """Retrain on train+val using best params; evaluate on test set; log to MLflow."""

    X_trainval = pd.concat([X_train, X_val], ignore_index=True)
    y_trainval = pd.concat([y_train, y_val], ignore_index=True)

    lgb_trainval = lgb.Dataset(
        X_trainval, label=y_trainval,
        categorical_feature=cat_cols or "auto",
        free_raw_data=False,
    )

    # Re-validate on the val set to pick threshold (never touch test for this)
    lgb_val_tmp = lgb.Dataset(
        X_val, label=y_val,
        categorical_feature=cat_cols or "auto",
        reference=lgb_trainval,
        free_raw_data=False,
    )

    final_params = {k: v for k, v in best_params.items()
                    if k not in ("n_estimators", "metric")}
    final_params.update({
        "objective":  "binary",
        "metric":     ["auc", "binary_logloss", "average_precision"],
        "verbosity":  -1,
        "n_jobs":     -1,
    })

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)

    with mlflow.start_run(run_name="lgbm_best_retrained") as run:
        t0 = time.time()

        # ── Train ─────────────────────────────────────────────────────────
        callbacks = [
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=100),
        ]
        final_model = lgb.train(
            final_params,
            train_set=lgb_trainval,
            num_boost_round=best_iteration + 100,   # slight buffer
            valid_sets=[lgb_val_tmp],
            valid_names=["val"],
            callbacks=callbacks,
        )
        train_time = time.time() - t0

        # ── Val probabilities → threshold ────────────────────────────────
        val_proba = final_model.predict(X_val, num_iteration=final_model.best_iteration)
        best_thresh, best_f1_val = best_f1_threshold(y_val.values, val_proba)

        # ── Test evaluation ──────────────────────────────────────────────
        test_proba = final_model.predict(X_test, num_iteration=final_model.best_iteration)
        test_pred  = (test_proba >= best_thresh).astype(int)

        test_auc    = roc_auc_score(y_test, test_proba)
        test_ap     = average_precision_score(y_test, test_proba)
        test_f1     = f1_score(y_test, test_pred, zero_division=0)
        test_brier  = brier_score_loss(y_test, test_proba)
        val_auc     = roc_auc_score(y_val, val_proba)
        val_ap      = average_precision_score(y_val, val_proba)

        # ── Log to MLflow ────────────────────────────────────────────────
        mlflow.log_params({
            **{k: v for k, v in final_params.items()
               if k not in ("verbosity", "n_jobs")},
            "best_iteration": final_model.best_iteration,
            "threshold":      round(best_thresh, 4),
            "data_source":    data_source,
            "optuna_study_run_id": mlflow_run_id_study,
        })
        mlflow.log_metrics({
            "train_time_seconds": round(train_time, 1),
            "val_roc_auc":        round(val_auc,   6),
            "val_avg_precision":  round(val_ap,    6),
            "val_f1_at_thresh":   round(best_f1_val, 6),
            "test_roc_auc":       round(test_auc,  6),
            "test_avg_precision": round(test_ap,   6),
            "test_f1":            round(test_f1,   6),
            "test_brier":         round(test_brier, 6),
        })
        # Save model to temp file then log as artifact (avoids MLflow 3.x path issues)
        _tmp_model_path = Path("/tmp/lgbm_tuned_tmp.txt")
        final_model.save_model(str(_tmp_model_path))
        mlflow.log_artifact(str(_tmp_model_path), artifact_path="model")

        run_id = run.info.run_id

    # ── Save artifacts ────────────────────────────────────────────────────
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = ARTIFACTS_DIR / "lightgbm_tuned.txt"
    final_model.save_model(str(model_path))

    metadata = {
        "model_type":          "LightGBM (Optuna-tuned)",
        "mlflow_run_id":       run_id,
        "best_iteration":      final_model.best_iteration,
        "threshold":           round(best_thresh, 4),
        "feature_columns":     X_train.columns.tolist(),
        "categorical_cols":    cat_cols,
        "data_source":         data_source,
        "split_strategy":      split_strategy,
        "val_roc_auc":         round(val_auc,  6),
        "val_avg_precision":   round(val_ap,   6),
        "test_roc_auc":        round(test_auc, 6),
        "test_avg_precision":  round(test_ap,  6),
        "test_f1":             round(test_f1,  6),
        "test_brier":          round(test_brier, 6),
        "best_params":         {k: v for k, v in final_params.items()
                                if k not in ("verbosity", "n_jobs", "objective",
                                             "metric", "random_state")},
    }
    meta_path = ARTIFACTS_DIR / "tuning_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2))

    # ── Console summary ───────────────────────────────────────────────────
    print(f"\n  Data source   : {data_source}")
    print(f"  Features      : {X_train.shape[1]}")
    print(f"  Best iteration: {final_model.best_iteration}")
    print(f"  Train time    : {train_time:.1f}s")
    print(f"  Model  → {model_path.relative_to(PROJECT_ROOT)}")
    print(f"  Meta   → {meta_path.relative_to(PROJECT_ROOT)}")
    print(f"  MLflow → {run_id}")

    # Full threshold comparison on the test set
    test_refiner = ThresholdRefiner(y_test.values, test_proba, n_thresholds=500)
    all_strategies = test_refiner.compare_all()
    current_metrics = test_refiner._metrics_at(best_thresh)
    print_report(
        all_strategies,
        roc_auc=test_auc,
        pr_auc=test_ap,
        ks_stat=test_refiner.ks_stat,
        current_threshold=best_thresh,
        current_metrics=current_metrics,
        n_test=len(y_test),
    )

    print("Classification Report (Test Set — max_f1 threshold):")
    print(classification_report(y_test, test_pred,
                                 target_names=["Fully Paid", "Default"]))


# ══════════════════════════════════════════════════════════════════════════
# 6.  Main
# ══════════════════════════════════════════════════════════════════════════

def main(n_trials: int = 50, timeout: int | None = None) -> None:
    print("=" * 60)
    print("  LightGBM Hyperparameter Tuning — Optuna")
    print(f"  Trials: {n_trials}  |  Timeout: {timeout or 'none'}s  |  Split: {SPLIT_STRATEGY}")
    print("=" * 60)

    # ── Load data ─────────────────────────────────────────────────────────
    print("\n[1/5] Loading data …")
    X, y, issue_d, data_source = load_data()
    print(f"      Source: {data_source}")
    print(f"      Shape : {X.shape[0]:,} rows × {X.shape[1]} features")
    print(f"      Default rate: {y.mean()*100:.2f}%")

    # ── Split ─────────────────────────────────────────────────────────────
    print("\n[2/5] Splitting data …")
    X_train, X_val, X_test, y_train, y_val, y_test = make_splits(X, y, issue_d=issue_d)
    print_split_summary(issue_d, y)

    neg  = (y_train == 0).sum()
    pos  = (y_train == 1).sum()
    scale_pos_weight = float(neg / pos)
    print(f"      scale_pos_weight (neg/pos) = {scale_pos_weight:.2f}")

    # ── LightGBM datasets ────────────────────────────────────────────────
    print("\n[3/5] Building lgb.Dataset objects …")
    lgb_train, lgb_val, cat_cols = make_lgb_datasets(X_train, y_train, X_val, y_val)
    print(f"      Categorical cols : {len(cat_cols)}")

    # ── Optuna study ──────────────────────────────────────────────────────
    print(f"\n[4/5] Running Optuna ({n_trials} trials) …  (this may take a few minutes)\n")
    objective = build_objective(lgb_train, lgb_val, X_val, y_val, scale_pos_weight)

    # TPE sampler with a median pruner for efficiency
    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE, multivariate=True)
    study   = optuna.create_study(direction="maximize", sampler=sampler)

    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)

    with mlflow.start_run(run_name="lgbm_optuna_study") as study_run:
        study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True,
            n_jobs=1,  # parallelise at LightGBM level (n_jobs=-1 in params)
        )

        # Log study-level results
        best_trial = study.best_trial
        mlflow.log_params({
            "n_trials_requested": n_trials,
            "n_trials_completed": len(study.trials),
            "data_source":        data_source,
            **{f"best_{k}": v for k, v in best_trial.params.items()},
        })
        mlflow.log_metrics({
            "best_val_roc_auc":       best_trial.value,
            "best_val_avg_precision": best_trial.user_attrs.get("val_avg_precision", 0),
        })
        study_run_id = study_run.info.run_id

    print(f"\n  ✓ Best trial #{best_trial.number}  "
          f"val ROC-AUC = {best_trial.value:.5f}")
    print(f"  Best params:")
    for k, v in best_trial.params.items():
        print(f"    {k:25s}: {v}")

    # ── Retrain & evaluate ────────────────────────────────────────────────
    print("\n[5/5] Retraining final model on train+val …")
    best_params = {
        **best_trial.params,
        "objective":     "binary",
        "metric":        ["auc", "binary_logloss", "average_precision"],
        "boosting_type": "gbdt",
        "random_state":  RANDOM_STATE,
        "n_jobs":        -1,
        "verbosity":     -1,
        "subsample_freq": 1,
    }
    best_iteration = best_trial.user_attrs.get("best_iteration", 500)

    train_final_model(
        best_params     = best_params,
        best_iteration  = best_iteration,
        X_train=X_train, X_val=X_val, X_test=X_test,
        y_train=y_train, y_val=y_val, y_test=y_test,
        cat_cols        = cat_cols,
        data_source     = data_source,
        mlflow_run_id_study = study_run_id,
        split_strategy  = SPLIT_STRATEGY,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LightGBM Optuna Tuning")
    parser.add_argument("--n-trials", type=int, default=50,
                        help="Number of Optuna trials (default: 50)")
    parser.add_argument("--timeout",  type=int, default=None,
                        help="Max seconds for tuning (optional wall-clock cap)")
    args = parser.parse_args()
    main(n_trials=args.n_trials, timeout=args.timeout)
