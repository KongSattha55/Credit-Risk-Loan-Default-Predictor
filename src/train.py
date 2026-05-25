"""
Training pipeline for LendingClub Loan Default Predictor.
Reads model configuration from configs/model.yaml.

Usage:
    python src/train.py                          # train lightgbm (default)
    python src/train.py --model logistic_regression
    python src/train.py --model random_forest
    python src/train.py --model xgboost
    python src/train.py --model lightgbm

Outputs:
    • MLflow run logged under experiment "loan-default-predictor"
    • mlruns/artifacts/<model_name>_model.*     (saved model)
    • mlruns/artifacts/<model_name>_metadata.json
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
import mlflow
import mlflow.sklearn
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ── Paths ─────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).resolve().parents[1]
CONFIG_PATH   = PROJECT_ROOT / "configs" / "model.yaml"
INTERIM_PATH  = PROJECT_ROOT / "data" / "interim" / "loans_features.parquet"
CLEANED_PATH  = PROJECT_ROOT / "data" / "processed" / "loans_cleaned.parquet"
MLFLOW_URI    = str(PROJECT_ROOT / "mlruns")
ARTIFACTS_DIR = PROJECT_ROOT / "mlruns" / "artifacts"

# ── Constants ──────────────────────────────────────────────────────────────
TARGET       = "default"
RANDOM_STATE = 42

# Columns to drop before training (leaky, redundant, high-cardinality string)
ALWAYS_DROP = [
    TARGET,
    "funded_amnt",
    "funded_amnt_inv",
    "fico_range_high",
    "emp_title",
]


# ══════════════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════════════

def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


# ══════════════════════════════════════════════════════════════════════════
# Data
# ══════════════════════════════════════════════════════════════════════════

def load_data() -> tuple[pd.DataFrame, pd.Series, str]:
    if INTERIM_PATH.exists():
        df = pd.read_parquet(INTERIM_PATH)
        source = "interim (feature-engineered)"
    elif CLEANED_PATH.exists():
        df = pd.read_parquet(CLEANED_PATH)
        source = "processed (cleaned)"
    else:
        raise FileNotFoundError(
            "No data found. Run src/data_cleaning.py first, "
            "then optionally src/feature_engineering.py."
        )

    drop_cols  = [c for c in ALWAYS_DROP if c in df.columns]
    feature_cols = [c for c in df.columns if c not in drop_cols + [TARGET]]

    X = df[feature_cols].copy()
    y = df[TARGET].astype("int32")

    return X, y, source


def encode_categoricals(X_train: pd.DataFrame,
                         X_val: pd.DataFrame,
                         X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Label-encode object/string columns. Fit on train only."""
    cat_cols = X_train.select_dtypes(include=["object", "str", "category"]).columns.tolist()
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col].astype(str))
        X_val[col]   = le.transform(X_val[col].astype(str).map(
            lambda x, le=le: x if x in le.classes_ else le.classes_[0]))
        X_test[col]  = le.transform(X_test[col].astype(str).map(
            lambda x, le=le: x if x in le.classes_ else le.classes_[0]))
        encoders[col] = le
        X_train[col]  = X_train[col].astype("float32")
        X_val[col]    = X_val[col].astype("float32")
        X_test[col]   = X_test[col].astype("float32")
    return X_train, X_val, X_test


def make_splits(X: pd.DataFrame, y: pd.Series, cfg: dict):
    test_size = cfg["data"]["test_size"]
    val_size  = cfg["data"]["val_size"]
    rs        = cfg["data"]["random_state"]

    # val_frac is fraction of the temp (non-test) set
    val_frac = val_size / (1.0 - test_size)

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=rs
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_frac, stratify=y_temp, random_state=rs
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# ══════════════════════════════════════════════════════════════════════════
# Models
# ══════════════════════════════════════════════════════════════════════════

def build_model(model_name: str, cfg: dict):
    params = cfg["models"][model_name]

    if model_name == "logistic_regression":
        return LogisticRegression(**params)

    if model_name == "random_forest":
        return RandomForestClassifier(**params)

    if model_name == "xgboost":
        from xgboost import XGBClassifier
        # xgboost 2.x removed use_label_encoder; eval_metric goes in the constructor
        xgb_params = {k: v for k, v in params.items() if k != "eval_metric"}
        return XGBClassifier(
            **xgb_params,
            eval_metric=params.get("eval_metric", "auc"),
        )

    if model_name == "lightgbm":
        import lightgbm as lgb
        return lgb.LGBMClassifier(**params)

    raise ValueError(f"Unknown model: {model_name}. "
                     f"Choose from: logistic_regression, random_forest, xgboost, lightgbm")


# ══════════════════════════════════════════════════════════════════════════
# Threshold optimisation
# ══════════════════════════════════════════════════════════════════════════

def best_f1_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> tuple[float, float]:
    thresholds = np.linspace(0.01, 0.99, 200)
    f1s = [f1_score(y_true, (y_proba >= t).astype(int), zero_division=0)
           for t in thresholds]
    idx = int(np.argmax(f1s))
    return float(thresholds[idx]), float(f1s[idx])


# ══════════════════════════════════════════════════════════════════════════
# Training & evaluation
# ══════════════════════════════════════════════════════════════════════════

def train_and_evaluate(model_name: str, cfg: dict) -> None:
    experiment = cfg["training"]["experiment_name"]
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(experiment)

    # ── Load data ─────────────────────────────────────────────────────────
    print("[1/4] Loading data …")
    X, y, source = load_data()
    print(f"      Source: {source}")
    print(f"      Shape : {X.shape[0]:,} rows × {X.shape[1]} features")
    print(f"      Default rate: {y.mean()*100:.2f}%")

    # ── Split ─────────────────────────────────────────────────────────────
    print("[2/4] Splitting data …")
    X_train, X_val, X_test, y_train, y_val, y_test = make_splits(X, y, cfg)
    print(f"      Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

    # Encode categoricals (sklearn models need numeric input)
    X_train, X_val, X_test = encode_categoricals(X_train, X_val, X_test)

    # ── Train ─────────────────────────────────────────────────────────────
    print(f"[3/4] Training {model_name} …")
    model = build_model(model_name, cfg)

    with mlflow.start_run(run_name=model_name) as run:
        t0 = time.time()

        if model_name == "xgboost":
            # xgboost 2.x: verbose is a fit kwarg; no callbacks needed for basic eval
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        elif model_name == "lightgbm":
            import lightgbm as lgb
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.log_evaluation(period=0)],
            )
        else:
            model.fit(X_train, y_train)

        train_time = time.time() - t0

        # ── Evaluate ──────────────────────────────────────────────────────
        print("[4/4] Evaluating …")
        val_proba  = model.predict_proba(X_val)[:, 1]
        test_proba = model.predict_proba(X_test)[:, 1]

        best_thresh, best_f1_val = best_f1_threshold(y_val.values, val_proba)
        test_pred = (test_proba >= best_thresh).astype(int)

        val_auc   = roc_auc_score(y_val, val_proba)
        val_ap    = average_precision_score(y_val, val_proba)
        test_auc  = roc_auc_score(y_test, test_proba)
        test_ap   = average_precision_score(y_test, test_proba)
        test_f1   = f1_score(y_test, test_pred, zero_division=0)
        test_brier = brier_score_loss(y_test, test_proba)

        # ── Log to MLflow ─────────────────────────────────────────────────
        mlflow.log_params({
            **cfg["models"][model_name],
            "model_name":      model_name,
            "threshold":       round(best_thresh, 4),
            "data_source":     source,
            "n_train":         len(X_train),
            "n_val":           len(X_val),
            "n_test":          len(X_test),
            "n_features":      X_train.shape[1],
        })
        mlflow.log_metrics({
            "train_time_seconds": round(train_time, 1),
            "val_roc_auc":        round(val_auc,    6),
            "val_avg_precision":  round(val_ap,     6),
            "val_f1_at_thresh":   round(best_f1_val, 6),
            "test_roc_auc":       round(test_auc,   6),
            "test_avg_precision": round(test_ap,    6),
            "test_f1":            round(test_f1,    6),
            "test_brier":         round(test_brier, 6),
        })
        mlflow.sklearn.log_model(model, artifact_path="model")

        run_id = run.info.run_id

    # ── Save artifacts ────────────────────────────────────────────────────
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    import pickle
    model_path = ARTIFACTS_DIR / f"{model_name}_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    metadata = {
        "model_type":      model_name,
        "mlflow_run_id":   run_id,
        "threshold":       round(best_thresh, 4),
        "feature_columns": X_train.columns.tolist(),
        "data_source":     source,
        "val_roc_auc":     round(val_auc,    6),
        "val_avg_precision": round(val_ap,   6),
        "test_roc_auc":    round(test_auc,   6),
        "test_avg_precision": round(test_ap, 6),
        "test_f1":         round(test_f1,    6),
        "test_brier":      round(test_brier, 6),
        "params":          cfg["models"][model_name],
    }
    meta_path = ARTIFACTS_DIR / f"{model_name}_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2))

    # ── Console summary ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  {model_name.upper()} — RESULTS")
    print("=" * 60)
    print(f"  Data source       : {source}")
    print(f"  Features          : {X_train.shape[1]}")
    print(f"  Train time        : {train_time:.1f}s")
    print(f"  Optimal threshold : {best_thresh:.4f}")
    print()
    print("  ── Val Set ──")
    print(f"  ROC-AUC           : {val_auc:.4f}")
    print(f"  Avg Precision     : {val_ap:.4f}")
    print(f"  F1 @ threshold    : {best_f1_val:.4f}")
    print()
    print("  ── Test Set ──")
    print(f"  ROC-AUC           : {test_auc:.4f}")
    print(f"  Avg Precision     : {test_ap:.4f}")
    print(f"  F1                : {test_f1:.4f}")
    print(f"  Brier Score       : {test_brier:.4f}")
    print()
    print(f"  Model    → {model_path.relative_to(PROJECT_ROOT)}")
    print(f"  Metadata → {meta_path.relative_to(PROJECT_ROOT)}")
    print(f"  MLflow   → {run_id}")
    print("=" * 60)
    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, test_pred,
                                target_names=["Fully Paid", "Default"]))


# ══════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Train a loan default predictor")
    parser.add_argument(
        "--model",
        type=str,
        default="lightgbm",
        choices=["logistic_regression", "random_forest", "xgboost", "lightgbm"],
        help="Model to train (default: lightgbm)",
    )
    args = parser.parse_args()

    cfg = load_config()
    train_and_evaluate(args.model, cfg)


if __name__ == "__main__":
    main()
