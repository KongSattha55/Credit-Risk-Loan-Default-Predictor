"""
src/baseline.py
───────────────
Trivial baseline model: Logistic Regression on just FICO score + grade.

The point is to quantify the *uplift* of the tuned LightGBM model over the
simplest sensible credit-risk baseline. If a two-feature logistic regression
gets 68 % ROC-AUC and the tuned LightGBM gets 72 %, the 80+ feature pipeline
is buying 4 points of AUC — useful to know when justifying complexity.

Features:
    fico_range_low   (numeric — lowest of the reported FICO range at origination)
    grade            (A–G, ordinal — LendingClub's own risk grade)

Split: same as the tuned model (reads split_strategy from tuning_metadata.json).

Outputs:
    mlruns/artifacts/baseline_metrics.json
        {
          "model": "logreg_fico_grade",
          "features": [...],
          "val_roc_auc":  ...,
          "val_pr_auc":   ...,
          "val_ks_stat":  ...,
          "val_log_loss": ...,
          "test_roc_auc": ...,
          "test_pr_auc":  ...,
          "test_ks_stat": ...,
          "test_log_loss":...,
          "uplift_vs_tuned": {"test_roc_auc_delta": ..., "test_pr_auc_delta": ...}
        }

Usage:
    python src/baseline.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    log_loss,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference_fe import TARGET  # noqa: E402

INTERIM_PATH = PROJECT_ROOT / "data" / "interim"   / "loans_features.parquet"
CLEANED_PATH = PROJECT_ROOT / "data" / "processed" / "loans_cleaned.parquet"
META_PATH    = PROJECT_ROOT / "mlruns" / "artifacts" / "tuning_metadata.json"
OUT_PATH     = PROJECT_ROOT / "mlruns" / "artifacts" / "baseline_metrics.json"

RANDOM_STATE = 42
GRADE_MAP = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7}


def _load_frame() -> tuple[pd.DataFrame, pd.Series | None]:
    """Load the smallest frame that contains fico_range_low + grade(_enc) + target."""
    if INTERIM_PATH.exists():
        df = pd.read_parquet(
            INTERIM_PATH,
            columns=[c for c in ["fico_range_low", "grade_enc", "issue_d", TARGET]
                     if c is not None],
        )
        if "grade_enc" not in df.columns:
            raise KeyError("grade_enc missing from interim parquet — rerun feature_engineering.py")
        df = df.rename(columns={"grade_enc": "grade_ord"})
    elif CLEANED_PATH.exists():
        df = pd.read_parquet(CLEANED_PATH, columns=["fico_range_low", "grade", "issue_d", TARGET])
        df["grade_ord"] = df["grade"].map(GRADE_MAP).astype("float32")
        df = df.drop(columns=["grade"])
    else:
        raise FileNotFoundError("Run data_cleaning.py (or feature_engineering.py) first.")

    issue_d = df["issue_d"] if "issue_d" in df.columns else None
    if issue_d is not None:
        df = df.drop(columns=["issue_d"])

    df = df.dropna(subset=["fico_range_low", "grade_ord", TARGET])
    return df, issue_d


def _splits(df: pd.DataFrame, issue_d: pd.Series | None, strategy: str):
    """Return (X_train, X_val, X_test, y_train, y_val, y_test)."""
    X = df[["fico_range_low", "grade_ord"]].astype("float32")
    y = df[TARGET].astype("int32")

    if strategy == "chronological":
        if issue_d is None:
            raise ValueError("chronological split requires issue_d column.")
        order = np.argsort(issue_d.loc[df.index].values)
        n = len(X)
        n_test = int(n * 0.20)
        n_val  = int(n * 0.10)
        train_idx = order[: n - n_test - n_val]
        val_idx   = order[n - n_test - n_val : n - n_test]
        test_idx  = order[n - n_test :]
        return (
            X.iloc[train_idx], X.iloc[val_idx], X.iloc[test_idx],
            y.iloc[train_idx], y.iloc[val_idx], y.iloc[test_idx],
        )

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.125, stratify=y_temp, random_state=RANDOM_STATE
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def _ks(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    return float(np.max(tpr - fpr))


def _scores(y_true: np.ndarray, y_proba: np.ndarray) -> dict:
    return {
        "roc_auc":  round(float(roc_auc_score(y_true, y_proba)), 5),
        "pr_auc":   round(float(average_precision_score(y_true, y_proba)), 5),
        "ks_stat":  round(_ks(y_true, y_proba), 5),
        "log_loss": round(float(log_loss(y_true, y_proba)), 5),
    }


def main() -> None:
    print("=" * 60)
    print("  Baseline — Logistic Regression on FICO + grade")
    print("=" * 60)

    meta = json.loads(META_PATH.read_text()) if META_PATH.exists() else {}
    strategy = meta.get("split_strategy", "random")
    print(f"  Split strategy : {strategy}")

    print("\n[1/4] Loading data …")
    df, issue_d = _load_frame()
    print(f"  Rows after NA drop : {len(df):,}")

    print("[2/4] Splitting …")
    X_train, X_val, X_test, y_train, y_val, y_test = _splits(df, issue_d, strategy)
    print(f"  Train / Val / Test : {len(X_train):,} / {len(X_val):,} / {len(X_test):,}")
    print(f"  Default rate (train): {y_train.mean():.3%}")

    print("[3/4] Fitting logistic regression (class_weight='balanced') …")
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_STATE,
        )),
    ])
    model.fit(X_train, y_train)
    coefs = dict(zip(X_train.columns, model.named_steps["logreg"].coef_[0].round(4)))
    intercept = float(model.named_steps["logreg"].intercept_[0])
    print(f"  Coefficients  : {coefs}")
    print(f"  Intercept     : {intercept:+.4f}")

    val_proba  = model.predict_proba(X_val)[:, 1]
    test_proba = model.predict_proba(X_test)[:, 1]
    val_scores  = _scores(y_val.values,  val_proba)
    test_scores = _scores(y_test.values, test_proba)

    print("\n[4/4] Results")
    print(f"  Val   — ROC-AUC {val_scores['roc_auc']:.4f}  PR-AUC {val_scores['pr_auc']:.4f}  "
          f"KS {val_scores['ks_stat']:.4f}  log-loss {val_scores['log_loss']:.4f}")
    print(f"  Test  — ROC-AUC {test_scores['roc_auc']:.4f}  PR-AUC {test_scores['pr_auc']:.4f}  "
          f"KS {test_scores['ks_stat']:.4f}  log-loss {test_scores['log_loss']:.4f}")

    uplift = {}
    tuned_roc = meta.get("test_roc_auc")
    tuned_pr  = meta.get("test_pr_auc") or meta.get("test_average_precision")
    if tuned_roc is not None:
        uplift["test_roc_auc_delta"] = round(float(tuned_roc) - test_scores["roc_auc"], 5)
        print(f"\n  Tuned LightGBM test ROC-AUC : {tuned_roc:.4f}")
        print(f"  Uplift vs baseline (ROC-AUC): {uplift['test_roc_auc_delta']:+.4f}")
    if tuned_pr is not None:
        uplift["test_pr_auc_delta"] = round(float(tuned_pr) - test_scores["pr_auc"], 5)
        print(f"  Uplift vs baseline (PR-AUC) : {uplift['test_pr_auc_delta']:+.4f}")

    out = {
        "model":          "logreg_fico_grade",
        "features":       list(X_train.columns),
        "split_strategy": strategy,
        "coefficients":   coefs,
        "intercept":      round(intercept, 5),
        "val_roc_auc":   val_scores["roc_auc"],
        "val_pr_auc":    val_scores["pr_auc"],
        "val_ks_stat":   val_scores["ks_stat"],
        "val_log_loss":  val_scores["log_loss"],
        "test_roc_auc":  test_scores["roc_auc"],
        "test_pr_auc":   test_scores["pr_auc"],
        "test_ks_stat":  test_scores["ks_stat"],
        "test_log_loss": test_scores["log_loss"],
        "uplift_vs_tuned": uplift,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f"\n  Metrics → {OUT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
