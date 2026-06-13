"""
Model comparison workflow for the LendingClub loan default predictor.

Evaluates all comparison models on the same time-based issue_d split using
leakage-free features only.

Outputs:
    artifacts/model_comparison.csv
    artifacts/model_comparison.json

Usage:
    python src/model_comparison.py
    python src/model_comparison.py --max-train-rows 200000
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
import warnings
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.calibration_classes import LGBMBoosterWrapper, PreFitCalibratedClassifier  # noqa: F401,E402
from src.leakage import TARGET, feature_columns, leakage_columns_present  # noqa: E402
from src.splits import SPLIT_STRATEGY, split_xy_by_issue_date  # noqa: E402

INTERIM_PATH = PROJECT_ROOT / "data" / "interim" / "loans_features.parquet"
CLEANED_PATH = PROJECT_ROOT / "data" / "processed" / "loans_cleaned.parquet"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_ARTIFACTS_DIR = PROJECT_ROOT / "mlruns" / "artifacts"
CALIBRATED_SIGMOID_PATH = PROJECT_ROOT / "models" / "lightgbm_calibrated_sigmoid.pkl"
OUT_CSV = ARTIFACTS_DIR / "model_comparison.csv"
OUT_JSON = ARTIFACTS_DIR / "model_comparison.json"
RANDOM_STATE = 42


def load_frame() -> tuple[pd.DataFrame, str]:
    if INTERIM_PATH.exists():
        return pd.read_parquet(INTERIM_PATH), "interim (feature-engineered)"
    if CLEANED_PATH.exists():
        return pd.read_parquet(CLEANED_PATH), "processed (cleaned)"
    raise FileNotFoundError("No data found. Run src/data_cleaning.py and src/feature_engineering.py first.")


def maybe_subsample_train(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    max_train_rows: int | None,
) -> tuple[pd.DataFrame, pd.Series]:
    if not max_train_rows or len(X_train) <= max_train_rows:
        return X_train, y_train
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(X_train), size=max_train_rows, replace=False)
    return X_train.iloc[idx].copy(), y_train.iloc[idx].copy()


def best_f1_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> tuple[float, float]:
    thresholds = np.linspace(0.01, 0.99, 200)
    f1_scores = [f1_score(y_true, y_proba >= threshold, zero_division=0) for threshold in thresholds]
    best_idx = int(np.argmax(f1_scores))
    return float(thresholds[best_idx]), float(f1_scores[best_idx])


def score_model(
    model_name: str,
    y_test: pd.Series,
    y_proba: np.ndarray,
    threshold: float,
    training_time_seconds: float,
    n_train: int,
    n_test: int,
    feature_count: int,
    notes: str = "",
) -> dict[str, Any]:
    y_true = y_test.to_numpy()
    clipped = np.clip(np.asarray(y_proba, dtype=float), 1e-6, 1 - 1e-6)
    y_pred = (clipped >= threshold).astype(int)
    return {
        "model": model_name,
        "roc_auc": round(float(roc_auc_score(y_true, clipped)), 6),
        "pr_auc": round(float(average_precision_score(y_true, clipped)), 6),
        "average_precision": round(float(average_precision_score(y_true, clipped)), 6),
        "brier_score": round(float(brier_score_loss(y_true, clipped)), 6),
        "log_loss": round(float(log_loss(y_true, clipped)), 6),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 6),
        "threshold": round(float(threshold), 6),
        "training_time_seconds": round(float(training_time_seconds), 3),
        "n_train": int(n_train),
        "n_test": int(n_test),
        "feature_count": int(feature_count),
        "notes": notes,
    }


def fit_predict(
    model_name: str,
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_test: pd.DataFrame,
) -> tuple[np.ndarray, float, float]:
    start = time.time()
    if model_name == "XGBoost":
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    elif model_name == "LightGBM":
        import lightgbm as lgb

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.log_evaluation(period=0)],
        )
    else:
        model.fit(X_train, y_train)
    train_time = time.time() - start
    val_proba = model.predict_proba(X_val)[:, 1]
    threshold, _ = best_f1_threshold(y_val.to_numpy(), val_proba)
    test_proba = model.predict_proba(X_test)[:, 1]
    return test_proba, threshold, train_time


def comparison_summary(results: list[dict[str, Any]]) -> dict[str, str]:
    rows = pd.DataFrame(results)
    ranking = rows.sort_values(["roc_auc", "pr_auc"], ascending=False).iloc[0]
    calibrated = rows.sort_values(["brier_score", "log_loss"], ascending=True).iloc[0]
    production = rows.loc[rows["model"] == "Calibrated LightGBM sigmoid"]
    production_model = (
        "Calibrated LightGBM sigmoid"
        if not production.empty
        else str(calibrated["model"])
    )
    return {
        "ranking": f"{ranking['model']} has the strongest ranking performance by ROC-AUC/PR-AUC.",
        "calibrated_probability": f"{calibrated['model']} has the best probability quality by Brier score/log loss.",
        "interpretability": "FICO/grade logistic baseline is the most interpretable because it uses only FICO and LendingClub grade.",
        "production_recommendation": (
            f"Keep {production_model} as the production probability model unless a future comparison "
            "shows a clearly better calibrated model with equal or better ranking and acceptable complexity."
        ),
    }


def main(max_train_rows: int | None = None) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    df, source = load_frame()
    if "issue_d" not in df.columns:
        raise ValueError("Model comparison requires issue_d for the centralized time-based split.")

    features = feature_columns(df.columns, extra_drop=["issue_d"])
    leaks = leakage_columns_present(features)
    if leaks:
        raise ValueError(f"Leakage columns found in comparison features: {leaks}")

    X = df[features].copy()
    y = df[TARGET].astype("int32")
    X_train, X_val, X_test, y_train, y_val, y_test = split_xy_by_issue_date(X, y, df["issue_d"])
    X_train_fit, y_train_fit = maybe_subsample_train(X_train, y_train, max_train_rows)

    results: list[dict[str, Any]] = []

    baseline_features = [col for col in ["fico_range_low", "grade_enc"] if col in X_train.columns]
    if len(baseline_features) != 2:
        raise ValueError("FICO/grade baseline requires fico_range_low and grade_enc features.")

    models = [
        (
            "FICO/grade logistic baseline",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)),
                ]
            ),
            baseline_features,
        ),
        (
            "Logistic Regression",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", LogisticRegression(class_weight="balanced", max_iter=500, random_state=RANDOM_STATE)),
                ]
            ),
            features,
        ),
        (
            "Random Forest",
            RandomForestClassifier(
                n_estimators=80,
                max_depth=10,
                min_samples_leaf=100,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            features,
        ),
        (
            "XGBoost",
            None,
            features,
        ),
        (
            "LightGBM",
            None,
            features,
        ),
    ]

    for model_name, model, model_features in models:
        print(f"Training {model_name} ...")
        if model_name == "XGBoost":
            from xgboost import XGBClassifier

            model = XGBClassifier(
                n_estimators=180,
                max_depth=4,
                learning_rate=0.06,
                subsample=0.85,
                colsample_bytree=0.85,
                eval_metric="auc",
                tree_method="hist",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        elif model_name == "LightGBM":
            import lightgbm as lgb

            model = lgb.LGBMClassifier(
                n_estimators=250,
                max_depth=6,
                learning_rate=0.05,
                num_leaves=63,
                subsample=0.85,
                colsample_bytree=0.85,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbose=-1,
            )

        test_proba, threshold, train_time = fit_predict(
            model_name,
            model,
            X_train_fit[model_features],
            y_train_fit,
            X_val[model_features],
            y_val,
            X_test[model_features],
        )
        results.append(
            score_model(
                model_name,
                y_test,
                test_proba,
                threshold,
                train_time,
                len(X_train_fit),
                len(X_test),
                len(model_features),
                notes="trained in comparison workflow",
            )
        )

    print("Scoring Calibrated LightGBM sigmoid ...")
    if not CALIBRATED_SIGMOID_PATH.exists():
        raise FileNotFoundError("Missing models/lightgbm_calibrated_sigmoid.pkl. Run src/calibrate.py first.")
    with open(CALIBRATED_SIGMOID_PATH, "rb") as f:
        calibrated = pickle.load(f)
    start = time.time()
    calibrated_proba = calibrated.predict_proba(X_test[features])[:, 1]
    calibrated_time = time.time() - start
    cal_threshold, _ = best_f1_threshold(y_val.to_numpy(), calibrated.predict_proba(X_val[features])[:, 1])
    results.append(
        score_model(
            "Calibrated LightGBM sigmoid",
            y_test,
            calibrated_proba,
            cal_threshold,
            calibrated_time,
            len(X_train),
            len(X_test),
            len(features),
            notes="pretrained production calibrated sigmoid artifact; time is scoring/loading comparison time",
        )
    )

    result_df = pd.DataFrame(results).sort_values("roc_auc", ascending=False)
    result_df.to_csv(OUT_CSV, index=False)
    summary = comparison_summary(results)
    payload = {
        "data_source": source,
        "split_strategy": SPLIT_STRATEGY,
        "max_train_rows": max_train_rows,
        "feature_columns": features,
        "leakage_columns_present": leaks,
        "results": results,
        "summary": summary,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2))

    print("\nModel comparison:")
    print(result_df.to_string(index=False))
    print("\nSummary:")
    for key, value in summary.items():
        print(f"- {key}: {value}")
    print(f"\nSaved {OUT_CSV.relative_to(PROJECT_ROOT)}")
    print(f"Saved {OUT_JSON.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare loan default models on the same time-based split")
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=None,
        help="Optional cap for training rows to speed up exploratory comparisons.",
    )
    args = parser.parse_args()
    main(max_train_rows=args.max_train_rows)
