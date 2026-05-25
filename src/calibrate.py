"""
src/calibrate.py
────────────────
Probability calibration for the tuned LightGBM model.

LightGBM trained with scale_pos_weight (or class_weight) produces probabilities
that are systematically inflated — good for ranking/AUC, wrong for decisions
that depend on the raw probability (risk tiers, expected-loss calculations).

This script fits an isotonic regression calibrator on the validation-set
probabilities and saves it to mlruns/artifacts/calibrator.pkl. The API and app
can load it and apply `calibrator.transform(raw_probs)` before showing a score.

Usage:
    python src/calibrate.py
    python src/calibrate.py --method sigmoid    # Platt scaling instead

Outputs:
    • mlruns/artifacts/calibrator.pkl
    • tuning_metadata.json updated with {"calibration_method", "calibration_brier_pre/post"}
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split

PROJECT_ROOT  = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference_fe import ALWAYS_DROP, TARGET  # noqa: E402

INTERIM_PATH  = PROJECT_ROOT / "data" / "interim"   / "loans_features.parquet"
CLEANED_PATH  = PROJECT_ROOT / "data" / "processed" / "loans_cleaned.parquet"
MODEL_PATH    = PROJECT_ROOT / "mlruns" / "artifacts" / "lightgbm_tuned.txt"
META_PATH     = PROJECT_ROOT / "mlruns" / "artifacts" / "tuning_metadata.json"
OUT_PATH      = PROJECT_ROOT / "mlruns" / "artifacts" / "calibrator.pkl"

RANDOM_STATE = 42


def _load_val_split() -> tuple[pd.DataFrame, pd.Series]:
    """Reproduce the val split used during tuning (matches tune_lightgbm.py)."""
    if INTERIM_PATH.exists():
        df = pd.read_parquet(INTERIM_PATH)
    elif CLEANED_PATH.exists():
        df = pd.read_parquet(CLEANED_PATH)
    else:
        raise FileNotFoundError("Run data_cleaning.py / feature_engineering.py first.")

    issue_d = df["issue_d"] if "issue_d" in df.columns else None
    drop_cols = [c for c in (list(ALWAYS_DROP) + ["issue_d"]) if c in df.columns]
    feature_cols = [c for c in df.columns if c not in drop_cols + [TARGET]]
    X = df[feature_cols].copy()
    y = df[TARGET].astype("int32")
    for col in X.select_dtypes(include=["object"]).columns:
        X[col] = X[col].astype("category")

    meta = json.loads(META_PATH.read_text())
    strategy = meta.get("split_strategy", "random")

    if strategy == "chronological" and issue_d is not None:
        order = np.argsort(issue_d.values)
        n = len(X)
        n_test = int(n * 0.20)
        n_val  = int(n * 0.10)
        val_idx = order[n - n_test - n_val : n - n_test]
        return X.iloc[val_idx], y.iloc[val_idx]

    X_temp, _, y_temp, _ = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    _, X_val, _, y_val = train_test_split(
        X_temp, y_temp, test_size=0.125, stratify=y_temp, random_state=RANDOM_STATE
    )
    return X_val, y_val


def main(method: str = "isotonic") -> None:
    print(f"[1/4] Loading model & val split  (method={method}) …")
    model = lgb.Booster(model_file=str(MODEL_PATH))
    X_val, y_val = _load_val_split()

    # Align columns to the model's feature list from metadata
    meta = json.loads(META_PATH.read_text())
    features = meta["feature_columns"]
    X_val = X_val[[c for c in features if c in X_val.columns]]

    print(f"      Val size: {len(X_val):,}")

    print("[2/4] Generating raw probabilities on val …")
    raw = model.predict(X_val, num_iteration=meta.get("best_iteration", 0) or 0)
    pre_brier = brier_score_loss(y_val, raw)

    print(f"[3/4] Fitting {method} calibrator …")
    if method == "isotonic":
        cal = IsotonicRegression(out_of_bounds="clip")
        cal.fit(raw, y_val)
        cal_proba = cal.transform(raw)
    elif method == "sigmoid":
        # Platt scaling: logistic on the raw score
        cal = LogisticRegression()
        cal.fit(raw.reshape(-1, 1), y_val)
        cal_proba = cal.predict_proba(raw.reshape(-1, 1))[:, 1]
    else:
        raise ValueError(f"Unknown method: {method}")

    post_brier = brier_score_loss(y_val, cal_proba)

    print(f"      Brier (val)  pre: {pre_brier:.5f}  →  post: {post_brier:.5f}   "
          f"(improvement: {pre_brier - post_brier:+.5f})")

    print("[4/4] Saving calibrator + updating metadata …")
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "wb") as f:
        pickle.dump({"method": method, "calibrator": cal}, f)

    meta["calibration_method"]      = method
    meta["calibration_brier_pre"]   = round(float(pre_brier), 6)
    meta["calibration_brier_post"]  = round(float(post_brier), 6)
    META_PATH.write_text(json.dumps(meta, indent=2))

    print(f"      Calibrator → {OUT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"      Metadata updated with calibration stats.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate LightGBM probabilities")
    parser.add_argument("--method", choices=["isotonic", "sigmoid"], default="isotonic")
    args = parser.parse_args()
    main(method=args.method)
