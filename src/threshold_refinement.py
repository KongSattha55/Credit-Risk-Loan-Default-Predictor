"""
src/threshold_refinement.py
───────────────────────────
Precision / Recall threshold analysis for the LightGBM loan-default model.

Reconstructs the deterministic time-based test split (`issue_d >= 2017-01-01`),
runs predictions, then evaluates threshold strategies:

  1. max_f1          — maximise F1  (current default)
  2. max_precision   — highest precision s.t. recall ≥ --min-recall
  3. max_recall      — highest recall s.t. precision ≥ --min-precision
  4. youden          — Youden's J statistic (sensitivity + specificity − 1)
  5. cost_sensitive  — minimise expected misclassification cost

Usage:
    python src/threshold_refinement.py
    python src/threshold_refinement.py --strategy max_precision --min-recall 0.60
    python src/threshold_refinement.py --strategy cost_sensitive --fn-cost 5 --fp-cost 1
    python src/threshold_refinement.py --strategy youden --save
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).resolve().parents[1]
INTERIM_PATH  = PROJECT_ROOT / "data" / "interim"    / "loans_features.parquet"
CLEANED_PATH  = PROJECT_ROOT / "data" / "processed"  / "loans_cleaned.parquet"
ARTIFACTS_DIR = PROJECT_ROOT / "mlruns" / "artifacts"
MODEL_PATH    = ARTIFACTS_DIR / "lightgbm_tuned.txt"
META_PATH     = ARTIFACTS_DIR / "tuning_metadata.json"

import sys as _sys
_sys.path.insert(0, str(PROJECT_ROOT))
from src.leakage import TARGET as _TARGET, feature_columns  # noqa: E402
from src.splits import ensure_issue_datetime, time_split_masks  # noqa: E402

TARGET       = _TARGET

STRATEGIES = ("max_f1", "max_precision", "max_recall", "youden", "cost_sensitive", "max_ks")


# ══════════════════════════════════════════════════════════════════════════
# Data helpers  (mirrors tune_lightgbm.py — same split = same test set)
# ══════════════════════════════════════════════════════════════════════════

def _load_data() -> tuple[pd.DataFrame, pd.Series, pd.Series | None]:
    """Returns (X, y, issue_d_or_None)."""
    if INTERIM_PATH.exists():
        df = pd.read_parquet(INTERIM_PATH)
    elif CLEANED_PATH.exists():
        df = pd.read_parquet(CLEANED_PATH)
    else:
        raise FileNotFoundError("Run src/data_cleaning.py first.")

    if "issue_d" not in df.columns:
        raise ValueError("Time-based threshold split requires issue_d. Re-run data cleaning and feature engineering.")
    issue_d = ensure_issue_datetime(df["issue_d"])
    feature_cols = feature_columns(df.columns, extra_drop=["issue_d"])

    X = df[feature_cols].copy()
    y = df[TARGET].astype("int32")

    for col in X.select_dtypes(include=["object"]).columns:
        X[col] = X[col].astype("category")

    return X, y, issue_d


def _test_split(
    X: pd.DataFrame,
    y: pd.Series,
    issue_d: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return the identical test set used during training."""
    _, _, test_mask = time_split_masks(issue_d)
    return X.loc[test_mask], y.loc[test_mask]


def _val_split(
    X: pd.DataFrame,
    y: pd.Series,
    issue_d: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return the identical validation set used during training."""
    _, val_mask, _ = time_split_masks(issue_d)
    return X.loc[val_mask], y.loc[val_mask]


# ══════════════════════════════════════════════════════════════════════════
# Threshold strategies
# ══════════════════════════════════════════════════════════════════════════

class ThresholdRefiner:
    """
    Compute and compare multiple threshold strategies on a given
    (y_true, y_proba) pair.

    Parameters
    ----------
    y_true : array-like of int  — ground truth labels
    y_proba : array-like of float — predicted default probabilities
    n_thresholds : int — resolution of the sweep grid
    """

    def __init__(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        n_thresholds: int = 500,
    ):
        self.y_true   = np.asarray(y_true)
        self.y_proba  = np.asarray(y_proba)

        # ── Precompute curve arrays ────────────────────────────────────────
        self.thresholds = np.linspace(0.001, 0.999, n_thresholds)

        self.precisions = np.array([
            precision_score(self.y_true, (self.y_proba >= t).astype(int), zero_division=0)
            for t in self.thresholds
        ])
        self.recalls = np.array([
            recall_score(self.y_true, (self.y_proba >= t).astype(int), zero_division=0)
            for t in self.thresholds
        ])
        self.f1s = np.array([
            f1_score(self.y_true, (self.y_proba >= t).astype(int), zero_division=0)
            for t in self.thresholds
        ])

        # Specificity = TN / (TN + FP)
        self.specificities = np.array([
            self._specificity((self.y_proba >= t).astype(int))
            for t in self.thresholds
        ])

        # sklearn PR curve (more stable endpoints)
        self._pr_prec, self._pr_rec, self._pr_thresh = precision_recall_curve(
            self.y_true, self.y_proba
        )
        self._roc_fpr, self._roc_tpr, self._roc_thresh = roc_curve(
            self.y_true, self.y_proba
        )

        # Aggregate metrics
        self.roc_auc = roc_auc_score(self.y_true, self.y_proba)
        self.pr_auc  = average_precision_score(self.y_true, self.y_proba)

        # Kolmogorov–Smirnov statistic: max separation between the
        # cumulative distributions of defaulter vs non-defaulter scores.
        # Equivalent to max(TPR − FPR) along the ROC curve.
        ks_curve      = self._roc_tpr - self._roc_fpr
        ks_idx        = int(np.argmax(ks_curve))
        self.ks_stat  = float(ks_curve[ks_idx])
        self.ks_thresh = float(self._roc_thresh[ks_idx])

    # ── helpers ────────────────────────────────────────────────────────────

    def _specificity(self, y_pred: np.ndarray) -> float:
        tn, fp, fn, tp = confusion_matrix(self.y_true, y_pred, labels=[0, 1]).ravel()
        return tn / (tn + fp) if (tn + fp) > 0 else 0.0

    def _metrics_at(self, thresh: float) -> dict:
        y_pred = (self.y_proba >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(self.y_true, y_pred, labels=[0, 1]).ravel()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        return {
            "threshold":   round(float(thresh), 4),
            "precision":   round(prec, 4),
            "recall":      round(rec,  4),
            "specificity": round(spec, 4),
            "f1":          round(f1,   4),
            "tp": int(tp), "fp": int(fp),
            "fn": int(fn), "tn": int(tn),
        }

    # ── strategies ─────────────────────────────────────────────────────────

    def max_f1(self) -> dict:
        """Maximise F1 score (harmonic mean of precision and recall)."""
        idx = int(np.argmax(self.f1s))
        return {"strategy": "max_f1", **self._metrics_at(self.thresholds[idx])}

    def max_precision(self, min_recall: float = 0.50) -> dict:
        """
        Highest precision subject to recall ≥ min_recall.
        Use when false positives (wrongly rejected applicants) are costly.
        """
        mask = self.recalls >= min_recall
        if not mask.any():
            raise ValueError(
                f"No threshold achieves recall ≥ {min_recall:.2f}. "
                "Lower --min-recall."
            )
        idx = int(np.argmax(np.where(mask, self.precisions, -np.inf)))
        return {
            "strategy":   "max_precision",
            "constraint": f"recall ≥ {min_recall:.2f}",
            **self._metrics_at(self.thresholds[idx]),
        }

    def max_recall(self, min_precision: float = 0.35) -> dict:
        """
        Highest recall subject to precision ≥ min_precision.
        Use when false negatives (missed defaults) are costly.
        """
        mask = self.precisions >= min_precision
        if not mask.any():
            raise ValueError(
                f"No threshold achieves precision ≥ {min_precision:.2f}. "
                "Lower --min-precision."
            )
        idx = int(np.argmax(np.where(mask, self.recalls, -np.inf)))
        return {
            "strategy":   "max_recall",
            "constraint": f"precision ≥ {min_precision:.2f}",
            **self._metrics_at(self.thresholds[idx]),
        }

    def youden(self) -> dict:
        """
        Youden's J = sensitivity + specificity − 1.
        Maximises the geometric distance from the ROC diagonal.
        Strategy-neutral: does not favour precision over recall.
        """
        j_stats = self.recalls + self.specificities - 1
        idx = int(np.argmax(j_stats))
        return {
            "strategy": "youden",
            "j_stat":   round(float(j_stats[idx]), 4),
            **self._metrics_at(self.thresholds[idx]),
        }

    def cost_sensitive(self, fp_cost: float = 1.0, fn_cost: float = 5.0) -> dict:
        """
        Minimise expected misclassification cost:
            cost(t) = FP(t) × fp_cost + FN(t) × fn_cost

        Default: fn_cost=5 × fp_cost (an undetected default costs 5× a
        wrongful rejection), reflecting the asymmetric risk in lending.
        """
        costs = np.array([
            self._cost_at(t, fp_cost, fn_cost) for t in self.thresholds
        ])
        idx = int(np.argmin(costs))
        return {
            "strategy":  "cost_sensitive",
            "fp_cost":   fp_cost,
            "fn_cost":   fn_cost,
            "min_cost":  round(float(costs[idx]), 1),
            **self._metrics_at(self.thresholds[idx]),
        }

    def max_ks(self) -> dict:
        """
        Threshold at maximum Kolmogorov–Smirnov separation
        (max TPR − FPR on the ROC curve).  Standard credit-risk metric
        for assessing discriminatory power.
        """
        return {
            "strategy": "max_ks",
            "ks_stat":  round(self.ks_stat, 4),
            **self._metrics_at(self.ks_thresh),
        }

    def _cost_at(self, thresh: float, fp_cost: float, fn_cost: float) -> float:
        y_pred = (self.y_proba >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(self.y_true, y_pred, labels=[0, 1]).ravel()
        return float(fp) * fp_cost + float(fn) * fn_cost

    # ── bulk comparison ────────────────────────────────────────────────────

    def compare_all(
        self,
        min_recall: float      = 0.50,
        min_precision: float   = 0.35,
        fp_cost: float         = 1.0,
        fn_cost: float         = 5.0,
    ) -> list[dict]:
        return [
            self.max_f1(),
            self.max_precision(min_recall),
            self.max_recall(min_precision),
            self.youden(),
            self.cost_sensitive(fp_cost, fn_cost),
            self.max_ks(),
        ]

    # ── data accessors for plotting ────────────────────────────────────────

    @property
    def pr_curve(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(precision, recall, thresholds) from sklearn."""
        return self._pr_prec, self._pr_rec, self._pr_thresh

    @property
    def roc_curve(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(fpr, tpr, thresholds) from sklearn."""
        return self._roc_fpr, self._roc_tpr, self._roc_thresh

    @property
    def sweep(self) -> pd.DataFrame:
        """Full metric sweep as a DataFrame — useful for plotting."""
        return pd.DataFrame({
            "threshold":   self.thresholds,
            "precision":   self.precisions,
            "recall":      self.recalls,
            "f1":          self.f1s,
            "specificity": self.specificities,
        })


# ══════════════════════════════════════════════════════════════════════════
# Reporting
# ══════════════════════════════════════════════════════════════════════════

_COL_W = 16

def _fmt_row(d: dict) -> str:
    return (
        f"  {d['strategy']:<20}"
        f"{d['threshold']:>10.4f}"
        f"{d['precision']:>12.4f}"
        f"{d['recall']:>10.4f}"
        f"{d['f1']:>10.4f}"
        f"{d['specificity']:>13.4f}"
        f"{d['tp']:>8,}"
        f"{d['fp']:>8,}"
        f"{d['fn']:>8,}"
        f"{d['tn']:>10,}"
    )


def print_report(
    results: list[dict],
    roc_auc: float,
    pr_auc: float,
    ks_stat: float,
    current_threshold: float,
    current_metrics: dict,
    n_test: int,
) -> None:
    header = (
        f"  {'Strategy':<20}"
        f"{'Threshold':>10}"
        f"{'Precision':>12}"
        f"{'Recall':>10}"
        f"{'F1':>10}"
        f"{'Specificity':>13}"
        f"{'TP':>8}"
        f"{'FP':>8}"
        f"{'FN':>8}"
        f"{'TN':>10}"
    )
    sep = "  " + "─" * (len(header) - 2)

    print()
    print("=" * len(header))
    print("  THRESHOLD REFINEMENT — PRECISION / RECALL TRADE-OFF")
    print("=" * len(header))
    print(f"  Test set : {n_test:,} samples")
    print(f"  ROC-AUC  : {roc_auc:.5f}   |   PR-AUC (Avg Precision): {pr_auc:.5f}")
    print(f"  KS-stat  : {ks_stat:.5f}   (max separation between default / non-default score CDFs)")
    print()
    print("  Current (saved) threshold")
    print(sep)
    print(header)
    print(sep)
    print(_fmt_row({"strategy": "► current", **current_metrics}))
    print()
    print("  Candidate strategies")
    print(sep)
    print(header)
    print(sep)
    for r in results:
        constraint = r.get("constraint", "")
        extra = (
            f"  [J={r['j_stat']:.4f}]" if "j_stat" in r
            else f"  [KS={r['ks_stat']:.4f}]" if "ks_stat" in r
            else f"  [cost: FP×{r['fp_cost']}, FN×{r['fn_cost']}]" if "fn_cost" in r
            else f"  [{constraint}]" if constraint
            else ""
        )
        print(_fmt_row(r) + extra)
    print(sep)
    print()
    print("  Interpretation guide")
    print("  ─────────────────────")
    print("  Precision  = of loans flagged as default, % that truly default")
    print("               (high → fewer wrongful rejections of good borrowers)")
    print("  Recall     = of all actual defaults, % the model catches")
    print("               (high → fewer undetected defaults slip through)")
    print("  Specificity= of all good loans, % correctly approved")
    print("  Youden's J = sensitivity + specificity − 1  (ROC-optimal)")
    print("  KS-stat    = max(TPR − FPR)  (credit-risk discrimination)")
    print()


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Precision/recall threshold refinement for the LightGBM model"
    )
    parser.add_argument(
        "--strategy",
        choices=list(STRATEGIES) + ["all"],
        default="all",
        help="Which strategy to run (default: show all)",
    )
    parser.add_argument(
        "--min-recall", type=float, default=0.50,
        help="Minimum recall for max_precision strategy (default: 0.50)",
    )
    parser.add_argument(
        "--min-precision", type=float, default=0.35,
        help="Minimum precision for max_recall strategy (default: 0.35)",
    )
    parser.add_argument(
        "--fp-cost", type=float, default=1.0,
        help="Relative cost of a false positive for cost_sensitive (default: 1.0)",
    )
    parser.add_argument(
        "--fn-cost", type=float, default=5.0,
        help="Relative cost of a false negative for cost_sensitive (default: 5.0)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Overwrite threshold in tuning_metadata.json with chosen strategy result",
    )
    args = parser.parse_args()

    # ── Load model + metadata ──────────────────────────────────────────────
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}. Run tune_lightgbm.py first.")

    print("Loading model …")
    model    = lgb.Booster(model_file=str(MODEL_PATH))
    meta     = json.loads(META_PATH.read_text())
    features = meta["feature_columns"]
    saved_thresh = meta["threshold"]

    # ── Reconstruct test set ───────────────────────────────────────────────
    print("Reconstructing test split …")
    X, y, issue_d = _load_data()
    strategy = meta.get("split_strategy", "time_based_issue_d")
    print(f"  Split strategy (from metadata): {strategy}")
    X_test, y_test = _test_split(X, y, issue_d=issue_d)

    # Align columns to what model expects
    X_test = X_test[features]

    print(f"  Test size : {len(y_test):,}  |  Default rate: {y_test.mean():.3%}")

    # ── Predict ────────────────────────────────────────────────────────────
    print("Generating predictions …")
    y_proba = model.predict(X_test, num_iteration=meta.get("best_iteration", 0) or 0)

    # ── Initialise refiner ─────────────────────────────────────────────────
    print("Computing threshold sweep (500 points) …\n")
    refiner = ThresholdRefiner(y_test.values, y_proba, n_thresholds=500)

    current_metrics = refiner._metrics_at(saved_thresh)

    # ── Run strategies ─────────────────────────────────────────────────────
    if args.strategy == "all":
        results = refiner.compare_all(
            min_recall=args.min_recall,
            min_precision=args.min_precision,
            fp_cost=args.fp_cost,
            fn_cost=args.fn_cost,
        )
        chosen = results[0]  # default to max_f1 if --save without --strategy
    else:
        fn = getattr(refiner, args.strategy)
        if args.strategy == "max_precision":
            result = fn(min_recall=args.min_recall)
        elif args.strategy == "max_recall":
            result = fn(min_precision=args.min_precision)
        elif args.strategy == "cost_sensitive":
            result = fn(fp_cost=args.fp_cost, fn_cost=args.fn_cost)
        else:
            result = fn()
        results = [result]
        chosen  = result

    print_report(
        results,
        roc_auc=refiner.roc_auc,
        pr_auc=refiner.pr_auc,
        ks_stat=refiner.ks_stat,
        current_threshold=saved_thresh,
        current_metrics=current_metrics,
        n_test=len(y_test),
    )

    # ── Optionally save ────────────────────────────────────────────────────
    if args.save:
        new_thresh  = chosen["threshold"]
        new_metrics = refiner._metrics_at(new_thresh)

        meta["threshold"]         = new_thresh
        meta["threshold_strategy"] = chosen["strategy"]
        meta["test_f1"]           = new_metrics["f1"]
        meta["test_ks_stat"]      = round(refiner.ks_stat, 5)
        META_PATH.write_text(json.dumps(meta, indent=2))

        print(f"  Saved threshold {new_thresh:.4f} ({chosen['strategy']}) → {META_PATH.name}")
        print(f"  Updated test_f1: {new_metrics['f1']:.4f}")
        print()


if __name__ == "__main__":
    main()
