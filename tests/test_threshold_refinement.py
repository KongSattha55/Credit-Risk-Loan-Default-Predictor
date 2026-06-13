"""
Unit tests for src/threshold_refinement.ThresholdRefiner.
Run: pytest tests/test_threshold_refinement.py -v
"""

import numpy as np
import pytest

from src.threshold_refinement import ThresholdRefiner


# ── Fixtures ───────────────────────────────────────────────────────────────

def _make_data(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Realistic (non-perfectly-separable) binary data with ~21% positives."""
    rng = np.random.default_rng(seed)
    n   = 2000
    y   = (rng.random(n) < 0.21).astype(int)
    # Large noise makes the distributions overlap heavily
    noise   = rng.random(n) * 0.65
    y_proba = np.clip(y * 0.35 + noise, 0, 1).astype(float)
    return y, y_proba


@pytest.fixture(scope="module")
def refiner():
    y, p = _make_data()
    return ThresholdRefiner(y, p, n_thresholds=200)


# ── Sweep DataFrame ────────────────────────────────────────────────────────

class TestSweep:
    def test_sweep_shape(self, refiner):
        df = refiner.sweep
        assert len(df) == 200
        assert set(df.columns) == {"threshold", "precision", "recall", "f1", "specificity"}

    def test_precision_monotone_increasing_with_threshold(self, refiner):
        # Precision generally rises as threshold rises; allow small non-monotone noise
        prec = refiner.sweep["precision"].values
        # At least 80% of consecutive pairs should be non-decreasing
        diffs = np.diff(prec)
        assert (diffs >= 0).mean() >= 0.80

    def test_recall_monotone_decreasing_with_threshold(self, refiner):
        rec = refiner.sweep["recall"].values
        diffs = np.diff(rec)
        assert (diffs <= 0).mean() >= 0.80

    def test_all_scores_in_unit_interval(self, refiner):
        df = refiner.sweep
        for col in ("precision", "recall", "f1", "specificity"):
            assert df[col].between(0, 1).all(), f"{col} has values outside [0,1]"


# ── Strategy: max_f1 ───────────────────────────────────────────────────────

class TestMaxF1:
    def test_returns_dict_with_required_keys(self, refiner):
        r = refiner.max_f1()
        for key in ("strategy", "threshold", "precision", "recall", "f1", "tp", "fp", "fn", "tn"):
            assert key in r, f"Missing key: {key}"

    def test_strategy_name(self, refiner):
        assert refiner.max_f1()["strategy"] == "max_f1"

    def test_threshold_in_unit_interval(self, refiner):
        t = refiner.max_f1()["threshold"]
        assert 0 < t < 1

    def test_f1_is_near_maximum(self, refiner):
        result = refiner.max_f1()
        best_f1 = refiner.f1s.max()
        # F1 at chosen threshold should be within 0.005 of the sweep max
        assert result["f1"] >= best_f1 - 0.005


# ── Strategy: max_precision ────────────────────────────────────────────────

class TestMaxPrecision:
    def test_recall_constraint_satisfied(self, refiner):
        min_r = 0.40
        r = refiner.max_precision(min_recall=min_r)
        assert r["recall"] >= min_r - 1e-4   # floating-point tolerance

    def test_precision_is_maximised_under_constraint(self, refiner):
        min_r = 0.40
        r = refiner.max_precision(min_recall=min_r)
        # Any threshold satisfying recall >= min_r should have precision <= chosen
        mask = refiner.recalls >= min_r
        max_attainable = refiner.precisions[mask].max()
        assert r["precision"] >= max_attainable - 0.01

    def test_impossible_constraint_raises(self, refiner):
        with pytest.raises(ValueError, match="No threshold achieves recall"):
            refiner.max_precision(min_recall=1.001)  # mathematically impossible


# ── Strategy: max_recall ───────────────────────────────────────────────────

class TestMaxRecall:
    def test_precision_constraint_satisfied(self, refiner):
        min_p = 0.25
        r = refiner.max_recall(min_precision=min_p)
        assert r["precision"] >= min_p - 1e-4

    def test_recall_is_maximised_under_constraint(self, refiner):
        min_p = 0.25
        r = refiner.max_recall(min_precision=min_p)
        mask = refiner.precisions >= min_p
        max_attainable = refiner.recalls[mask].max()
        assert r["recall"] >= max_attainable - 0.01

    def test_impossible_constraint_raises(self, refiner):
        with pytest.raises(ValueError, match="No threshold achieves precision"):
            refiner.max_recall(min_precision=1.001)  # mathematically impossible


# ── Strategy: youden ───────────────────────────────────────────────────────

class TestYouden:
    def test_returns_j_stat(self, refiner):
        r = refiner.youden()
        assert "j_stat" in r
        assert 0 <= r["j_stat"] <= 1

    def test_j_stat_equals_sensitivity_plus_specificity_minus_one(self, refiner):
        r = refiner.youden()
        expected_j = r["recall"] + r["specificity"] - 1
        assert r["j_stat"] == pytest.approx(expected_j, abs=1e-3)


# ── Strategy: cost_sensitive ───────────────────────────────────────────────

class TestCostSensitive:
    def test_returns_cost_fields(self, refiner):
        r = refiner.cost_sensitive(fp_cost=1.0, fn_cost=5.0)
        assert "fp_cost" in r
        assert "fn_cost" in r
        assert "min_cost" in r

    def test_high_fn_cost_favours_lower_threshold(self, refiner):
        """Penalising FN heavily should push toward a lower threshold (higher recall)."""
        r_equal  = refiner.cost_sensitive(fp_cost=1.0, fn_cost=1.0)
        r_fn_heavy = refiner.cost_sensitive(fp_cost=1.0, fn_cost=10.0)
        assert r_fn_heavy["threshold"] <= r_equal["threshold"] + 0.05

    def test_high_fp_cost_favours_higher_threshold(self, refiner):
        """Penalising FP heavily should push toward a higher threshold (higher precision)."""
        r_equal  = refiner.cost_sensitive(fp_cost=1.0, fn_cost=1.0)
        r_fp_heavy = refiner.cost_sensitive(fp_cost=10.0, fn_cost=1.0)
        assert r_fp_heavy["threshold"] >= r_equal["threshold"] - 0.05


# ── compare_all ────────────────────────────────────────────────────────────

class TestCompareAll:
    def test_returns_six_results(self, refiner):
        results = refiner.compare_all()
        assert len(results) == 6

    def test_strategies_are_distinct(self, refiner):
        results  = refiner.compare_all()
        names    = [r["strategy"] for r in results]
        assert len(set(names)) == 6

    def test_all_thresholds_in_unit_interval(self, refiner):
        for r in refiner.compare_all():
            assert 0 < r["threshold"] < 1, f"{r['strategy']} threshold out of range"


# ── _metrics_at ────────────────────────────────────────────────────────────

class TestMetricsAt:
    def test_tp_fp_fn_tn_sum_to_total(self, refiner):
        r = refiner._metrics_at(0.30)
        total = r["tp"] + r["fp"] + r["fn"] + r["tn"]
        assert total == len(refiner.y_true)

    def test_precision_recall_consistent_with_tp_fp_fn(self, refiner):
        r = refiner._metrics_at(0.30)
        tp, fp, fn = r["tp"], r["fp"], r["fn"]
        expected_prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        expected_rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        assert r["precision"] == pytest.approx(expected_prec, abs=1e-4)
        assert r["recall"]    == pytest.approx(expected_rec,  abs=1e-4)
