"""
Unit tests for src/data_cleaning.py helper functions.
Run: pytest tests/test_data_cleaning.py -v
"""

import numpy as np
import pandas as pd
import pytest

from src.data_cleaning import parse_term, parse_emp_length, parse_pct, months_since


# ── parse_term ─────────────────────────────────────────────────────────────

class TestParseTerm:
    def test_extracts_36(self):
        s = pd.Series([" 36 months"])
        assert parse_term(s).iloc[0] == 36.0

    def test_extracts_60(self):
        s = pd.Series([" 60 months"])
        assert parse_term(s).iloc[0] == 60.0

    def test_missing_becomes_nan(self):
        s = pd.Series([None])
        assert pd.isna(parse_term(s).iloc[0])

    def test_multiple_values(self):
        s = pd.Series([" 36 months", " 60 months", None])
        result = parse_term(s)
        assert result.iloc[0] == 36.0
        assert result.iloc[1] == 60.0
        assert pd.isna(result.iloc[2])


# ── parse_emp_length ───────────────────────────────────────────────────────

class TestParseEmpLength:
    def test_ten_plus_years(self):
        s = pd.Series(["10+ years"])
        assert parse_emp_length(s).iloc[0] == 10.0

    def test_less_than_one_year(self):
        s = pd.Series(["< 1 year"])
        assert parse_emp_length(s).iloc[0] == 0.0

    def test_two_years(self):
        s = pd.Series(["2 years"])
        assert parse_emp_length(s).iloc[0] == 2.0

    def test_n_a_becomes_nan(self):
        s = pd.Series(["n/a"])
        result = parse_emp_length(s)
        # "n/a" has no digit — should produce NaN
        assert pd.isna(result.iloc[0])

    def test_multiple_values(self):
        s = pd.Series(["< 1 year", "5 years", "10+ years"])
        result = parse_emp_length(s)
        assert result.tolist() == [0.0, 5.0, 10.0]


# ── parse_pct ──────────────────────────────────────────────────────────────

class TestParsePct:
    def test_string_with_percent(self):
        s = pd.Series(["13.5%", "7.99%"])
        result = parse_pct(s)
        assert pytest.approx(result.iloc[0]) == 13.5
        assert pytest.approx(result.iloc[1]) == 7.99

    def test_already_numeric(self):
        s = pd.Series([13.5, 7.99])
        result = parse_pct(s)
        assert pytest.approx(result.iloc[0]) == 13.5

    def test_invalid_becomes_nan(self):
        s = pd.Series(["N/A"])
        result = parse_pct(s)
        assert pd.isna(result.iloc[0])


# ── months_since ───────────────────────────────────────────────────────────

class TestMonthsSince:
    def test_known_delta(self):
        s = pd.Series(["Jan-2019"])
        ref = pd.Timestamp("2019-07-01")
        result = months_since(s, ref)
        # ~6 months
        assert result.iloc[0] == pytest.approx(6, abs=1)

    def test_same_month(self):
        s = pd.Series(["Jan-2019"])
        ref = pd.Timestamp("2019-01-15")
        result = months_since(s, ref)
        assert result.iloc[0] == pytest.approx(0, abs=1)

    def test_invalid_date_becomes_nan(self):
        s = pd.Series(["bad-date"])
        ref = pd.Timestamp("2019-01-01")
        result = months_since(s, ref)
        assert pd.isna(result.iloc[0])
