"""Tests for factor loading and factor-trade analysis."""

import numpy as np
import pandas as pd
import pytest
import tempfile
import os

from src.data.factor_loader import load_factors, describe_factors, _normalize_quarter
from src.model.factor_analysis import (
    analyze_factor_trade_relationship,
    build_institution_profiles,
    predict_trades_from_factors,
    _merge_factors_with_trades,
)


# ---------------------------------------------------------------------------
# Factor loader tests
# ---------------------------------------------------------------------------

class TestNormalizeQuarter:
    def test_standard_format(self):
        assert _normalize_quarter("2024-Q1") == "2024-Q1"
        assert _normalize_quarter("2024-Q4") == "2024-Q4"

    def test_no_dash(self):
        assert _normalize_quarter("2024Q1") == "2024-Q1"
        assert _normalize_quarter("2024Q3") == "2024-Q3"

    def test_reversed_format(self):
        assert _normalize_quarter("Q1-2024") == "2024-Q1"
        assert _normalize_quarter("Q4 2024") == "2024-Q4"

    def test_invalid(self):
        assert _normalize_quarter("hello") is None
        assert _normalize_quarter("2024-Q5") is None
        assert _normalize_quarter("") is None


class TestLoadFactors:
    def test_csv_with_quarter_and_ticker(self, tmp_path):
        csv = tmp_path / "factors.csv"
        csv.write_text(
            "ticker,quarter,combined_ratio,momentum\n"
            "PGR,2024-Q1,95.5,0.12\n"
            "PGR,2024-Q2,96.1,0.08\n"
            "TRV,2024-Q1,98.0,-0.03\n"
            "TRV,2024-Q2,97.5,0.05\n"
        )
        df = load_factors(str(csv))
        assert len(df) == 4
        assert "quarter" in df.columns
        assert "ticker" in df.columns
        assert "combined_ratio" in df.columns
        assert "momentum" in df.columns
        assert df["quarter"].iloc[0] == "2024-Q1"

    def test_csv_macro_no_ticker(self, tmp_path):
        csv = tmp_path / "macro.csv"
        csv.write_text(
            "quarter,treasury_10y,credit_spread\n"
            "2024-Q1,4.25,1.10\n"
            "2024-Q2,4.50,1.05\n"
        )
        df = load_factors(str(csv))
        assert len(df) == 2
        assert "ticker" not in df.columns
        assert "treasury_10y" in df.columns

    def test_date_column_converted_to_quarter(self, tmp_path):
        csv = tmp_path / "dated.csv"
        csv.write_text(
            "date,factor_a\n"
            "2024-03-31,1.5\n"
            "2024-06-30,2.0\n"
            "2024-09-30,2.5\n"
        )
        df = load_factors(str(csv))
        assert len(df) == 3
        assert df["quarter"].iloc[0] == "2024-Q1"
        assert df["quarter"].iloc[1] == "2024-Q2"
        assert df["quarter"].iloc[2] == "2024-Q3"

    def test_parquet_format(self, tmp_path):
        pq = tmp_path / "factors.parquet"
        data = pd.DataFrame({
            "ticker": ["PGR", "TRV"],
            "quarter": ["2024-Q1", "2024-Q1"],
            "cr": [95.0, 98.0],
        })
        data.to_parquet(pq)
        df = load_factors(str(pq))
        assert len(df) == 2
        assert "cr" in df.columns

    def test_no_quarter_column_raises(self, tmp_path):
        csv = tmp_path / "bad.csv"
        csv.write_text("ticker,value\nPGR,100\n")
        with pytest.raises(ValueError, match="No quarter or date column"):
            load_factors(str(csv))

    def test_no_numeric_columns_raises(self, tmp_path):
        csv = tmp_path / "nonnumeric.csv"
        csv.write_text("quarter,category\n2024-Q1,high\n2024-Q2,low\n")
        with pytest.raises(ValueError, match="No numeric factor columns"):
            load_factors(str(csv))

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_factors("/nonexistent/path.csv")

    def test_alternative_quarter_formats(self, tmp_path):
        csv = tmp_path / "alt.csv"
        csv.write_text(
            "period,val\n"
            "Q1-2024,1.0\n"
            "Q2-2024,2.0\n"
        )
        df = load_factors(str(csv))
        assert df["quarter"].iloc[0] == "2024-Q1"


class TestDescribeFactors:
    def test_basic_description(self):
        df = pd.DataFrame({
            "quarter": ["2024-Q1", "2024-Q2"],
            "ticker": ["PGR", "TRV"],
            "factor_a": [1.0, 2.0],
            "factor_b": [3.0, 4.0],
        })
        info = describe_factors(df)
        assert info["n_rows"] == 2
        assert info["n_quarters"] == 2
        assert info["has_ticker"] is True
        assert info["n_factors"] == 2
        assert "factor_a" in info["factor_names"]


# ---------------------------------------------------------------------------
# Factor-trade analysis tests
# ---------------------------------------------------------------------------

def _make_holdings_with_factors():
    """Create synthetic holdings + factors with a known relationship.

    factor_a is positively correlated with buying (high factor → more buys).
    factor_b is noise.
    """
    np.random.seed(42)
    n = 200
    quarters = [f"20{15 + i // 20}-Q{(i % 4) + 1}" for i in range(n)]

    institutions = ["fidelity"] * 100 + ["capital_group"] * 100
    inst_names = ["FMR LLC (Fidelity)"] * 100 + ["Capital Group"] * 100
    styles = ["active"] * 100 + ["active"] * 100
    tickers = np.random.choice(["PGR", "TRV", "ALL", "CB", "HIG"], n)

    # Factor a: positively correlated with buying
    factor_a = np.random.randn(n)
    # Make buying more likely when factor_a is high
    buy_prob = 1 / (1 + np.exp(-(factor_a * 1.5)))
    bought = np.random.binomial(1, buy_prob)
    delta_shares = np.where(bought, np.random.randint(1000, 50000, n), -np.random.randint(1000, 50000, n))
    delta_pct = delta_shares / 100000 * 100

    # Factor b: pure noise
    factor_b = np.random.randn(n)

    holdings = pd.DataFrame({
        "institution": institutions,
        "institution_name": inst_names,
        "style": styles,
        "ticker": tickers,
        "quarter": quarters,
        "shares": np.abs(delta_shares) + 100000,
        "value": np.abs(delta_shares) * 50,
        "delta_shares": delta_shares.astype(float),
        "delta_pct": delta_pct,
    })

    factors = pd.DataFrame({
        "ticker": tickers,
        "quarter": quarters,
        "factor_a": factor_a,
        "factor_b": factor_b,
    })

    return holdings, factors


class TestMergeFactorsWithTrades:
    def test_merge_with_ticker(self):
        holdings, factors = _make_holdings_with_factors()
        merged = _merge_factors_with_trades(holdings, factors)
        assert len(merged) > 0
        assert "bought" in merged.columns
        assert "factor_a" in merged.columns

    def test_merge_macro_factors(self):
        holdings, factors = _make_holdings_with_factors()
        # Drop ticker from factors to make them macro
        macro = factors.drop(columns=["ticker"]).drop_duplicates(subset=["quarter"])
        merged = _merge_factors_with_trades(holdings, macro)
        assert len(merged) > 0
        assert "factor_a" in merged.columns

    def test_empty_holdings(self):
        _, factors = _make_holdings_with_factors()
        empty = pd.DataFrame(columns=["institution", "ticker", "quarter", "delta_shares"])
        result = _merge_factors_with_trades(empty, factors)
        assert result.empty


class TestAnalyzeFactorTradeRelationship:
    def test_detects_significant_factor(self):
        holdings, factors = _make_holdings_with_factors()
        results = analyze_factor_trade_relationship(holdings, factors)
        assert len(results) > 0

        # factor_a should be significant (it drives buying)
        factor_a_result = next((r for r in results if r.factor_name == "factor_a"), None)
        assert factor_a_result is not None
        assert factor_a_result.correlation > 0  # positive correlation
        assert factor_a_result.is_significant  # p < 0.05

    def test_noise_factor_less_significant(self):
        holdings, factors = _make_holdings_with_factors()
        results = analyze_factor_trade_relationship(holdings, factors)

        factor_a = next(r for r in results if r.factor_name == "factor_a")
        factor_b = next((r for r in results if r.factor_name == "factor_b"), None)

        # factor_a should have stronger correlation than factor_b
        if factor_b:
            assert abs(factor_a.correlation) > abs(factor_b.correlation)

    def test_quartile_rates_populated(self):
        holdings, factors = _make_holdings_with_factors()
        results = analyze_factor_trade_relationship(holdings, factors)
        factor_a = next(r for r in results if r.factor_name == "factor_a")

        assert "Q1" in factor_a.quartile_buy_rates
        assert "Q4" in factor_a.quartile_buy_rates
        # Q4 buy rate should be higher than Q1 (factor_a drives buying)
        assert factor_a.quartile_buy_rates["Q4"] > factor_a.quartile_buy_rates["Q1"]

    def test_empty_data_returns_empty(self):
        empty = pd.DataFrame(columns=["institution", "ticker", "quarter", "delta_shares"])
        factors = pd.DataFrame(columns=["quarter", "factor_a"])
        assert analyze_factor_trade_relationship(empty, factors) == []


class TestBuildInstitutionProfiles:
    def test_creates_profiles(self):
        holdings, factors = _make_holdings_with_factors()
        profiles = build_institution_profiles(holdings, factors, min_trades=10)
        assert len(profiles) > 0
        assert "fidelity" in profiles or "capital_group" in profiles

    def test_profile_has_required_fields(self):
        holdings, factors = _make_holdings_with_factors()
        profiles = build_institution_profiles(holdings, factors, min_trades=10)

        for inst, profile in profiles.items():
            assert profile.institution == inst
            assert profile.n_trades >= 10
            assert 0 <= profile.model_auc <= 1
            assert len(profile.feature_importances) > 0
            assert profile.model is not None
            assert profile.scaler is not None

    def test_auc_above_random(self):
        holdings, factors = _make_holdings_with_factors()
        profiles = build_institution_profiles(holdings, factors, min_trades=10)
        for profile in profiles.values():
            # With a strong signal, AUC should be above random
            assert profile.model_auc >= 0.5

    def test_insufficient_data_skipped(self):
        holdings, factors = _make_holdings_with_factors()
        # Require many trades — should reduce number of profiles
        profiles = build_institution_profiles(holdings, factors, min_trades=500)
        assert len(profiles) == 0


class TestPredictTradesFromFactors:
    def test_produces_predictions(self):
        holdings, factors = _make_holdings_with_factors()
        profiles = build_institution_profiles(holdings, factors, min_trades=10)

        # Use latest quarter as "current"
        current = factors[factors["quarter"] == factors["quarter"].max()]
        predictions = predict_trades_from_factors(profiles, current)

        assert len(predictions) > 0
        assert "p_buy" in predictions.columns
        assert "predicted_action" in predictions.columns
        assert all(0 <= p <= 1 for p in predictions["p_buy"])

    def test_predicted_action_values(self):
        holdings, factors = _make_holdings_with_factors()
        profiles = build_institution_profiles(holdings, factors, min_trades=10)
        current = factors[factors["quarter"] == factors["quarter"].max()]
        predictions = predict_trades_from_factors(profiles, current)

        valid_actions = {"LIKELY BUYING", "LIKELY SELLING", "NEUTRAL"}
        assert all(a in valid_actions for a in predictions["predicted_action"])

    def test_empty_profiles(self):
        current = pd.DataFrame({"quarter": ["2024-Q1"], "factor_a": [1.0]})
        result = predict_trades_from_factors({}, current)
        assert result.empty

    def test_macro_factors_prediction(self):
        holdings, factors = _make_holdings_with_factors()
        profiles = build_institution_profiles(holdings, factors, min_trades=10)

        # Macro factors (no ticker)
        current = pd.DataFrame({
            "quarter": ["2024-Q4"],
            "factor_a": [2.0],
            "factor_b": [0.5],
        })
        predictions = predict_trades_from_factors(profiles, current)
        assert len(predictions) > 0
