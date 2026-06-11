"""Tests for trade convergence analysis (pure functions, no network)."""

import numpy as np
import pandas as pd
import pytest

from src.model.convergence import (
    build_trade_factor_panel,
    add_volume_context,
    compute_factor_alignment,
    find_similar_quarters,
    compute_convergence,
    quarterly_market_data,
    factor_market_impact,
    player_market_impact,
    summarize_readability,
)
from src.data.snowflake_factors import AXIOMA_FACTORS


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_holdings():
    return pd.DataFrame({
        "institution": ["wellington", "wellington", "berkshire", "wellington"],
        "institution_name": ["Wellington", "Wellington", "Berkshire", "Wellington"],
        "style": ["active", "active", "active", "active"],
        "ticker": ["PGR", "TRV", "PGR", "PGR"],
        "quarter": ["2024-Q1", "2024-Q1", "2024-Q1", "2024-Q2"],
        "delta_shares": [1000, -500, 200, 800],
        "delta_pct": [5.0, -3.0, 1.0, 4.0],
        "shares": [10000, 5000, 2000, 10800],
        "value": [100, 50, 20, 110],
        "portfolio_weight": [0.1, 0.05, 0.02, 0.11],
    })


def _make_factor_history():
    records = []
    for quarter in ["2024-Q1", "2024-Q2"]:
        for i, ticker in enumerate(["PGR", "TRV", "CB"]):
            record = {"ticker": ticker, "quarter": quarter}
            for j, f in enumerate(AXIOMA_FACTORS):
                record[f] = (i - 1) * 0.5 + j * 0.01
            records.append(record)
    return pd.DataFrame(records)


class _FakeScaler:
    def transform(self, X):
        return np.asarray(X, dtype=float)


class _FakeModel:
    """p_buy = sigmoid(weights · factors)."""
    def __init__(self, weights):
        self.weights = np.asarray(weights, dtype=float)

    def predict_proba(self, X):
        z = np.asarray(X, dtype=float) @ self.weights
        p = 1 / (1 + np.exp(-z))
        return np.column_stack([1 - p, p])


class _FakeProfile:
    def __init__(self, factor_names, weights):
        self.factor_names = factor_names
        self.model = _FakeModel(weights)
        self.scaler = _FakeScaler()


# ---------------------------------------------------------------------------
# Panel construction
# ---------------------------------------------------------------------------

class TestBuildTradeFactorPanel:
    def test_empty_inputs(self):
        assert build_trade_factor_panel(pd.DataFrame(), pd.DataFrame()).empty

    def test_direction_flag(self):
        panel = build_trade_factor_panel(_make_holdings(), _make_factor_history())
        pgr = panel[(panel["ticker"] == "PGR") & (panel["institution"] == "wellington")
                    & (panel["quarter"] == "2024-Q1")]
        trv = panel[panel["ticker"] == "TRV"]
        assert pgr["direction"].iloc[0] == 1
        assert trv["direction"].iloc[0] == -1

    def test_factors_merged(self):
        panel = build_trade_factor_panel(_make_holdings(), _make_factor_history())
        for f in AXIOMA_FACTORS:
            assert f in panel.columns
            assert f"{f}_z" in panel.columns

    def test_zscores_centered_per_quarter(self):
        panel = build_trade_factor_panel(_make_holdings(), _make_factor_history())
        q1 = panel[panel["quarter"] == "2024-Q1"]
        # Cross-sectional z-scores should be mean ~0 within the quarter
        assert q1["value_z"].mean() == pytest.approx(0.0, abs=1e-9)

    def test_skips_zero_delta(self):
        holdings = _make_holdings()
        holdings.loc[0, "delta_shares"] = 0
        panel = build_trade_factor_panel(holdings, _make_factor_history())
        assert len(panel) == 3


# ---------------------------------------------------------------------------
# Volume context
# ---------------------------------------------------------------------------

class TestAddVolumeContext:
    def test_no_profiles(self):
        panel = build_trade_factor_panel(_make_holdings(), _make_factor_history())
        out = add_volume_context(panel, pd.DataFrame())
        assert not out["unusual_volume"].any()

    def test_unusual_volume_flag(self):
        panel = build_trade_factor_panel(_make_holdings(), _make_factor_history())
        profiles = pd.DataFrame({
            "ticker": ["PGR", "TRV"],
            "quarter": ["2024-Q1", "2024-Q1"],
            "institution": ["wellington", "wellington"],
            "vol_mean_z": [0.5, 0.0],
            "vol_trend": [0.02, -0.01],
            "high_vol_day_frac": [0.30, 0.02],
            "consecutive_high_days": [8, 1],
        })
        out = add_volume_context(panel, profiles)
        pgr = out[(out["ticker"] == "PGR") & (out["quarter"] == "2024-Q1")]
        trv = out[out["ticker"] == "TRV"]
        assert pgr["volume_confirms"].all()
        assert not trv["volume_confirms"].any()


# ---------------------------------------------------------------------------
# Factor alignment
# ---------------------------------------------------------------------------

class TestComputeFactorAlignment:
    def test_no_profiles(self):
        panel = build_trade_factor_panel(_make_holdings(), _make_factor_history())
        out = compute_factor_alignment(panel, {})
        assert out["p_buy_from_factors"].isna().all()
        assert (out["factor_alignment_score"] == 0).all()

    def test_buy_with_high_pbuy_is_aligned(self):
        panel = build_trade_factor_panel(_make_holdings(), _make_factor_history())
        # Profile that buys when "value" is high. PGR has value ~ -0.5+...,
        # use big weight so sign of value drives p_buy decisively.
        profiles = {"wellington": _FakeProfile(["value"], [10.0])}
        out = compute_factor_alignment(panel, profiles)

        wel = out[out["institution"] == "wellington"]
        for _, row in wel.iterrows():
            p = row["p_buy_from_factors"]
            assert pd.notna(p)
            predicted_buy = p >= 0.5
            actually_bought = row["direction"] > 0
            assert row["factor_aligned"] == (predicted_buy == actually_bought)
            # p_buy == 0.5 exactly means the playbook has no opinion (score 0)
            if row["factor_aligned"]:
                assert row["factor_alignment_score"] >= 0
            else:
                assert row["factor_alignment_score"] <= 0

    def test_alignment_score_bounded(self):
        panel = build_trade_factor_panel(_make_holdings(), _make_factor_history())
        profiles = {"wellington": _FakeProfile(["value"], [100.0])}
        out = compute_factor_alignment(panel, profiles)
        assert out["factor_alignment_score"].abs().max() <= 1.0


# ---------------------------------------------------------------------------
# Cross-quarter similarity
# ---------------------------------------------------------------------------

class TestFindSimilarQuarters:
    def _panel_with_repeated_setup(self, directions):
        """One institution, same ticker, identical factor setup each quarter."""
        quarters = [f"2023-Q{q}" for q in range(1, len(directions) + 1)]
        records = []
        for quarter, direction in zip(quarters, directions):
            record = {
                "institution": "wellington",
                "institution_name": "Wellington",
                "style": "active",
                "ticker": "PGR",
                "quarter": quarter,
                "direction": direction,
                "delta_shares": 100 * direction,
            }
            for f in AXIOMA_FACTORS:
                record[f"{f}_z"] = 1.0  # identical setup every quarter
            records.append(record)
        return pd.DataFrame(records)

    def test_consistent_history_high_consistency(self):
        panel = self._panel_with_repeated_setup([1, 1, 1, 1, 1])
        out = find_similar_quarters(panel)
        # Every similar quarter has the same direction
        assert (out["historical_consistency"].dropna() == 1.0).all()

    def test_contrarian_history_low_consistency(self):
        panel = self._panel_with_repeated_setup([1, -1, -1, -1, -1])
        out = find_similar_quarters(panel)
        # The lone buy disagrees with all its similar (sell) quarters
        first = out[out["quarter"] == "2023-Q1"].iloc[0]
        assert first["historical_consistency"] == 0.0

    def test_single_trade_no_comparables(self):
        panel = self._panel_with_repeated_setup([1])
        out = find_similar_quarters(panel)
        assert out["historical_consistency"].isna().all()


# ---------------------------------------------------------------------------
# Convergence
# ---------------------------------------------------------------------------

class TestComputeConvergence:
    def _row(self, direction, align, vol, consist):
        return pd.DataFrame([{
            "ticker": "PGR",
            "direction": direction,
            "factor_alignment_score": align,
            "volume_confirms": vol,
            "historical_consistency": consist,
        }])

    def test_full_convergence_buy(self):
        out = compute_convergence(self._row(1, 0.9, True, 1.0))
        assert out["verdict"].iloc[0] == "CONVERGED — BUYING"
        assert out["convergence_score"].iloc[0] >= 0.7

    def test_full_convergence_sell(self):
        out = compute_convergence(self._row(-1, 0.9, True, 1.0))
        assert out["verdict"].iloc[0] == "CONVERGED — SELLING"

    def test_divergent(self):
        out = compute_convergence(self._row(1, -0.8, False, 0.0))
        assert out["verdict"].iloc[0] == "DIVERGENT"

    def test_partial(self):
        out = compute_convergence(self._row(1, 0.5, False, 0.8))
        assert out["verdict"].iloc[0] in ("PARTIAL", "DIVERGENT", "CONVERGED — BUYING")
        score = out["convergence_score"].iloc[0]
        assert 0 <= score <= 1


# ---------------------------------------------------------------------------
# Market data + factor impact
# ---------------------------------------------------------------------------

class TestQuarterlyMarketData:
    def test_returns_and_dollar_volume(self):
        q1 = pd.bdate_range("2024-01-02", "2024-03-28")
        q2 = pd.bdate_range("2024-04-01", "2024-06-28")
        hist = pd.DataFrame({
            "Close": [100.0] * len(q1) + [110.0] * len(q2),
            "Volume": [1000] * (len(q1) + len(q2)),
        }, index=q1.append(q2))

        md = quarterly_market_data({"PGR": hist})
        q2_row = md[(md["ticker"] == "PGR") & (md["quarter"] == "2024-Q2")].iloc[0]
        assert q2_row["ret"] == pytest.approx(0.10)
        assert q2_row["dollar_volume"] == pytest.approx(110.0 * 1000 * len(q2))


class TestFactorMarketImpact:
    def test_recovers_known_factor(self):
        # 6 tickers, value exposure drives next-quarter returns at 5% per 1σ.
        tickers = [f"T{i}" for i in range(6)]
        values = np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
        quarters = [f"2023-Q{q}" for q in range(1, 5)] + ["2024-Q1", "2024-Q2"]

        fh_records = []
        md_records = []
        for qi, quarter in enumerate(quarters):
            z = (values - values.mean()) / values.std()
            for ticker, v, vz in zip(tickers, values, z):
                fh_records.append({
                    "ticker": ticker, "quarter": quarter,
                    "value": v, "momentum": 0.5,  # constant → no impact
                })
                md_records.append({
                    "ticker": ticker, "quarter": quarter,
                    # return in THIS quarter driven by PRIOR quarter exposure;
                    # exposures are constant so this is simply 0.05 * z
                    "ret": 0.05 * vz,
                    "dollar_volume": 1e9, "avg_close": 100.0,
                })

        impacts = factor_market_impact(pd.DataFrame(fh_records), pd.DataFrame(md_records))
        by_name = {fi.factor: fi for fi in impacts}

        assert "value" in by_name
        assert by_name["value"].avg_impact_bps == pytest.approx(500, rel=0.05)
        assert by_name["value"].hit_rate == 1.0
        if "momentum" in by_name:
            assert abs(by_name["momentum"].avg_impact_bps) < 1.0

    def test_empty_inputs(self):
        assert factor_market_impact(pd.DataFrame(), pd.DataFrame()) == []


# ---------------------------------------------------------------------------
# Player impact
# ---------------------------------------------------------------------------

class TestPlayerMarketImpact:
    def test_buying_into_strength(self):
        panel = pd.DataFrame({
            "institution": ["wellington"] * 4,
            "institution_name": ["Wellington"] * 4,
            "style": ["active"] * 4,
            "ticker": ["PGR"] * 4,
            "quarter": ["2023-Q1", "2023-Q2", "2023-Q3", "2023-Q4"],
            "direction": [1, 1, -1, -1],
            "delta_shares": [1000, 2000, -1500, -500],
        })
        market = pd.DataFrame({
            "ticker": ["PGR"] * 4,
            "quarter": ["2023-Q1", "2023-Q2", "2023-Q3", "2023-Q4"],
            "ret": [0.05, 0.03, -0.04, -0.02],  # up when buying, down when selling
            "dollar_volume": [1e8] * 4,
            "avg_close": [100.0] * 4,
        })
        impacts = player_market_impact(panel, market)
        assert len(impacts) == 1
        pi = impacts[0]
        assert pi.impact_bps_when_buying > 0
        assert pi.impact_bps_when_selling < 0
        assert pi.n_trades == 4
        # participation: avg(|delta| * 100 / 1e8) = avg(1000,2000,1500,500)*100/1e8
        expected = np.mean([1000, 2000, 1500, 500]) * 100.0 / 1e8 * 100
        assert pi.avg_participation_pct == pytest.approx(expected, rel=0.01)

    def test_empty_inputs(self):
        assert player_market_impact(pd.DataFrame(), pd.DataFrame()) == []


# ---------------------------------------------------------------------------
# Readability summary
# ---------------------------------------------------------------------------

class TestSummarizeReadability:
    def test_aggregates_per_institution(self):
        panel = pd.DataFrame({
            "institution": ["a", "a", "b"],
            "institution_name": ["A", "A", "B"],
            "style": ["active", "active", "passive"],
            "convergence_score": [0.8, 0.6, 0.3],
            "factor_aligned": [True, True, False],
            "volume_confirms": [True, False, False],
        })
        out = summarize_readability(panel)
        assert len(out) == 2
        a = out[out["institution"] == "a"].iloc[0]
        assert a["avg_convergence"] == pytest.approx(0.7)
        assert a["n_trades"] == 2
        # Most readable institution sorts first
        assert out.iloc[0]["institution"] == "a"

    def test_empty(self):
        assert summarize_readability(pd.DataFrame()).empty
