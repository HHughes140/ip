"""Tests for crowding signal computation (pure functions, no Snowflake)."""

import numpy as np
import pandas as pd
import pytest

from src.data.snowflake_crowding import (
    compute_crowding_signals,
    build_quarterly_history,
    _composite_timeseries,
    CROWDING_PROVIDERS,
)


def _make_raw(days: int = 120, tickers: tuple = ("PGR", "TRV", "CB")) -> pd.DataFrame:
    """Synthetic raw crowding panel.

    PGR's crowding RISES relative to the cross-section over time (starts at
    the bottom, ends at the top), so its composite delta should be positive.
    TRV starts relatively high and gets overtaken, so its delta is negative.
    """
    dates = pd.bdate_range(end="2026-06-01", periods=days)
    offset = {"PGR": 0.0, "TRV": 0.5, "CB": -0.5}
    trend = {"PGR": 1.0, "TRV": 0.0, "CB": 0.0}
    records = []
    for provider, scale in [("ms", 1.0), ("jpm", 10.0), ("ubs", 0.5), ("citi", 5.0), ("nr", 2.0)]:
        for ticker in tickers:
            for i, date in enumerate(dates):
                base = offset[ticker] + trend[ticker] * (i / days)
                records.append({
                    "ticker": ticker,
                    "date": date,
                    "provider": provider,
                    "crowding_score": (base + 0.1 * np.sin(i)) * scale,
                })
    return pd.DataFrame(records)


class TestCompositeTimeseries:
    def test_shape_and_columns(self):
        raw = _make_raw()
        comp = _composite_timeseries(raw)
        assert set(comp.columns) == {"ticker", "date", "composite", "n_providers"}
        assert comp["ticker"].nunique() == 3

    def test_provider_count(self):
        raw = _make_raw()
        comp = _composite_timeseries(raw)
        assert (comp["n_providers"] == 5).all()

    def test_scale_invariance(self):
        # Providers report on wildly different scales; z-scoring should make
        # the composite comparable regardless of provider scale.
        raw = _make_raw()
        comp = _composite_timeseries(raw)
        last = comp[comp["date"] == comp["date"].max()]
        pgr = last[last["ticker"] == "PGR"]["composite"].iloc[0]
        trv = last[last["ticker"] == "TRV"]["composite"].iloc[0]
        assert pgr > trv


class TestComputeCrowdingSignals:
    def test_empty_input(self):
        assert compute_crowding_signals(pd.DataFrame()).empty

    def test_one_row_per_ticker(self):
        signals = compute_crowding_signals(_make_raw())
        assert len(signals) == 3
        assert signals["ticker"].is_unique

    def test_expected_columns(self):
        signals = compute_crowding_signals(_make_raw())
        for col in ["crowding_composite", "crowding_delta_1m", "crowding_delta_3m",
                    "crowding_z", "n_providers"]:
            assert col in signals.columns
        for provider in CROWDING_PROVIDERS:
            assert f"crowding_{provider}" in signals.columns

    def test_rising_crowding_positive_delta(self):
        signals = compute_crowding_signals(_make_raw())
        pgr = signals[signals["ticker"] == "PGR"].iloc[0]
        trv = signals[signals["ticker"] == "TRV"].iloc[0]
        assert pgr["crowding_delta_1m"] > 0
        assert trv["crowding_delta_1m"] < 0

    def test_crowding_z_is_cross_sectional(self):
        signals = compute_crowding_signals(_make_raw())
        # z-scores across the universe should be centered near 0
        assert signals["crowding_z"].mean() == pytest.approx(0.0, abs=1e-6)

    def test_missing_provider_is_nan(self):
        raw = _make_raw()
        raw = raw[raw["provider"] != "ubs"]
        signals = compute_crowding_signals(raw)
        assert signals["crowding_ubs"].isna().all()
        assert (signals["n_providers"] == 4).all()


class TestBuildQuarterlyHistory:
    def test_empty_input(self):
        assert build_quarterly_history(pd.DataFrame()).empty

    def test_quarter_format(self):
        history = build_quarterly_history(_make_raw(days=200))
        assert history["quarter"].str.match(r"^\d{4}-Q[1-4]$").all()

    def test_one_row_per_ticker_quarter(self):
        history = build_quarterly_history(_make_raw(days=200))
        assert not history.duplicated(subset=["ticker", "quarter"]).any()

    def test_has_demand_model_features(self):
        history = build_quarterly_history(_make_raw(days=200))
        assert "crowding_composite" in history.columns
        assert "crowding_delta_1m" in history.columns
