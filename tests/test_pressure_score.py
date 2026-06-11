"""Tests for pressure score computation."""

import pandas as pd
import numpy as np
import pytest

from src.model.pressure_score import (
    compute_pressure_scores, _sigmoid_scale, PressureResult, DEFAULT_WEIGHTS,
)
from src.model.volume_model import VolumePrediction
from src.universe import INSURANCE_UNIVERSE


class TestSigmoidScale:
    def test_zero(self):
        assert _sigmoid_scale(0) == 0.0

    def test_positive(self):
        assert _sigmoid_scale(2.0) > 0
        assert _sigmoid_scale(2.0) < 1

    def test_negative(self):
        assert _sigmoid_scale(-2.0) < 0
        assert _sigmoid_scale(-2.0) > -1

    def test_symmetry(self):
        assert _sigmoid_scale(1.0) == pytest.approx(-_sigmoid_scale(-1.0))

    def test_large_value_bounded(self):
        assert _sigmoid_scale(100.0) <= 1.0
        assert _sigmoid_scale(-100.0) >= -1.0


class TestPressureScore:
    def _make_inputs(self):
        residuals = pd.DataFrame({
            "ticker": ["PGR", "TRV"],
            "residual_z": [2.0, -1.5],
            "residual_combined": [0.15, -0.10],
        })
        volume_signals = pd.DataFrame({
            "ticker": ["PGR", "TRV"],
            "volume_zscore": [1.5, 0.3],
            "cum_anomaly_5d": [3.0, -1.0],
        })
        options_signals = pd.DataFrame({
            "ticker": ["PGR", "TRV"],
            "pc_volume_ratio": [0.7, 1.3],
        })
        etf_signals = pd.DataFrame({
            "etf_ticker": ["KIE"],
            "flow_zscore": [1.0],
        })
        factors_df = pd.DataFrame({
            "ticker": ["PGR", "TRV"],
            "earnings_revision": [5.0, -3.0],
        })
        ownership = pd.DataFrame({
            "ticker": ["PGR", "TRV"],
            "hhi_delta": [0.005, -0.003],
        })
        vol_preds = [
            VolumePrediction("PGR", 0.75, "UP", 0.8),
            VolumePrediction("TRV", 0.60, "DOWN", 0.6),
        ]
        return residuals, volume_signals, options_signals, etf_signals, factors_df, ownership, vol_preds

    def test_produces_results(self):
        args = self._make_inputs()
        results = compute_pressure_scores(*args)
        assert len(results) == 2
        assert all(isinstance(r, PressureResult) for r in results)

    def test_score_range(self):
        args = self._make_inputs()
        results = compute_pressure_scores(*args)
        for r in results:
            assert -100 <= r.score <= 100

    def test_positive_residual_positive_score(self):
        args = self._make_inputs()
        results = compute_pressure_scores(*args)
        pgr = [r for r in results if r.ticker == "PGR"][0]
        assert pgr.score > 0
        assert pgr.direction == "ACCUMULATE"

    def test_negative_residual_negative_score(self):
        args = self._make_inputs()
        results = compute_pressure_scores(*args)
        trv = [r for r in results if r.ticker == "TRV"][0]
        assert trv.score < 0
        assert trv.direction == "DISTRIBUTE"

    def test_components_populated(self):
        args = self._make_inputs()
        results = compute_pressure_scores(*args)
        for r in results:
            assert "residual" in r.components
            assert "volume_anomaly" in r.components
            assert "options_signal" in r.components
            assert "crowding" in r.components

    def test_crowding_neutral_when_missing(self):
        args = self._make_inputs()
        results = compute_pressure_scores(*args)
        for r in results:
            assert r.components["crowding"] == 0.0

    def test_crowding_rising_is_positive(self):
        args = self._make_inputs()
        crowding = pd.DataFrame({
            "ticker": ["PGR", "TRV"],
            "crowding_composite": [0.5, -0.3],
            "crowding_delta_1m": [0.8, -0.6],
            "crowding_z": [0.5, -0.4],
        })
        results = compute_pressure_scores(*args, crowding_signals=crowding)
        pgr = [r for r in results if r.ticker == "PGR"][0]
        trv = [r for r in results if r.ticker == "TRV"][0]
        assert pgr.components["crowding"] > 0
        assert trv.components["crowding"] < 0

    def test_crowding_saturation_dampens(self):
        # Same delta, but one name is already extremely crowded — the
        # saturated name should get a smaller positive signal.
        args = self._make_inputs()
        crowding = pd.DataFrame({
            "ticker": ["PGR", "TRV"],
            "crowding_composite": [3.0, 0.0],
            "crowding_delta_1m": [0.8, 0.8],
            "crowding_z": [2.0, 0.0],
        })
        results = compute_pressure_scores(*args, crowding_signals=crowding)
        pgr = [r for r in results if r.ticker == "PGR"][0]
        trv = [r for r in results if r.ticker == "TRV"][0]
        assert 0 < pgr.components["crowding"] < trv.components["crowding"]


class TestDefaultWeights:
    def test_weights_sum_to_one(self):
        assert sum(DEFAULT_WEIGHTS.values()) == pytest.approx(1.0)

    def test_has_eight_components(self):
        assert len(DEFAULT_WEIGHTS) == 8
        assert "crowding" in DEFAULT_WEIGHTS


class TestPressureResultStrength:
    def test_strong(self):
        r = PressureResult("X", "X", 75, "ACCUMULATE", 0.9, {}, 2.0, 0.8, [])
        assert r.strength == "STRONG"

    def test_moderate(self):
        r = PressureResult("X", "X", 45, "ACCUMULATE", 0.7, {}, 1.0, 0.6, [])
        assert r.strength == "MODERATE"

    def test_weak(self):
        r = PressureResult("X", "X", 25, "ACCUMULATE", 0.5, {}, 0.5, 0.4, [])
        assert r.strength == "WEAK"

    def test_negligible(self):
        r = PressureResult("X", "X", 5, "NEUTRAL", 0.3, {}, 0.1, 0.2, [])
        assert r.strength == "NEGLIGIBLE"
