"""Tests for demand model training data construction."""

import pandas as pd
import numpy as np
import pytest

from src.model.demand_model import build_training_data, _prepare_features


class TestBuildTrainingData:
    def test_creates_bought_flag(self):
        holdings = pd.DataFrame({
            "institution": ["a", "a"],
            "ticker": ["PGR", "TRV"],
            "quarter": ["2024-Q1", "2024-Q1"],
            "delta_shares": [1000, -500],
            "style": ["active", "active"],
        })
        factors = pd.DataFrame({
            "ticker": ["PGR", "TRV"],
            "quarter": ["2024-Q1", "2024-Q1"],
            "value": [0.5, -0.2],
            "momentum": [1.1, -0.3],
        })
        result = build_training_data(holdings, factors)
        pgr = result[result["ticker"] == "PGR"]
        trv = result[result["ticker"] == "TRV"]
        assert pgr["bought"].iloc[0] == 1
        assert trv["bought"].iloc[0] == 0

    def test_empty_input(self):
        result = build_training_data(pd.DataFrame(), pd.DataFrame())
        assert result.empty

    def test_skips_nan_deltas(self):
        holdings = pd.DataFrame({
            "institution": ["a", "a"],
            "ticker": ["PGR", "TRV"],
            "quarter": ["2024-Q1", "2024-Q1"],
            "delta_shares": [1000, np.nan],
            "style": ["active", "active"],
        })
        factors = pd.DataFrame({
            "ticker": ["PGR", "TRV"],
            "quarter": ["2024-Q1", "2024-Q1"],
            "value": [0.5, -0.2],
        })
        result = build_training_data(holdings, factors)
        assert len(result) == 1

    def test_merges_crowding(self):
        holdings = pd.DataFrame({
            "institution": ["a", "a"],
            "ticker": ["PGR", "TRV"],
            "quarter": ["2024-Q1", "2024-Q1"],
            "delta_shares": [1000, -500],
            "style": ["active", "active"],
        })
        factors = pd.DataFrame({
            "ticker": ["PGR", "TRV"],
            "quarter": ["2024-Q1", "2024-Q1"],
            "value": [0.5, -0.2],
        })
        crowding = pd.DataFrame({
            "ticker": ["PGR", "TRV"],
            "quarter": ["2024-Q1", "2024-Q1"],
            "crowding_composite": [1.2, -0.8],
            "crowding_delta_1m": [0.3, -0.1],
        })
        result = build_training_data(holdings, factors, crowding=crowding)
        assert "crowding_composite" in result.columns
        assert "crowding_delta_1m" in result.columns
        pgr = result[result["ticker"] == "PGR"]
        assert pgr["crowding_composite"].iloc[0] == 1.2

    def test_crowding_optional(self):
        holdings = pd.DataFrame({
            "institution": ["a"],
            "ticker": ["PGR"],
            "quarter": ["2024-Q1"],
            "delta_shares": [1000],
            "style": ["active"],
        })
        factors = pd.DataFrame({
            "ticker": ["PGR"],
            "quarter": ["2024-Q1"],
            "value": [0.5],
        })
        result = build_training_data(holdings, factors, crowding=None)
        assert len(result) == 1


class TestPrepareFeatures:
    def test_fills_nan_with_median(self):
        df = pd.DataFrame({
            "value": [0.5, 1.0, np.nan],
            "momentum": [1.0, np.nan, -0.5],
        })
        X, features = _prepare_features(df)
        assert not X.isna().any().any()
        assert "value" in features

    def test_only_known_features(self):
        df = pd.DataFrame({
            "value": [0.5],
            "random_col": [42],
        })
        X, features = _prepare_features(df)
        assert "value" in features
        assert "random_col" not in features

    def test_includes_axioma_and_crowding_features(self):
        df = pd.DataFrame({
            "value": [0.5],
            "short_momentum": [-0.2],
            "fx_sensitivity": [0.1],
            "crowding_composite": [1.0],
            "crowding_delta_1m": [0.2],
        })
        X, features = _prepare_features(df)
        for f in ["value", "short_momentum", "fx_sensitivity",
                  "crowding_composite", "crowding_delta_1m"]:
            assert f in features
