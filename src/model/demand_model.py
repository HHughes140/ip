"""Institutional demand model — predict expected position changes from factors.

For each institution style (passive, active), trains:
    P(institution increases position) = f(factor_1, factor_2, ..., factor_n)

Uses logistic regression for interpretability. Walk-forward cross-validation
to avoid lookahead bias.

The model doesn't predict returns — it predicts institutional behavior.
The residual (observed - expected) is the signal.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)

FACTOR_FEATURES = [
    # Axioma risk model exposures
    "value", "growth", "momentum", "short_momentum", "volatility",
    "size", "leverage", "profitability", "earnings_yield",
    "dividend_yield", "liquidity", "market_sensitivity", "fx_sensitivity",
    # Macro rate environment (FRED)
    "treasury_10y", "credit_spread", "spread_2s10s",
    # Crowding — leading indicator of institutional entry
    "crowding_composite", "crowding_delta_1m",
]


@dataclass
class DemandModelResult:
    style: str                    # "passive" or "active"
    auc_walkforward: float        # Walk-forward AUC
    n_train: int
    n_features: int
    feature_importances: dict[str, float]  # feature → coefficient magnitude


def build_training_data(
    holdings: pd.DataFrame,
    factors: pd.DataFrame,
    rates: pd.DataFrame | None = None,
    crowding: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Merge holdings deltas with factor snapshots to create training panel.

    Each row: (institution, ticker, quarter, bought_flag, factor_1, ..., factor_n)
    """
    if holdings.empty or factors.empty:
        return pd.DataFrame()

    # Create binary target: did institution increase position?
    df = holdings[holdings["delta_shares"].notna()].copy()
    df["bought"] = (df["delta_shares"] > 0).astype(int)

    # Merge factors by ticker × quarter
    if "quarter" in factors.columns:
        df = df.merge(factors, on=["ticker", "quarter"], how="left", suffixes=("", "_factor"))
    else:
        # If factors are a current snapshot, broadcast to all quarters
        # (only usable for scoring, not training)
        for col in FACTOR_FEATURES:
            if col in factors.columns:
                ticker_factor = factors.set_index("ticker")[col]
                df[col] = df["ticker"].map(ticker_factor)

    # Merge crowding history (quarter-end composite + 1m delta)
    if crowding is not None and not crowding.empty and "quarter" in crowding.columns:
        crowd_cols = [c for c in ["crowding_composite", "crowding_delta_1m"]
                      if c in crowding.columns]
        if crowd_cols:
            df = df.merge(
                crowding[["ticker", "quarter"] + crowd_cols],
                on=["ticker", "quarter"], how="left", suffixes=("", "_crowd"),
            )

    # Merge rate environment if available
    if rates is not None and not rates.empty:
        # Get quarterly rate snapshots (last day of each quarter)
        rate_cols = [c for c in FACTOR_FEATURES if c in rates.columns]
        if rate_cols:
            rates_q = rates[rate_cols].resample("QE").last()
            rates_q.index = rates_q.index.to_period("Q").to_timestamp()
            # Map quarter strings to timestamps for merge
            # This is a simplification — in production you'd parse quarter properly

    return df


def _prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Select and clean feature columns for modeling."""
    available = [f for f in FACTOR_FEATURES if f in df.columns]
    X = df[available].copy()

    # Fill NaN with column median (simple imputation)
    for col in available:
        X[col] = pd.to_numeric(X[col], errors="coerce")
        median_val = X[col].median()
        X[col] = X[col].fillna(median_val if pd.notna(median_val) else 0)

    return X, available


def train_walk_forward(
    df: pd.DataFrame,
    style: str,
    min_train_quarters: int = 8,
) -> tuple[LogisticRegression, StandardScaler, DemandModelResult] | None:
    """Train with walk-forward cross-validation.

    For each quarter t, train on quarters [0, t-1], predict quarter t.
    Compute out-of-sample AUC.
    """
    # Filter to institution style
    style_df = df[df["style"] == style].copy() if "style" in df.columns else df.copy()

    if style_df.empty:
        logger.warning("No data for style '%s'", style)
        return None

    X, features = _prepare_features(style_df)
    y = style_df["bought"]

    if len(features) < 3 or len(y) < 50:
        logger.warning("Insufficient data for %s model: %d rows, %d features",
                       style, len(y), len(features))
        return None

    # Sort by quarter for walk-forward
    quarters = style_df["quarter"].unique()
    quarters = sorted(quarters)

    if len(quarters) < min_train_quarters + 2:
        logger.warning("Only %d quarters available for %s, need %d+2",
                       len(quarters), style, min_train_quarters)
        return None

    # Walk-forward predictions
    all_preds = []
    all_actuals = []

    for i in range(min_train_quarters, len(quarters)):
        train_quarters = set(quarters[:i])
        test_quarter = quarters[i]

        train_mask = style_df["quarter"].isin(train_quarters)
        test_mask = style_df["quarter"] == test_quarter

        X_train = X[train_mask]
        y_train = y[train_mask]
        X_test = X[test_mask]
        y_test = y[test_mask]

        if len(y_train) < 20 or len(y_test) < 5:
            continue

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = LogisticRegression(max_iter=1000, C=0.1, random_state=42)
        model.fit(X_train_scaled, y_train)

        preds = model.predict_proba(X_test_scaled)[:, 1]
        all_preds.extend(preds)
        all_actuals.extend(y_test.values)

    # Compute walk-forward AUC
    if len(set(all_actuals)) < 2:
        logger.warning("Single class in walk-forward predictions for %s", style)
        auc = 0.5
    else:
        auc = roc_auc_score(all_actuals, all_preds)

    # Fit final model on all data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LogisticRegression(max_iter=1000, C=0.1, random_state=42)
    model.fit(X_scaled, y)

    # Feature importances (absolute coefficient values)
    importances = {
        feat: round(abs(coef), 4)
        for feat, coef in zip(features, model.coef_[0])
    }
    importances = dict(sorted(importances.items(), key=lambda x: -x[1]))

    result = DemandModelResult(
        style=style,
        auc_walkforward=round(auc, 4),
        n_train=len(y),
        n_features=len(features),
        feature_importances=importances,
    )

    logger.info("%s demand model: AUC=%.3f, n=%d, features=%d",
                style, auc, len(y), len(features))

    return model, scaler, result


def predict_expected(
    model: LogisticRegression,
    scaler: StandardScaler,
    factors: pd.DataFrame,
    features: list[str],
) -> pd.Series:
    """Predict P(buy) for each stock given current factors."""
    X, available = _prepare_features(factors)
    # Align features
    for f in features:
        if f not in X.columns:
            X[f] = 0
    X = X[features]

    X_scaled = scaler.transform(X)
    proba = model.predict_proba(X_scaled)[:, 1]
    return pd.Series(proba, index=factors.index, name="expected_buy_prob")


def fit_and_save(
    holdings: pd.DataFrame,
    factors: pd.DataFrame,
    model_dir: str,
    rates: pd.DataFrame | None = None,
    crowding: pd.DataFrame | None = None,
) -> dict[str, DemandModelResult]:
    """Fit demand models for both passive and active styles, persist."""
    training_data = build_training_data(holdings, factors, rates, crowding)

    if training_data.empty:
        logger.warning("No training data available")
        return {}

    results = {}
    models = {}

    for style in ["passive", "active"]:
        outcome = train_walk_forward(training_data, style)
        if outcome is None:
            continue
        model, scaler, result = outcome
        results[style] = result
        models[style] = {"model": model, "scaler": scaler, "result": result}

    # Persist
    path = Path(model_dir)
    path.mkdir(parents=True, exist_ok=True)
    joblib.dump(models, path / "demand_models.joblib")
    logger.info("Saved demand models to %s", path)

    return results


def load_models(model_dir: str) -> dict | None:
    path = Path(model_dir) / "demand_models.joblib"
    if path.exists():
        return joblib.load(path)
    return None
