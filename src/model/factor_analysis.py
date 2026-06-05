"""Factor-trade analysis — discover which factors drive institutional trading.

Given user-provided factors and 13F holdings data, this module answers:
  - Which factors correlate with institutional buying/selling?
  - Which factors drive each specific institution's behavior?
  - Given current factor values, what is each institution likely doing?

This is distinct from demand_model.py: that module predicts P(buy) from
built-in factors. This module lets users bring arbitrary factors and
discovers relationships automatically.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


@dataclass
class FactorTradeResult:
    """Result of analyzing one factor's relationship to institutional trades."""
    factor_name: str
    correlation: float          # Point-biserial correlation with buy/sell
    p_value: float
    quartile_buy_rates: dict    # {"Q1": 0.35, "Q2": 0.45, "Q3": 0.55, "Q4": 0.72}
    n_observations: int
    direction: str              # "buy_when_high", "buy_when_low", or "neutral"

    @property
    def is_significant(self) -> bool:
        return self.p_value < 0.05

    @property
    def stars(self) -> str:
        if self.p_value < 0.001:
            return "***"
        if self.p_value < 0.01:
            return "**"
        if self.p_value < 0.05:
            return "*"
        return ""


@dataclass
class InstitutionFactorProfile:
    """Per-institution profile: which factors predict their trading behavior."""
    institution: str
    institution_name: str
    style: str
    n_trades: int
    significant_factors: list[FactorTradeResult]
    model_auc: float
    feature_importances: dict[str, float]  # {factor: signed coefficient}
    model: LogisticRegression | None = field(default=None, repr=False)
    scaler: StandardScaler | None = field(default=None, repr=False)
    factor_names: list[str] = field(default_factory=list)


def _merge_factors_with_trades(
    holdings: pd.DataFrame,
    factors: pd.DataFrame,
) -> pd.DataFrame:
    """Merge factor data with holdings trades.

    Creates a 'bought' binary flag from delta_shares.
    Handles both per-ticker and macro (no ticker) factors.
    """
    df = holdings[holdings["delta_shares"].notna()].copy()
    if df.empty:
        return pd.DataFrame()

    df["bought"] = (df["delta_shares"] > 0).astype(int)

    has_ticker = "ticker" in factors.columns

    if has_ticker:
        merged = df.merge(factors, on=["ticker", "quarter"], how="inner")
    else:
        # Macro factors — broadcast to all tickers
        merged = df.merge(factors, on="quarter", how="inner")

    # Identify factor columns
    meta_cols = {
        "institution", "institution_name", "style", "ticker", "quarter",
        "filing_date", "shares", "value", "delta_shares", "delta_pct",
        "portfolio_weight", "bought", "investment_discretion",
    }
    factor_cols = [c for c in merged.columns if c not in meta_cols]

    logger.info(
        "Merged: %d trade-factor observations, %d factors",
        len(merged), len(factor_cols),
    )

    return merged


def _compute_factor_trade_correlation(
    merged: pd.DataFrame,
    factor_col: str,
    institution: str | None = None,
) -> FactorTradeResult | None:
    """Compute point-biserial correlation between a factor and buy/sell."""
    df = merged.dropna(subset=[factor_col, "bought"]).copy()

    if institution:
        df = df[df["institution"] == institution]

    if len(df) < 10:
        return None

    # Ensure numeric
    df[factor_col] = pd.to_numeric(df[factor_col], errors="coerce")
    df = df.dropna(subset=[factor_col])

    if len(df) < 10 or df[factor_col].std() == 0:
        return None

    # Point-biserial correlation
    corr, p_value = stats.pointbiserialr(df["bought"], df[factor_col])

    # Quartile analysis
    try:
        df["quartile"] = pd.qcut(df[factor_col], 4, labels=["Q1", "Q2", "Q3", "Q4"])
    except ValueError:
        # Not enough unique values for quartiles — use what we can
        df["quartile"] = pd.qcut(df[factor_col].rank(method="first"), 4, labels=["Q1", "Q2", "Q3", "Q4"])

    quartile_rates = df.groupby("quartile", observed=True)["bought"].mean().to_dict()

    # Determine direction
    q1_rate = quartile_rates.get("Q1", 0.5)
    q4_rate = quartile_rates.get("Q4", 0.5)
    if corr > 0.05:
        direction = "buy_when_high"
    elif corr < -0.05:
        direction = "buy_when_low"
    else:
        direction = "neutral"

    return FactorTradeResult(
        factor_name=factor_col,
        correlation=round(corr, 4),
        p_value=round(p_value, 6),
        quartile_buy_rates={k: round(v, 3) for k, v in quartile_rates.items()},
        n_observations=len(df),
        direction=direction,
    )


def analyze_factor_trade_relationship(
    holdings: pd.DataFrame,
    factors: pd.DataFrame,
    min_observations: int = 20,
) -> list[FactorTradeResult]:
    """Analyze how each factor correlates with institutional buying/selling.

    Returns a list of FactorTradeResult, sorted by significance.
    This looks at all institutions pooled together.
    """
    merged = _merge_factors_with_trades(holdings, factors)
    if merged.empty:
        logger.warning("No overlapping trade-factor data")
        return []

    meta_cols = {
        "institution", "institution_name", "style", "ticker", "quarter",
        "filing_date", "shares", "value", "delta_shares", "delta_pct",
        "portfolio_weight", "bought", "investment_discretion",
    }
    factor_cols = [c for c in merged.columns if c not in meta_cols]

    results = []
    for col in factor_cols:
        result = _compute_factor_trade_correlation(merged, col)
        if result and result.n_observations >= min_observations:
            results.append(result)

    results.sort(key=lambda r: r.p_value)
    return results


def build_institution_profiles(
    holdings: pd.DataFrame,
    factors: pd.DataFrame,
    min_trades: int = 20,
) -> dict[str, InstitutionFactorProfile]:
    """Build a factor sensitivity profile for each institution.

    For each institution with enough trades, fits a logistic regression
    to learn which factors predict their trading behavior.
    """
    merged = _merge_factors_with_trades(holdings, factors)
    if merged.empty:
        return {}

    meta_cols = {
        "institution", "institution_name", "style", "ticker", "quarter",
        "filing_date", "shares", "value", "delta_shares", "delta_pct",
        "portfolio_weight", "bought", "investment_discretion",
    }
    factor_cols = [c for c in merged.columns if c not in meta_cols]

    if not factor_cols:
        logger.warning("No factor columns available for profiling")
        return {}

    profiles = {}

    for inst_key, inst_df in merged.groupby("institution"):
        inst_name = inst_df["institution_name"].iloc[0] if "institution_name" in inst_df.columns else inst_key
        style = inst_df["style"].iloc[0] if "style" in inst_df.columns else "unknown"

        # Need both buys and sells and enough data
        if len(inst_df) < min_trades:
            continue
        if inst_df["bought"].nunique() < 2:
            continue

        # Prepare features
        X = inst_df[factor_cols].copy()
        for col in factor_cols:
            X[col] = pd.to_numeric(X[col], errors="coerce")
            median_val = X[col].median()
            X[col] = X[col].fillna(median_val if pd.notna(median_val) else 0)

        y = inst_df["bought"].values

        # Drop zero-variance columns
        nonzero_var = [c for c in factor_cols if X[c].std() > 0]
        if len(nonzero_var) < 1:
            continue
        X = X[nonzero_var]

        # Walk-forward AUC
        quarters = sorted(inst_df["quarter"].unique())
        min_train_q = max(4, len(quarters) // 3)

        all_preds = []
        all_actuals = []

        for i in range(min_train_q, len(quarters)):
            train_mask = inst_df["quarter"].isin(set(quarters[:i]))
            test_mask = inst_df["quarter"] == quarters[i]

            X_train = X[train_mask.values]
            y_train = y[train_mask.values]
            X_test = X[test_mask.values]
            y_test = y[test_mask.values]

            if len(y_train) < 10 or len(y_test) < 3:
                continue
            if len(set(y_train)) < 2:
                continue

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            model = LogisticRegression(max_iter=1000, C=0.1, random_state=42)
            model.fit(X_train_s, y_train)

            preds = model.predict_proba(X_test_s)[:, 1]
            all_preds.extend(preds)
            all_actuals.extend(y_test)

        if len(set(all_actuals)) < 2:
            auc = 0.5
        elif len(all_preds) < 5:
            auc = 0.5
        else:
            auc = roc_auc_score(all_actuals, all_preds)

        # Fit final model on all data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = LogisticRegression(max_iter=1000, C=0.1, random_state=42)
        model.fit(X_scaled, y)

        # Signed coefficients — positive means factor increases P(buy)
        importances = {
            feat: round(float(coef), 4)
            for feat, coef in zip(nonzero_var, model.coef_[0])
        }
        importances = dict(sorted(importances.items(), key=lambda x: -abs(x[1])))

        # Per-factor univariate significance
        sig_factors = []
        for col in nonzero_var:
            result = _compute_factor_trade_correlation(merged, col, institution=inst_key)
            if result:
                sig_factors.append(result)
        sig_factors.sort(key=lambda r: r.p_value)

        profiles[inst_key] = InstitutionFactorProfile(
            institution=inst_key,
            institution_name=inst_name,
            style=style,
            n_trades=len(inst_df),
            significant_factors=sig_factors,
            model_auc=round(auc, 4),
            feature_importances=importances,
            model=model,
            scaler=scaler,
            factor_names=nonzero_var,
        )

        logger.info(
            "%s (%s): AUC=%.3f, %d trades, %d significant factors",
            inst_name, style, auc, len(inst_df),
            sum(1 for f in sig_factors if f.is_significant),
        )

    return profiles


def predict_trades_from_factors(
    profiles: dict[str, InstitutionFactorProfile],
    current_factors: pd.DataFrame,
) -> pd.DataFrame:
    """Given current factor values, predict what each institution will do.

    current_factors: DataFrame with 'ticker' and factor columns for the current period.
    Returns DataFrame with columns: institution, institution_name, ticker, p_buy, predicted_action.
    """
    if not profiles:
        return pd.DataFrame()

    has_ticker = "ticker" in current_factors.columns
    records = []

    for inst_key, profile in profiles.items():
        if profile.model is None or profile.scaler is None:
            continue

        if has_ticker:
            tickers = current_factors["ticker"].unique()
        else:
            tickers = ["ALL"]

        for ticker in tickers:
            if has_ticker:
                row_data = current_factors[current_factors["ticker"] == ticker]
            else:
                row_data = current_factors

            if row_data.empty:
                continue

            # Prepare features
            X = pd.DataFrame()
            for col in profile.factor_names:
                if col in row_data.columns:
                    X[col] = pd.to_numeric(row_data[col], errors="coerce").fillna(0)
                else:
                    X[col] = 0

            if X.empty:
                continue

            X_scaled = profile.scaler.transform(X)
            p_buy = profile.model.predict_proba(X_scaled)[:, 1].mean()

            if p_buy > 0.6:
                action = "LIKELY BUYING"
            elif p_buy < 0.4:
                action = "LIKELY SELLING"
            else:
                action = "NEUTRAL"

            records.append({
                "institution": inst_key,
                "institution_name": profile.institution_name,
                "style": profile.style,
                "ticker": ticker,
                "p_buy": round(p_buy, 3),
                "predicted_action": action,
                "model_auc": profile.model_auc,
            })

    if not records:
        return pd.DataFrame()

    return pd.DataFrame(records).sort_values(["ticker", "p_buy"], ascending=[True, False])
