"""Factor data pipeline — fundamentals, momentum, volatility, beta.

Default provider uses yfinance (free). Defines a FactorProvider protocol
so paid sources (Capital IQ, FactSet, Refinitiv) can be swapped in.

Factors computed per stock per quarter:
    market_cap, pe_trailing, pe_forward, pb, ev_ebitda, roe,
    dividend_yield, momentum_3m, momentum_6m, momentum_12m,
    beta_1y, volatility_60d, earnings_revision
"""

from __future__ import annotations

import logging
from typing import Protocol

import numpy as np
import pandas as pd
import yfinance as yf

from src.data import cache
from src.universe import ALL_TICKERS

logger = logging.getLogger(__name__)

NAMESPACE = "factors"

FACTOR_COLUMNS = [
    "ticker", "quarter", "market_cap", "pe_trailing", "pe_forward",
    "pb", "ev_ebitda", "roe", "dividend_yield",
    "momentum_3m", "momentum_6m", "momentum_12m",
    "beta_1y", "volatility_60d", "earnings_revision",
]


# ---------------------------------------------------------------------------
# Provider protocol (for pluggable paid sources)
# ---------------------------------------------------------------------------

class FactorProvider(Protocol):
    def get_fundamentals(self, ticker: str) -> dict: ...
    def get_price_history(self, ticker: str, period: str) -> pd.DataFrame: ...


# ---------------------------------------------------------------------------
# yfinance default provider
# ---------------------------------------------------------------------------

def _get_fundamentals_yf(ticker: str) -> dict:
    """Pull fundamental factors from yfinance."""
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
    except Exception as e:
        logger.warning("yfinance info failed for %s: %s", ticker, e)
        return {}

    return {
        "market_cap": info.get("marketCap"),
        "pe_trailing": info.get("trailingPE"),
        "pe_forward": info.get("forwardPE"),
        "pb": info.get("priceToBook"),
        "ev_ebitda": info.get("enterpriseToEbitda"),
        "roe": info.get("returnOnEquity"),
        "dividend_yield": info.get("dividendYield"),
        "beta_1y": info.get("beta"),
    }


def _get_price_history_yf(ticker: str, period: str = "2y") -> pd.DataFrame:
    """Pull price history from yfinance."""
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period=period, auto_adjust=True)
        if hist.empty:
            return pd.DataFrame()
        hist.index = hist.index.tz_localize(None)
        return hist
    except Exception as e:
        logger.warning("yfinance history failed for %s: %s", ticker, e)
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Derived factor calculations
# ---------------------------------------------------------------------------

def _compute_momentum(hist: pd.DataFrame) -> dict:
    """Compute 3M, 6M, 12M price momentum from history."""
    if hist.empty or "Close" not in hist.columns:
        return {"momentum_3m": None, "momentum_6m": None, "momentum_12m": None}

    close = hist["Close"].dropna()
    if len(close) < 22:
        return {"momentum_3m": None, "momentum_6m": None, "momentum_12m": None}

    current = close.iloc[-1]

    def _ret(days: int) -> float | None:
        if len(close) >= days:
            return (current / close.iloc[-days] - 1) * 100
        return None

    return {
        "momentum_3m": _ret(63),
        "momentum_6m": _ret(126),
        "momentum_12m": _ret(252),
    }


def _compute_volatility(hist: pd.DataFrame, window: int = 60) -> float | None:
    """Compute realized volatility (annualized) over a rolling window."""
    if hist.empty or "Close" not in hist.columns:
        return None
    returns = hist["Close"].pct_change().dropna()
    if len(returns) < window:
        return None
    vol = returns.iloc[-window:].std() * np.sqrt(252) * 100
    return round(vol, 2)


def _compute_earnings_revision(ticker: str) -> float | None:
    """Estimate earnings revision from analyst estimates.

    Compares current consensus EPS to the estimate from 90 days ago.
    Returns percent change in consensus. Positive = upward revision.
    """
    try:
        t = yf.Ticker(ticker)
        # yfinance provides analyst recommendations and earnings estimates
        estimates = t.earnings_estimate
        if estimates is not None and not estimates.empty:
            # Use current year estimate change as proxy
            if "growth" in estimates.columns:
                return float(estimates["growth"].iloc[0]) * 100
        return None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def snapshot_stock(ticker: str) -> dict:
    """Compute a full factor snapshot for a single stock."""
    fundamentals = _get_fundamentals_yf(ticker)
    hist = _get_price_history_yf(ticker, period="2y")
    momentum = _compute_momentum(hist)
    vol = _compute_volatility(hist)
    revision = _compute_earnings_revision(ticker)

    record = {"ticker": ticker}
    record.update(fundamentals)
    record.update(momentum)
    record["volatility_60d"] = vol
    record["earnings_revision"] = revision

    return record


def refresh_all(
    data_dir: str,
    tickers: list[str] | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """Pull factor snapshots for the full insurance universe.

    Returns DataFrame with one row per stock and columns for each factor.
    """
    if not force and not cache.is_stale(data_dir, NAMESPACE, "current_snapshot"):
        cached = cache.load(data_dir, NAMESPACE, "current_snapshot")
        if cached is not None:
            logger.info("Factor cache is fresh")
            return cached

    if tickers is None:
        tickers = ALL_TICKERS

    records = []
    for ticker in tickers:
        logger.info("Pulling factors for %s", ticker)
        try:
            snap = snapshot_stock(ticker)
            # Tag with current quarter
            now = pd.Timestamp.now()
            quarter = (now.month - 1) // 3 + 1
            snap["quarter"] = f"{now.year}-Q{quarter}"
            records.append(snap)
        except Exception as e:
            logger.warning("Failed to get factors for %s: %s", ticker, e)

    df = pd.DataFrame(records)
    cache.save(df, data_dir, NAMESPACE, "current_snapshot")
    logger.info("Factor snapshot: %d stocks", len(df))
    return df


def load_snapshot(data_dir: str) -> pd.DataFrame | None:
    return cache.load(data_dir, NAMESPACE, "current_snapshot")
