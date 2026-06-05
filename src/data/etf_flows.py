"""Insurance ETF flow estimation — KIE and IAK.

Estimates net flows into insurance-sector ETFs and translates those
flows into per-stock impact using approximate ETF weights.

Since true AUM/flow data requires paid sources, we estimate flows
from price × volume changes relative to NAV-implied moves.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import yfinance as yf

from src.data import cache

logger = logging.getLogger(__name__)

NAMESPACE = "etf_flows"

# Insurance ETFs to track
INSURANCE_ETFS = {
    "KIE": {
        "name": "SPDR S&P Insurance ETF",
        "style": "equal_weight",
    },
    "IAK": {
        "name": "iShares U.S. Insurance ETF",
        "style": "market_cap_weight",
    },
}


def _estimate_flows(ticker: str, period: str = "3mo") -> pd.DataFrame:
    """Estimate daily ETF flows from volume and price data.

    Flow proxy: dollar volume × sign(close - open) as a rough
    directional indicator. Not precise, but captures the sign and
    magnitude of capital movement.
    """
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period=period, auto_adjust=True)
        if hist.empty:
            return pd.DataFrame()
        hist.index = hist.index.tz_localize(None)
    except Exception as e:
        logger.warning("ETF data failed for %s: %s", ticker, e)
        return pd.DataFrame()

    df = hist[["Open", "Close", "Volume"]].copy()
    df["dollar_volume"] = df["Volume"] * df["Close"]
    df["direction"] = np.sign(df["Close"] - df["Open"])
    df["flow_proxy"] = df["dollar_volume"] * df["direction"]

    # Rolling cumulative flow
    df["cum_flow_5d"] = df["flow_proxy"].rolling(5).sum()
    df["cum_flow_20d"] = df["flow_proxy"].rolling(20).sum()

    # Normalize to z-score
    df["flow_zscore"] = (
        (df["flow_proxy"] - df["flow_proxy"].rolling(20).mean())
        / df["flow_proxy"].rolling(20).std().replace(0, np.nan)
    )

    return df


def get_etf_signals() -> dict[str, dict]:
    """Get latest flow signals for all insurance ETFs."""
    results = {}
    for etf_ticker, meta in INSURANCE_ETFS.items():
        flows = _estimate_flows(etf_ticker)
        if flows.empty:
            results[etf_ticker] = {
                "name": meta["name"],
                "flow_zscore": None,
                "cum_flow_5d": None,
                "cum_flow_20d": None,
            }
            continue

        latest = flows.iloc[-1]
        results[etf_ticker] = {
            "name": meta["name"],
            "flow_zscore": round(latest.get("flow_zscore", 0), 2),
            "cum_flow_5d": latest.get("cum_flow_5d", 0),
            "cum_flow_20d": latest.get("cum_flow_20d", 0),
            "direction": int(latest.get("direction", 0)),
        }

    return results


def refresh(data_dir: str, force: bool = False) -> pd.DataFrame:
    """Pull ETF flow signals and cache."""
    if not force and not cache.is_stale(data_dir, NAMESPACE, "latest", max_age_hours=4):
        cached = cache.load(data_dir, NAMESPACE, "latest")
        if cached is not None:
            return cached

    signals = get_etf_signals()
    records = []
    for etf_ticker, sig in signals.items():
        sig["etf_ticker"] = etf_ticker
        records.append(sig)

    df = pd.DataFrame(records)
    cache.save(df, data_dir, NAMESPACE, "latest")
    return df


def load_latest(data_dir: str) -> pd.DataFrame | None:
    return cache.load(data_dir, NAMESPACE, "latest")
