"""Daily volume pipeline — anomaly detection and volume z-scores.

For each insurance stock:
    - Pull daily volume history
    - Compute rolling 20-day mean and std
    - Volume z-score = (volume - 20d_mean) / 20d_std
    - Flag spikes (z > 2.0)
    - Cumulative volume anomaly over 5d, 10d, 20d windows
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import yfinance as yf

from src.data import cache
from src.universe import ALL_TICKERS

logger = logging.getLogger(__name__)

NAMESPACE = "volume"

SPIKE_THRESHOLD = 2.0


def _pull_volume(ticker: str, period: str = "6mo") -> pd.DataFrame:
    """Pull daily volume from yfinance."""
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period=period, auto_adjust=True)
        if hist.empty:
            return pd.DataFrame()
        hist.index = hist.index.tz_localize(None)
        return hist[["Volume", "Close"]].copy()
    except Exception as e:
        logger.warning("Volume pull failed for %s: %s", ticker, e)
        return pd.DataFrame()


def compute_volume_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Compute volume z-score, spike flags, and cumulative anomaly."""
    if df.empty or "Volume" not in df.columns:
        return df

    df = df.copy()
    df["vol_20d_mean"] = df["Volume"].rolling(20).mean()
    df["vol_20d_std"] = df["Volume"].rolling(20).std()
    df["volume_zscore"] = (
        (df["Volume"] - df["vol_20d_mean"]) / df["vol_20d_std"].replace(0, np.nan)
    )
    df["volume_spike"] = df["volume_zscore"] > SPIKE_THRESHOLD

    # Cumulative anomaly: sum of z-scores over rolling windows
    df["cum_anomaly_5d"] = df["volume_zscore"].rolling(5).sum()
    df["cum_anomaly_10d"] = df["volume_zscore"].rolling(10).sum()
    df["cum_anomaly_20d"] = df["volume_zscore"].rolling(20).sum()

    # Dollar volume
    df["dollar_volume"] = df["Volume"] * df["Close"]

    return df


def get_latest_signals(ticker: str) -> dict:
    """Get the most recent volume signals for a single stock."""
    raw = _pull_volume(ticker)
    if raw.empty:
        return {"ticker": ticker}

    signals = compute_volume_signals(raw)
    latest = signals.iloc[-1]

    return {
        "ticker": ticker,
        "volume": int(latest.get("Volume", 0)),
        "volume_zscore": round(latest.get("volume_zscore", 0), 2),
        "volume_spike": bool(latest.get("volume_spike", False)),
        "cum_anomaly_5d": round(latest.get("cum_anomaly_5d", 0), 2),
        "cum_anomaly_10d": round(latest.get("cum_anomaly_10d", 0), 2),
        "cum_anomaly_20d": round(latest.get("cum_anomaly_20d", 0), 2),
        "dollar_volume": latest.get("dollar_volume", 0),
    }


def refresh_all(
    data_dir: str,
    tickers: list[str] | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """Pull volume signals for the full universe."""
    if not force and not cache.is_stale(data_dir, NAMESPACE, "latest", max_age_hours=4):
        cached = cache.load(data_dir, NAMESPACE, "latest")
        if cached is not None:
            return cached

    if tickers is None:
        tickers = ALL_TICKERS

    records = []
    for ticker in tickers:
        logger.info("Volume signals for %s", ticker)
        signals = get_latest_signals(ticker)
        records.append(signals)

    df = pd.DataFrame(records)
    cache.save(df, data_dir, NAMESPACE, "latest")
    return df


def load_latest(data_dir: str) -> pd.DataFrame | None:
    return cache.load(data_dir, NAMESPACE, "latest")
