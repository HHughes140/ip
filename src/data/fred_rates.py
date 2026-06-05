"""Interest rate and credit spread data from FRED.

Insurance-specific macro factors:
    - 10Y Treasury yield (investment income sensitivity)
    - 2Y Treasury yield
    - 2s10s spread (yield curve)
    - BAA-AAA credit spread (risk appetite / credit conditions)
"""

from __future__ import annotations

import logging

import pandas as pd
from fredapi import Fred

from src.data import cache

logger = logging.getLogger(__name__)

NAMESPACE = "rates"

SERIES = {
    "treasury_10y": {"fred_id": "DGS10", "description": "10-Year Treasury Yield"},
    "treasury_2y": {"fred_id": "DGS2", "description": "2-Year Treasury Yield"},
    "baa_yield": {"fred_id": "DBAA", "description": "Moody's BAA Corporate Bond Yield"},
    "aaa_yield": {"fred_id": "DAAA", "description": "Moody's AAA Corporate Bond Yield"},
}


def refresh(api_key: str, data_dir: str, force: bool = False) -> pd.DataFrame:
    """Pull rate series from FRED, compute derived spreads, return daily panel."""
    if not force and not cache.is_stale(data_dir, NAMESPACE, "daily_panel"):
        cached = cache.load(data_dir, NAMESPACE, "daily_panel")
        if cached is not None:
            return cached

    fred = Fred(api_key=api_key)
    frames = {}

    for name, info in SERIES.items():
        logger.info("Fetching %s (%s)", info["fred_id"], info["description"])
        try:
            s = fred.get_series(info["fred_id"], observation_start="2010-01-01")
            frames[name] = s
        except Exception as e:
            logger.warning("Failed to fetch %s: %s", info["fred_id"], e)

    if not frames:
        return pd.DataFrame()

    df = pd.DataFrame(frames)
    df.index.name = "date"
    df.index = pd.to_datetime(df.index)

    # Derived spreads
    if "treasury_10y" in df.columns and "treasury_2y" in df.columns:
        df["spread_2s10s"] = df["treasury_10y"] - df["treasury_2y"]
    if "baa_yield" in df.columns and "aaa_yield" in df.columns:
        df["credit_spread"] = df["baa_yield"] - df["aaa_yield"]

    # Forward-fill missing days (weekends/holidays)
    df = df.ffill()

    # Compute rate changes
    for col in ["treasury_10y", "treasury_2y", "credit_spread", "spread_2s10s"]:
        if col in df.columns:
            df[f"{col}_chg_1m"] = df[col].diff(21)
            df[f"{col}_chg_3m"] = df[col].diff(63)

    cache.save(df, data_dir, NAMESPACE, "daily_panel")
    logger.info("Rate data: %d days", len(df))
    return df


def get_latest(data_dir: str) -> dict:
    """Get the most recent rate snapshot."""
    df = load_panel(data_dir)
    if df is None or df.empty:
        return {}

    latest = df.iloc[-1]
    return {k: round(v, 3) if pd.notna(v) else None for k, v in latest.items()}


def load_panel(data_dir: str) -> pd.DataFrame | None:
    return cache.load(data_dir, NAMESPACE, "daily_panel")
