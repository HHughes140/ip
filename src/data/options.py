"""Options activity pipeline — put/call ratios and unusual activity detection.

For each insurance stock:
    - Pull current options chain from yfinance
    - Compute put/call volume ratio and OI ratio
    - Compare to historical average
    - Flag unusual activity (large deviations)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import yfinance as yf

from src.data import cache
from src.universe import ALL_TICKERS

logger = logging.getLogger(__name__)

NAMESPACE = "options"


def _get_options_summary(ticker: str) -> dict:
    """Compute put/call ratios from the current options chain."""
    result = {
        "ticker": ticker,
        "pc_volume_ratio": None,
        "pc_oi_ratio": None,
        "total_call_volume": 0,
        "total_put_volume": 0,
        "total_call_oi": 0,
        "total_put_oi": 0,
        "near_term_pc_ratio": None,
        "options_available": False,
    }

    try:
        t = yf.Ticker(ticker)
        expirations = t.options
        if not expirations:
            return result

        result["options_available"] = True

        total_call_vol = 0
        total_put_vol = 0
        total_call_oi = 0
        total_put_oi = 0
        near_call_vol = 0
        near_put_vol = 0

        # Process first 4 expirations (near-term focus)
        for i, exp in enumerate(expirations[:4]):
            try:
                chain = t.option_chain(exp)
            except Exception:
                continue

            calls = chain.calls
            puts = chain.puts

            cv = int(calls["volume"].sum()) if "volume" in calls.columns else 0
            pv = int(puts["volume"].sum()) if "volume" in puts.columns else 0
            coi = int(calls["openInterest"].sum()) if "openInterest" in calls.columns else 0
            poi = int(puts["openInterest"].sum()) if "openInterest" in puts.columns else 0

            total_call_vol += cv
            total_put_vol += pv
            total_call_oi += coi
            total_put_oi += poi

            if i == 0:
                near_call_vol = cv
                near_put_vol = pv

        result["total_call_volume"] = total_call_vol
        result["total_put_volume"] = total_put_vol
        result["total_call_oi"] = total_call_oi
        result["total_put_oi"] = total_put_oi

        if total_call_vol > 0:
            result["pc_volume_ratio"] = round(total_put_vol / total_call_vol, 3)
        if total_call_oi > 0:
            result["pc_oi_ratio"] = round(total_put_oi / total_call_oi, 3)
        if near_call_vol > 0:
            result["near_term_pc_ratio"] = round(near_put_vol / near_call_vol, 3)

    except Exception as e:
        logger.warning("Options data failed for %s: %s", ticker, e)

    return result


def refresh_all(
    data_dir: str,
    tickers: list[str] | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """Pull options signals for the full universe."""
    if not force and not cache.is_stale(data_dir, NAMESPACE, "latest", max_age_hours=12):
        cached = cache.load(data_dir, NAMESPACE, "latest")
        if cached is not None:
            return cached

    if tickers is None:
        tickers = ALL_TICKERS

    records = []
    for ticker in tickers:
        logger.info("Options signals for %s", ticker)
        summary = _get_options_summary(ticker)
        records.append(summary)

    df = pd.DataFrame(records)
    cache.save(df, data_dir, NAMESPACE, "latest")
    return df


def load_latest(data_dir: str) -> pd.DataFrame | None:
    return cache.load(data_dir, NAMESPACE, "latest")
