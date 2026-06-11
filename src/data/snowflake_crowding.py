"""Crowding scores from prime broker desks (MS, JPM, UBS, Citi) + NR composite.

Crowding shows who's getting INTO trades — it leads the 13F tape by weeks,
because positioning data updates daily/weekly while 13Fs lag a quarter-end
by ~45 days. Rising crowding in a name means institutions are entering now.

Provider scores arrive on different scales, so each provider is z-scored
cross-sectionally (per date, across the universe) before averaging into a
composite.

Output schema (one row per ticker):
    ticker, crowding_ms, crowding_jpm, crowding_ubs, crowding_citi,
    crowding_nr, crowding_composite, crowding_delta_1m, crowding_delta_3m,
    crowding_z, n_providers
"""

from __future__ import annotations

import datetime as dt
import logging

import numpy as np
import pandas as pd

from src.data import cache
from src.data.snowflake_utilities import read_sql
from src.universe import ALL_TICKERS

logger = logging.getLogger(__name__)

NAMESPACE = "crowding"

CROWDING_PROVIDERS = ["ms", "jpm", "ubs", "citi", "nr"]

# Column names in the Snowflake crowding table — adjust if the real schema
# differs.
COL_TICKER = "TICKER"
COL_DATE = "DATE"
COL_PROVIDER = "PROVIDER"
COL_SCORE = "CROWDING_SCORE"


def _tickers_to_sql(tickers: list[str]) -> str:
    cleaned = [t.upper().replace("'", "") for t in tickers]
    return "(" + ", ".join(f"'{t}'" for t in cleaned) + ")"


def _conn_kwargs(snowflake_cfg: dict) -> dict:
    return {
        "warehouse": snowflake_cfg.get("warehouse", "WHSE_TEAM_WILHELM_001"),
        "database": snowflake_cfg.get("database", "DB_TEAM_WILHELM_001"),
        "schema": snowflake_cfg.get("schema", "PUBLIC"),
    }


def fetch_crowding(
    tickers: list[str],
    start_date: str,
    end_date: str,
    snowflake_cfg: dict,
) -> pd.DataFrame:
    """Pull raw crowding scores from Snowflake.

    Returns long-format DataFrame: ticker, date, provider, crowding_score.
    """
    table = snowflake_cfg.get("tables", {}).get("crowding", "")
    if not table or table == "FILL_ME_IN":
        raise ValueError(
            "snowflake.tables.crowding is not configured in config.yaml"
        )

    query = f"""
        SELECT
            {COL_TICKER} AS ticker,
            {COL_DATE} AS date,
            {COL_PROVIDER} AS provider,
            {COL_SCORE} AS crowding_score
        FROM {table}
        WHERE {COL_TICKER} IN {_tickers_to_sql(tickers)}
          AND {COL_DATE} >= '{start_date}'
          AND {COL_DATE} < '{end_date}'
        ORDER BY {COL_TICKER}, {COL_DATE}
    """
    df = read_sql(query, to_lower=True, **_conn_kwargs(snowflake_cfg))
    if df.empty:
        return df

    df["ticker"] = df["ticker"].str.upper()
    df["provider"] = df["provider"].str.lower()
    df["date"] = pd.to_datetime(df["date"])
    df["crowding_score"] = pd.to_numeric(df["crowding_score"], errors="coerce")
    return df.dropna(subset=["crowding_score"])


def _composite_timeseries(raw: pd.DataFrame) -> pd.DataFrame:
    """Build a daily composite crowding series per ticker.

    Each provider is z-scored cross-sectionally (per provider, per date,
    across tickers) to normalize scale differences, then averaged across
    providers.

    Returns: ticker, date, composite, n_providers.
    """
    df = raw.copy()

    # Cross-sectional z-score per provider per date
    grouped = df.groupby(["provider", "date"])["crowding_score"]
    mean = grouped.transform("mean")
    std = grouped.transform("std").replace(0, np.nan)
    df["score_z"] = (df["crowding_score"] - mean) / std
    # Single-ticker dates have no cross-section — fall back to raw score
    df["score_z"] = df["score_z"].fillna(0.0)

    composite = (
        df.groupby(["ticker", "date"])
        .agg(composite=("score_z", "mean"), n_providers=("provider", "nunique"))
        .reset_index()
    )
    return composite.sort_values(["ticker", "date"])


def compute_crowding_signals(raw: pd.DataFrame) -> pd.DataFrame:
    """Compute per-ticker crowding signals from raw provider scores.

    Returns one row per ticker with latest provider scores, composite,
    1m/3m deltas, and a cross-sectional z-score.
    """
    if raw.empty:
        return pd.DataFrame()

    composite = _composite_timeseries(raw)

    records = []
    for ticker, group in composite.groupby("ticker"):
        series = group.set_index("date")["composite"].sort_index()
        latest = series.iloc[-1]

        def _delta(days_back: int) -> float:
            target = series.index[-1] - pd.Timedelta(days=days_back)
            prior = series[series.index <= target]
            if prior.empty:
                return np.nan
            return float(latest - prior.iloc[-1])

        record = {
            "ticker": ticker,
            "crowding_composite": round(float(latest), 4),
            "crowding_delta_1m": round(_delta(30), 4) if pd.notna(_delta(30)) else np.nan,
            "crowding_delta_3m": round(_delta(91), 4) if pd.notna(_delta(91)) else np.nan,
            "n_providers": int(group.iloc[-1]["n_providers"]),
        }

        # Latest raw score per provider
        ticker_raw = raw[raw["ticker"] == ticker]
        for provider in CROWDING_PROVIDERS:
            prov = ticker_raw[ticker_raw["provider"] == provider]
            if prov.empty:
                record[f"crowding_{provider}"] = np.nan
            else:
                record[f"crowding_{provider}"] = float(
                    prov.sort_values("date")["crowding_score"].iloc[-1]
                )

        records.append(record)

    df = pd.DataFrame(records)

    # Cross-sectional z of the latest composite across the universe
    comp = df["crowding_composite"]
    std = comp.std()
    df["crowding_z"] = (comp - comp.mean()) / std if std and std > 0 else 0.0
    df["crowding_z"] = df["crowding_z"].round(4)

    return df


def build_quarterly_history(raw: pd.DataFrame) -> pd.DataFrame:
    """Quarter-end composite crowding per ticker, for demand model training.

    Returns: ticker, quarter (YYYY-QN), crowding_composite, crowding_delta_1m.
    """
    if raw.empty:
        return pd.DataFrame()

    composite = _composite_timeseries(raw)
    composite["quarter"] = (
        composite["date"].dt.year.astype(str)
        + "-Q"
        + composite["date"].dt.quarter.astype(str)
    )

    records = []
    for (ticker, quarter), group in composite.groupby(["ticker", "quarter"]):
        group = group.sort_values("date")
        latest = float(group["composite"].iloc[-1])

        # ~1 month before quarter end
        target = group["date"].iloc[-1] - pd.Timedelta(days=30)
        prior = group[group["date"] <= target]
        delta_1m = latest - float(prior["composite"].iloc[-1]) if not prior.empty else np.nan

        records.append({
            "ticker": ticker,
            "quarter": quarter,
            "crowding_composite": round(latest, 4),
            "crowding_delta_1m": round(delta_1m, 4) if pd.notna(delta_1m) else np.nan,
        })

    return pd.DataFrame(records)


def refresh_all(
    data_dir: str,
    snowflake_cfg: dict,
    tickers: list[str] | None = None,
    start_year: int = 2015,
    force: bool = False,
) -> pd.DataFrame:
    """Pull crowding from Snowflake; compute latest signals + quarterly history."""
    if not force and not cache.is_stale(data_dir, NAMESPACE, "latest", max_age_hours=12):
        cached = cache.load(data_dir, NAMESPACE, "latest")
        if cached is not None:
            logger.info("Crowding cache is fresh")
            return cached

    if tickers is None:
        tickers = ALL_TICKERS

    end = dt.date.today() + dt.timedelta(days=1)
    start = dt.date(start_year, 1, 1)

    logger.info("Pulling crowding scores from Snowflake for %d tickers", len(tickers))
    raw = fetch_crowding(tickers, start.isoformat(), end.isoformat(), snowflake_cfg)
    if raw.empty:
        logger.warning("No crowding data returned from Snowflake")
        return pd.DataFrame()

    signals = compute_crowding_signals(raw)
    cache.save(signals, data_dir, NAMESPACE, "latest")

    history = build_quarterly_history(raw)
    cache.save(history, data_dir, NAMESPACE, "quarterly_history")

    logger.info(
        "Crowding: %d tickers, %d ticker-quarters of history",
        len(signals), len(history),
    )
    return signals


def load_signals(data_dir: str) -> pd.DataFrame | None:
    return cache.load(data_dir, NAMESPACE, "latest")


def load_history(data_dir: str) -> pd.DataFrame | None:
    return cache.load(data_dir, NAMESPACE, "quarterly_history")
