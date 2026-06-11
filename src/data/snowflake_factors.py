"""Axioma risk model factors from Snowflake — replaces yfinance fundamentals.

Pulls daily Axioma factor exposures per ticker, then builds quarter-end
snapshots so the demand model can train walk-forward on true historical
exposures (an improvement over the old single-snapshot broadcast).

Factors:
    value, growth, momentum, short_momentum, volatility, size, leverage,
    profitability, earnings_yield, dividend_yield, liquidity,
    market_sensitivity, fx_sensitivity

Output schemas:
    history:  ticker, quarter (YYYY-QN), + 13 factor columns
    snapshot: ticker, quarter, + 13 factor columns (latest quarter only)
"""

from __future__ import annotations

import datetime as dt
import logging

import pandas as pd

from src.data import cache
from src.data.snowflake_utilities import read_sql
from src.universe import ALL_TICKERS

logger = logging.getLogger(__name__)

NAMESPACE = "factors"

AXIOMA_FACTORS = [
    "value",
    "growth",
    "momentum",
    "short_momentum",
    "volatility",
    "size",
    "leverage",
    "profitability",
    "earnings_yield",
    "dividend_yield",
    "liquidity",
    "market_sensitivity",
    "fx_sensitivity",
]

# Column names in the Snowflake Axioma table — adjust if the real schema
# differs. Maps our factor name -> Snowflake column name.
COL_TICKER = "TICKER"
COL_DATE = "DATE"
AXIOMA_COLUMN_MAP = {f: f.upper() for f in AXIOMA_FACTORS}


def _tickers_to_sql(tickers: list[str]) -> str:
    cleaned = [t.upper().replace("'", "") for t in tickers]
    return "(" + ", ".join(f"'{t}'" for t in cleaned) + ")"


def _conn_kwargs(snowflake_cfg: dict) -> dict:
    return {
        "warehouse": snowflake_cfg.get("warehouse", "WHSE_TEAM_WILHELM_001"),
        "database": snowflake_cfg.get("database", "DB_TEAM_WILHELM_001"),
        "schema": snowflake_cfg.get("schema", "PUBLIC"),
    }


def fetch_axioma_exposures(
    tickers: list[str],
    start_date: str,
    end_date: str,
    snowflake_cfg: dict,
) -> pd.DataFrame:
    """Pull daily Axioma factor exposures from Snowflake.

    Returns long-format DataFrame: ticker, date, + factor columns.
    """
    table = snowflake_cfg.get("tables", {}).get("axioma_factors", "")
    if not table or table == "FILL_ME_IN":
        raise ValueError(
            "snowflake.tables.axioma_factors is not configured in config.yaml"
        )

    factor_select = ",\n            ".join(
        f"{col} AS {name}" for name, col in AXIOMA_COLUMN_MAP.items()
    )
    query = f"""
        SELECT
            {COL_TICKER} AS ticker,
            {COL_DATE} AS date,
            {factor_select}
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
    df["date"] = pd.to_datetime(df["date"])
    for f in AXIOMA_FACTORS:
        if f in df.columns:
            df[f] = pd.to_numeric(df[f], errors="coerce")
    return df


def build_quarterly_snapshots(daily_exposures: pd.DataFrame) -> pd.DataFrame:
    """Reduce daily exposures to quarter-end snapshots per ticker.

    Takes the last available observation per ticker per quarter so the
    demand model can train walk-forward on point-in-time exposures.

    Returns: ticker, quarter (YYYY-QN), + factor columns.
    """
    if daily_exposures.empty:
        return pd.DataFrame()

    df = daily_exposures.copy()
    df["quarter"] = (
        df["date"].dt.year.astype(str)
        + "-Q"
        + df["date"].dt.quarter.astype(str)
    )

    df = df.sort_values("date")
    snapshots = df.groupby(["ticker", "quarter"], as_index=False).last()

    cols = ["ticker", "quarter"] + [f for f in AXIOMA_FACTORS if f in snapshots.columns]
    return snapshots[cols]


def refresh_all(
    data_dir: str,
    snowflake_cfg: dict,
    tickers: list[str] | None = None,
    start_year: int = 2015,
    force: bool = False,
) -> pd.DataFrame:
    """Pull Axioma exposures and build quarterly history + current snapshot.

    Returns the current snapshot (latest quarter, one row per ticker).
    """
    if not force and not cache.is_stale(data_dir, NAMESPACE, "current_snapshot"):
        cached = cache.load(data_dir, NAMESPACE, "current_snapshot")
        if cached is not None:
            logger.info("Axioma factor cache is fresh")
            return cached

    if tickers is None:
        tickers = ALL_TICKERS

    end = dt.date.today() + dt.timedelta(days=1)
    start = dt.date(start_year, 1, 1)

    logger.info("Pulling Axioma exposures from Snowflake for %d tickers", len(tickers))
    daily = fetch_axioma_exposures(
        tickers, start.isoformat(), end.isoformat(), snowflake_cfg
    )
    if daily.empty:
        logger.warning("No Axioma exposure data returned from Snowflake")
        return pd.DataFrame()

    history = build_quarterly_snapshots(daily)
    cache.save(history, data_dir, NAMESPACE, "axioma_history")
    logger.info(
        "Axioma history: %d ticker-quarters across %d quarters",
        len(history), history["quarter"].nunique(),
    )

    latest_quarter = history["quarter"].max()
    snapshot = history[history["quarter"] == latest_quarter].reset_index(drop=True)
    cache.save(snapshot, data_dir, NAMESPACE, "current_snapshot")
    logger.info("Axioma snapshot (%s): %d stocks", latest_quarter, len(snapshot))

    return snapshot


def load_snapshot(data_dir: str) -> pd.DataFrame | None:
    """Latest-quarter exposures, one row per ticker (for scoring)."""
    return cache.load(data_dir, NAMESPACE, "current_snapshot")


def load_history(data_dir: str) -> pd.DataFrame | None:
    """Full quarterly exposure history (for demand model training)."""
    return cache.load(data_dir, NAMESPACE, "axioma_history")
