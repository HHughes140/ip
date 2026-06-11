"""Daily price + volume from Snowflake — replaces yfinance as the market
data source.

Pulls daily OHLCV per ticker from the configured Snowflake table, then
reuses the existing volume signal logic (z-scores, spike flags, cumulative
anomalies) from src.data.volume so the output schema is unchanged:

    ticker, volume, volume_zscore, volume_spike, cum_anomaly_5d,
    cum_anomaly_10d, cum_anomaly_20d, dollar_volume

Also provides per-ticker OHLCV history (indexed by date, with "Volume" and
"Close" columns) for the accumulation fingerprint pipeline.
"""

from __future__ import annotations

import datetime as dt
import logging

import pandas as pd

from src.data import cache
from src.data.snowflake_utilities import read_sql
from src.data.volume import compute_volume_signals
from src.universe import ALL_TICKERS

logger = logging.getLogger(__name__)

NAMESPACE = "volume"

# Column names in the Snowflake price/volume table — adjust if the real
# table schema differs.
COL_TICKER = "TICKER"
COL_DATE = "DATE"
COL_CLOSE = "CLOSE"
COL_VOLUME = "VOLUME"


def _tickers_to_sql(tickers: list[str]) -> str:
    cleaned = [t.upper().replace("'", "") for t in tickers]
    return "(" + ", ".join(f"'{t}'" for t in cleaned) + ")"


def _conn_kwargs(snowflake_cfg: dict) -> dict:
    return {
        "warehouse": snowflake_cfg.get("warehouse", "WHSE_TEAM_WILHELM_001"),
        "database": snowflake_cfg.get("database", "DB_TEAM_WILHELM_001"),
        "schema": snowflake_cfg.get("schema", "PUBLIC"),
    }


def fetch_daily_data(
    tickers: list[str],
    start_date: str,
    end_date: str,
    snowflake_cfg: dict,
) -> pd.DataFrame:
    """Pull daily OHLCV for a set of tickers from Snowflake.

    Returns long-format DataFrame with columns: ticker, date, close, volume.
    """
    table = snowflake_cfg.get("tables", {}).get("price_volume", "")
    if not table or table == "FILL_ME_IN":
        raise ValueError(
            "snowflake.tables.price_volume is not configured in config.yaml"
        )

    query = f"""
        SELECT
            {COL_TICKER} AS ticker,
            {COL_DATE} AS date,
            {COL_CLOSE} AS close,
            {COL_VOLUME} AS volume
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
    return df


def get_ticker_history(
    ticker: str,
    start_date: str,
    end_date: str,
    snowflake_cfg: dict,
) -> pd.DataFrame:
    """Daily history for one ticker, shaped like the old yfinance output:
    indexed by date with capitalized "Volume" and "Close" columns.

    Used by the accumulation fingerprint pipeline.
    """
    df = fetch_daily_data([ticker], start_date, end_date, snowflake_cfg)
    if df.empty:
        return pd.DataFrame()

    out = df.set_index("date")[["volume", "close"]].rename(
        columns={"volume": "Volume", "close": "Close"}
    )
    return out.sort_index()


def get_all_histories(
    tickers: list[str],
    start_date: str,
    end_date: str,
    snowflake_cfg: dict,
) -> dict[str, pd.DataFrame]:
    """Daily histories for many tickers in a single query.

    Returns {ticker: DataFrame indexed by date with [Volume, Close]}.
    """
    df = fetch_daily_data(tickers, start_date, end_date, snowflake_cfg)
    if df.empty:
        return {}

    out = {}
    for ticker, group in df.groupby("ticker"):
        hist = group.set_index("date")[["volume", "close"]].rename(
            columns={"volume": "Volume", "close": "Close"}
        )
        out[ticker] = hist.sort_index()
    return out


def _latest_signals(hist: pd.DataFrame, ticker: str) -> dict:
    """Compute latest volume signals for one ticker's history."""
    if hist.empty:
        return {"ticker": ticker}

    signals = compute_volume_signals(hist)
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
    snowflake_cfg: dict,
    tickers: list[str] | None = None,
    force: bool = False,
) -> pd.DataFrame:
    """Pull volume signals for the full universe from Snowflake."""
    if not force and not cache.is_stale(data_dir, NAMESPACE, "latest", max_age_hours=4):
        cached = cache.load(data_dir, NAMESPACE, "latest")
        if cached is not None:
            return cached

    if tickers is None:
        tickers = ALL_TICKERS

    end = dt.date.today() + dt.timedelta(days=1)
    start = end - dt.timedelta(days=185)  # ~6 months

    logger.info("Pulling daily price/volume from Snowflake for %d tickers", len(tickers))
    histories = get_all_histories(
        tickers, start.isoformat(), end.isoformat(), snowflake_cfg
    )

    records = []
    for ticker in tickers:
        hist = histories.get(ticker, pd.DataFrame())
        if hist.empty:
            logger.warning("No Snowflake price/volume data for %s", ticker)
        records.append(_latest_signals(hist, ticker))

    df = pd.DataFrame(records)
    cache.save(df, data_dir, NAMESPACE, "latest")
    return df


def load_latest(data_dir: str) -> pd.DataFrame | None:
    return cache.load(data_dir, NAMESPACE, "latest")
