"""Flexible factor data loader.

Accepts CSV, Excel, or Parquet files containing factor data.
Auto-detects date/quarter and ticker columns, normalizes quarter
format to match 13F data (YYYY-QN), and identifies numeric factor columns.

Supports two layouts:
  - Per-ticker: columns include ticker + quarter + factor values
  - Macro/broad: columns include quarter + factor values (no ticker → applied to all)
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Column name patterns for auto-detection
_QUARTER_PATTERNS = ["quarter", "qtr", "period", "reporting_quarter"]
_DATE_PATTERNS = ["date", "reporting_date", "as_of", "asof", "timestamp"]
_TICKER_PATTERNS = ["ticker", "symbol", "stock", "cusip", "company"]

# Quarter format patterns
_QTR_REGEX_YYYY_QN = re.compile(r"^(\d{4})[- ]?Q([1-4])$", re.IGNORECASE)
_QTR_REGEX_QN_YYYY = re.compile(r"^Q([1-4])[- ]?(\d{4})$", re.IGNORECASE)
_QTR_REGEX_YYYYQN = re.compile(r"^(\d{4})Q([1-4])$", re.IGNORECASE)


def _normalize_quarter(val: str) -> str | None:
    """Convert various quarter formats to standardized YYYY-QN."""
    val = str(val).strip()

    # Already in YYYY-QN format
    m = _QTR_REGEX_YYYY_QN.match(val)
    if m:
        return f"{m.group(1)}-Q{m.group(2)}"

    # QN-YYYY or Q1 2024
    m = _QTR_REGEX_QN_YYYY.match(val)
    if m:
        return f"{m.group(2)}-Q{m.group(1)}"

    # 2024Q1
    m = _QTR_REGEX_YYYYQN.match(val)
    if m:
        return f"{m.group(1)}-Q{m.group(2)}"

    return None


def _date_to_quarter(dt: pd.Timestamp) -> str:
    """Convert a date to YYYY-QN format."""
    return f"{dt.year}-Q{(dt.month - 1) // 3 + 1}"


def _detect_column(df: pd.DataFrame, patterns: list[str]) -> str | None:
    """Find a column matching any of the given name patterns (case-insensitive)."""
    lower_cols = {c.lower().strip(): c for c in df.columns}
    for pattern in patterns:
        if pattern in lower_cols:
            return lower_cols[pattern]
    # Partial match
    for pattern in patterns:
        for lower, original in lower_cols.items():
            if pattern in lower:
                return original
    return None


def load_factors(path: str) -> pd.DataFrame:
    """Load factor data from a file.

    Accepts CSV, Excel (.xlsx/.xls), or Parquet.
    Auto-detects quarter/date and ticker columns.
    Returns DataFrame with standardized 'quarter' column,
    optional 'ticker' column, and numeric factor columns.

    Raises ValueError if the file cannot be parsed or has no usable data.
    """
    filepath = Path(path)
    if not filepath.exists():
        raise FileNotFoundError(f"Factor file not found: {path}")

    suffix = filepath.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(filepath)
    elif suffix in (".xlsx", ".xls"):
        df = pd.read_excel(filepath)
    elif suffix == ".parquet":
        df = pd.read_parquet(filepath)
    else:
        raise ValueError(f"Unsupported file format: {suffix} (use .csv, .xlsx, or .parquet)")

    if df.empty:
        raise ValueError("Factor file is empty")

    logger.info("Loaded %d rows, %d columns from %s", len(df), len(df.columns), filepath.name)

    # --- Detect and normalize quarter/date column ---
    quarter_col = _detect_column(df, _QUARTER_PATTERNS)
    date_col = _detect_column(df, _DATE_PATTERNS) if quarter_col is None else None

    if quarter_col:
        # Try to normalize quarter values
        normalized = df[quarter_col].astype(str).map(_normalize_quarter)
        if normalized.notna().sum() == 0:
            # Maybe it's a date column mislabeled as quarter
            try:
                dates = pd.to_datetime(df[quarter_col])
                df["quarter"] = dates.map(_date_to_quarter)
            except Exception:
                raise ValueError(
                    f"Column '{quarter_col}' doesn't contain recognizable quarter or date values. "
                    f"Expected formats: 2024-Q1, Q1-2024, 2024Q1, or dates."
                )
        else:
            df["quarter"] = normalized
        if quarter_col != "quarter":
            df = df.drop(columns=[quarter_col])
    elif date_col:
        try:
            dates = pd.to_datetime(df[date_col])
            df["quarter"] = dates.map(_date_to_quarter)
        except Exception:
            raise ValueError(f"Column '{date_col}' doesn't contain parseable dates")
        df = df.drop(columns=[date_col])
    else:
        raise ValueError(
            "No quarter or date column found. Include a column named 'quarter', 'date', "
            "'period', or 'reporting_date'."
        )

    # Drop rows where quarter couldn't be parsed
    df = df.dropna(subset=["quarter"])

    # --- Detect ticker column ---
    ticker_col = _detect_column(df, _TICKER_PATTERNS)
    has_ticker = False
    if ticker_col:
        df["ticker"] = df[ticker_col].astype(str).str.upper().str.strip()
        if ticker_col != "ticker":
            df = df.drop(columns=[ticker_col])
        has_ticker = True

    # --- Identify numeric factor columns ---
    meta_cols = {"quarter", "ticker"}
    factor_cols = []
    for col in df.columns:
        if col in meta_cols:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            factor_cols.append(col)
        else:
            # Try coercing to numeric
            coerced = pd.to_numeric(df[col], errors="coerce")
            if coerced.notna().sum() > len(df) * 0.5:
                df[col] = coerced
                factor_cols.append(col)
            else:
                logger.info("Skipping non-numeric column: %s", col)

    if not factor_cols:
        raise ValueError("No numeric factor columns found in the data")

    # Keep only relevant columns
    keep = ["quarter"] + (["ticker"] if has_ticker else []) + factor_cols
    df = df[keep].copy()

    logger.info(
        "Parsed %d factors: %s | %s | %d quarters",
        len(factor_cols),
        ", ".join(factor_cols),
        "per-ticker" if has_ticker else "macro (no ticker)",
        df["quarter"].nunique(),
    )

    return df


def describe_factors(df: pd.DataFrame) -> dict:
    """Return a summary of loaded factor data."""
    meta_cols = {"quarter", "ticker"}
    factor_cols = [c for c in df.columns if c not in meta_cols]

    return {
        "n_rows": len(df),
        "n_quarters": df["quarter"].nunique(),
        "quarters": sorted(df["quarter"].unique()),
        "has_ticker": "ticker" in df.columns,
        "n_tickers": df["ticker"].nunique() if "ticker" in df.columns else 0,
        "factor_names": factor_cols,
        "n_factors": len(factor_cols),
    }
