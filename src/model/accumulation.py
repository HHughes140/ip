"""Accumulation detection — learn institutional execution fingerprints from
historical 13F + daily volume, then detect them in the current tape.

Key insight: insurance stocks are institutionally dominated. When Wellington
builds a KNSL position over 3 quarters, their execution leaves a statistical
footprint in daily volume. We can learn that footprint and detect it in
real-time, even though the next 13F is 45 days away.

Architecture:
1. VolumeProfile: for each (institution, ticker, quarter) in 13F history,
   compute the daily volume profile that accompanied the position change.
2. ExecutionFingerprint: per institution, learn the statistical signature
   of their accumulation vs. distribution vs. no-change quarters.
3. FingerprintMatcher: score current daily volume against each institution's
   learned fingerprint to estimate P(accumulating now).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.data import cache

logger = logging.getLogger(__name__)

NAMESPACE = "accumulation"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AccumulationSignal:
    ticker: str
    institution: str
    institution_name: str
    style: str

    # 13F history
    consecutive_buys: int
    consecutive_sells: int
    avg_quarterly_delta_pct: float
    total_change_pct: float
    latest_quarter: str

    # Fingerprint match
    volume_confirms: bool
    continuation_probability: float

    # Learned fingerprint detail
    fingerprint_match_score: float = 0.0
    volume_features_used: dict = field(default_factory=dict)

    @property
    def direction(self) -> str:
        if self.consecutive_buys >= 2:
            return "ACCUMULATING"
        elif self.consecutive_sells >= 2:
            return "DISTRIBUTING"
        return "UNCLEAR"


@dataclass
class ExecutionFingerprint:
    """Learned volume signature for an institution's accumulation behavior."""
    institution: str
    model: LogisticRegression | None
    scaler: StandardScaler | None
    n_training_samples: int
    auc: float
    feature_importances: dict[str, float]


# ---------------------------------------------------------------------------
# Volume profile features
# ---------------------------------------------------------------------------

VOLUME_PROFILE_FEATURES = [
    "vol_mean_z",           # Mean volume z-score during the quarter
    "vol_std_z",            # Volatility of volume z-score (choppy = distribution, smooth = accumulation)
    "vol_trend",            # Linear slope of volume over the quarter (ramping up or down)
    "vol_autocorr",         # Autocorrelation at lag-1 (persistent = systematic execution)
    "high_vol_day_frac",    # Fraction of days with volume z > 1.5
    "low_vol_day_frac",     # Fraction of days with volume z < -0.5
    "vol_skew",             # Skewness of daily volume (right-skewed = occasional large blocks)
    "vol_kurtosis",         # Kurtosis (fat tails = large block trades)
    "close_to_close_corr",  # Correlation between volume and absolute price change
    "up_day_vol_ratio",     # Ratio of volume on up days vs down days
    "end_of_quarter_surge", # Volume in last 2 weeks vs first 10 weeks
    "intraweek_pattern",    # Std of weekday average volumes (Mon vs Fri skew)
    "consecutive_high_days", # Max consecutive days above 1.2x average volume
    "dollar_vol_trend",     # Trend in dollar volume (captures both price and vol)
]


def compute_volume_profile(daily_data: pd.DataFrame) -> dict[str, float]:
    """Compute volume profile features for a period of daily data.

    Args:
        daily_data: DataFrame with columns [Volume, Close] indexed by date.
                    Should cover roughly one quarter (~63 trading days).

    Returns:
        Dict of feature_name → value.
    """
    if daily_data.empty or len(daily_data) < 20:
        return {f: np.nan for f in VOLUME_PROFILE_FEATURES}

    vol = daily_data["Volume"].astype(float)
    close = daily_data["Close"].astype(float) if "Close" in daily_data.columns else pd.Series(dtype=float)

    # Z-score volume relative to its own mean/std in this window
    vol_mean = vol.mean()
    vol_std = vol.std()
    if vol_std == 0 or vol_mean == 0:
        vol_z = pd.Series(0.0, index=vol.index)
    else:
        vol_z = (vol - vol_mean) / vol_std

    features = {}

    # Basic distribution stats
    features["vol_mean_z"] = float(vol_z.mean())
    features["vol_std_z"] = float(vol_z.std()) if len(vol_z) > 1 else 0.0
    features["vol_skew"] = float(vol_z.skew()) if len(vol_z) > 2 else 0.0
    features["vol_kurtosis"] = float(vol_z.kurtosis()) if len(vol_z) > 3 else 0.0

    # Trend: linear regression slope of volume z-scores over time
    x = np.arange(len(vol_z))
    if len(x) > 1:
        slope = np.polyfit(x, vol_z.values, 1)[0]
        features["vol_trend"] = float(slope)
    else:
        features["vol_trend"] = 0.0

    # Autocorrelation at lag 1 — persistent elevated volume = systematic execution
    if len(vol_z) > 2:
        features["vol_autocorr"] = float(vol_z.autocorr(lag=1))
        if pd.isna(features["vol_autocorr"]):
            features["vol_autocorr"] = 0.0
    else:
        features["vol_autocorr"] = 0.0

    # Fraction of high/low volume days
    features["high_vol_day_frac"] = float((vol_z > 1.5).sum() / len(vol_z))
    features["low_vol_day_frac"] = float((vol_z < -0.5).sum() / len(vol_z))

    # Price-volume correlation
    if not close.empty and len(close) > 5:
        abs_returns = close.pct_change().abs().dropna()
        if len(abs_returns) == len(vol_z) - 1:
            corr = vol_z.iloc[1:].reset_index(drop=True).corr(abs_returns.reset_index(drop=True))
            features["close_to_close_corr"] = float(corr) if pd.notna(corr) else 0.0
        else:
            features["close_to_close_corr"] = 0.0

        # Up-day vs down-day volume ratio
        returns = close.pct_change().dropna()
        up_days = returns > 0
        if up_days.sum() > 0 and (~up_days).sum() > 0:
            up_vol = vol.iloc[1:][up_days.values].mean()
            down_vol = vol.iloc[1:][~up_days.values].mean()
            features["up_day_vol_ratio"] = float(up_vol / down_vol) if down_vol > 0 else 1.0
        else:
            features["up_day_vol_ratio"] = 1.0
    else:
        features["close_to_close_corr"] = 0.0
        features["up_day_vol_ratio"] = 1.0

    # End-of-quarter surge: last 2 weeks vs first 10 weeks
    n = len(vol)
    cutoff = max(1, n - 10)
    early_mean = vol.iloc[:cutoff].mean()
    late_mean = vol.iloc[cutoff:].mean()
    features["end_of_quarter_surge"] = float(late_mean / early_mean) if early_mean > 0 else 1.0

    # Intraweek pattern — institutions often concentrate execution on specific days
    if hasattr(daily_data.index, "dayofweek"):
        weekday_means = vol.groupby(daily_data.index.dayofweek).mean()
        features["intraweek_pattern"] = float(weekday_means.std() / weekday_means.mean()) if weekday_means.mean() > 0 else 0.0
    else:
        features["intraweek_pattern"] = 0.0

    # Max consecutive high-volume days (>1.2x average)
    above_avg = (vol > vol_mean * 1.2).astype(int)
    if above_avg.sum() > 0:
        streaks = above_avg.groupby((above_avg != above_avg.shift()).cumsum())
        max_streak = streaks.sum().max()
        features["consecutive_high_days"] = float(max_streak)
    else:
        features["consecutive_high_days"] = 0.0

    # Dollar volume trend
    if not close.empty and len(close) > 1:
        dollar_vol = vol * close
        dv_z = (dollar_vol - dollar_vol.mean()) / (dollar_vol.std() if dollar_vol.std() > 0 else 1)
        x_dv = np.arange(len(dv_z))
        features["dollar_vol_trend"] = float(np.polyfit(x_dv, dv_z.values, 1)[0])
    else:
        features["dollar_vol_trend"] = 0.0

    return features


# ---------------------------------------------------------------------------
# Historical profile construction
# ---------------------------------------------------------------------------

def build_historical_profiles(
    holdings: pd.DataFrame,
    price_fetcher,
) -> pd.DataFrame:
    """Build volume profiles for every (institution, ticker, quarter) in 13F history.

    Args:
        holdings: 13F holdings with delta_shares computed.
        price_fetcher: callable(ticker, start_date, end_date) -> DataFrame with [Volume, Close]

    Returns:
        DataFrame with columns: institution, ticker, quarter, action, + all profile features
    """
    if holdings.empty or "delta_shares" not in holdings.columns:
        return pd.DataFrame()

    df = holdings[holdings["delta_shares"].notna()].copy()

    # Classify action
    df["action"] = "hold"
    df.loc[df["delta_shares"] > 0, "action"] = "buy"
    df.loc[df["delta_shares"] < 0, "action"] = "sell"

    records = []

    for _, row in df.iterrows():
        quarter = row["quarter"]  # e.g., "2024-Q2"
        ticker = row["ticker"]

        # Parse quarter to date range
        try:
            year = int(quarter[:4])
            q = int(quarter[-1])
            start_month = (q - 1) * 3 + 1
            end_month = q * 3
            start_date = f"{year}-{start_month:02d}-01"
            if end_month == 12:
                end_date = f"{year}-12-31"
            else:
                end_date = f"{year}-{end_month + 1:02d}-01"
        except (ValueError, IndexError):
            continue

        # Fetch daily volume for this quarter
        try:
            daily = price_fetcher(ticker, start_date, end_date)
        except Exception:
            continue

        if daily.empty:
            continue

        profile = compute_volume_profile(daily)
        profile["institution"] = row["institution"]
        profile["ticker"] = ticker
        profile["quarter"] = quarter
        profile["action"] = row["action"]
        profile["delta_shares"] = row["delta_shares"]
        profile["delta_pct"] = row.get("delta_pct", 0)
        records.append(profile)

    return pd.DataFrame(records)


def build_profiles_from_cache(
    holdings: pd.DataFrame,
    data_dir: str,
    snowflake_cfg: dict,
) -> pd.DataFrame:
    """Build profiles using Snowflake daily data, with caching.

    Fetches full daily history per ticker from Snowflake once, then slices
    per quarter to compute volume profiles.
    """
    from src.data import snowflake_price_volume

    cached = cache.load(data_dir, NAMESPACE, "historical_profiles")
    if cached is not None:
        return cached

    # Fetch per-ticker history lazily, cached for the run
    _price_cache: dict[str, pd.DataFrame] = {}

    def _fetch(ticker: str, start: str, end: str) -> pd.DataFrame:
        if ticker not in _price_cache:
            try:
                _price_cache[ticker] = snowflake_price_volume.get_ticker_history(
                    ticker, "2014-01-01", pd.Timestamp.now().strftime("%Y-%m-%d"),
                    snowflake_cfg,
                )
            except Exception as e:
                logger.warning("Snowflake history fetch failed for %s: %s", ticker, e)
                _price_cache[ticker] = pd.DataFrame()

        hist = _price_cache.get(ticker, pd.DataFrame())
        if hist.empty:
            return pd.DataFrame()

        mask = (hist.index >= start) & (hist.index < end)
        return hist.loc[mask, ["Volume", "Close"]]

    profiles = build_historical_profiles(holdings, _fetch)

    if not profiles.empty:
        cache.save(profiles, data_dir, NAMESPACE, "historical_profiles")

    return profiles


# ---------------------------------------------------------------------------
# Fingerprint learning
# ---------------------------------------------------------------------------

def learn_fingerprints(
    profiles: pd.DataFrame,
    min_samples: int = 20,
) -> dict[str, ExecutionFingerprint]:
    """Learn execution fingerprints per institution.

    For each institution, train a classifier:
        P(action = buy) = f(volume_profile_features)

    This learns what volume looks like when that institution is accumulating
    vs distributing vs holding steady.
    """
    if profiles.empty:
        return {}

    fingerprints = {}

    for institution in profiles["institution"].unique():
        inst_data = profiles[profiles["institution"] == institution]

        # Binary target: buying or not
        y = (inst_data["action"] == "buy").astype(int)

        # Need both classes
        if y.sum() < 5 or (1 - y).sum() < 5:
            logger.info("Skipping %s: insufficient buy/sell diversity", institution)
            continue

        feature_cols = [f for f in VOLUME_PROFILE_FEATURES if f in inst_data.columns]
        X = inst_data[feature_cols].copy()

        # Clean: fill NaN with 0, convert to numeric
        for col in feature_cols:
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)

        if len(X) < min_samples:
            logger.info("Skipping %s: only %d samples", institution, len(X))
            continue

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = LogisticRegression(max_iter=1000, C=0.1, random_state=42)
        model.fit(X_scaled, y)

        # Score on training data (walk-forward would be better with more data)
        from sklearn.metrics import roc_auc_score
        try:
            proba = model.predict_proba(X_scaled)[:, 1]
            auc = roc_auc_score(y, proba)
        except ValueError:
            auc = 0.5

        importances = {
            feat: round(abs(coef), 4)
            for feat, coef in zip(feature_cols, model.coef_[0])
        }
        importances = dict(sorted(importances.items(), key=lambda x: -x[1]))

        fingerprints[institution] = ExecutionFingerprint(
            institution=institution,
            model=model,
            scaler=scaler,
            n_training_samples=len(X),
            auc=round(auc, 3),
            feature_importances=importances,
        )

        logger.info(
            "Learned fingerprint for %s: AUC=%.3f, n=%d, top features: %s",
            institution, auc, len(X),
            ", ".join(f"{k}={v}" for k, v in list(importances.items())[:3]),
        )

    return fingerprints


# ---------------------------------------------------------------------------
# Real-time matching
# ---------------------------------------------------------------------------

def match_current_volume(
    ticker: str,
    fingerprints: dict[str, ExecutionFingerprint],
    current_daily: pd.DataFrame,
) -> dict[str, float]:
    """Score current volume profile against each institution's fingerprint.

    Args:
        ticker: Stock ticker
        fingerprints: Learned fingerprints per institution
        current_daily: Recent daily data (last ~60 trading days) with [Volume, Close]

    Returns:
        Dict of {institution: P(accumulating)}
    """
    if current_daily.empty or not fingerprints:
        return {}

    profile = compute_volume_profile(current_daily)

    results = {}

    for inst_key, fp in fingerprints.items():
        if fp.model is None or fp.scaler is None:
            continue

        feature_cols = list(fp.scaler.feature_names_in_)
        X = pd.DataFrame([{f: profile.get(f, 0) for f in feature_cols}])

        for col in feature_cols:
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)

        X_scaled = fp.scaler.transform(X)
        prob = float(fp.model.predict_proba(X_scaled)[0, 1])
        results[inst_key] = round(prob, 3)

    return results


# ---------------------------------------------------------------------------
# Streak detection (kept from original)
# ---------------------------------------------------------------------------

def _compute_streaks(holdings: pd.DataFrame) -> pd.DataFrame:
    """Identify consecutive buy/sell streaks per institution x ticker."""
    if holdings.empty or "delta_shares" not in holdings.columns:
        return pd.DataFrame()

    df = holdings.sort_values(["institution", "ticker", "quarter"]).copy()
    df = df[df["delta_shares"].notna()]

    records = []

    for (inst, ticker), group in df.groupby(["institution", "ticker"]):
        group = group.sort_values("quarter")
        if len(group) < 2:
            continue

        deltas = group["delta_shares"].values
        quarters = group["quarter"].values

        consecutive_buys = 0
        consecutive_sells = 0

        for i in range(len(deltas) - 1, -1, -1):
            if deltas[i] > 0:
                if consecutive_sells > 0:
                    break
                consecutive_buys += 1
            elif deltas[i] < 0:
                if consecutive_buys > 0:
                    break
                consecutive_sells += 1
            else:
                break

        streak_len = max(consecutive_buys, consecutive_sells)
        if streak_len >= 2:
            streak_deltas = group["delta_pct"].iloc[-streak_len:]
            avg_delta = streak_deltas.mean() if not streak_deltas.isna().all() else 0
            total_change = streak_deltas.sum() if not streak_deltas.isna().all() else 0
        else:
            avg_delta = group["delta_pct"].iloc[-1] if pd.notna(group["delta_pct"].iloc[-1]) else 0
            total_change = avg_delta

        latest = group.iloc[-1]

        records.append({
            "institution": inst,
            "institution_name": latest.get("institution_name", inst),
            "style": latest.get("style", "unknown"),
            "ticker": ticker,
            "consecutive_buys": consecutive_buys,
            "consecutive_sells": consecutive_sells,
            "avg_quarterly_delta_pct": round(avg_delta, 2),
            "total_change_pct": round(total_change, 2),
            "latest_quarter": quarters[-1],
            "latest_shares": int(latest["shares"]),
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def detect_accumulation(
    holdings: pd.DataFrame,
    volume_signals: pd.DataFrame,
    fingerprints: dict[str, ExecutionFingerprint] | None = None,
    current_daily: dict[str, pd.DataFrame] | None = None,
    min_streak: int = 2,
) -> list[AccumulationSignal]:
    """Detect accumulation using streaks + learned fingerprints.

    If fingerprints are provided, uses learned execution signatures to
    score continuation probability. Otherwise falls back to volume z-score
    heuristics.
    """
    streaks = _compute_streaks(holdings)
    if streaks.empty:
        return []

    streaks = streaks[
        (streaks["consecutive_buys"] >= min_streak) |
        (streaks["consecutive_sells"] >= min_streak)
    ]

    if streaks.empty:
        return []

    results = []

    for _, row in streaks.iterrows():
        ticker = row["ticker"]
        inst_key = row["institution"]

        # Try fingerprint match first
        fp_prob = None
        fp_score = 0.0
        vol_features = {}

        if fingerprints and current_daily and ticker in current_daily:
            inst_fp = fingerprints.get(inst_key)
            if inst_fp and inst_fp.model is not None:
                match_results = match_current_volume(
                    ticker, {inst_key: inst_fp}, current_daily[ticker]
                )
                if inst_key in match_results:
                    fp_prob = match_results[inst_key]
                    fp_score = fp_prob

                    # Get the volume features for detail
                    daily = current_daily[ticker]
                    if not daily.empty:
                        vol_features = compute_volume_profile(daily)

        # Determine continuation probability
        if fp_prob is not None:
            # Use learned fingerprint — more reliable
            prob = fp_prob
            confirms = (
                (row["consecutive_buys"] >= 2 and prob > 0.6) or
                (row["consecutive_sells"] >= 2 and prob < 0.4)
            )
        else:
            # Fallback: volume z-score heuristic
            vol_row = volume_signals[volume_signals["ticker"] == ticker] if not volume_signals.empty else pd.DataFrame()

            if not vol_row.empty:
                vol_z = vol_row.iloc[0].get("volume_zscore", 0)
                if pd.isna(vol_z):
                    vol_z = 0

                is_buying = row["consecutive_buys"] >= 2
                if is_buying:
                    if vol_z > 0.5:
                        prob = min(0.5 + 0.1 * row["consecutive_buys"] + 0.05 * vol_z, 0.95)
                        confirms = True
                    elif vol_z > -0.5:
                        prob = 0.5 + 0.05 * row["consecutive_buys"]
                        confirms = False
                    else:
                        prob = max(0.3, 0.5 - 0.1 * abs(vol_z))
                        confirms = False
                else:
                    if vol_z > 1.0:
                        prob = min(0.5 + 0.1 * row["consecutive_sells"] + 0.05 * vol_z, 0.95)
                        confirms = True
                    else:
                        prob = 0.5 + 0.03 * row["consecutive_sells"]
                        confirms = vol_z > 0.5
            else:
                prob = 0.5
                confirms = False

        results.append(AccumulationSignal(
            ticker=ticker,
            institution=inst_key,
            institution_name=row["institution_name"],
            style=row["style"],
            consecutive_buys=int(row["consecutive_buys"]),
            consecutive_sells=int(row["consecutive_sells"]),
            avg_quarterly_delta_pct=row["avg_quarterly_delta_pct"],
            total_change_pct=row["total_change_pct"],
            latest_quarter=row["latest_quarter"],
            volume_confirms=bool(confirms),
            continuation_probability=round(prob, 3),
            fingerprint_match_score=round(fp_score, 3),
            volume_features_used=vol_features,
        ))

    results.sort(
        key=lambda s: (s.continuation_probability, max(s.consecutive_buys, s.consecutive_sells)),
        reverse=True,
    )
    return results


def summarize_by_stock(signals: list[AccumulationSignal]) -> pd.DataFrame:
    """Aggregate signals to per-stock level."""
    if not signals:
        return pd.DataFrame()

    by_ticker: dict[str, list[AccumulationSignal]] = {}
    for s in signals:
        by_ticker.setdefault(s.ticker, []).append(s)

    records = []
    for ticker, sigs in by_ticker.items():
        accumulators = [s for s in sigs if s.direction == "ACCUMULATING"]
        distributors = [s for s in sigs if s.direction == "DISTRIBUTING"]

        avg_prob = np.mean([s.continuation_probability for s in sigs])
        confirmed = sum(1 for s in sigs if s.volume_confirms)
        avg_fp = np.mean([s.fingerprint_match_score for s in sigs if s.fingerprint_match_score > 0]) if any(s.fingerprint_match_score > 0 for s in sigs) else 0

        net = len(accumulators) - len(distributors)

        records.append({
            "ticker": ticker,
            "n_accumulating": len(accumulators),
            "n_distributing": len(distributors),
            "net_direction": "ACCUMULATE" if net > 0 else ("DISTRIBUTE" if net < 0 else "MIXED"),
            "avg_continuation_prob": round(avg_prob, 3),
            "avg_fingerprint_score": round(avg_fp, 3),
            "volume_confirmed_count": confirmed,
            "top_accumulator": accumulators[0].institution_name if accumulators else None,
            "top_distributor": distributors[0].institution_name if distributors else None,
        })

    return pd.DataFrame(records).sort_values("avg_continuation_prob", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_fingerprints(fingerprints: dict[str, ExecutionFingerprint], model_dir: str) -> None:
    path = Path(model_dir)
    path.mkdir(parents=True, exist_ok=True)
    joblib.dump(fingerprints, path / "fingerprints.joblib")


def load_fingerprints(model_dir: str) -> dict[str, ExecutionFingerprint] | None:
    path = Path(model_dir) / "fingerprints.joblib"
    if path.exists():
        return joblib.load(path)
    return None
