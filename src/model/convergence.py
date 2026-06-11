"""Trade convergence analysis — reverse-engineer how each institution trades.

Transposes 13F trades against quarterly Axioma factor exposures, then checks
three independent reads on every trade:

1. Volume — did the quarter show an unusual execution footprint?
2. Factor alignment — does the trade match the factors that institution is
   known to trade on (learned per-institution factor profile)?
3. Historical similarity — in past quarters with a similar factor setup,
   did they trade the same direction?

When all three converge, you know how they're trading.

Also estimates:
- Market impact of each FACTOR: Fama-MacBeth style cross-sectional
  regressions of quarterly returns on prior-quarter exposures.
- Market impact of each PLAYER: participation rate (their trade vs the
  tape), same-quarter price move when they buy vs sell, and the volume
  footprint they leave.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data.snowflake_factors import AXIOMA_FACTORS

logger = logging.getLogger(__name__)

# Volume profile features used for the unusual-volume read
_VOLUME_CONTEXT_COLS = [
    "vol_mean_z", "vol_trend", "high_vol_day_frac", "consecutive_high_days",
]


# ---------------------------------------------------------------------------
# 1. Trade × factor panel (the "transpose")
# ---------------------------------------------------------------------------

def build_trade_factor_panel(
    holdings: pd.DataFrame,
    factor_history: pd.DataFrame,
) -> pd.DataFrame:
    """One row per (institution, ticker, quarter) trade with that quarter's
    factor exposures.

    Adds:
        direction: +1 (buy) / -1 (sell)
        {factor}_z: exposure z-scored cross-sectionally within the quarter,
                    so factor setups are comparable across quarters.
    """
    if holdings.empty or factor_history.empty:
        return pd.DataFrame()

    trades = holdings[
        holdings["delta_shares"].notna() & (holdings["delta_shares"] != 0)
    ].copy()
    if trades.empty:
        return pd.DataFrame()

    trades["direction"] = np.sign(trades["delta_shares"]).astype(int)

    # 13F holdings carry a "value" column (position market value) which
    # collides with the Axioma "value" factor — rename it first.
    if "value" in trades.columns:
        trades = trades.rename(columns={"value": "position_value"})

    factors_present = [f for f in AXIOMA_FACTORS if f in factor_history.columns]
    panel = trades.merge(
        factor_history[["ticker", "quarter"] + factors_present],
        on=["ticker", "quarter"],
        how="inner",
    )
    if panel.empty:
        logger.warning("No overlap between trades and factor history")
        return panel

    # Per-quarter cross-sectional z-scores
    for f in factors_present:
        grouped = panel.groupby("quarter")[f]
        mean = grouped.transform("mean")
        std = grouped.transform("std").replace(0, np.nan)
        panel[f"{f}_z"] = ((panel[f] - mean) / std).fillna(0.0)

    logger.info(
        "Trade-factor panel: %d trades, %d factors, %d quarters",
        len(panel), len(factors_present), panel["quarter"].nunique(),
    )
    return panel


# ---------------------------------------------------------------------------
# 2. Unusual volume context
# ---------------------------------------------------------------------------

def add_volume_context(
    panel: pd.DataFrame,
    volume_profiles: pd.DataFrame,
) -> pd.DataFrame:
    """Merge per ticker-quarter volume signatures; flag unusual volume.

    volume_profiles: the accumulation historical profiles (one row per
    institution × ticker × quarter, but the volume features are per
    ticker-quarter, so we dedupe).
    """
    panel = panel.copy()

    if volume_profiles is None or volume_profiles.empty:
        panel["unusual_volume"] = False
        panel["volume_confirms"] = False
        return panel

    cols = [c for c in _VOLUME_CONTEXT_COLS if c in volume_profiles.columns]
    vol = (
        volume_profiles[["ticker", "quarter"] + cols]
        .drop_duplicates(subset=["ticker", "quarter"])
    )
    panel = panel.merge(vol, on=["ticker", "quarter"], how="left")

    # Unusual volume: a meaningful share of high-volume days, or a sustained
    # ramp with consecutive elevated days — the execution footprint.
    high_frac = panel.get("high_vol_day_frac", pd.Series(np.nan, index=panel.index))
    vol_trend = panel.get("vol_trend", pd.Series(np.nan, index=panel.index))
    consec = panel.get("consecutive_high_days", pd.Series(np.nan, index=panel.index))

    panel["unusual_volume"] = (
        (high_frac.fillna(0) > 0.15)
        | ((vol_trend.fillna(0) > 0) & (consec.fillna(0) >= 5))
    )
    # The trade happened — does the tape show it?
    panel["volume_confirms"] = panel["unusual_volume"]

    return panel


# ---------------------------------------------------------------------------
# 3. Factor alignment — does the trade match their known factor playbook?
# ---------------------------------------------------------------------------

def compute_factor_alignment(
    panel: pd.DataFrame,
    profiles: dict,
) -> pd.DataFrame:
    """Score each trade's quarter factors through the institution's learned
    factor profile (factor_analysis.InstitutionFactorProfile).

    Adds:
        p_buy_from_factors: what their factor playbook predicted
        factor_aligned: trade direction matches the prediction
        factor_alignment_score: signed strength, -1..+1
    """
    panel = panel.copy()
    panel["p_buy_from_factors"] = np.nan
    panel["factor_aligned"] = False
    panel["factor_alignment_score"] = 0.0

    if not profiles:
        return panel

    for inst_key, group in panel.groupby("institution"):
        profile = profiles.get(inst_key)
        if profile is None or profile.model is None or profile.scaler is None:
            continue

        X = pd.DataFrame(index=group.index)
        for col in profile.factor_names:
            if col in group.columns:
                X[col] = pd.to_numeric(group[col], errors="coerce").fillna(0)
            else:
                X[col] = 0.0

        X_scaled = profile.scaler.transform(X)
        p_buy = profile.model.predict_proba(X_scaled)[:, 1]

        panel.loc[group.index, "p_buy_from_factors"] = p_buy
        predicted_dir = np.where(p_buy >= 0.5, 1, -1)
        aligned = predicted_dir == group["direction"].values
        panel.loc[group.index, "factor_aligned"] = aligned
        # Signed: positive when the trade matches their playbook, scaled by
        # how decisive the playbook signal was.
        panel.loc[group.index, "factor_alignment_score"] = (
            np.abs(p_buy - 0.5) * 2 * np.where(aligned, 1, -1)
        )

    return panel


# ---------------------------------------------------------------------------
# 4. Cross-quarter similarity — did they do the same thing in similar setups?
# ---------------------------------------------------------------------------

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def find_similar_quarters(
    panel: pd.DataFrame,
    top_k: int = 5,
    min_same_ticker: int = 3,
) -> pd.DataFrame:
    """For each trade, find the institution's most similar historical factor
    setups and check whether they traded the same direction.

    Compares z-scored factor vectors with cosine similarity. Prefers the same
    ticker's history when it has at least `min_same_ticker` other trades,
    otherwise uses the institution's full history.

    Adds:
        historical_consistency: similarity-weighted fraction of top-k similar
                                quarters where the direction matched (0-1)
        n_similar_quarters: how many comparables were used
        top_similar_quarter: the single most similar (quarter, ticker)
    """
    panel = panel.copy()
    z_cols = [f"{f}_z" for f in AXIOMA_FACTORS if f"{f}_z" in panel.columns]

    panel["historical_consistency"] = np.nan
    panel["n_similar_quarters"] = 0
    panel["top_similar_quarter"] = None

    if not z_cols:
        return panel

    vectors = panel[z_cols].to_numpy(dtype=float)

    for inst_key, inst_group in panel.groupby("institution"):
        inst_idx = inst_group.index.to_numpy()

        for idx in inst_idx:
            row = panel.loc[idx]

            # Candidate history: same ticker preferred, else institution-wide.
            # Exclude the trade's own quarter to avoid self-matching.
            same_ticker = inst_group[
                (inst_group["ticker"] == row["ticker"])
                & (inst_group["quarter"] != row["quarter"])
            ]
            if len(same_ticker) >= min_same_ticker:
                candidates = same_ticker
            else:
                candidates = inst_group[inst_group["quarter"] != row["quarter"]]

            if candidates.empty:
                continue

            target = vectors[panel.index.get_loc(idx)]
            sims = []
            for c_idx in candidates.index:
                sim = _cosine(target, vectors[panel.index.get_loc(c_idx)])
                sims.append((c_idx, sim))

            # Top-k by similarity, only positively-similar setups
            sims = [s for s in sims if s[1] > 0]
            sims.sort(key=lambda x: -x[1])
            sims = sims[:top_k]
            if not sims:
                continue

            total_sim = sum(s for _, s in sims)
            matched = sum(
                s for c_idx, s in sims
                if panel.loc[c_idx, "direction"] == row["direction"]
            )
            consistency = matched / total_sim if total_sim > 0 else 0.0

            best_idx = sims[0][0]
            panel.loc[idx, "historical_consistency"] = round(consistency, 3)
            panel.loc[idx, "n_similar_quarters"] = len(sims)
            panel.loc[idx, "top_similar_quarter"] = (
                f"{panel.loc[best_idx, 'quarter']}:{panel.loc[best_idx, 'ticker']}"
            )

    return panel


# ---------------------------------------------------------------------------
# 5. Convergence — when all three reads agree, you know how they're trading
# ---------------------------------------------------------------------------

def compute_convergence(panel: pd.DataFrame) -> pd.DataFrame:
    """Combine factor alignment + volume confirmation + historical
    consistency into a convergence score and verdict per trade.
    """
    panel = panel.copy()

    scores = []
    verdicts = []

    for _, row in panel.iterrows():
        components = []

        align = row.get("factor_alignment_score")
        if pd.notna(align):
            components.append((align + 1) / 2)  # -1..1 -> 0..1

        if "volume_confirms" in panel.columns:
            components.append(1.0 if row["volume_confirms"] else 0.0)

        consistency = row.get("historical_consistency")
        if pd.notna(consistency):
            components.append(float(consistency))

        score = float(np.mean(components)) if components else np.nan
        scores.append(round(score, 3) if pd.notna(score) else np.nan)

        if pd.isna(score):
            verdicts.append("INSUFFICIENT DATA")
        elif score >= 0.7:
            word = "BUYING" if row["direction"] > 0 else "SELLING"
            verdicts.append(f"CONVERGED — {word}")
        elif score >= 0.5:
            verdicts.append("PARTIAL")
        else:
            verdicts.append("DIVERGENT")

    panel["convergence_score"] = scores
    panel["verdict"] = verdicts
    return panel


def summarize_readability(panel: pd.DataFrame) -> pd.DataFrame:
    """Per institution: how readable/predictable is their trading?

    Avg convergence across their trades — high means their trades reliably
    show up in factors + volume + history, i.e. you can see them coming.
    """
    if panel.empty or "convergence_score" not in panel.columns:
        return pd.DataFrame()

    out = (
        panel.dropna(subset=["convergence_score"])
        .groupby(["institution", "institution_name", "style"])
        .agg(
            avg_convergence=("convergence_score", "mean"),
            n_trades=("convergence_score", "size"),
            pct_factor_aligned=("factor_aligned", "mean"),
            pct_volume_confirmed=("volume_confirms", "mean"),
        )
        .reset_index()
        .sort_values("avg_convergence", ascending=False)
    )
    out["avg_convergence"] = out["avg_convergence"].round(3)
    out["pct_factor_aligned"] = out["pct_factor_aligned"].round(3)
    out["pct_volume_confirmed"] = out["pct_volume_confirmed"].round(3)
    return out


# ---------------------------------------------------------------------------
# Quarterly market data from daily histories
# ---------------------------------------------------------------------------

def quarterly_market_data(
    daily_histories: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Quarterly returns, dollar volume, and average price per ticker.

    daily_histories: {ticker: DataFrame indexed by date with [Volume, Close]}

    Returns: ticker, quarter, ret (quarter-over-quarter close return),
             dollar_volume (sum over quarter), avg_close.
    """
    records = []
    for ticker, hist in daily_histories.items():
        if hist is None or hist.empty or "Close" not in hist.columns:
            continue
        df = hist.sort_index()
        quarters = (
            df.index.to_series().dt.year.astype(str)
            + "-Q"
            + df.index.to_series().dt.quarter.astype(str)
        )
        grouped = df.groupby(quarters.values)
        q_close = grouped["Close"].last()
        q_dollar = (df["Close"] * df["Volume"]).groupby(quarters.values).sum()
        q_avg = grouped["Close"].mean()

        rets = q_close.pct_change()
        for quarter in q_close.index:
            records.append({
                "ticker": ticker,
                "quarter": quarter,
                "ret": rets.loc[quarter],
                "dollar_volume": float(q_dollar.loc[quarter]),
                "avg_close": float(q_avg.loc[quarter]),
            })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 6. Market impact of each factor (Fama-MacBeth)
# ---------------------------------------------------------------------------

@dataclass
class FactorImpact:
    factor: str
    avg_impact_bps: float      # mean quarterly return per +1σ exposure
    t_stat: float
    hit_rate: float            # fraction of quarters with same-sign impact
    last_quarter_impact_bps: float
    n_quarters: int


def factor_market_impact(
    factor_history: pd.DataFrame,
    market_data: pd.DataFrame,
) -> list[FactorImpact]:
    """Cross-sectional quarterly regressions of returns on prior-quarter
    exposures (Fama-MacBeth). Shows how much each factor moved these stocks.

    factor_history: ticker, quarter, + AXIOMA_FACTORS columns
    market_data: output of quarterly_market_data (ticker, quarter, ret)
    """
    if factor_history.empty or market_data.empty:
        return []

    factors_present = [f for f in AXIOMA_FACTORS if f in factor_history.columns]
    if not factors_present:
        return []

    # Exposure at end of quarter q predicts return in the NEXT quarter —
    # shift quarters forward to align exposures with forward returns.
    quarters = sorted(set(factor_history["quarter"]) | set(market_data["quarter"]))
    next_quarter = {q: quarters[i + 1] for i, q in enumerate(quarters[:-1])}

    exposures = factor_history.copy()
    exposures["ret_quarter"] = exposures["quarter"].map(next_quarter)
    merged = exposures.merge(
        market_data[["ticker", "quarter", "ret"]].rename(
            columns={"quarter": "ret_quarter"}
        ),
        on=["ticker", "ret_quarter"],
        how="inner",
    ).dropna(subset=["ret"])

    if merged.empty:
        return []

    # Per-quarter cross-sectional OLS: ret ~ intercept + z-scored exposures
    coef_rows = []
    for quarter, group in merged.groupby("ret_quarter"):
        if len(group) < len(factors_present) + 2:
            continue

        X = group[factors_present].to_numpy(dtype=float)
        # Cross-sectional z-score
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std[std == 0] = 1.0
        Xz = (X - mean) / std

        y = group["ret"].to_numpy(dtype=float)
        A = np.column_stack([np.ones(len(y)), Xz])
        coefs, *_ = np.linalg.lstsq(A, y, rcond=None)
        coef_rows.append({"quarter": quarter, **dict(zip(factors_present, coefs[1:]))})

    if not coef_rows:
        return []

    coef_df = pd.DataFrame(coef_rows).sort_values("quarter")

    results = []
    for f in factors_present:
        series = coef_df[f].dropna()
        if series.empty:
            continue
        mean = series.mean()
        se = series.std() / np.sqrt(len(series)) if len(series) > 1 else np.nan
        t_stat = mean / se if se and se > 0 else 0.0
        hit_rate = float((np.sign(series) == np.sign(mean)).mean()) if mean != 0 else 0.5

        results.append(FactorImpact(
            factor=f,
            avg_impact_bps=round(mean * 10000, 1),
            t_stat=round(float(t_stat), 2),
            hit_rate=round(hit_rate, 3),
            last_quarter_impact_bps=round(float(series.iloc[-1]) * 10000, 1),
            n_quarters=len(series),
        ))

    results.sort(key=lambda r: -abs(r.avg_impact_bps))
    return results


# ---------------------------------------------------------------------------
# 7. Market impact of each player
# ---------------------------------------------------------------------------

@dataclass
class PlayerImpact:
    institution: str
    institution_name: str
    style: str
    avg_participation_pct: float      # their trade $ as % of the quarter's tape
    impact_bps_when_buying: float     # avg same-quarter return when they bought
    impact_bps_when_selling: float    # avg same-quarter return when they sold
    volume_footprint: float           # avg high-vol-day fraction in traded quarters
    n_trades: int


def player_market_impact(
    panel: pd.DataFrame,
    market_data: pd.DataFrame,
) -> list[PlayerImpact]:
    """Per-institution market impact from the trade panel + market data.

    Participation: |delta_shares| × avg quarter price / quarter dollar volume.
    Impact: same-quarter return split by trade direction — who moves price
    when they trade vs who trades invisibly.
    """
    if panel.empty or market_data.empty:
        return []

    df = panel.merge(
        market_data[["ticker", "quarter", "ret", "dollar_volume", "avg_close"]],
        on=["ticker", "quarter"],
        how="left",
    )

    df["trade_dollars"] = df["delta_shares"].abs() * df["avg_close"]
    df["participation"] = df["trade_dollars"] / df["dollar_volume"].replace(0, np.nan)

    results = []
    for (inst, name, style), group in df.groupby(
        ["institution", "institution_name", "style"]
    ):
        buys = group[group["direction"] > 0]
        sells = group[group["direction"] < 0]

        buy_ret = buys["ret"].mean() if not buys.empty else np.nan
        sell_ret = sells["ret"].mean() if not sells.empty else np.nan
        participation = group["participation"].mean()
        footprint = (
            group["high_vol_day_frac"].mean()
            if "high_vol_day_frac" in group.columns
            else np.nan
        )

        results.append(PlayerImpact(
            institution=inst,
            institution_name=name,
            style=style,
            avg_participation_pct=(
                round(float(participation) * 100, 3) if pd.notna(participation) else 0.0
            ),
            impact_bps_when_buying=(
                round(float(buy_ret) * 10000, 1) if pd.notna(buy_ret) else 0.0
            ),
            impact_bps_when_selling=(
                round(float(sell_ret) * 10000, 1) if pd.notna(sell_ret) else 0.0
            ),
            volume_footprint=(
                round(float(footprint), 3) if pd.notna(footprint) else 0.0
            ),
            n_trades=len(group),
        ))

    results.sort(key=lambda r: -r.avg_participation_pct)
    return results


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_convergence_analysis(
    holdings: pd.DataFrame,
    factor_history: pd.DataFrame,
    volume_profiles: pd.DataFrame,
    daily_histories: dict[str, pd.DataFrame],
    profiles: dict | None = None,
    top_k: int = 5,
) -> dict:
    """End-to-end: panel -> volume -> alignment -> similarity -> convergence
    -> factor impact -> player impact.

    Returns dict with keys: panel, readability, factor_impacts, player_impacts.
    """
    panel = build_trade_factor_panel(holdings, factor_history)
    if panel.empty:
        return {"panel": panel, "readability": pd.DataFrame(),
                "factor_impacts": [], "player_impacts": []}

    panel = add_volume_context(panel, volume_profiles)

    if profiles is None:
        from src.model import factor_analysis
        factors_present = [f for f in AXIOMA_FACTORS if f in factor_history.columns]
        profiles = factor_analysis.build_institution_profiles(
            holdings, factor_history[["ticker", "quarter"] + factors_present],
        )

    panel = compute_factor_alignment(panel, profiles)
    panel = find_similar_quarters(panel, top_k=top_k)
    panel = compute_convergence(panel)

    market_data = quarterly_market_data(daily_histories)
    factor_impacts = factor_market_impact(factor_history, market_data)
    player_impacts = player_market_impact(panel, market_data)
    readability = summarize_readability(panel)

    return {
        "panel": panel,
        "readability": readability,
        "factor_impacts": factor_impacts,
        "player_impacts": player_impacts,
    }
