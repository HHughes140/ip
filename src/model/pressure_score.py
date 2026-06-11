"""Composite Institutional Pressure Score (IPS).

Combines all signal components into a single score per stock:
    IPS = w1·residual_z + w2·crowding + w3·volume_anomaly
        + w4·ownership_concentration_Δ + w5·etf_flow_impact
        + w6·options_signal + w7·earnings_revision + w8·volume_prediction

Score range: [-100, +100]
    +100 = strong accumulation signal
    -100 = strong distribution signal
       0 = neutral / no signal

Confidence level reflects signal agreement across components.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.model.volume_model import VolumePrediction

logger = logging.getLogger(__name__)


@dataclass
class PressureResult:
    ticker: str
    name: str
    score: float              # [-100, +100]
    direction: str            # ACCUMULATE, DISTRIBUTE, NEUTRAL
    confidence: float         # 0-1
    components: dict[str, float]

    # Detail
    residual_z: float
    volume_spike_prob: float
    top_institutions: list[str]  # Institutions driving the residual

    @property
    def strength(self) -> str:
        """Human-readable signal strength."""
        a = abs(self.score)
        if a >= 70:
            return "STRONG"
        elif a >= 40:
            return "MODERATE"
        elif a >= 20:
            return "WEAK"
        return "NEGLIGIBLE"


# Default component weights — can be calibrated on forward 13F changes
DEFAULT_WEIGHTS = {
    "residual": 0.25,
    "crowding": 0.15,
    "volume_anomaly": 0.125,
    "options_signal": 0.125,
    "ownership_concentration": 0.10,
    "etf_flow": 0.075,
    "earnings_revision": 0.075,
    "volume_prediction": 0.10,
}


def _sigmoid_scale(x: float, scale: float = 2.0) -> float:
    """Map any real number to [-1, 1] via scaled tanh."""
    return float(np.tanh(x / scale))


def compute_pressure_scores(
    residuals: pd.DataFrame,
    volume_signals: pd.DataFrame,
    options_signals: pd.DataFrame,
    etf_signals: pd.DataFrame,
    factors: pd.DataFrame,
    ownership_changes: pd.DataFrame,
    volume_predictions: list[VolumePrediction],
    holdings: pd.DataFrame | None = None,
    crowding_signals: pd.DataFrame | None = None,
    weights: dict[str, float] | None = None,
) -> list[PressureResult]:
    """Compute the Institutional Pressure Score for each stock.

    All inputs are DataFrames indexed/keyed by ticker.
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    # Index volume predictions by ticker
    vol_pred_map = {vp.ticker: vp for vp in volume_predictions}

    # Get stock names from universe
    from src.universe import INSURANCE_UNIVERSE
    name_map = {t: s.name for t, s in INSURANCE_UNIVERSE.items()}

    results = []

    for _, row in residuals.iterrows():
        ticker = row["ticker"]
        components = {}

        # 1. Residual demand (strongest signal)
        res_z = row.get("residual_z", 0)
        components["residual"] = _sigmoid_scale(res_z, scale=2.0)

        # 1b. Crowding — leading indicator of institutional entry.
        # Rising crowding (delta) = institutions entering now = accumulation
        # pressure. The level (crowding_z) dampens the signal when a name is
        # already saturated — extremely crowded names have less room to add.
        crowd_row = (
            crowding_signals[crowding_signals["ticker"] == ticker]
            if crowding_signals is not None and not crowding_signals.empty
            else pd.DataFrame()
        )
        if not crowd_row.empty:
            delta_1m = crowd_row.iloc[0].get("crowding_delta_1m")
            crowd_z = crowd_row.iloc[0].get("crowding_z", 0)
            if pd.notna(delta_1m):
                crowd_z = crowd_z if pd.notna(crowd_z) else 0
                damping = 1 - 0.5 * np.clip(crowd_z, 0, 2) / 2
                components["crowding"] = _sigmoid_scale(delta_1m * damping, scale=1.0)
            else:
                components["crowding"] = 0.0
        else:
            components["crowding"] = 0.0

        # 2. Volume anomaly
        vol_row = volume_signals[volume_signals["ticker"] == ticker]
        if not vol_row.empty:
            cum_anom = vol_row.iloc[0].get("cum_anomaly_5d", 0)
            if pd.notna(cum_anom):
                # Sign should match residual direction
                vol_direction = np.sign(res_z) if res_z != 0 else 1
                components["volume_anomaly"] = _sigmoid_scale(
                    abs(cum_anom) * vol_direction, scale=5.0
                )
            else:
                components["volume_anomaly"] = 0.0
        else:
            components["volume_anomaly"] = 0.0

        # 3. Ownership concentration change
        own_row = ownership_changes[ownership_changes["ticker"] == ticker] if not ownership_changes.empty else pd.DataFrame()
        if not own_row.empty:
            hhi_delta = own_row.iloc[-1].get("hhi_delta", 0)
            if pd.notna(hhi_delta):
                components["ownership_concentration"] = _sigmoid_scale(hhi_delta, scale=0.01)
            else:
                components["ownership_concentration"] = 0.0
        else:
            components["ownership_concentration"] = 0.0

        # 4. ETF flow impact (sector-wide signal)
        if not etf_signals.empty:
            avg_flow_z = etf_signals["flow_zscore"].mean() if "flow_zscore" in etf_signals.columns else 0
            if pd.notna(avg_flow_z):
                components["etf_flow"] = _sigmoid_scale(avg_flow_z, scale=2.0)
            else:
                components["etf_flow"] = 0.0
        else:
            components["etf_flow"] = 0.0

        # 5. Options signal (P/C ratio deviation)
        opt_row = options_signals[options_signals["ticker"] == ticker] if not options_signals.empty else pd.DataFrame()
        if not opt_row.empty:
            pcr = opt_row.iloc[0].get("pc_volume_ratio")
            if pd.notna(pcr):
                # P/C > 1 → bearish sentiment (negative pressure)
                # P/C < 1 → bullish sentiment (positive pressure)
                components["options_signal"] = _sigmoid_scale(-(pcr - 1.0), scale=0.5)
            else:
                components["options_signal"] = 0.0
        else:
            components["options_signal"] = 0.0

        # 6. Earnings revision
        fac_row = factors[factors["ticker"] == ticker] if not factors.empty else pd.DataFrame()
        if not fac_row.empty:
            rev = fac_row.iloc[0].get("earnings_revision")
            if pd.notna(rev):
                components["earnings_revision"] = _sigmoid_scale(rev, scale=10.0)
            else:
                components["earnings_revision"] = 0.0
        else:
            components["earnings_revision"] = 0.0

        # 7. Volume prediction
        vp = vol_pred_map.get(ticker)
        if vp:
            # Convert probability to [-1, 1] based on predicted direction
            dir_sign = 1.0 if vp.direction == "UP" else -1.0
            components["volume_prediction"] = (vp.spike_probability - 0.5) * 2 * dir_sign
        else:
            components["volume_prediction"] = 0.0

        # Weighted combination
        raw_score = sum(
            components.get(k, 0) * v for k, v in weights.items()
        )

        # Scale to [-100, +100]
        score = round(raw_score * 100, 1)
        score = max(-100, min(100, score))

        # Direction
        if score > 10:
            direction = "ACCUMULATE"
        elif score < -10:
            direction = "DISTRIBUTE"
        else:
            direction = "NEUTRAL"

        # Confidence: how many components agree on direction
        signs = [np.sign(v) for v in components.values() if v != 0]
        if signs:
            dominant_sign = np.sign(sum(signs))
            agreement = sum(1 for s in signs if s == dominant_sign) / len(signs)
        else:
            agreement = 0.0

        # Identify top institutions driving the residual
        top_insts = []
        if holdings is not None and not holdings.empty:
            ticker_holdings = holdings[
                (holdings["ticker"] == ticker) &
                (holdings["delta_shares"].notna())
            ]
            if not ticker_holdings.empty:
                latest_q = ticker_holdings["quarter"].max()
                latest = ticker_holdings[ticker_holdings["quarter"] == latest_q]
                latest_sorted = latest.sort_values("delta_shares", ascending=False)
                top_insts = latest_sorted["institution_name"].head(3).tolist()

        results.append(PressureResult(
            ticker=ticker,
            name=name_map.get(ticker, ticker),
            score=score,
            direction=direction,
            confidence=round(agreement, 2),
            components={k: round(v, 3) for k, v in components.items()},
            residual_z=round(res_z, 3) if pd.notna(res_z) else 0,
            volume_spike_prob=vp.spike_probability if vp else 0,
            top_institutions=top_insts,
        ))

    # Sort by absolute score descending
    results.sort(key=lambda r: abs(r.score), reverse=True)
    return results
