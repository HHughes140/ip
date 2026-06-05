"""Pull real 13F + volume data and find confirmable accumulation/distribution patterns."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

from src.data import edgar_13f
from src.universe import INSTITUTION_REGISTRY, INSURANCE_UNIVERSE
from src.model.accumulation import _compute_streaks, compute_volume_profile


def main():
    data_dir = "data"

    # --- Step 1: Pull real 13F holdings from EDGAR ---
    print("=" * 70)
    print("STEP 1: Pulling 13F holdings from SEC EDGAR")
    print("=" * 70)

    all_holdings = []
    for inst_key, inst in INSTITUTION_REGISTRY.items():
        print(f"\n  Pulling {inst.name} (CIK: {inst.cik})...")
        try:
            df = edgar_13f.extract_holdings(inst_key, data_dir, start_year=2023, force=True)
            if not df.empty:
                print(f"    Found {len(df)} insurance stock holdings across {df['ticker'].nunique()} tickers")
                print(f"    Quarters: {sorted(df['quarter'].unique())}")
                all_holdings.append(df)
            else:
                print(f"    No insurance holdings found")
        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()

    if not all_holdings:
        print("\nNo holdings data retrieved. Check network/EDGAR access.")
        return

    combined = pd.concat(all_holdings, ignore_index=True)
    combined = edgar_13f.compute_deltas(combined)

    print(f"\n{'=' * 70}")
    print(f"TOTAL: {len(combined)} holding records across {combined['institution'].nunique()} institutions")
    print(f"Tickers covered: {sorted(combined['ticker'].unique())}")
    print(f"Quarters: {sorted(combined['quarter'].unique())}")

    # --- Step 2: Find accumulation/distribution streaks ---
    print(f"\n{'=' * 70}")
    print("STEP 2: Finding accumulation/distribution streaks")
    print("=" * 70)

    streaks = _compute_streaks(combined)
    if streaks.empty:
        print("No streaks found.")
    else:
        accum = streaks[streaks["consecutive_buys"] >= 2].sort_values("consecutive_buys", ascending=False)
        distrib = streaks[streaks["consecutive_sells"] >= 2].sort_values("consecutive_sells", ascending=False)

        print(f"\nACCUMULATION STREAKS (2+ consecutive quarters of buying):")
        print("-" * 70)
        if accum.empty:
            print("  None found")
        else:
            for _, row in accum.head(25).iterrows():
                total_chg = row.get('total_delta_pct', 0)
                chg_str = f"{total_chg:+.1f}%" if pd.notna(total_chg) else "N/A"
                print(f"  {row['ticker']:6s} <- {row.get('institution_name', row['institution']):25s} "
                      f"{row['consecutive_buys']}Q streak  chg={chg_str}  "
                      f"last_q={row.get('last_quarter', '?')}")

        print(f"\nDISTRIBUTION STREAKS (2+ consecutive quarters of selling):")
        print("-" * 70)
        if distrib.empty:
            print("  None found")
        else:
            for _, row in distrib.head(25).iterrows():
                total_chg = row.get('total_delta_pct', 0)
                chg_str = f"{total_chg:+.1f}%" if pd.notna(total_chg) else "N/A"
                print(f"  {row['ticker']:6s} <- {row.get('institution_name', row['institution']):25s} "
                      f"{row['consecutive_sells']}Q streak  chg={chg_str}  "
                      f"last_q={row.get('last_quarter', '?')}")

    # --- Step 3: Pull current volume ---
    print(f"\n{'=' * 70}")
    print("STEP 3: Cross-referencing with current volume data")
    print("=" * 70)

    import yfinance as yf

    tickers_in_data = sorted(combined["ticker"].unique())
    print(f"\nChecking volume for {len(tickers_in_data)} tickers...")

    for ticker in tickers_in_data:
        try:
            t = yf.Ticker(ticker)
            hist = t.history(period="3mo", auto_adjust=True)
            if hist.empty:
                continue
            hist.index = hist.index.tz_localize(None)
            daily = hist[["Volume", "Close"]]
            profile = compute_volume_profile(daily)

            vol_20d = daily["Volume"].rolling(20).mean().iloc[-1]
            vol_20d_std = daily["Volume"].rolling(20).std().iloc[-1]
            current_vol = daily["Volume"].iloc[-1]
            vol_z = (current_vol - vol_20d) / vol_20d_std if vol_20d_std > 0 else 0

            flag = " ** UNUSUAL **" if abs(vol_z) > 1.5 else ""
            trend_flag = " (trending UP)" if profile.get("vol_trend", 0) > 0.01 else (
                " (trending DOWN)" if profile.get("vol_trend", 0) < -0.01 else ""
            )
            print(f"  {ticker:6s}: Vol z={vol_z:+.2f}  trend={profile.get('vol_trend', 0):+.4f}"
                  f"  autocorr={profile.get('vol_autocorr', 0):.2f}{flag}{trend_flag}")
        except Exception as e:
            print(f"  {ticker:6s}: Error - {e}")

    # --- Step 4: Raw holdings for latest quarter ---
    print(f"\n{'=' * 70}")
    print("STEP 4: RAW 13F DATA — RECENT QUARTERS")
    print("=" * 70)

    latest_q = combined["quarter"].max()
    prev_q = sorted(combined["quarter"].unique())[-2] if len(combined["quarter"].unique()) > 1 else None

    # Show latest quarter positions
    latest = combined[combined["quarter"] == latest_q].copy()
    latest = latest.sort_values(["ticker", "institution"])

    print(f"\nLatest quarter in data: {latest_q}")
    print(f"{'Ticker':6s}  {'Institution':25s}  {'Shares':>12s}  {'Delta':>12s}  {'Chg%':>8s}  {'Value $M':>10s}")
    print("-" * 80)

    for _, row in latest.iterrows():
        delta = row.get("delta_shares", np.nan)
        pct = row.get("delta_pct", np.nan)
        val = row.get("value", 0)
        delta_str = f"{delta:>+12,.0f}" if pd.notna(delta) else "         N/A"
        pct_str = f"{pct:>+7.1f}%" if pd.notna(pct) else "     N/A"
        val_str = f"{val/1e6:>9.1f}M"
        inst_name = row.get("institution_name", row["institution"])
        print(f"  {row['ticker']:6s}  {inst_name:25s}  {row['shares']:>12,d}  {delta_str}  {pct_str}  {val_str}")

    # --- Step 5: Biggest movers ---
    print(f"\n{'=' * 70}")
    print("STEP 5: BIGGEST POSITION CHANGES (Latest Quarter)")
    print("=" * 70)

    movers = latest.dropna(subset=["delta_shares"]).copy()
    if not movers.empty:
        movers["abs_delta"] = movers["delta_shares"].abs()
        movers = movers.sort_values("abs_delta", ascending=False)

        print(f"\n{'Ticker':6s}  {'Institution':25s}  {'Delta Shares':>14s}  {'Chg%':>8s}  {'Direction':12s}")
        print("-" * 75)
        for _, row in movers.head(20).iterrows():
            direction = "BUYING" if row["delta_shares"] > 0 else "SELLING"
            color = direction
            inst_name = row.get("institution_name", row["institution"])
            print(f"  {row['ticker']:6s}  {inst_name:25s}  {row['delta_shares']:>+14,.0f}  "
                  f"{row.get('delta_pct', 0):>+7.1f}%  {direction}")


if __name__ == "__main__":
    main()
