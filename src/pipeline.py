"""CLI entry point — orchestrate ingest, model, score, and report.

Usage:
    pressure run              # Full pipeline
    pressure ingest           # Pull 13F, Axioma factors, volume, crowding, options, ETF, rates
    pressure model            # Fit demand model
    pressure score            # Compute pressure scores
    pressure report           # Generate report
    pressure detail TICKER    # Deep dive on a single stock
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click
import pandas as pd
import yaml

from src.universe import INSTITUTION_REGISTRY, ALL_TICKERS, INSURANCE_UNIVERSE
from src.data import edgar_13f, options, etf_flows, fred_rates, cache
from src.data import snowflake_price_volume, snowflake_factors, snowflake_crowding
from src.data import factor_loader
from src.model import demand_model, residual, volume_model, pressure_score, accumulation
from src.model import factor_analysis, convergence
from src import report

logger = logging.getLogger(__name__)


def _load_config(config_path: str = "config.yaml") -> dict:
    local = Path(config_path).with_name("config.local.yaml")
    if local.exists():
        config_path = str(local)
    path = Path(config_path)
    if not path.exists():
        click.echo(f"Config not found: {config_path}", err=True)
        sys.exit(1)
    with open(path) as f:
        return yaml.safe_load(f)


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


@click.group()
@click.option("--config", default="config.yaml", help="Path to config file")
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging")
@click.pass_context
def cli(ctx, config, verbose):
    """Institutional Pressure Score — Insurance Equities."""
    _setup_logging(verbose)
    ctx.ensure_object(dict)
    ctx.obj["config"] = _load_config(config)
    ctx.obj["data_dir"] = ctx.obj["config"].get("data_dir", "data")
    ctx.obj["model_dir"] = ctx.obj["config"].get("model_dir", "data/models")
    ctx.obj["snowflake"] = ctx.obj["config"].get("snowflake", {})


@cli.command()
@click.option("--force", is_flag=True, help="Ignore cache, re-download everything")
@click.pass_context
def ingest(ctx, force):
    """Pull data from all sources (13F, factors, volume, crowding, options, ETF, rates)."""
    config = ctx.obj["config"]
    data_dir = ctx.obj["data_dir"]
    snowflake_cfg = ctx.obj["snowflake"]
    start_year = config.get("history", {}).get("start_year", 2015)

    # 13F holdings
    click.echo("Ingesting 13F holdings...")
    holdings = edgar_13f.refresh_all(data_dir, force=force)
    click.echo(f"  13F: {len(holdings)} holdings records")

    # Axioma factors (Snowflake)
    click.echo("Pulling Axioma factors from Snowflake...")
    fac = snowflake_factors.refresh_all(
        data_dir, snowflake_cfg, start_year=start_year, force=force
    )
    click.echo(f"  Factors: {len(fac)} stocks")

    # Volume signals (Snowflake)
    click.echo("Computing volume signals from Snowflake...")
    vol = snowflake_price_volume.refresh_all(data_dir, snowflake_cfg, force=force)
    click.echo(f"  Volume: {len(vol)} stocks")

    # Crowding scores (Snowflake — MS/JPM/UBS/Citi/NR)
    click.echo("Pulling crowding scores from Snowflake...")
    crowd = snowflake_crowding.refresh_all(
        data_dir, snowflake_cfg, start_year=start_year, force=force
    )
    click.echo(f"  Crowding: {len(crowd)} stocks")

    # Options activity
    click.echo("Pulling options data...")
    opts = options.refresh_all(data_dir, force=force)
    click.echo(f"  Options: {len(opts)} stocks")

    # ETF flows
    click.echo("Estimating ETF flows...")
    etf = etf_flows.refresh(data_dir, force=force)
    click.echo(f"  ETFs: {len(etf)} tracked")

    # Interest rates
    api_key = config.get("fred", {}).get("api_key", "")
    if api_key and api_key != "YOUR_FRED_API_KEY_HERE":
        click.echo("Pulling rate data...")
        rates = fred_rates.refresh(api_key, data_dir, force=force)
        click.echo(f"  Rates: {len(rates)} days")
    else:
        click.echo("  Skipping rates (no FRED API key)")

    click.echo("Data ingestion complete.")


@cli.command()
@click.pass_context
def model(ctx):
    """Fit institutional demand models."""
    data_dir = ctx.obj["data_dir"]
    model_dir = ctx.obj["model_dir"]

    holdings = edgar_13f.load_holdings(data_dir)
    # Use quarterly Axioma history for walk-forward training; fall back to
    # the current snapshot if history is unavailable.
    fac = snowflake_factors.load_history(data_dir)
    if fac is None or fac.empty:
        fac = snowflake_factors.load_snapshot(data_dir)
    rates_df = fred_rates.load_panel(data_dir)
    crowding_df = snowflake_crowding.load_history(data_dir)

    if holdings is None or fac is None:
        click.echo("Error: Run 'pressure ingest' first.", err=True)
        sys.exit(1)

    click.echo("Fitting demand models...")
    results = demand_model.fit_and_save(holdings, fac, model_dir, rates_df, crowding_df)

    for style, res in results.items():
        click.echo(
            f"  {style}: AUC={res.auc_walkforward:.3f}, n={res.n_train}, "
            f"features={res.n_features}"
        )
        top3 = list(res.feature_importances.items())[:3]
        for feat, imp in top3:
            click.echo(f"    {feat}: {imp:.4f}")


@cli.command("learn")
@click.pass_context
def learn_fingerprints(ctx):
    """Learn execution fingerprints from historical 13F + daily volume.

    For each institution, learns the volume signature that accompanies
    accumulation vs. distribution. Requires 13F data to be ingested first.
    """
    data_dir = ctx.obj["data_dir"]
    model_dir = ctx.obj["model_dir"]
    snowflake_cfg = ctx.obj["snowflake"]

    holdings = edgar_13f.load_holdings(data_dir)
    if holdings is None or holdings.empty:
        click.echo("Error: Run 'pressure ingest' first.", err=True)
        sys.exit(1)

    click.echo("Building historical volume profiles (this may take a while)...")
    profiles = accumulation.build_profiles_from_cache(holdings, data_dir, snowflake_cfg)

    if profiles.empty:
        click.echo("No profiles could be built. Check that holdings data has position changes.")
        return

    click.echo(f"  Built {len(profiles)} quarterly volume profiles")

    click.echo("Learning execution fingerprints...")
    fingerprints = accumulation.learn_fingerprints(profiles)

    if not fingerprints:
        click.echo("  Could not learn any fingerprints (insufficient data per institution)")
        return

    accumulation.save_fingerprints(fingerprints, model_dir)

    for inst_key, fp in fingerprints.items():
        click.echo(f"  {inst_key}: AUC={fp.auc:.3f}, n={fp.n_training_samples}")
        top3 = list(fp.feature_importances.items())[:3]
        for feat, imp in top3:
            click.echo(f"    {feat}: {imp:.4f}")

    click.echo(f"\nSaved {len(fingerprints)} fingerprints.")


@cli.command()
@click.pass_context
def score(ctx):
    """Compute current Institutional Pressure Scores."""
    data_dir = ctx.obj["data_dir"]
    snowflake_cfg = ctx.obj["snowflake"]

    # Load all data
    holdings = edgar_13f.load_holdings(data_dir)
    fac = snowflake_factors.load_snapshot(data_dir)
    fac = fac if fac is not None else pd.DataFrame()
    vol = snowflake_price_volume.load_latest(data_dir)
    vol = vol if vol is not None else pd.DataFrame()
    crowd = snowflake_crowding.load_signals(data_dir)
    crowd = crowd if crowd is not None else pd.DataFrame()
    opts = options.load_latest(data_dir)
    opts = opts if opts is not None else pd.DataFrame()
    etf = etf_flows.load_latest(data_dir)
    etf = etf if etf is not None else pd.DataFrame()

    if holdings is None or holdings.empty:
        click.echo("Error: No holdings data. Run 'pressure ingest' first.", err=True)
        sys.exit(1)

    model_dir = ctx.obj["model_dir"]

    # Use trained demand model if available, otherwise neutral baseline
    click.echo("Computing institutional residuals...")
    trained = demand_model.load_models(model_dir)
    holdings_with_expected = holdings.copy()

    if trained:
        click.echo("  Using trained demand model for expected probabilities")
        # Apply trained model to compute expected buy probability per holding
        for style_key in ["passive", "active"]:
            if style_key not in trained:
                continue
            m = trained[style_key]
            style_mask = holdings_with_expected["style"] == style_key
            style_holdings = holdings_with_expected[style_mask]
            if style_holdings.empty:
                continue

            features = [f for f in demand_model.FACTOR_FEATURES if f in fac.columns]
            if features and not fac.empty:
                merged = style_holdings.merge(fac[["ticker"] + features], on="ticker", how="left")
                X, used_features = demand_model._prepare_features(merged)
                for f in m["scaler"].feature_names_in_:
                    if f not in X.columns:
                        X[f] = 0
                X = X[list(m["scaler"].feature_names_in_)]
                X_scaled = m["scaler"].transform(X)
                proba = m["model"].predict_proba(X_scaled)[:, 1]
                holdings_with_expected.loc[style_mask, "expected_buy_prob"] = proba
    else:
        click.echo("  No trained model found — using neutral baseline (run 'pressure model' first for better residuals)")
        holdings_with_expected["expected_buy_prob"] = 0.5

    resid = residual.compute_institution_residuals(
        holdings_with_expected,
        holdings_with_expected,
    )
    agg_resid = residual.aggregate_residuals(resid)

    if agg_resid.empty:
        click.echo("Error: Could not compute residuals.", err=True)
        sys.exit(1)

    # Ownership concentration
    ownership = residual.compute_ownership_concentration(holdings)

    # Accumulation detection — link 13F streaks to current volume
    click.echo("Detecting accumulation patterns...")
    fingerprints = accumulation.load_fingerprints(model_dir)

    # Fetch current daily data for fingerprint matching
    current_daily: dict[str, pd.DataFrame] = {}
    if fingerprints:
        click.echo("  Using learned execution fingerprints")
        import datetime as dt
        tickers_in_holdings = list(holdings["ticker"].unique())
        end = dt.date.today() + dt.timedelta(days=1)
        start = end - dt.timedelta(days=95)  # ~3 months
        try:
            current_daily = snowflake_price_volume.get_all_histories(
                tickers_in_holdings, start.isoformat(), end.isoformat(), snowflake_cfg
            )
        except Exception as e:
            click.echo(f"  Warning: Snowflake daily fetch failed: {e}")
    else:
        click.echo("  No fingerprints found — using volume heuristics (run 'pressure learn' to train)")

    accum_signals = accumulation.detect_accumulation(
        holdings, vol,
        fingerprints=fingerprints,
        current_daily=current_daily if current_daily else None,
    )
    accum_summary = accumulation.summarize_by_stock(accum_signals)

    # Volume predictions
    click.echo("Predicting volume spikes...")
    vol_features = volume_model.build_volume_features(agg_resid, fac, vol, opts, crowd)
    vol_preds = volume_model.predict_volume_spikes(vol_features)

    # Composite pressure score
    click.echo("Computing pressure scores...")
    results = pressure_score.compute_pressure_scores(
        residuals=agg_resid,
        volume_signals=vol,
        options_signals=opts,
        etf_signals=etf,
        factors=fac,
        ownership_changes=ownership,
        volume_predictions=vol_preds,
        holdings=holdings,
        crowding_signals=crowd,
    )

    # Output
    report.print_terminal_report(results)

    # Print accumulation signals if any
    if accum_signals:
        report.print_accumulation_report(accum_signals, accum_summary)

    md_path = report.generate_markdown_report(
        results, f"{data_dir}/reports/pressure_latest.md"
    )
    click.echo(f"Report saved to {md_path}")


@cli.command("report")
@click.pass_context
def generate_report(ctx):
    """Generate report from latest scores."""
    ctx.invoke(score)


@cli.command()
@click.argument("ticker")
@click.pass_context
def detail(ctx, ticker):
    """Deep dive on a single stock's pressure score."""
    ticker = ticker.upper()
    if ticker not in INSURANCE_UNIVERSE:
        click.echo(f"Unknown ticker: {ticker}", err=True)
        click.echo(f"Available: {', '.join(ALL_TICKERS)}")
        sys.exit(1)

    data_dir = ctx.obj["data_dir"]

    # Recompute scores (or load cached — for now recompute)
    holdings = edgar_13f.load_holdings(data_dir)
    fac = snowflake_factors.load_snapshot(data_dir)
    fac = fac if fac is not None else pd.DataFrame()
    vol = snowflake_price_volume.load_latest(data_dir)
    vol = vol if vol is not None else pd.DataFrame()
    crowd = snowflake_crowding.load_signals(data_dir)
    crowd = crowd if crowd is not None else pd.DataFrame()
    opts = options.load_latest(data_dir)
    opts = opts if opts is not None else pd.DataFrame()
    etf = etf_flows.load_latest(data_dir)
    etf = etf if etf is not None else pd.DataFrame()

    if holdings is None or holdings.empty:
        click.echo("Error: No data. Run 'pressure ingest' first.", err=True)
        sys.exit(1)

    holdings_with_expected = holdings.copy()
    holdings_with_expected["expected_buy_prob"] = 0.5

    resid = residual.compute_institution_residuals(
        holdings_with_expected, holdings_with_expected
    )
    agg_resid = residual.aggregate_residuals(resid)
    ownership = residual.compute_ownership_concentration(holdings)

    vol_features = volume_model.build_volume_features(agg_resid, fac, vol, opts, crowd)
    vol_preds = volume_model.predict_volume_spikes(vol_features)

    results = pressure_score.compute_pressure_scores(
        residuals=agg_resid,
        volume_signals=vol,
        options_signals=opts,
        etf_signals=etf,
        factors=fac,
        ownership_changes=ownership,
        volume_predictions=vol_preds,
        holdings=holdings,
        crowding_signals=crowd,
    )

    # Find the specific ticker
    target = [r for r in results if r.ticker == ticker]
    if not target:
        click.echo(f"No data available for {ticker}")
        return

    report.print_detail_report(target[0])


@cli.command()
@click.option("--ticker", default=None, help="Filter to a specific ticker")
@click.option("--institution", default=None, help="Filter to a specific institution")
@click.option("--all-quarters", is_flag=True, help="Show all quarters, not just the latest")
@click.pass_context
def converge(ctx, ticker, institution, all_quarters):
    """Trade convergence analysis — reverse-engineer how institutions trade.

    Transposes 13F trades against Axioma factor exposures, then checks
    direction vs unusual volume, vs the factors each institution is known
    to trade on, and vs similar historical quarters. When everything
    converges, you know how they're trading. Also shows the market impact
    of each factor and each player.
    """
    data_dir = ctx.obj["data_dir"]
    snowflake_cfg = ctx.obj["snowflake"]

    holdings = edgar_13f.load_holdings(data_dir)
    factor_history = snowflake_factors.load_history(data_dir)

    if holdings is None or holdings.empty:
        click.echo("Error: No holdings data. Run 'pressure ingest' first.", err=True)
        sys.exit(1)
    if factor_history is None or factor_history.empty:
        click.echo("Error: No Axioma factor history. Run 'pressure ingest' first.", err=True)
        sys.exit(1)

    # Optional filters
    if ticker:
        ticker = ticker.upper()
        holdings = holdings[holdings["ticker"] == ticker]
        if holdings.empty:
            click.echo(f"No holdings data for {ticker}", err=True)
            sys.exit(1)
    if institution:
        institution = institution.lower()
        holdings = holdings[holdings["institution"] == institution]
        if holdings.empty:
            click.echo(f"No holdings data for institution '{institution}'", err=True)
            sys.exit(1)

    # Volume profiles per ticker-quarter (Snowflake-backed, cached)
    click.echo("Loading volume profiles...")
    volume_profiles = cache.load(data_dir, accumulation.NAMESPACE, "historical_profiles")
    if volume_profiles is None:
        click.echo("  No cached profiles — building from Snowflake (this may take a while)")
        volume_profiles = accumulation.build_profiles_from_cache(
            holdings, data_dir, snowflake_cfg
        )

    # Daily histories for quarterly returns + dollar volume
    click.echo("Pulling daily market data from Snowflake...")
    import datetime as dt
    tickers_needed = list(holdings["ticker"].unique())
    end = dt.date.today() + dt.timedelta(days=1)
    daily_histories = snowflake_price_volume.get_all_histories(
        tickers_needed, "2014-01-01", end.isoformat(), snowflake_cfg
    )

    click.echo("Running convergence analysis...")
    results = convergence.run_convergence_analysis(
        holdings=holdings,
        factor_history=factor_history,
        volume_profiles=volume_profiles if volume_profiles is not None else pd.DataFrame(),
        daily_histories=daily_histories,
    )

    if results["panel"].empty:
        click.echo("No overlapping trade-factor data to analyze.")
        return

    report.print_convergence_report(
        results["panel"],
        results["readability"],
        results["factor_impacts"],
        results["player_impacts"],
        latest_only=not all_quarters,
    )

    md_path = report.generate_convergence_markdown(
        results["panel"],
        results["readability"],
        results["factor_impacts"],
        results["player_impacts"],
        f"{data_dir}/reports/convergence_latest.md",
    )
    click.echo(f"Report saved to {md_path}")


@cli.command("analyze-factors")
@click.argument("factors_path", type=click.Path(exists=True))
@click.option("--institution", default=None, help="Filter to a specific institution")
@click.option("--ticker", default=None, help="Filter to a specific ticker")
@click.option("--min-trades", default=20, help="Minimum trades per institution for profiling")
@click.pass_context
def analyze_factors(ctx, factors_path, institution, ticker, min_trades):
    """Analyze which factors drive institutional trading.

    Provide a CSV/Excel/Parquet file with factor data. The system will
    automatically detect quarter/date columns, ticker columns, and
    numeric factor columns, then correlate them with 13F trading history.

    \b
    Examples:
        pressure analyze-factors data/my_factors.csv
        pressure analyze-factors data/factors.xlsx --institution fidelity
        pressure analyze-factors data/macro.csv --ticker PGR
    """
    data_dir = ctx.obj["data_dir"]

    # Load 13F holdings
    holdings = edgar_13f.load_holdings(data_dir)
    if holdings is None or holdings.empty:
        click.echo("Error: No holdings data. Run 'pressure ingest' first.", err=True)
        sys.exit(1)

    # Load user factors
    click.echo(f"Loading factors from {factors_path}...")
    try:
        user_factors = factor_loader.load_factors(factors_path)
    except (ValueError, FileNotFoundError) as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    info = factor_loader.describe_factors(user_factors)
    click.echo(f"  {info['n_factors']} factors: {', '.join(info['factor_names'])}")
    click.echo(f"  {info['n_quarters']} quarters: {info['quarters'][0]} to {info['quarters'][-1]}")
    if info['has_ticker']:
        click.echo(f"  {info['n_tickers']} tickers (per-stock factors)")
    else:
        click.echo("  Macro factors (applied to all tickers)")

    # Optional filters
    if ticker:
        ticker = ticker.upper()
        holdings = holdings[holdings["ticker"] == ticker]
        if info['has_ticker']:
            user_factors = user_factors[user_factors["ticker"] == ticker]
        if holdings.empty:
            click.echo(f"No holdings data for {ticker}", err=True)
            sys.exit(1)

    if institution:
        institution = institution.lower()
        holdings = holdings[holdings["institution"] == institution]
        if holdings.empty:
            click.echo(f"No holdings data for institution '{institution}'", err=True)
            sys.exit(1)

    # Step 1: Overall factor-trade correlations
    click.echo("\nAnalyzing factor-trade relationships...")
    overall_results = factor_analysis.analyze_factor_trade_relationship(
        holdings, user_factors, min_observations=min_trades,
    )

    # Step 2: Per-institution profiles
    click.echo("Building institution factor profiles...")
    profiles = factor_analysis.build_institution_profiles(
        holdings, user_factors, min_trades=min_trades,
    )

    # Step 3: Predict current regime
    # Use the latest quarter of factor data as "current"
    latest_quarter = user_factors["quarter"].max()
    current_factors = user_factors[user_factors["quarter"] == latest_quarter]
    predictions = factor_analysis.predict_trades_from_factors(profiles, current_factors)

    # Output
    report.print_factor_report(overall_results, profiles, predictions)


@cli.command()
@click.option("--force", is_flag=True)
@click.pass_context
def run(ctx, force):
    """Full pipeline: ingest -> model -> learn -> score -> report."""
    ctx.invoke(ingest, force=force)
    ctx.invoke(model)
    ctx.invoke(learn_fingerprints)
    ctx.invoke(score)


if __name__ == "__main__":
    cli()
