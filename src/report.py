"""Report generator — terminal and markdown output for pressure scores."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich.panel import Panel

from src.model.pressure_score import PressureResult
from src.model.accumulation import AccumulationSignal
from src.model.factor_analysis import FactorTradeResult, InstitutionFactorProfile

logger = logging.getLogger(__name__)


def print_terminal_report(results: list[PressureResult]) -> None:
    """Print a rich terminal table of pressure scores."""
    console = Console()

    console.print()
    console.print(
        Panel(
            f"[bold]Institutional Pressure Score — Insurance Equities[/bold]\n"
            f"Generated: {datetime.now():%Y-%m-%d %H:%M}",
            style="cyan",
        )
    )

    # Main score table
    table = Table(show_header=True, header_style="bold")
    table.add_column("Ticker", style="bold", width=6)
    table.add_column("Name", width=22)
    table.add_column("IPS", justify="right", width=7)
    table.add_column("Direction", justify="center", width=12)
    table.add_column("Strength", justify="center", width=10)
    table.add_column("Confidence", justify="right", width=10)
    table.add_column("Vol Spike P", justify="right", width=10)
    table.add_column("Res Z", justify="right", width=7)

    for r in results:
        # Color-code direction
        if r.direction == "ACCUMULATE":
            dir_style = "bold green"
        elif r.direction == "DISTRIBUTE":
            dir_style = "bold red"
        else:
            dir_style = "dim"

        # Color-code strength
        strength_styles = {
            "STRONG": "bold",
            "MODERATE": "",
            "WEAK": "dim",
            "NEGLIGIBLE": "dim italic",
        }

        # Color-code score
        if r.score > 0:
            score_str = f"+{r.score:.0f}"
            score_style = "green" if r.score > 20 else ""
        elif r.score < 0:
            score_str = f"{r.score:.0f}"
            score_style = "red" if r.score < -20 else ""
        else:
            score_str = "0"
            score_style = "dim"

        table.add_row(
            r.ticker,
            r.name,
            Text(score_str, style=score_style),
            Text(r.direction, style=dir_style),
            Text(r.strength, style=strength_styles.get(r.strength, "")),
            f"{r.confidence:.0%}",
            f"{r.volume_spike_prob:.0%}",
            f"{r.residual_z:+.2f}",
        )

    console.print(table)

    # Actionable signals — stocks with strong scores
    strong = [r for r in results if r.strength in ("STRONG", "MODERATE")]
    if strong:
        console.print()
        console.print("[bold cyan]Actionable Signals[/bold cyan]")
        console.print()

        for r in strong:
            dir_word = "accumulating" if r.direction == "ACCUMULATE" else "distributing"
            color = "green" if r.direction == "ACCUMULATE" else "red"

            console.print(f"  [{color} bold]{r.ticker}[/{color} bold] ({r.name})")
            console.print(f"    Score: {r.score:+.0f} — Institutions {dir_word}")

            if r.top_institutions:
                console.print(f"    Key movers: {', '.join(r.top_institutions)}")

            # Component breakdown
            top_components = sorted(
                r.components.items(), key=lambda x: abs(x[1]), reverse=True
            )[:3]
            drivers = ", ".join(
                f"{k}: {v:+.2f}" for k, v in top_components
            )
            console.print(f"    Drivers: {drivers}")
            console.print()

    # ETF sector flow summary
    console.print("[dim]Score range: -100 (distribution) to +100 (accumulation)[/dim]")
    console.print()


def print_accumulation_report(
    signals: list[AccumulationSignal],
    summary: "pd.DataFrame",
) -> None:
    """Print accumulation/distribution patterns detected from 13F streaks + volume."""
    console = Console()
    import pandas as pd

    console.print()
    console.print(Panel(
        "[bold]Accumulation / Distribution Detection[/bold]\n"
        "13F position streaks cross-referenced with current volume",
        style="cyan",
    ))

    # Per-stock summary
    if not summary.empty:
        table = Table(show_header=True, header_style="bold")
        table.add_column("Ticker", width=6)
        table.add_column("Direction", width=12)
        table.add_column("# Accum", justify="right", width=8)
        table.add_column("# Distrib", justify="right", width=8)
        table.add_column("Cont. Prob", justify="right", width=10)
        table.add_column("Vol Confirmed", justify="right", width=13)
        table.add_column("Top Accumulator", width=22)

        for _, row in summary.iterrows():
            dir_style = "green" if row["net_direction"] == "ACCUMULATE" else (
                "red" if row["net_direction"] == "DISTRIBUTE" else "yellow"
            )
            table.add_row(
                row["ticker"],
                Text(row["net_direction"], style=dir_style),
                str(row["n_accumulating"]),
                str(row["n_distributing"]),
                f"{row['avg_continuation_prob']:.0%}",
                str(row["volume_confirmed_count"]),
                str(row.get("top_accumulator", "—") or "—"),
            )

        console.print(table)

    # Top individual signals
    active_accum = [s for s in signals if s.direction == "ACCUMULATING" and s.style == "active"]
    if active_accum:
        console.print()
        console.print("[bold]Active Fund Accumulation Streaks[/bold]")
        for s in active_accum[:10]:
            vol_icon = "volume confirms" if s.volume_confirms else "volume inconclusive"
            color = "green" if s.volume_confirms else "yellow"
            console.print(
                f"  [bold]{s.ticker}[/bold] ← {s.institution_name}: "
                f"{s.consecutive_buys}Q streak, {s.total_change_pct:+.1f}% total, "
                f"[{color}]{vol_icon}[/{color}] "
                f"(P={s.continuation_probability:.0%})"
            )
    console.print()


def print_detail_report(result: PressureResult) -> None:
    """Print a detailed deep-dive for a single stock."""
    console = Console()
    color = "green" if result.direction == "ACCUMULATE" else ("red" if result.direction == "DISTRIBUTE" else "white")

    console.print()
    console.print(Panel(
        f"[bold]{result.ticker}[/bold] — {result.name}\n"
        f"[{color}]IPS: {result.score:+.0f} | {result.direction} | {result.strength}[/{color}]",
    ))

    # Component table
    comp_table = Table(title="Score Components", show_header=True)
    comp_table.add_column("Component", width=25)
    comp_table.add_column("Value", justify="right", width=10)
    comp_table.add_column("Contribution", justify="right", width=12)

    from src.model.pressure_score import DEFAULT_WEIGHTS
    for comp_name, comp_val in sorted(result.components.items(), key=lambda x: abs(x[1]), reverse=True):
        weight = DEFAULT_WEIGHTS.get(comp_name, 0)
        contribution = comp_val * weight * 100
        comp_table.add_row(
            comp_name.replace("_", " ").title(),
            f"{comp_val:+.3f}",
            f"{contribution:+.1f}",
        )

    console.print(comp_table)

    # Institution activity
    if result.top_institutions:
        console.print()
        console.print("[bold]Top Institutions:[/bold]")
        for inst in result.top_institutions:
            console.print(f"  - {inst}")

    console.print()
    console.print(f"Volume spike probability: {result.volume_spike_prob:.0%}")
    console.print(f"Residual z-score: {result.residual_z:+.3f}")
    console.print(f"Signal confidence: {result.confidence:.0%}")
    console.print()


def generate_markdown_report(results: list[PressureResult], output_path: str) -> Path:
    """Generate a markdown report file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Institutional Pressure Score Report",
        "",
        f"**Generated:** {datetime.now():%Y-%m-%d %H:%M}",
        "",
        "## Scores",
        "",
        "| Ticker | Name | IPS | Direction | Strength | Confidence | Vol Spike P |",
        "|--------|------|-----|-----------|----------|------------|-------------|",
    ]

    for r in results:
        score_str = f"{r.score:+.0f}"
        lines.append(
            f"| {r.ticker} | {r.name} | {score_str} | {r.direction} | "
            f"{r.strength} | {r.confidence:.0%} | {r.volume_spike_prob:.0%} |"
        )

    lines.append("")

    # Actionable signals
    strong = [r for r in results if r.strength in ("STRONG", "MODERATE")]
    if strong:
        lines.append("## Actionable Signals")
        lines.append("")
        for r in strong:
            dir_word = "accumulating" if r.direction == "ACCUMULATE" else "distributing"
            lines.append(f"### {r.ticker} ({r.name}) — IPS {r.score:+.0f}")
            lines.append("")
            lines.append(f"Institutions are **{dir_word}**. Residual z-score: {r.residual_z:+.3f}")
            if r.top_institutions:
                lines.append(f"\nKey movers: {', '.join(r.top_institutions)}")
            lines.append("")
            top_comp = sorted(r.components.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
            lines.append("**Drivers:**")
            for k, v in top_comp:
                lines.append(f"- {k.replace('_', ' ').title()}: {v:+.3f}")
            lines.append("")

    path.write_text("\n".join(lines))
    logger.info("Markdown report written to %s", path)
    return path


def print_convergence_report(
    panel: "pd.DataFrame",
    readability: "pd.DataFrame",
    factor_impacts: list,
    player_impacts: list,
    latest_only: bool = True,
) -> None:
    """Print trade convergence analysis: who's trading, how readable they
    are, and the market impact of each factor and player."""
    import pandas as pd
    console = Console()

    console.print()
    console.print(Panel(
        "[bold]Trade Convergence Analysis[/bold]\n"
        "13F direction × unusual volume × known factors × similar quarters",
        style="cyan",
    ))

    # --- 1. Convergence table (latest quarter trades) ---
    if not panel.empty and "convergence_score" in panel.columns:
        display = panel.dropna(subset=["convergence_score"])
        if latest_only and not display.empty:
            latest_q = display["quarter"].max()
            display = display[display["quarter"] == latest_q]
            title = f"Trade Convergence — {latest_q}"
        else:
            title = "Trade Convergence"

        display = display.sort_values("convergence_score", ascending=False)

        table = Table(title=title, show_header=True, header_style="bold")
        table.add_column("Ticker", width=6)
        table.add_column("Institution", width=20)
        table.add_column("Dir", justify="center", width=5)
        table.add_column("Factor Align", justify="right", width=12)
        table.add_column("Vol Confirms", justify="center", width=12)
        table.add_column("Hist Consist", justify="right", width=12)
        table.add_column("Converge", justify="right", width=9)
        table.add_column("Verdict", width=20)

        for _, row in display.head(30).iterrows():
            dir_text = Text("BUY", style="green") if row["direction"] > 0 else Text("SELL", style="red")
            verdict = row["verdict"]
            if verdict.startswith("CONVERGED"):
                v_style = "bold green" if "BUYING" in verdict else "bold red"
            elif verdict == "PARTIAL":
                v_style = "yellow"
            else:
                v_style = "dim"

            align = row.get("factor_alignment_score")
            consist = row.get("historical_consistency")
            table.add_row(
                row["ticker"],
                str(row.get("institution_name", row["institution"])),
                dir_text,
                f"{align:+.2f}" if pd.notna(align) else "—",
                "yes" if row.get("volume_confirms") else "no",
                f"{consist:.0%}" if pd.notna(consist) else "—",
                f"{row['convergence_score']:.2f}",
                Text(verdict, style=v_style),
            )

        console.print(table)

    # --- 2. Factor market impact ---
    if factor_impacts:
        console.print()
        table = Table(
            title="Market Impact per Factor (quarterly return per +1σ exposure)",
            show_header=True, header_style="bold",
        )
        table.add_column("Factor", width=20)
        table.add_column("Avg Impact (bps)", justify="right", width=16)
        table.add_column("t-stat", justify="right", width=8)
        table.add_column("Hit Rate", justify="right", width=9)
        table.add_column("Last Qtr (bps)", justify="right", width=14)
        table.add_column("N Qtrs", justify="right", width=7)

        for fi in factor_impacts:
            style = "green" if fi.avg_impact_bps > 0 else "red"
            sig = "bold " + style if abs(fi.t_stat) >= 2 else style
            table.add_row(
                fi.factor.replace("_", " ").title(),
                Text(f"{fi.avg_impact_bps:+.1f}", style=sig),
                f"{fi.t_stat:+.2f}",
                f"{fi.hit_rate:.0%}",
                f"{fi.last_quarter_impact_bps:+.1f}",
                str(fi.n_quarters),
            )

        console.print(table)

    # --- 3. Player market impact + readability ---
    if player_impacts:
        console.print()
        read_map = {}
        if readability is not None and not readability.empty:
            read_map = readability.set_index("institution")["avg_convergence"].to_dict()

        table = Table(
            title="Market Impact per Player",
            show_header=True, header_style="bold",
        )
        table.add_column("Institution", width=20)
        table.add_column("Style", width=8)
        table.add_column("Participation", justify="right", width=13)
        table.add_column("Buy Impact (bps)", justify="right", width=16)
        table.add_column("Sell Impact (bps)", justify="right", width=17)
        table.add_column("Vol Footprint", justify="right", width=13)
        table.add_column("Readability", justify="right", width=11)
        table.add_column("Trades", justify="right", width=7)

        for pi in player_impacts:
            readab = read_map.get(pi.institution)
            table.add_row(
                pi.institution_name,
                pi.style,
                f"{pi.avg_participation_pct:.2f}%",
                Text(f"{pi.impact_bps_when_buying:+.0f}",
                     style="green" if pi.impact_bps_when_buying > 0 else "red"),
                Text(f"{pi.impact_bps_when_selling:+.0f}",
                     style="red" if pi.impact_bps_when_selling < 0 else "green"),
                f"{pi.volume_footprint:.2f}",
                f"{readab:.2f}" if readab is not None else "—",
                str(pi.n_trades),
            )

        console.print(table)

    console.print()


def generate_convergence_markdown(
    panel: "pd.DataFrame",
    readability: "pd.DataFrame",
    factor_impacts: list,
    player_impacts: list,
    output_path: str,
) -> Path:
    """Write the convergence analysis to markdown."""
    import pandas as pd
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Trade Convergence Report",
        "",
        f"**Generated:** {datetime.now():%Y-%m-%d %H:%M}",
        "",
    ]

    if not panel.empty and "convergence_score" in panel.columns:
        display = panel.dropna(subset=["convergence_score"])
        if not display.empty:
            latest_q = display["quarter"].max()
            display = display[display["quarter"] == latest_q].sort_values(
                "convergence_score", ascending=False
            )
            lines += [
                f"## Trade Convergence — {latest_q}",
                "",
                "| Ticker | Institution | Dir | Factor Align | Vol | Hist | Score | Verdict |",
                "|--------|-------------|-----|--------------|-----|------|-------|---------|",
            ]
            for _, row in display.iterrows():
                align = row.get("factor_alignment_score")
                consist = row.get("historical_consistency")
                lines.append(
                    f"| {row['ticker']} | {row.get('institution_name', row['institution'])} | "
                    f"{'BUY' if row['direction'] > 0 else 'SELL'} | "
                    f"{f'{align:+.2f}' if pd.notna(align) else '—'} | "
                    f"{'yes' if row.get('volume_confirms') else 'no'} | "
                    f"{f'{consist:.0%}' if pd.notna(consist) else '—'} | "
                    f"{row['convergence_score']:.2f} | {row['verdict']} |"
                )
            lines.append("")

    if factor_impacts:
        lines += [
            "## Market Impact per Factor",
            "",
            "| Factor | Avg Impact (bps/1σ) | t-stat | Hit Rate | Last Qtr (bps) | N |",
            "|--------|---------------------|--------|----------|----------------|---|",
        ]
        for fi in factor_impacts:
            lines.append(
                f"| {fi.factor.replace('_', ' ').title()} | {fi.avg_impact_bps:+.1f} | "
                f"{fi.t_stat:+.2f} | {fi.hit_rate:.0%} | "
                f"{fi.last_quarter_impact_bps:+.1f} | {fi.n_quarters} |"
            )
        lines.append("")

    if player_impacts:
        read_map = {}
        if readability is not None and not readability.empty:
            read_map = readability.set_index("institution")["avg_convergence"].to_dict()
        lines += [
            "## Market Impact per Player",
            "",
            "| Institution | Style | Participation | Buy Impact | Sell Impact | Vol Footprint | Readability | Trades |",
            "|-------------|-------|---------------|------------|-------------|---------------|-------------|--------|",
        ]
        for pi in player_impacts:
            readab = read_map.get(pi.institution)
            lines.append(
                f"| {pi.institution_name} | {pi.style} | {pi.avg_participation_pct:.2f}% | "
                f"{pi.impact_bps_when_buying:+.0f} bps | {pi.impact_bps_when_selling:+.0f} bps | "
                f"{pi.volume_footprint:.2f} | "
                f"{f'{readab:.2f}' if readab is not None else '—'} | {pi.n_trades} |"
            )
        lines.append("")

    path.write_text("\n".join(lines))
    logger.info("Convergence report written to %s", path)
    return path


def print_factor_report(
    overall_results: list[FactorTradeResult],
    profiles: dict[str, InstitutionFactorProfile],
    predictions: "pd.DataFrame",
) -> None:
    """Print factor-trade analysis results."""
    import pandas as pd
    console = Console()

    console.print()
    console.print(Panel(
        "[bold]Factor-Trade Analysis[/bold]\n"
        "Which factors drive institutional trading in insurance stocks?",
        style="cyan",
    ))

    # --- Overall factor significance ---
    if overall_results:
        table = Table(title="Factor Significance (All Institutions)", show_header=True, header_style="bold")
        table.add_column("Factor", width=25)
        table.add_column("Corr", justify="right", width=8)
        table.add_column("p-value", justify="right", width=10)
        table.add_column("Sig", justify="center", width=5)
        table.add_column("Direction", width=16)
        table.add_column("Q1 Buy%", justify="right", width=8)
        table.add_column("Q4 Buy%", justify="right", width=8)
        table.add_column("N", justify="right", width=6)

        for r in overall_results:
            corr_style = "green" if r.correlation > 0 else ("red" if r.correlation < 0 else "")
            sig_style = "bold green" if r.is_significant else "dim"
            q1 = r.quartile_buy_rates.get("Q1", 0)
            q4 = r.quartile_buy_rates.get("Q4", 0)

            dir_text = r.direction.replace("_", " ")
            if r.direction == "buy_when_high":
                dir_text = "buy when high"
                dir_style = "green"
            elif r.direction == "buy_when_low":
                dir_text = "buy when low"
                dir_style = "red"
            else:
                dir_style = "dim"

            table.add_row(
                r.factor_name.replace("_", " ").title(),
                Text(f"{r.correlation:+.3f}", style=corr_style),
                f"{r.p_value:.4f}",
                Text(r.stars or "-", style=sig_style),
                Text(dir_text, style=dir_style),
                f"{q1:.0%}",
                f"{q4:.0%}",
                str(r.n_observations),
            )

        console.print(table)
    else:
        console.print("[dim]No significant factor-trade relationships found.[/dim]")

    # --- Per-institution profiles ---
    if profiles:
        console.print()
        console.print("[bold cyan]Institution Factor Profiles[/bold cyan]")
        console.print()

        for inst_key, profile in sorted(profiles.items(), key=lambda x: -x[1].model_auc):
            auc_style = "green" if profile.model_auc > 0.6 else ("yellow" if profile.model_auc > 0.55 else "dim")
            console.print(
                f"  [bold]{profile.institution_name}[/bold] ({profile.style})  "
                f"AUC=[{auc_style}]{profile.model_auc:.3f}[/{auc_style}]  "
                f"n={profile.n_trades}"
            )

            # Top 5 factors by importance
            top_factors = list(profile.feature_importances.items())[:5]
            if top_factors:
                parts = []
                for fname, coef in top_factors:
                    color = "green" if coef > 0 else "red"
                    parts.append(f"[{color}]{fname}: {coef:+.3f}[/{color}]")
                console.print(f"    Factors: {', '.join(parts)}")

            # Significant univariate factors
            sig = [f for f in profile.significant_factors if f.is_significant]
            if sig:
                sig_names = [f"{f.factor_name}{f.stars}" for f in sig[:5]]
                console.print(f"    Significant: {', '.join(sig_names)}")

            console.print()

    # --- Current predictions ---
    if predictions is not None and not predictions.empty:
        console.print()
        pred_table = Table(title="Current Factor Regime Predictions", show_header=True, header_style="bold")
        pred_table.add_column("Ticker", width=6)
        pred_table.add_column("Institution", width=25)
        pred_table.add_column("P(Buy)", justify="right", width=8)
        pred_table.add_column("Prediction", width=15)
        pred_table.add_column("AUC", justify="right", width=6)

        for _, row in predictions.iterrows():
            if row["predicted_action"] == "LIKELY BUYING":
                action_style = "bold green"
            elif row["predicted_action"] == "LIKELY SELLING":
                action_style = "bold red"
            else:
                action_style = "dim"

            pred_table.add_row(
                row["ticker"],
                row["institution_name"],
                f"{row['p_buy']:.0%}",
                Text(row["predicted_action"], style=action_style),
                f"{row['model_auc']:.2f}",
            )

        console.print(pred_table)

    console.print()
