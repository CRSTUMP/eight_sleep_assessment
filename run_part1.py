#!/usr/bin/env python3
"""Run Part 1: clustering + anomaly detection pipeline.

Usage:
  python run_part1.py
  python run_part1.py --force-recompute      # ignore embedding cache
  python run_part1.py --output my_results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

# Windows console defaults to charmap; force UTF-8 so emoji in Slack alerts render
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("httpx").setLevel(logging.WARNING)

console = Console()

SEVERITY_COLOR = {
    "critical": "red",
    "high": "orange3",
    "medium": "yellow",
    "low": "green",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Eight Sleep CX Anomaly Detection")
    parser.add_argument("--force-recompute", action="store_true", help="Recompute embeddings even if cached")
    parser.add_argument("--output", default="results_part1.json", help="JSON output path")
    args = parser.parse_args()

    _check_env()

    from part1.pipeline import run_pipeline

    console.print(Panel("[bold blue]Eight Sleep CX Intelligence — Part 1: Clustering & Anomaly Detection[/bold blue]"))

    results = run_pipeline(force_recompute=args.force_recompute)

    _print_clusters(results)
    _print_anomalies(results, day=2)
    _print_anomalies(results, day=3)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    console.print(f"\n[green]✓ Full results saved to {args.output}[/green]")


def _print_clusters(results: dict) -> None:
    clusters = results["clusters"]
    day_stats = results["day_stats"]

    table = Table(title="Discovered Issue Clusters", show_header=True, header_style="bold cyan")
    table.add_column("ID", width=4, style="dim")
    table.add_column("Label", style="bold", min_width=30)
    table.add_column("Description", max_width=45)
    table.add_column("Total", justify="right", width=7)
    table.add_column("Day 1", justify="right", width=7)
    table.add_column("Day 2", justify="right", width=7)
    table.add_column("Day 3", justify="right", width=7)

    sorted_clusters = sorted(clusters.items(), key=lambda x: x[1]["ticket_count"], reverse=True)
    for cid_str, cl in sorted_clusters:
        # cluster_counts keys are integers (from model_dump); cid_str is the string key
        cid_int = int(cid_str)
        d1 = day_stats.get("1", {}).get("cluster_counts", {}).get(cid_int, 0)
        d2 = day_stats.get("2", {}).get("cluster_counts", {}).get(cid_int, 0)
        d3 = day_stats.get("3", {}).get("cluster_counts", {}).get(cid_int, 0)
        table.add_row(
            cid_str,
            cl["label"],
            cl["description"][:44],
            str(cl["ticket_count"]),
            str(d1),
            str(d2),
            str(d3),
        )
    console.print(table)


def _print_anomalies(results: dict, day: int) -> None:
    key = f"anomalies_day{day}"
    anomalies = results.get(key, [])
    if not anomalies:
        console.print(f"\n[dim]No anomalies detected on Day {day}.[/dim]")
        return

    console.print(f"\n[bold red]Day {day} Anomalies — {len(anomalies)} detected[/bold red]")
    for a in anomalies:
        color = SEVERITY_COLOR.get(a["severity"], "white")
        sign = "+" if a["magnitude_pct"] >= 0 else ""
        console.print(
            Panel(
                f"[{color}]{a['severity'].upper()}[/{color}]  "
                f"[bold]{sign}{a['magnitude_pct']:.0f}%[/bold]  "
                f"({a['current_count']} tickets vs {a['baseline_count']} baseline)\n\n"
                f"[italic]{a['cluster_description']}[/italic]\n\n"
                f"[dim]Slack alert:[/dim]\n{a['slack_message']}",
                title=f"[bold]{a['cluster_label']}[/bold]  [{a['anomaly_type']}]",
                border_style=color,
                expand=False,
            )
        )


def _check_env() -> None:
    import os
    missing = [v for v in ("GEMINI_API_KEY", "GROQ_API_KEY") if not os.getenv(v)]
    if missing:
        # pydantic-settings will also check .env file — just warn here
        console.print(f"[yellow]Warning: {', '.join(missing)} not set in environment. Checking .env file...[/yellow]")


if __name__ == "__main__":
    main()
