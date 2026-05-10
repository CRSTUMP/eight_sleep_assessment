#!/usr/bin/env python3
"""Run Part 2: interactive RAG chatbot for CX ticket resolution.

Usage:
  python run_part2.py                    # free-form chat, no specific ticket
  python run_part2.py --ticket TICKET_ID # anchor chat to a specific open ticket
  python run_part2.py --rebuild          # force rebuild ChromaDB index

The chatbot:
  1. Loads all tickets and reuses Part 1's embedding cache if present.
  2. Builds (or loads) the ChromaDB knowledge base of resolved tickets.
  3. On each turn: embeds the query, retrieves top-5 resolved tickets,
     passes them as context to Groq Llama 3.3-70b, streams the response.
"""

from __future__ import annotations

import argparse
import logging
import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger("httpx").setLevel(logging.WARNING)

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(description="Eight Sleep CX Chatbot")
    parser.add_argument("--ticket", default=None, help="Ticket ID to anchor the session on")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild ChromaDB index from scratch")
    args = parser.parse_args()

    from part1.embeddings import get_or_create_embeddings, load_tickets
    from part2 import knowledge_base as kb
    from part2.chatbot import CXChatbot

    console.print(Panel("[bold blue]Eight Sleep CX Intelligence — Part 2: Ticket Resolution Chatbot[/bold blue]"))

    # Load tickets and embeddings
    console.print("[dim]Loading tickets...[/dim]")
    tickets = load_tickets()
    console.print("[dim]Loading/computing embeddings...[/dim]")
    embeddings = get_or_create_embeddings(tickets)

    # Build or load ChromaDB
    if args.rebuild:
        console.print("[dim]Rebuilding knowledge base...[/dim]")
        collection = kb.build(tickets, embeddings)
    else:
        try:
            collection = kb.load()
            console.print(f"[green]Knowledge base loaded ({collection.count()} resolved tickets indexed)[/green]")
        except Exception:
            console.print("[yellow]No existing index found — building now...[/yellow]")
            collection = kb.build(tickets, embeddings)

    # Resolve anchor ticket
    open_ticket = None
    if args.ticket:
        ticket_map = {t.id: t for t in tickets}
        open_ticket = ticket_map.get(args.ticket)
        if open_ticket is None:
            console.print(f"[red]Ticket ID '{args.ticket}' not found in dataset.[/red]")
            sys.exit(1)
        console.print(Panel(open_ticket.to_chat_context(), title=f"[bold]Anchored Ticket: {open_ticket.id}[/bold]"))

    bot = CXChatbot(collection, tickets)

    console.print("\n[bold green]Chatbot ready.[/bold green] Type your question (or 'exit' to quit, 'reset' to clear history).\n")

    while True:
        try:
            user_input = Prompt.ask("[bold cyan]Agent[/bold cyan]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Session ended.[/dim]")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit", "q"):
            break
        if user_input.lower() == "reset":
            bot.reset()
            console.print("[dim]History cleared.[/dim]")
            continue

        console.print("[bold magenta]Assistant[/bold magenta]: ", end="")
        try:
            bot.chat(user_input, open_ticket=open_ticket)
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")

    console.print("[dim]Goodbye.[/dim]")


if __name__ == "__main__":
    main()
