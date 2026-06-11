"""Typer-based CLI for the travel-planner demo."""

from __future__ import annotations

import logging
import time
import uuid
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)

import typer
from rich.console import Console
from rich.panel import Panel
from rich.pretty import Pretty
from rich.table import Table

from .agent import build_travel_agent
from .config import load_settings
from .store import ns_past_trips, ns_preferences

app = typer.Typer(
    help="DeepAgents + MongoDB Atlas travel planner demo.",
    no_args_is_help=True,
)
console = Console()


def _print_event(event: dict) -> None:
    """Render a streamed LangGraph event for the user."""
    node = list(event.keys())[0] if event else None
    if not node:
        return
    payload = event[node]
    msgs = payload.get("messages") if isinstance(payload, dict) else None
    if not msgs:
        return
    last = msgs[-1]
    role = getattr(last, "type", "?")
    name = getattr(last, "name", None) or node
    content = getattr(last, "content", "")
    if isinstance(content, list):
        # Tool-result content blocks come through as lists; flatten to text.
        content = "\n".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    snippet = (content or "").strip()
    if len(snippet) > 600:
        snippet = snippet[:600] + " …"
    if not snippet:
        return
    console.print(
        Panel(
            snippet,
            title=f"[bold]{name}[/bold] · {role}",
            border_style="cyan" if role == "ai" else "yellow",
            expand=False,
        )
    )


@app.command()
def plan(
    user: str = typer.Option(..., help="User identifier (used for memory namespacing)."),
    query: str = typer.Option(..., help="Travel request in natural language."),
    thread_id: Optional[str] = typer.Option(
        None,
        help="Resume an existing conversation thread; otherwise a new one is created.",
    ),
    no_cache: bool = typer.Option(False, help="Disable the MongoDB semantic cache."),
    show_stream: bool = typer.Option(
        True, help="Print intermediate agent events as they stream."
    ),
):
    """Plan a trip — runs the full deepagent pipeline."""
    settings = load_settings()
    bundle = build_travel_agent(settings, enable_cache=not no_cache)
    thread_id = thread_id or f"thread-{user}-{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread_id, "user_id": user}}

    console.rule(f"[bold magenta]Travel Planner · user={user} · thread={thread_id}")
    console.print(f"[dim]Cache enabled: {not no_cache}[/dim]")
    console.print(f"[bold]Query:[/bold] {query}\n")

    user_message = (
        f"{query}\n\n"
        "Plan the trip end-to-end using your subagents and MongoDB memory."
    )

    start = time.time()
    try:
        for event in bundle.agent.stream(
            {"messages": [{"role": "user", "content": user_message}]},
            config=config,
            stream_mode="updates",
        ):
            if show_stream:
                _print_event(event)
        elapsed = time.time() - start

        # Fetch the final state to print the compiled itinerary, if any.
        final_state = bundle.agent.get_state(config).values
        files = final_state.get("files", {}) if isinstance(final_state, dict) else {}
        # deepagents normalises virtual filesystem paths to start with "/".
        itinerary = (
            files.get("/itinerary.md")
            or files.get("itinerary.md")
        )
        if itinerary:
            console.rule("[bold green]itinerary.md")
            text = (
                itinerary
                if isinstance(itinerary, str)
                else itinerary.get("content", "")
            )
            console.print(text)
        else:
            console.print("[yellow]No itinerary.md was produced.[/yellow]")

        console.rule()
        console.print(f"[bold]Elapsed:[/bold] {elapsed:.1f}s · thread_id={thread_id}")
        console.print(
            "[dim]Re-run with --thread-id "
            f"{thread_id} to continue this conversation.[/dim]"
        )
    finally:
        # Clear the session-scoped semantic cache entries for this thread.
        # This prevents stale cache entries from accumulating in MongoDB and
        # ensures the cache never crosses session boundaries.
        if bundle.cache is not None:
            bundle.cache.clear_session(thread_id)
            console.print(
                f"[dim]Session cache cleared for thread_id={thread_id}[/dim]"
            )
        bundle.close()


@app.command()
def inspect(user: str = typer.Option(..., help="User identifier.")):
    """Inspect what MongoDB Atlas remembers about a user."""
    settings = load_settings()
    bundle = build_travel_agent(settings, enable_cache=False)
    try:
        store = bundle.store
        prefs = store.search(ns_preferences(user), query="preferences", limit=20)
        trips = store.search(ns_past_trips(user), query="trips", limit=20)

        table = Table(title=f"Stored preferences for {user}")
        table.add_column("Content")
        table.add_column("Stored at")
        for r in prefs:
            v = r.value or {}
            ts = v.get("stored_at")
            ts_str = time.strftime("%Y-%m-%d %H:%M", time.localtime(ts)) if ts else "-"
            table.add_row(v.get("content", ""), ts_str)
        console.print(table)

        table2 = Table(title=f"Past trips for {user}")
        table2.add_column("Summary")
        table2.add_column("Stored at")
        for r in trips:
            v = r.value or {}
            ts = v.get("stored_at")
            ts_str = time.strftime("%Y-%m-%d %H:%M", time.localtime(ts)) if ts else "-"
            table2.add_row(v.get("content", ""), ts_str)
        console.print(table2)

        # Last checkpoint for any thread of this user.
        db = bundle.client[settings.mongo.db]
        latest = list(
            db[settings.mongo.checkpoints]
            .find({"thread_id": {"$regex": f"^thread-{user}-"}})
            .sort([("step", -1)])
            .limit(1)
        )
        if latest:
            console.rule(f"[bold]Last checkpoint for {user}")
            console.print(Pretty({k: v for k, v in latest[0].items() if k != "checkpoint"}))
    finally:
        bundle.close()


@app.command()
def bootstrap():
    """Convenience wrapper around scripts/bootstrap_atlas.py."""
    from scripts import bootstrap_atlas  # type: ignore

    bootstrap_atlas.main()


if __name__ == "__main__":
    app()
