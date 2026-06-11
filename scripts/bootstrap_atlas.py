"""Idempotent bootstrap: create collections, vector indexes, and seed data.

Usage:
    uv run python scripts/bootstrap_atlas.py

Steps performed:
    1. Connect to MongoDB Atlas.
    2. Create regular indexes on operational collections.
    3. Create Atlas Vector Search indexes for:
         - agent_store      (long-term memory)
         - semantic_cache   (LLM cache)
         - destinations_kb  (RAG)
    4. Wait until those vector indexes are queryable.
    5. Insert mocked flights/hotels/activities/restaurants.
    6. Embed and load the destinations knowledge base.

Safe to re-run — every step is idempotent.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from rich.console import Console  # noqa: E402

from deepagents_mongodb.config import load_settings  # noqa: E402
from deepagents_mongodb.knowledge.ingest import (  # noqa: E402
    ingest_knowledge_base,
    ingest_operational,
)
from deepagents_mongodb.mongo import (  # noqa: E402
    ensure_operational_indexes,
    ensure_vector_indexes,
    get_client,
    wait_for_vector_indexes_ready,
)

console = Console()


def main() -> None:
    settings = load_settings()
    console.rule("[bold magenta]Bootstrapping MongoDB Atlas")
    console.print(f"DB: [bold]{settings.mongo.db}[/bold]")

    client = get_client(settings)
    try:
        # Sanity check.
        client.admin.command("ping")
        console.print("[green]✓[/green] Connected to Atlas")

        console.print("• Creating operational indexes …")
        ensure_operational_indexes(client, settings.mongo)
        console.print("[green]✓[/green] Operational indexes ready")

        console.print("• Creating Atlas Vector Search indexes …")
        failures = ensure_vector_indexes(client, settings.mongo)
        if failures:
            console.print(
                "[red]✗ Some vector indexes could not be created.[/red] "
                "This usually means your Atlas cluster's search-index quota "
                "is full (M0/M2/M5 allow 3 search indexes per cluster)."
            )
            for qual, msg in failures:
                console.print(f"   - {qual}: {msg.splitlines()[0]}")
            console.print(
                "[yellow]Continuing without these indexes. List existing "
                "indexes and drop unused ones via\n"
                "  db.<collection>.dropSearchIndex(\"<index-name>\")\n"
                "then re-run this script.[/yellow]"
            )
        else:
            console.print("[green]✓[/green] Vector index models submitted")

        console.print("• Waiting for vector indexes to be queryable (up to 3 min) …")
        try:
            wait_for_vector_indexes_ready(client, settings.mongo)
            console.print("[green]✓[/green] All vector indexes are queryable")
        except TimeoutError as exc:
            console.print(f"[yellow]⚠ {exc}[/yellow]")

        console.print("• Inserting operational data (flights/hotels/activities/restaurants) …")
        counts = ingest_operational(client, settings)
        for name, n in counts.items():
            console.print(f"   - {name}: {n} docs")

        console.print("• Embedding + loading destination knowledge base …")
        n_kb = ingest_knowledge_base(client, settings)
        console.print(f"   - destinations_kb: {n_kb} docs")

        console.rule("[bold green]Bootstrap complete")
        console.print(
            "Next: try `uv run deepagents-mongodb plan --user alice --query "
            "\"Plan a 5-day Tokyo trip in May 2026, vegetarian, budget $4000\"`"
        )
    finally:
        client.close()


if __name__ == "__main__":
    main()
