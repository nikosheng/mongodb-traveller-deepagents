"""Wipe the demo database — collections only, indexes kept.

Usage:
    uv run python scripts/reset_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from rich.console import Console  # noqa: E402

from deepagents_mongodb.config import load_settings  # noqa: E402
from deepagents_mongodb.mongo import get_client  # noqa: E402

console = Console()


def main() -> None:
    settings = load_settings()
    cfg = settings.mongo
    client = get_client(settings)
    try:
        db = client[cfg.db]
        for coll_name in (
            cfg.checkpoints,
            cfg.checkpoint_writes,
            cfg.agent_store,
            cfg.semantic_cache,
            cfg.destinations_kb,
            cfg.flights,
            cfg.hotels,
            cfg.activities,
            cfg.restaurants,
        ):
            n = db[coll_name].delete_many({}).deleted_count
            console.print(f"[yellow]✗[/yellow] {coll_name}: deleted {n} docs")
        console.rule("[bold green]Reset complete")
        console.print("Run `python scripts/bootstrap_atlas.py` to re-seed.")
    finally:
        client.close()


if __name__ == "__main__":
    main()
