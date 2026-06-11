"""List every Atlas Vector Search index on the cluster.

Useful when the bootstrap fails with
``The maximum number of FTS indexes has been reached for this instance size.``

Usage:
    uv run python scripts/list_search_indexes.py
    uv run python scripts/list_search_indexes.py --drop <db>.<coll>::<index>
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import argparse  # noqa: E402

from rich.console import Console  # noqa: E402
from rich.table import Table  # noqa: E402

from deepagents_mongodb.config import load_settings  # noqa: E402
from deepagents_mongodb.mongo import get_client  # noqa: E402

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--drop",
        action="append",
        default=[],
        help=(
            "Drop the specified search index. Format: '<db>.<coll>::<index>'. "
            "Repeat for multiple."
        ),
    )
    args = parser.parse_args()

    settings = load_settings()
    client = get_client(settings)
    try:
        # Drop requested indexes first.
        for spec in args.drop:
            qual, _, index = spec.partition("::")
            db_name, _, coll_name = qual.partition(".")
            if not (db_name and coll_name and index):
                console.print(f"[red]Bad --drop value: {spec!r}[/red]")
                continue
            coll = client[db_name][coll_name]
            try:
                coll.drop_search_index(index)
                console.print(f"[yellow]dropped[/yellow] {spec}")
            except Exception as exc:
                console.print(f"[red]failed[/red] {spec}: {exc}")

        # Then list what remains.
        table = Table(title="Atlas Vector Search indexes")
        table.add_column("Database")
        table.add_column("Collection")
        table.add_column("Index")
        table.add_column("Type")
        table.add_column("Queryable")

        total = 0
        for db_name in client.list_database_names():
            if db_name in ("admin", "local", "config"):
                continue
            db = client[db_name]
            for coll_name in db.list_collection_names():
                try:
                    idxs = list(db[coll_name].list_search_indexes())
                except Exception:
                    idxs = []
                for i in idxs:
                    total += 1
                    table.add_row(
                        db_name,
                        coll_name,
                        i.get("name", ""),
                        i.get("type", ""),
                        "yes" if i.get("queryable") else "no",
                    )
        console.print(table)
        console.print(f"[bold]Total: {total}[/bold]")
    finally:
        client.close()


if __name__ == "__main__":
    main()
