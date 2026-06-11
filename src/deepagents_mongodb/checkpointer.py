"""MongoDB-backed LangGraph checkpointer (short-term memory).

The checkpointer persists the full LangGraph state — including messages,
the deepagents virtual filesystem, the todo list, and subagent traces — to
MongoDB after each step. This makes every conversation thread durable and
resumable across processes.
"""

from __future__ import annotations

from langgraph.checkpoint.mongodb import MongoDBSaver
from pymongo import MongoClient

from .config import Settings


def make_checkpointer(client: MongoClient, settings: Settings) -> MongoDBSaver:
    """Build a MongoDBSaver bound to the demo database."""
    return MongoDBSaver(
        client=client,
        db_name=settings.mongo.db,
        checkpoint_collection_name=settings.mongo.checkpoints,
        writes_collection_name=settings.mongo.checkpoint_writes,
    )
