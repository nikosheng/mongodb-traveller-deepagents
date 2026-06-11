"""MongoDB-backed LangGraph Store — long-term, semantically-searchable memory.

The Store is shared by the orchestrator and every subagent, so anything any
agent writes (e.g. "Alice is vegetarian, prefers boutique hotels") is
recallable by the others — even in a future, brand-new conversation.

Namespace conventions used by this demo
---------------------------------------
* ``("travelers", user_id, "preferences")`` — durable user preferences
* ``("travelers", user_id, "past_trips")``  — semantic summaries of past trips
* ``("agents", "shared_notes")``            — cross-subagent scratch facts
"""

from __future__ import annotations

from typing import Sequence

from langgraph.store.mongodb import MongoDBStore, create_vector_index_config
from pymongo import MongoClient

from .config import EMBEDDING_DIMS, Settings
from .llm import make_embeddings


PREFERENCES_NS = ("travelers", "{user_id}", "preferences")
PAST_TRIPS_NS = ("travelers", "{user_id}", "past_trips")
SHARED_NOTES_NS = ("agents", "shared_notes")


def make_store(settings: Settings, *, client: MongoClient | None = None) -> MongoDBStore:
    """Build a ``MongoDBStore`` bound to the demo cluster.

    We construct the store directly from a ``Collection`` rather than via
    ``MongoDBStore.from_conn_string`` because the latter is a context
    manager that closes its internal ``MongoClient`` on exit. We want the
    store to live for the duration of the agent run, so the caller owns
    the ``MongoClient`` and passes it (or we create one we keep alive on
    the returned object).
    """
    from .mongo import get_client  # local import to avoid cycle

    own_client = client
    if own_client is None:
        own_client = get_client(settings)

    index_config = create_vector_index_config(
        embed=make_embeddings(settings),
        dims=EMBEDDING_DIMS,
        fields=["content"],
        filters=["namespace"],
        name=settings.mongo.agent_store_index,
        relevance_score_fn="cosine",
    )

    db = own_client[settings.mongo.db]
    if settings.mongo.agent_store not in db.list_collection_names():
        db.create_collection(settings.mongo.agent_store)
    coll = db[settings.mongo.agent_store]

    store = MongoDBStore(collection=coll, index_config=index_config)
    # Stash the client so the caller can close it (and so it isn't GCed).
    store._owned_client = own_client  # type: ignore[attr-defined]
    return store


def close_store(store: MongoDBStore) -> None:
    """Close the client owned by a store created with ``make_store``."""
    client = getattr(store, "_owned_client", None)
    if client is not None:
        try:
            client.close()
        except Exception:
            pass


def ns_preferences(user_id: str) -> Sequence[str]:
    return ("travelers", user_id, "preferences")


def ns_past_trips(user_id: str) -> Sequence[str]:
    return ("travelers", user_id, "past_trips")
