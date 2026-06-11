"""MongoDB client + index bootstrap helpers.

This module is intentionally thin — it just constructs a `MongoClient` from
the settings and exposes helpers for creating the regular and Atlas Vector
Search indexes that the rest of the demo expects.
"""

from __future__ import annotations

import time
from typing import Iterable

import certifi
from pymongo import ASCENDING, MongoClient
from pymongo.collection import Collection
from pymongo.errors import OperationFailure
from pymongo.operations import SearchIndexModel

from .config import EMBEDDING_DIMS, MongoConfig, Settings


def get_client(settings: Settings) -> MongoClient:
    """Return a connected MongoClient.

    Uses ``certifi``'s CA bundle so connections work out-of-the-box on
    macOS Python installs that don't ship a system trust store.
    """
    return MongoClient(settings.mongo.uri, tlsCAFile=certifi.where())


# ---------------------------------------------------------------------------
# Regular B-tree indexes for the operational collections
# ---------------------------------------------------------------------------
def ensure_operational_indexes(client: MongoClient, cfg: MongoConfig) -> None:
    db = client[cfg.db]

    db[cfg.flights].create_index(
        [("origin", ASCENDING), ("destination", ASCENDING), ("depart_date", ASCENDING)]
    )
    db[cfg.hotels].create_index([("city", ASCENDING), ("price_per_night", ASCENDING)])
    db[cfg.activities].create_index([("city", ASCENDING), ("tags", ASCENDING)])
    db[cfg.restaurants].create_index(
        [("city", ASCENDING), ("dietary_tags", ASCENDING), ("price_level", ASCENDING)]
    )


# ---------------------------------------------------------------------------
# Atlas Vector Search indexes
# ---------------------------------------------------------------------------
def _vector_index_definition(
    *,
    path: str = "embedding",
    dims: int = EMBEDDING_DIMS,
    similarity: str = "cosine",
    filters: Iterable[str] = (),
) -> dict:
    fields: list[dict] = [
        {
            "type": "vector",
            "path": path,
            "numDimensions": dims,
            "similarity": similarity,
        }
    ]
    for f in filters:
        fields.append({"type": "filter", "path": f})
    return {"fields": fields}


def _ensure_collection(collection: Collection) -> None:
    """Create the collection if it doesn't exist.

    Atlas refuses ``createSearchIndexes`` against a non-existent collection,
    so we materialise it up-front via an idempotent ``createCollection``.
    """
    db = collection.database
    if collection.name not in db.list_collection_names():
        try:
            db.create_collection(collection.name)
        except OperationFailure as exc:
            # Race: another process created it. Ignore.
            if "already exists" not in str(exc).lower():
                raise


def _ensure_vector_index(
    collection: Collection,
    index_name: str,
    definition: dict,
) -> None:
    """Create the named Atlas Vector Search index if it does not exist."""
    _ensure_collection(collection)
    existing = {idx["name"] for idx in collection.list_search_indexes()}
    if index_name in existing:
        return
    model = SearchIndexModel(
        definition=definition,
        name=index_name,
        type="vectorSearch",
    )
    try:
        collection.create_search_index(model=model)
    except OperationFailure as exc:
        # If we hit "index already exists", we're done.
        if "already exists" in str(exc).lower():
            return
        raise


def _update_vector_index(
    collection: Collection,
    index_name: str,
    definition: dict,
) -> None:
    """Drop and recreate a named Atlas Vector Search index.

    Atlas does not support adding new filter fields to an existing index
    in-place.  This helper drops the old index (if present) and creates a
    fresh one with the new definition.  The collection remains queryable
    via the old index until the new one finishes building.

    Typical use: adding ``session_id`` as a filter field to the
    ``semantic_cache_vector_index`` after the initial bootstrap.
    """
    _ensure_collection(collection)
    existing = {idx["name"] for idx in collection.list_search_indexes()}
    if index_name in existing:
        collection.drop_search_index(index_name)
        # Wait briefly so Atlas registers the drop before we recreate.
        time.sleep(2)
    model = SearchIndexModel(
        definition=definition,
        name=index_name,
        type="vectorSearch",
    )
    collection.create_search_index(model=model)


def ensure_vector_indexes(
    client: MongoClient, cfg: MongoConfig
) -> list[tuple[str, str]]:
    """Create the three vector indexes used by the demo.

    Returns a list of ``(qualified_name, error_message)`` tuples for any
    indexes that could not be created. An empty list means everything is
    fine. The most common failure is Atlas search-index quota; if you hit
    it, list and drop unused indexes via ``db.collection.drop_search_index``.
    """
    db = client[cfg.db]

    targets = [
        # agent_store: MongoDBStore wraps each value under a `value` field
        # and requires `namespace_prefix` (joined namespace) as a filter.
        (
            db[cfg.agent_store],
            cfg.agent_store_index,
            _vector_index_definition(filters=["namespace_prefix"]),
        ),
        (
            db[cfg.semantic_cache],
            cfg.semantic_cache_index,
            # MongoDBAtlasSemanticCache filters every query by llm_string
            # (a hash of the model name + params) so cache entries from
            # different models don't collide.  Without this filter field
            # the aggregation raises "Path 'llm_string' needs to be indexed
            # as filter".
            # session_id scopes every lookup to the current LangGraph
            # thread_id so sessions cannot see each other's cache entries.
            _vector_index_definition(filters=["llm_string", "session_id"]),
        ),
        (
            db[cfg.destinations_kb],
            cfg.destinations_kb_index,
            _vector_index_definition(filters=["country", "region"]),
        ),
    ]

    failures: list[tuple[str, str]] = []
    for coll, name, definition in targets:
        try:
            _ensure_vector_index(coll, name, definition)
        except OperationFailure as exc:
            failures.append((f"{coll.database.name}.{coll.name}::{name}", str(exc)))
    return failures


def update_semantic_cache_index(
    client: MongoClient, cfg: MongoConfig
) -> None:
    """Drop and recreate the semantic cache vector index.

    Call this once after adding ``session_id`` as a new filter field.
    The operation is safe to re-run: if the index already has the correct
    definition it will be dropped and rebuilt (takes ~30 s on M0 clusters).
    """
    db = client[cfg.db]
    definition = _vector_index_definition(filters=["llm_string", "session_id"])
    _update_vector_index(db[cfg.semantic_cache], cfg.semantic_cache_index, definition)


def wait_for_vector_indexes_ready(
    client: MongoClient,
    cfg: MongoConfig,
    timeout_seconds: int = 180,
) -> None:
    """Block until all vector indexes are queryable."""
    db = client[cfg.db]
    targets = [
        (db[cfg.agent_store], cfg.agent_store_index),
        (db[cfg.semantic_cache], cfg.semantic_cache_index),
        (db[cfg.destinations_kb], cfg.destinations_kb_index),
    ]
    deadline = time.time() + timeout_seconds
    pending = list(targets)
    while pending and time.time() < deadline:
        still_pending = []
        for coll, name in pending:
            indexes = list(coll.list_search_indexes(name))
            if indexes and indexes[0].get("queryable"):
                continue
            still_pending.append((coll, name))
        pending = still_pending
        if pending:
            time.sleep(5)
    if pending:
        names = ", ".join(name for _, name in pending)
        raise TimeoutError(
            f"Vector indexes not ready in {timeout_seconds}s: {names}. "
            "They may still be building — try again in a minute."
        )
