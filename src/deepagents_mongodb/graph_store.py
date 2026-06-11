"""Store factory loaded by langgraph dev / LangGraph Platform.

``langgraph.json`` points ``store.path`` at :func:`create_mongo_store`.
The platform calls this zero-arg callable once at startup, then injects
the returned ``MongoDBStore`` instance as the store for every graph run —
making ``get_store()`` inside any tool return the real Atlas-backed store
instead of the default in-memory one.

This replaces the in-memory store used by ``langgraph dev`` without any
code change to the graph or the tools.
"""

from __future__ import annotations

import atexit

from deepagents_mongodb.config import load_settings
from deepagents_mongodb.store import close_store, make_store


def create_mongo_store():
    """Return a ``MongoDBStore`` backed by MongoDB Atlas.

    Called once by the LangGraph platform at server startup.
    An ``atexit`` handler is registered to close the underlying
    ``MongoClient`` cleanly on shutdown.
    """
    settings = load_settings()
    store = make_store(settings)
    atexit.register(close_store, store)
    return store
