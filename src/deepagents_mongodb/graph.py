"""LangGraph-CLI entry point for the travel-planner deepagent.

This module is loaded by ``langgraph dev`` (via ``langgraph.json``) so the
agent can be served over HTTP and driven from
`deep-agents-ui <https://github.com/langchain-ai/deep-agents-ui>`_.

Differences vs. :func:`deepagents_mongodb.agent.build_travel_agent`
(used by the Typer CLI):

* **No** ``MongoDBSaver`` checkpointer is passed in — ``langgraph dev``
  injects its own (in-memory in dev mode, Postgres in LangGraph Platform).
  For durable conversation state, keep using the Typer CLI.
* **No** ``MongoDBStore`` is passed in — ``langgraph dev`` / LangGraph Platform
  injects its own store. The memory tools call ``get_store()`` at runtime,
  so they will automatically use whichever store the platform provides.
  For long-term MongoDB-backed memory, keep using the Typer CLI.
* A module-level symbol ``agent`` is exported; ``langgraph.json`` points at
  ``./src/deepagents_mongodb/graph.py:agent``.

Callers **must** pass ``configurable={"user_id": "<name>"}`` in the run
config so that memory tools can scope data to the right traveler. Example::

    client.runs.create(
        thread_id="...",
        assistant_id="travel_planner",
        input={"messages": [...]},
        config={"configurable": {"user_id": "alice"}},
    )

Lifetime: the ``MongoClient`` used for non-memory tools lives for the
lifetime of the ``langgraph dev`` process. We register an ``atexit`` hook
to close it cleanly on shutdown.
"""

from __future__ import annotations

import atexit
import logging
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)

from deepagents import create_deep_agent
from langgraph.graph.state import CompiledStateGraph

from deepagents_mongodb.agent import ORCHESTRATOR_SYSTEM_PROMPT
from deepagents_mongodb.cache import enable_semantic_cache
from deepagents_mongodb.config import load_settings
from deepagents_mongodb.knowledge.retriever_tool import make_destination_retriever_tool
from deepagents_mongodb.llm import make_chat_model
from deepagents_mongodb.mongo import get_client
from deepagents_mongodb.subagents import build_subagents
from deepagents_mongodb.tools import (
    make_activity_tools,
    make_budget_tools,
    make_flight_tools,
    make_hotel_tools,
    make_memory_tools,
    make_restaurant_tools,
)

logger = logging.getLogger(__name__)


def _build_agent() -> CompiledStateGraph:
    """Wire all components and return a compiled deep agent (no checkpointer, no store)."""
    settings = load_settings(require_azure=True)

    # Session-scoped MongoDB semantic LLM cache — installed as LangChain's
    # global cache.  Each LangGraph thread_id gets its own isolated cache
    # bucket; entries from different sessions never mix.
    # Idempotent: safe even if langgraph dev re-imports this module.
    _cache = enable_semantic_cache(settings)

    client = get_client(settings)
    chat_model = make_chat_model(settings)

    # ----- tools -----
    flight_tools = make_flight_tools(client, settings)
    hotel_tools = make_hotel_tools(client, settings)
    activity_tools = make_activity_tools(client, settings)
    restaurant_tools = make_restaurant_tools(client, settings)
    budget_tools = make_budget_tools(client, settings)
    # Memory tools call get_store() at runtime — no store argument needed.
    memory_tools = make_memory_tools()
    destination_tool = make_destination_retriever_tool(client, settings)

    # ----- subagents -----
    subagents = build_subagents(
        destination_retriever_tool=destination_tool,
        flight_tools=flight_tools,
        hotel_tools=hotel_tools,
        activity_tools=activity_tools,
        restaurant_tools=restaurant_tools,
        budget_tools=budget_tools,
        memory_tools=memory_tools,
    )

    orchestrator_tools: list[Any] = list(memory_tools)

    compiled = create_deep_agent(
        model=chat_model,
        tools=orchestrator_tools,
        system_prompt=ORCHESTRATOR_SYSTEM_PROMPT,
        subagents=subagents,
        # NOTE: no checkpointer= and no store= — langgraph dev provides both.
        name="travel-orchestrator",
    )

    # ----- cleanup on process shutdown -----
    def _cleanup() -> None:
        logger.info("Closing MongoDB client held by the travel-planner graph.")
        try:
            client.close()
        except Exception as exc:  # pragma: no cover - best effort
            logger.warning("MongoClient.close failed: %s", exc)
        # Note: session cache entries are NOT wiped on shutdown intentionally.
        # Each session's entries are isolated by session_id and do not bleed
        # into other sessions.  They can be cleared via:
        #   _cache.clear_session(thread_id)
        # or by running scripts/reset_demo.py.

    atexit.register(_cleanup)

    return compiled


# Top-level symbol referenced by ``langgraph.json``.
agent: CompiledStateGraph = _build_agent()
