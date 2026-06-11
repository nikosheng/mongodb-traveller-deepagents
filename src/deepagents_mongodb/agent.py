"""Assemble the travel-planner deep agent.

This module is the integration seam: it constructs the chat model, the
MongoDB-backed checkpointer + store, all subagents and tools, and wires
them into a single ``create_deep_agent`` call.

The returned agent is a fully-compiled LangGraph ``CompiledStateGraph``;
callers invoke it with the standard LangGraph API.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from deepagents import create_deep_agent
from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.graph.state import CompiledStateGraph
from langgraph.store.mongodb import MongoDBStore
from pymongo import MongoClient

from .cache import SessionScopedSemanticCache
from .checkpointer import make_checkpointer
from .config import Settings, load_settings
from .knowledge.retriever_tool import make_destination_retriever_tool
from .llm import make_chat_model
from .mongo import get_client
from .store import close_store, make_store
from .subagents import build_subagents
from .tools import (
    make_activity_tools,
    make_budget_tools,
    make_flight_tools,
    make_hotel_tools,
    make_memory_tools,
    make_restaurant_tools,
)

ORCHESTRATOR_SYSTEM_PROMPT = """\
You are the **Travel Planning Orchestrator** for an AI travel-planning service.

You coordinate a team of specialised subagents (don't try to do their work
yourself). Your job is to:

1. **Recall context first.** At the start of every plan, call
   `recall_traveler_preferences` to pull what we already know about this
   traveler from MongoDB Atlas. Call `recall_past_trips` once if relevant.
   Stop after this — do NOT make memory calls in the middle of the plan;
   pass any recalled facts to subagents via their task descriptions instead.
2. **Plan.** Use `write_todos` to outline the work. Each subagent should
   be called at most once.
3. **Delegate** by calling the `task` tool with the appropriate subagent
   name and a concise task description that INCLUDES every fact the
   subagent needs (city, dates, budget, dietary tags, interests, etc.).
   Recommended order:
     - `destination_researcher` (visa, neighborhoods, transit)
     - `flight_researcher`
     - `hotel_researcher`
     - `activity_planner` — pass the traveler's interests/preferences
     - `restaurant_recommender` — pass dietary tags + budget
     - `itinerary_compiler` (LAST — give it a synthesis of the others'
       outputs as the task description)
4. **Learn.** When the user reveals a stable preference (diet, budget
   tolerance, accommodation style, accessibility needs), call
   `save_traveler_preference` once near the end.
5. **Save the itinerary details — ALWAYS, including after updates.** Every
   time the `itinerary_compiler` finishes (whether this is the initial plan
   OR an update to an existing plan), call `save_past_trip` with the full
   structured details extracted from the subagents' outputs. This overwrites
   the previous record for the same destination+date so the traveler always
   has the latest version. You MUST populate every field you have data for:
   - `destination`: city and country (e.g. "Tokyo, Japan")
   - `trip_summary`: 2-4 sentence narrative reflecting the FINAL itinerary
   - `start_date` / `end_date`: ISO dates from the trip
   - `total_budget_usd`: the total dollar amount from `compute_trip_budget`
   - `flight_summary`: airline, flight number, price, duration
   - `hotel_name`: exact name of the recommended hotel
   - `hotel_nightly_rate_usd`: nightly rate in USD
   - `restaurants`: list of restaurant names recommended (UPDATED list)
   - `activities`: list of activity names planned (UPDATED list)
   Skipping this call means the traveler cannot recall the updated details
   in a future session — this is a critical step, not optional.
6. **Reply.** Your final response should be a concise traveler-facing
   summary (≤ 200 words) referencing the `itinerary.md` file that the
   `itinerary_compiler` writes.

Note: memory tools automatically know which traveler you are serving — you
do NOT need to supply a user_id argument to any memory tool.
"""


@dataclass
class TravelAgent:
    """Bundle of agent + the resources it owns (so we can clean up)."""

    agent: CompiledStateGraph
    client: MongoClient
    checkpointer: MongoDBSaver
    store: MongoDBStore
    cache: Optional[SessionScopedSemanticCache] = field(default=None)

    def close(self) -> None:
        # The store owns its own MongoClient (see ``make_store``).
        close_store(self.store)
        try:
            self.client.close()
        except Exception:
            pass


def build_travel_agent(
    settings: Settings | None = None,
    *,
    enable_cache: bool = True,
) -> TravelAgent:
    """Build the full travel-planner deep agent wired to MongoDB Atlas."""
    settings = settings or load_settings(require_azure=True)

    # Optionally install the session-scoped semantic cache as LangChain's
    # global LLM cache.  We capture the instance so CLI callers can invoke
    # clear_session() when the conversation thread ends.
    _cache: Optional[SessionScopedSemanticCache] = None
    if enable_cache:
        from .cache import enable_semantic_cache  # local import to avoid cycles
        _cache = enable_semantic_cache(settings)

    client = get_client(settings)
    chat_model = make_chat_model(settings)

    # ----- persistence + memory -----
    checkpointer = make_checkpointer(client, settings)
    # The Store owns a *separate* MongoClient so its lifetime is independent
    # of this one and isn't tied to context-manager semantics.
    store = make_store(settings)

    # ----- tools -----
    flight_tools = make_flight_tools(client, settings)
    hotel_tools = make_hotel_tools(client, settings)
    activity_tools = make_activity_tools(client, settings)
    restaurant_tools = make_restaurant_tools(client, settings)
    budget_tools = make_budget_tools(client, settings)
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

    # ----- orchestrator -----
    # Top-level tools the orchestrator can call directly. Subagents'
    # specialised tools are NOT exposed here — the orchestrator must
    # delegate via the `task` tool.
    orchestrator_tools: list[Any] = list(memory_tools)

    agent = create_deep_agent(
        model=chat_model,
        tools=orchestrator_tools,
        system_prompt=ORCHESTRATOR_SYSTEM_PROMPT,
        subagents=subagents,
        checkpointer=checkpointer,
        store=store,
        name="travel-orchestrator",
    )

    return TravelAgent(
        agent=agent,
        client=client,
        checkpointer=checkpointer,
        store=store,
        cache=_cache,
    )
