"""SubAgent definitions for the travel planner.

Each subagent gets a focused system prompt and *only* the tools it needs —
this is the "least-privilege" pattern recommended by the deepagents docs and
keeps subagent context windows clean.
"""

from __future__ import annotations

from typing import Sequence

from langchain_core.tools import BaseTool


def build_subagents(
    *,
    destination_retriever_tool: BaseTool,
    flight_tools: Sequence[BaseTool],
    hotel_tools: Sequence[BaseTool],
    activity_tools: Sequence[BaseTool],
    restaurant_tools: Sequence[BaseTool],
    budget_tools: Sequence[BaseTool],
    memory_tools: Sequence[BaseTool],
) -> list[dict]:
    """Return the list of SubAgent dictionaries for ``create_deep_agent``.

    ``memory_tools`` are *not* given to every subagent — only those that
    benefit from semantic recall (the activity planner consults user prefs;
    the restaurant recommender does too). The orchestrator holds all of them.
    """
    # Memory tools live only on the orchestrator. Subagents receive any
    # relevant memory hits via their `task` description — this keeps the
    # number of embedding calls (which hit Voyage's free-tier rate limit)
    # to a minimum: the orchestrator recalls once, then dispatches.
    del memory_tools

    destination_researcher = {
        "name": "destination_researcher",
        "description": (
            "Use this subagent to gather facts about a destination city — "
            "best months to visit, visa rules, neighborhoods to stay in, "
            "and transit tips. Input: a question about the city."
        ),
        "system_prompt": (
            "You are a destination research specialist.\n"
            "Use `search_destination_knowledge` exactly once to look up "
            "facts in the MongoDB Atlas knowledge base for the requested "
            "city. Return a tight 4-6 sentence summary covering best season, "
            "visa, top neighborhoods to stay in, and transit. No preamble."
        ),
        "tools": [destination_retriever_tool],
    }

    flight_researcher = {
        "name": "flight_researcher",
        "description": (
            "Searches flights between two cities for given dates and budget. "
            "Returns the top options."
        ),
        "system_prompt": (
            "You are a flight research specialist.\n"
            "Call `search_flights` or `cheapest_flights_in_window` to find "
            "options that match the trip dates and budget. "
            "Return at most 3 best options as a bulleted list with "
            "airline, flight number, date, price, and duration. "
            "Pick the best option per the user's budget and call it out."
        ),
        "tools": [*flight_tools],
    }

    hotel_researcher = {
        "name": "hotel_researcher",
        "description": (
            "Finds hotels in a city for a given budget and style. Returns the "
            "top picks with nightly rates and ratings."
        ),
        "system_prompt": (
            "You are a hotel research specialist.\n"
            "Call `search_hotels` (with `max_price_per_night`, `min_rating`, "
            "`hotel_type` as appropriate) or `top_rated_hotels` to find good "
            "options. Return at most 3 hotels as a bulleted list with name, "
            "type, nightly price, rating, and notable amenities. "
            "Recommend a specific pick that fits the user's budget."
        ),
        "tools": [*hotel_tools],
    }

    activity_planner = {
        "name": "activity_planner",
        "description": (
            "Plans activities and attractions for the trip based on the "
            "traveler's stated interests (passed in by the orchestrator)."
        ),
        "system_prompt": (
            "You are an activities/attractions planner.\n"
            "The orchestrator will tell you the city and the traveler's "
            "interests/preferences in the task description. Call "
            "`find_activities` (filtered by relevant `interests` tags). "
            "Pick 4-6 activities that create a coherent itinerary (mix of "
            "food, culture, outdoor). Return a numbered list with price "
            "and duration. No preamble."
        ),
        "tools": [*activity_tools],
    }

    restaurant_recommender = {
        "name": "restaurant_recommender",
        "description": (
            "Recommends restaurants based on dietary restrictions, cuisine "
            "preferences, and budget — provided by the orchestrator."
        ),
        "system_prompt": (
            "You are a restaurant recommendation specialist.\n"
            "The orchestrator will tell you the city, dietary tags, and "
            "max price level in the task description. Call "
            "`find_restaurants` with those filters. Return 4-6 picks with "
            "name, cuisine, price level, and rating. Prefer higher "
            "ratings. No preamble."
        ),
        "tools": [*restaurant_tools],
    }

    itinerary_compiler = {
        "name": "itinerary_compiler",
        "description": (
            "Compiles the final day-by-day itinerary including budget "
            "breakdown. Writes or updates `itinerary.md` in the agent's "
            "virtual filesystem."
        ),
        "system_prompt": (
            "You are the itinerary compiler.\n"
            "You will be given the outputs of the flight, hotel, activity, "
            "and restaurant subagents. Your job is to:\n"
            "1. Call `compute_trip_budget` with the relevant numbers.\n"
            "2. Produce a day-by-day plan (Day 1, Day 2, …) interleaving "
            "activities and restaurants, respecting opening hours and "
            "travel times.\n"
            "3. Persist the final markdown (flights, hotel, day-by-day "
            "plan, budget table) to `itinerary.md` using the correct tool:\n"
            "   - If `itinerary.md` does NOT yet exist: call `write_file` "
            "with path=`itinerary.md`.\n"
            "   - If `itinerary.md` ALREADY EXISTS (e.g. this is an update "
            "request): call `read_file` first to get the current content, "
            "then call `edit_file` to replace the ENTIRE body with the new "
            "content. Do NOT call `write_file` on an existing file — it "
            "will fail with an error. Use `edit_file` with old_string set "
            "to the full current content and new_string set to the complete "
            "updated itinerary.\n"
            "Return a 5-line executive summary as your final answer."
        ),
        "tools": [*budget_tools],
    }

    return [
        destination_researcher,
        flight_researcher,
        hotel_researcher,
        activity_planner,
        restaurant_recommender,
        itinerary_compiler,
    ]
