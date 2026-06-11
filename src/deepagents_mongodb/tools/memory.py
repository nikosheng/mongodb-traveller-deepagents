"""Tools that read & write the shared long-term memory (MongoDBStore).

``user_id`` is read from the LangGraph run config (``configurable["user_id"]``)
at tool-call time — the LLM never needs to supply it as a parameter. This
eliminates the class of bug where the model passes a literal placeholder like
``{{user_id}}`` instead of the real name.

Preference storage design
--------------------------
All preferences for a user are stored in a **single document** keyed
``"profile"`` under the namespace ``("travelers", <user_id>, "preferences")``.
The document value is a flat dict of ``{ topic: preference_text, ... }``.

Saving a preference for an existing topic **overwrites** the old value —
there is never more than one document per user and no stale copies accumulate.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Optional

from langchain_core.tools import BaseTool, tool
from langgraph.config import get_config, get_store

_PROFILE_KEY = "profile"


def _get_user_id() -> str:
    """Return the user_id from the current LangGraph run config.

    The caller must have set ``configurable={"user_id": "<name>"}`` when
    invoking the graph. Raises ``ValueError`` if it is missing so the failure
    is explicit rather than silently writing to a wrong namespace.
    """
    config = get_config()
    user_id = config.get("configurable", {}).get("user_id")
    if not user_id:
        raise ValueError(
            "user_id not found in run config. "
            "Pass configurable={'user_id': '<name>'} when invoking the graph."
        )
    return user_id


def _upsert_profile(store, namespace: tuple, profile: dict) -> None:
    """Write the profile dict directly via the store's pymongo collection.

    We bypass ``store.put()`` because MongoDBStore.batch() has a bug where it
    unconditionally increments the embedding-vector index even when
    ``index=False`` is passed, causing an IndexError when no embeddable
    ``content`` field is present.

    We replicate the exact document schema that MongoDBStore uses so that
    ``store.get()`` can read the document back transparently.
    """
    ns_str = store.sep.join(namespace)
    now = datetime.now(tz=timezone.utc)
    store.collection.update_one(
        filter={"namespace_str": ns_str, "key": _PROFILE_KEY},
        update={
            "$set": {
                "namespace_str": ns_str,
                "value": profile,
                "updated_at": now,
            },
            "$setOnInsert": {
                "namespace": list(namespace),
                "key": _PROFILE_KEY,
                "created_at": now,
            },
        },
        upsert=True,
    )


def make_memory_tools() -> list[BaseTool]:
    @tool
    def save_traveler_preference(topic: str, preference: str) -> str:
        """Persist a traveler preference into MongoDB Atlas (long-term memory).

        All preferences for a user are kept in a single document, keyed by
        topic. Saving a preference for an existing topic overwrites the old
        value — no stale copies accumulate.

        Use this whenever the user reveals a stable preference — diet, budget
        tolerance, seat class, hotel style, accessibility need, etc.

        Args:
            topic: Short snake_case category, e.g. "seat", "diet", "hotel",
                "layovers", "budget". Use the same topic to update an existing
                preference.
            preference: Natural-language statement, e.g. "always window seat".
        """
        user_id = _get_user_id()
        store = get_store()
        ns = ("travelers", user_id, "preferences")

        # Read-modify-write: merge new topic into the existing profile doc.
        existing = store.get(namespace=ns, key=_PROFILE_KEY)
        profile: dict = dict(existing.value) if existing and existing.value else {}
        profile[topic] = preference
        profile["_updated_at"] = time.time()

        _upsert_profile(store, ns, profile)
        return f"Saved preference for {user_id} [{topic}]: {preference!r}"

    @tool
    def recall_traveler_preferences(topics: Optional[list[str]] = None) -> dict:
        """Recall the current traveler's preferences from MongoDB Atlas.

        Returns the full preference profile (all topics) or a subset if
        ``topics`` is specified.

        Args:
            topics: Optional list of topic keys to filter, e.g. ["seat", "diet"].
                    If omitted, all stored preferences are returned.

        Returns:
            A dict of ``{ topic: preference_text }`` (empty dict if nothing stored).
        """
        user_id = _get_user_id()
        store = get_store()
        ns = ("travelers", user_id, "preferences")
        item = store.get(namespace=ns, key=_PROFILE_KEY)
        if not item or not item.value:
            return {}
        profile = {k: v for k, v in item.value.items() if not k.startswith("_")}
        if topics:
            profile = {k: v for k, v in profile.items() if k in topics}
        return profile

    @tool
    def save_past_trip(
        destination: str,
        trip_summary: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        total_budget_usd: Optional[float] = None,
        flight_summary: Optional[str] = None,
        hotel_name: Optional[str] = None,
        hotel_nightly_rate_usd: Optional[float] = None,
        restaurants: Optional[list[str]] = None,
        activities: Optional[list[str]] = None,
    ) -> str:
        """Persist a completed/planned trip with full itinerary details for future recall.

        Store this after every itinerary is finalised so the traveler can ask
        questions like "what hotel did I book for Tokyo?" or "how much did my
        Seoul trip cost?" in a later session.

        Args:
            destination: City and country, e.g. "Tokyo, Japan".
            trip_summary: 2-4 sentence narrative summary of the trip.
            start_date: ISO date string, e.g. "2026-05-10".
            end_date: ISO date string, e.g. "2026-05-15".
            total_budget_usd: Total estimated trip cost in USD (flights + hotel +
                activities + meals combined).
            flight_summary: Brief description of chosen flight(s), e.g.
                "UA837 SFO→HND, $920 round-trip, 11h non-stop".
            hotel_name: Name of the recommended / booked hotel, e.g.
                "Claska Hotel Tokyo".
            hotel_nightly_rate_usd: Nightly room rate in USD.
            restaurants: List of recommended restaurant names, e.g.
                ["Sushi Saito", "Tempura Kondo"].
            activities: List of planned activity names, e.g.
                ["teamLab Planets", "Shinjuku Gyoen"].
        """
        user_id = _get_user_id()
        store = get_store()
        ns = ("travelers", user_id, "past_trips")

        # Build a rich flat text blob that the vector index embeds so semantic
        # search over phrases like "Tokyo hotel budget" surfaces this record.
        parts = [f"Trip to {destination}."]
        if start_date and end_date:
            parts.append(f"Dates: {start_date} to {end_date}.")
        if total_budget_usd is not None:
            parts.append(f"Total budget: ${total_budget_usd:,.0f} USD.")
        if flight_summary:
            parts.append(f"Flight: {flight_summary}.")
        if hotel_name:
            rate_str = (
                f" (${hotel_nightly_rate_usd:,.0f}/night)" if hotel_nightly_rate_usd else ""
            )
            parts.append(f"Hotel: {hotel_name}{rate_str}.")
        if restaurants:
            parts.append(f"Restaurants: {', '.join(restaurants)}.")
        if activities:
            parts.append(f"Activities: {', '.join(activities)}.")
        parts.append(trip_summary)
        content = " ".join(parts)

        value = {
            "content": content,
            "destination": destination,
            "trip_summary": trip_summary,
            "stored_at": time.time(),
        }
        if start_date:
            value["start_date"] = start_date
        if end_date:
            value["end_date"] = end_date
        if total_budget_usd is not None:
            value["total_budget_usd"] = total_budget_usd
        if flight_summary:
            value["flight_summary"] = flight_summary
        if hotel_name:
            value["hotel_name"] = hotel_name
        if hotel_nightly_rate_usd is not None:
            value["hotel_nightly_rate_usd"] = hotel_nightly_rate_usd
        if restaurants:
            value["restaurants"] = restaurants
        if activities:
            value["activities"] = activities

        store.put(namespace=ns, key=_key(destination + (start_date or "")), value=value)
        return (
            f"Saved past trip for {user_id}: {destination}"
            + (f", ${total_budget_usd:,.0f} total" if total_budget_usd else "")
            + (f", hotel: {hotel_name}" if hotel_name else "")
            + "."
        )

    @tool
    def recall_past_trips(query: str, limit: int = 3) -> list[dict]:
        """Semantically recall past trips for the current traveler.

        Returns structured records (destination, budget, hotel, restaurants,
        activities, dates, summary) so you can answer specific questions like
        "what was my Tokyo hotel?" or "how much did my last trip cost?".

        Args:
            query: Natural-language query, e.g. "previous Japan trips" or
                "Tokyo hotel budget".
            limit: Max number of past trips to return.
        """
        user_id = _get_user_id()
        store = get_store()
        ns = ("travelers", user_id, "past_trips")
        results = store.search(ns, query=query, limit=int(limit))
        trips = []
        for r in results:
            if not r.value:
                continue
            # Return the full structured record; omit internal fields.
            record = {k: v for k, v in r.value.items() if k != "stored_at"}
            trips.append(record)
        return trips

    @tool
    def save_shared_note(topic: str, note: str) -> str:
        """Save a cross-agent shared note (visible to every subagent).

        Args:
            topic: A short topic key, e.g. "tokyo_pricing_may2026".
            note: The note content.
        """
        store = get_store()
        ns = ("agents", "shared_notes")
        store.put(
            namespace=ns,
            key=_key(topic + note),
            value={"topic": topic, "content": note, "stored_at": time.time()},
        )
        return f"Saved shared note about {topic!r}."

    @tool
    def recall_shared_notes(query: str, limit: int = 5) -> list[dict]:
        """Semantically search shared notes left by other subagents."""
        store = get_store()
        ns = ("agents", "shared_notes")
        results = store.search(ns, query=query, limit=int(limit))
        return [r.value for r in results if r.value]

    return [
        save_traveler_preference,
        recall_traveler_preferences,
        save_past_trip,
        recall_past_trips,
        save_shared_note,
        recall_shared_notes,
    ]


def _key(text: str) -> str:
    """Stable short key derived from content (first 48 chars, url-safe)."""
    import hashlib
    return hashlib.sha1(text.encode()).hexdigest()[:16]
