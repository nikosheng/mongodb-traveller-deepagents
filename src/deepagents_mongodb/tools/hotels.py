"""Hotel search tools."""

from __future__ import annotations

from typing import Optional

from langchain_core.tools import BaseTool, tool
from pymongo import ASCENDING, DESCENDING, MongoClient

from ..config import Settings


def make_hotel_tools(client: MongoClient, settings: Settings) -> list[BaseTool]:
    coll = client[settings.mongo.db][settings.mongo.hotels]

    @tool
    def search_hotels(
        city: str,
        max_price_per_night: Optional[int] = None,
        min_rating: float = 0.0,
        hotel_type: Optional[str] = None,
        limit: int = 5,
    ) -> list[dict]:
        """Search hotels in a city, sorted by price ascending.

        Args:
            city: City name, e.g. "Tokyo".
            max_price_per_night: Max nightly price in USD.
            min_rating: Minimum rating 0.0–5.0.
            hotel_type: One of capsule, hostel, business, boutique, ryokan, luxury.
            limit: Max number of results.
        """
        query: dict = {"city": city, "rating": {"$gte": float(min_rating)}}
        if max_price_per_night is not None:
            query["price_per_night"] = {"$lte": int(max_price_per_night)}
        if hotel_type:
            query["type"] = hotel_type.lower()
        return list(
            coll.find(query, {"_id": 0})
            .sort([("price_per_night", ASCENDING)])
            .limit(int(limit))
        )

    @tool
    def top_rated_hotels(city: str, limit: int = 3) -> list[dict]:
        """Return the highest-rated hotels in a city (regardless of price)."""
        return list(
            coll.find({"city": city}, {"_id": 0})
            .sort([("rating", DESCENDING)])
            .limit(int(limit))
        )

    return [search_hotels, top_rated_hotels]
