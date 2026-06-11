"""Restaurant search tools."""

from __future__ import annotations

from typing import Optional

from langchain_core.tools import BaseTool, tool
from pymongo import DESCENDING, MongoClient

from ..config import Settings


def make_restaurant_tools(client: MongoClient, settings: Settings) -> list[BaseTool]:
    coll = client[settings.mongo.db][settings.mongo.restaurants]

    @tool
    def find_restaurants(
        city: str,
        dietary_tags: Optional[list[str]] = None,
        cuisine: Optional[str] = None,
        max_price_level: int = 4,
        limit: int = 6,
    ) -> list[dict]:
        """Find restaurants in a city with optional dietary / cuisine filters.

        Args:
            city: City name.
            dietary_tags: e.g. ["vegan", "vegetarian", "gluten-free"]. Any match.
            cuisine: e.g. "japanese", "italian", "thai".
            max_price_level: 1 (cheap) .. 4 (fine dining).
            limit: Max results.
        """
        query: dict = {"city": city, "price_level": {"$lte": int(max_price_level)}}
        if dietary_tags:
            query["dietary_tags"] = {"$in": [t.lower() for t in dietary_tags]}
        if cuisine:
            query["cuisine"] = cuisine.lower()
        return list(
            coll.find(query, {"_id": 0})
            .sort([("rating", DESCENDING)])
            .limit(int(limit))
        )

    return [find_restaurants]
