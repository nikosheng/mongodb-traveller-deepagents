"""Activity / attraction search tools."""

from __future__ import annotations

from typing import Optional

from langchain_core.tools import BaseTool, tool
from pymongo import ASCENDING, MongoClient

from ..config import Settings


def make_activity_tools(client: MongoClient, settings: Settings) -> list[BaseTool]:
    coll = client[settings.mongo.db][settings.mongo.activities]

    @tool
    def find_activities(
        city: str,
        interests: Optional[list[str]] = None,
        max_price_usd: Optional[int] = None,
        limit: int = 8,
    ) -> list[dict]:
        """Find activities/attractions in a city.

        Args:
            city: City name.
            interests: Optional list of tags to match (e.g. ["food", "history"]).
            max_price_usd: Optional cap per activity.
            limit: Max number of results.
        """
        query: dict = {"city": city}
        if interests:
            query["tags"] = {"$in": [t.lower() for t in interests]}
        if max_price_usd is not None:
            query["price_usd"] = {"$lte": int(max_price_usd)}
        return list(
            coll.find(query, {"_id": 0})
            .sort([("price_usd", ASCENDING)])
            .limit(int(limit))
        )

    return [find_activities]
