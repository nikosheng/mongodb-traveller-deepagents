"""Flight search tools — query the ``flights`` collection in MongoDB."""

from __future__ import annotations

from typing import Optional

from langchain_core.tools import BaseTool, tool
from pymongo import ASCENDING, MongoClient

from ..config import Settings


def make_flight_tools(client: MongoClient, settings: Settings) -> list[BaseTool]:
    coll = client[settings.mongo.db][settings.mongo.flights]

    @tool
    def search_flights(
        origin: str,
        destination: str,
        depart_date: str,
        max_price_usd: Optional[int] = None,
        limit: int = 5,
    ) -> list[dict]:
        """Search flights from origin city to destination city on a given date.

        Args:
            origin: City name (e.g. "San Francisco").
            destination: City name (e.g. "Tokyo").
            depart_date: ISO date string YYYY-MM-DD.
            max_price_usd: Optional price cap.
            limit: Max number of results.

        Returns:
            A list of flight documents, cheapest first.
        """
        query: dict = {
            "origin": origin,
            "destination": destination,
            "depart_date": depart_date,
        }
        if max_price_usd is not None:
            query["price_usd"] = {"$lte": int(max_price_usd)}
        cursor = (
            coll.find(query, {"_id": 0})
            .sort([("price_usd", ASCENDING)])
            .limit(int(limit))
        )
        return list(cursor)

    @tool
    def cheapest_flights_in_window(
        origin: str,
        destination: str,
        start_date: str,
        end_date: str,
        limit: int = 5,
    ) -> list[dict]:
        """Find the cheapest flights in a date range (inclusive).

        Args:
            origin: City name.
            destination: City name.
            start_date: ISO date string (inclusive).
            end_date: ISO date string (inclusive).
            limit: Max number of results.
        """
        cursor = (
            coll.find(
                {
                    "origin": origin,
                    "destination": destination,
                    "depart_date": {"$gte": start_date, "$lte": end_date},
                },
                {"_id": 0},
            )
            .sort([("price_usd", ASCENDING)])
            .limit(int(limit))
        )
        return list(cursor)

    return [search_flights, cheapest_flights_in_window]
