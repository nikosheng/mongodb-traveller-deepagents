"""Budgeting tool — pure Python helpers (no MongoDB needed)."""

from __future__ import annotations

from langchain_core.tools import BaseTool, tool
from pymongo import MongoClient

from ..config import Settings


def make_budget_tools(client: MongoClient, settings: Settings) -> list[BaseTool]:
    # Client/settings unused for now — passed for symmetry with other factories.
    del client, settings

    @tool
    def compute_trip_budget(
        flight_total_usd: float,
        hotel_per_night_usd: float,
        nights: int,
        daily_food_usd: float,
        daily_activities_usd: float,
        travellers: int = 1,
    ) -> dict:
        """Compute a trip-cost breakdown.

        Args:
            flight_total_usd: Total flight cost (already summed across travellers).
            hotel_per_night_usd: Nightly hotel rate.
            nights: Number of nights.
            daily_food_usd: Per-day food budget per traveller.
            daily_activities_usd: Per-day activities budget per traveller.
            travellers: Number of travellers.
        """
        hotel_total = float(hotel_per_night_usd) * int(nights)
        food_total = float(daily_food_usd) * int(nights) * int(travellers)
        activities_total = float(daily_activities_usd) * int(nights) * int(travellers)
        grand_total = float(flight_total_usd) + hotel_total + food_total + activities_total
        return {
            "flights": round(float(flight_total_usd), 2),
            "hotel": round(hotel_total, 2),
            "food": round(food_total, 2),
            "activities": round(activities_total, 2),
            "grand_total": round(grand_total, 2),
            "per_traveller": round(grand_total / max(travellers, 1), 2),
            "nights": int(nights),
            "travellers": int(travellers),
        }

    return [compute_trip_budget]
