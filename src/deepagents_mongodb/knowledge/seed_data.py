"""Mocked but self-consistent travel dataset.

Everything is deterministically generated from the cities defined here so a
re-run produces the same data. The data is intentionally compact (a handful
of items per city) so the demo runs quickly while still feeling realistic.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Iterable

# (city, country, iata, region)
CITIES: list[tuple[str, str, str, str]] = [
    ("Tokyo", "Japan", "HND", "Asia"),
    ("Kyoto", "Japan", "ITM", "Asia"),
    ("Osaka", "Japan", "KIX", "Asia"),
    ("Seoul", "South Korea", "ICN", "Asia"),
    ("Bangkok", "Thailand", "BKK", "Asia"),
    ("Singapore", "Singapore", "SIN", "Asia"),
    ("Paris", "France", "CDG", "Europe"),
    ("Rome", "Italy", "FCO", "Europe"),
    ("Barcelona", "Spain", "BCN", "Europe"),
    ("London", "United Kingdom", "LHR", "Europe"),
    ("New York", "United States", "JFK", "Americas"),
    ("San Francisco", "United States", "SFO", "Americas"),
    ("Vancouver", "Canada", "YVR", "Americas"),
    ("Mexico City", "Mexico", "MEX", "Americas"),
    ("Sydney", "Australia", "SYD", "Oceania"),
    ("Hong Kong", "China", "HKG", "Asia"),
    ("Shenzhen", "China", "SZX", "Asia"),
]


# ---------------------------------------------------------------------------
# Destination knowledge base (used by the RAG subagent)
# ---------------------------------------------------------------------------
DESTINATION_FACTS: dict[str, dict[str, str]] = {
    "Tokyo": {
        "overview": (
            "Tokyo is Japan's pulsating capital, blending ultra-modern districts "
            "like Shibuya and Shinjuku with historic temples in Asakusa. Best "
            "visited in late March–April for cherry blossoms and October–November "
            "for foliage. JR Yamanote Line connects the major hubs."
        ),
        "visa": "US/EU/UK passport holders get 90 days visa-free as of 2024.",
        "neighborhoods": (
            "Shibuya for nightlife, Ginza for luxury shopping, Asakusa for "
            "traditional Tokyo, Shimokitazawa for indie cafes, Akihabara for "
            "anime and electronics."
        ),
        "transit": "Suica/Pasmo IC card. Avoid Yamanote line at 8–9am rush hour.",
    },
    "Kyoto": {
        "overview": (
            "Kyoto preserves traditional Japan with 17 UNESCO sites, 1,600 "
            "temples and 400 shrines. Fushimi Inari and Kinkaku-ji are essentials. "
            "Hot and humid in summer; spectacular in mid-November."
        ),
        "visa": "Same 90-day visa-waiver as Tokyo.",
        "neighborhoods": "Gion for geisha culture, Arashiyama for bamboo, Higashiyama for traditional streets.",
        "transit": "Buses + bicycles preferred over the limited subway. JR Pass usable to Nara/Osaka.",
    },
    "Osaka": {
        "overview": (
            "Osaka is Japan's kitchen — takoyaki, okonomiyaki, kushikatsu. "
            "Dotonbori at night is iconic. Compact and walkable. Day-trip "
            "distance from Kyoto, Nara, Kobe."
        ),
        "visa": "Same 90-day visa-waiver as Tokyo.",
        "neighborhoods": "Namba/Dotonbori for food, Umeda for shopping, Tennoji for temples.",
        "transit": "ICOCA card, Midosuji subway line covers most of the city.",
    },
    "Seoul": {
        "overview": (
            "Seoul mixes Joseon palaces (Gyeongbokgung), K-pop districts "
            "(Hongdae, Gangnam), and the food markets of Gwangjang. Excellent "
            "metro coverage. Spring (April) and autumn (October) are ideal."
        ),
        "visa": "US/EU/UK passports: K-ETA required before flight, 90 days visa-free.",
        "neighborhoods": "Myeongdong (shopping), Itaewon (international), Bukchon (hanok), Hongdae (youth).",
        "transit": "T-money card. Metro is among the world's best.",
    },
    "Bangkok": {
        "overview": (
            "Bangkok is hot, chaotic, delicious. Grand Palace and Wat Pho by day; "
            "rooftop bars and night markets after dark. Avoid April (Songkran heat) "
            "unless you love the festival."
        ),
        "visa": "30 days visa-exempt for most western passports.",
        "neighborhoods": "Sukhumvit (international), Silom (business), Khao San (backpackers), Thonburi (canals).",
        "transit": "BTS Skytrain + MRT. Grab rideshare is widely used.",
    },
    "Singapore": {
        "overview": (
            "Singapore is hyper-organised, multi-cultural, and food-obsessed. "
            "Hawker centres (Maxwell, Old Airport Road) serve Michelin-quality "
            "meals for $5. Year-round warm and humid; brief afternoon storms."
        ),
        "visa": "90 days visa-free for most western passports.",
        "neighborhoods": "Marina Bay, Chinatown, Little India, Kampong Glam, Tiong Bahru.",
        "transit": "EZ-Link card. MRT is fast, clean, and goes to the airport.",
    },
    "Paris": {
        "overview": (
            "Paris rewards slow walking: each arrondissement has its own "
            "personality. Museums (Louvre, Orsay, Pompidou) require advance "
            "booking. May–June and September are ideal."
        ),
        "visa": "Schengen rules; 90/180 visa-free for US/UK passports.",
        "neighborhoods": "Marais (4th), Saint-Germain (6th), Montmartre (18th), Canal Saint-Martin (10th).",
        "transit": "Navigo Easy card. Metro line 1 covers the main sights.",
    },
    "Rome": {
        "overview": (
            "Rome is an open-air museum. Vatican, Colosseum, and Pantheon are "
            "musts. Skip August (locals leave, heat dominates); aim for May or "
            "October."
        ),
        "visa": "Schengen 90/180.",
        "neighborhoods": "Trastevere (food), Monti (boho), Prati (Vatican-adjacent), Testaccio (authentic).",
        "transit": "Walking + metro Line A/B. Watch for pickpockets on bus 64.",
    },
    "Barcelona": {
        "overview": (
            "Barcelona is Gaudí, beaches, and tapas. Sagrada Família must be "
            "booked weeks ahead. Late spring is perfect; August is hot and busy."
        ),
        "visa": "Schengen 90/180.",
        "neighborhoods": "Gothic Quarter, El Born, Gràcia, Barceloneta (beach).",
        "transit": "T-Casual 10-trip metro pass is the best value.",
    },
    "London": {
        "overview": (
            "London is huge — pick 2–3 areas per day. May to September offers "
            "the best weather but expect rain anytime. Theatre is world-class."
        ),
        "visa": "ETA required for many passports as of 2025; 6-month tourist allowance.",
        "neighborhoods": "South Bank, Shoreditch, Notting Hill, Camden, Greenwich.",
        "transit": "Oyster card or contactless. Avoid taxis at peak hours.",
    },
    "New York": {
        "overview": (
            "NYC is fast, vertical, and 24/7. Each borough is a different world. "
            "Fall (October) and spring (May) are ideal; summer is humid."
        ),
        "visa": "ESTA required for visa-waiver countries.",
        "neighborhoods": "Lower East Side, West Village, Williamsburg, Astoria, Harlem.",
        "transit": "OMNY tap or MetroCard. Subway 24/7 but slow at night.",
    },
    "San Francisco": {
        "overview": (
            "Cool, foggy, hilly. Best weather Sep–Oct, worst (foggy) Jun–Jul. "
            "Ferry Building, Mission, and Golden Gate Park are essential."
        ),
        "visa": "ESTA required.",
        "neighborhoods": "Mission (food), Hayes Valley, North Beach, Castro, Richmond.",
        "transit": "Clipper card. Muni + BART. Bring layers for any season.",
    },
    "Vancouver": {
        "overview": (
            "Mountains and ocean in one city. July–September is reliably dry; "
            "skiing 30 min away at Cypress/Grouse in winter."
        ),
        "visa": "eTA required for visa-exempt visitors.",
        "neighborhoods": "Gastown, Yaletown, Mount Pleasant, Kitsilano.",
        "transit": "Compass Card. SkyTrain to/from the airport.",
    },
    "Mexico City": {
        "overview": (
            "CDMX is high-altitude, culturally dense, and very affordable. "
            "Roma/Condesa for foodies, Coyoacán for history. Dry season Nov–May."
        ),
        "visa": "180-day tourist permit on arrival for most western passports.",
        "neighborhoods": "Roma Norte, Condesa, Polanco, Coyoacán, Centro.",
        "transit": "Metro is cheap and crowded; Uber is the safer rideshare option.",
    },
    "Sydney": {
        "overview": (
            "Sydney has world-class beaches and harbour walks. Bondi to Coogee "
            "coastal trail is unmissable. Best Sep–Nov and Mar–May."
        ),
        "visa": "ETA required for visa-waiver passports.",
        "neighborhoods": "Surry Hills, Newtown, Bondi, Manly, The Rocks.",
        "transit": "Opal card. Ferries are scenic transit.",
    },
    "Hong Kong": {
        "overview": (
            "Hong Kong is a dazzling fusion of East and West — a dense, vertical "
            "city with world-class dim sum, hiking trails on Lantau and the Dragon's "
            "Back, and one of Asia's best transport systems. Spring (Mar–May) and "
            "autumn (Oct–Nov) are most pleasant; summer is hot and typhoon-prone."
        ),
        "visa": (
            "Most western passport holders get 90 days visa-free. "
            "No visa required for onward connections through HKG."
        ),
        "neighborhoods": (
            "Tsim Sha Tsui for harbourfront and museums, Mong Kok for street "
            "markets, Central and SoHo for dining and nightlife, Sham Shui Po "
            "for local food, Stanley for beaches."
        ),
        "transit": (
            "Octopus card covers MTR, buses, trams, and ferries. "
            "Airport Express from HKG to Central in 24 minutes. "
            "Star Ferry across Victoria Harbour is iconic and cheap."
        ),
    },
    "Shenzhen": {
        "overview": (
            "Shenzhen grew from a fishing village to a tech megacity in 40 years. "
            "Home to Huawei, Tencent, and DJI, it is China's innovation hub. "
            "Dafen Oil Painting Village, OCT Loft creative park, and the adjacent "
            "hiking trails in Meilin and Tanglang Mountain are popular. "
            "Best visited Oct–Dec for cooler, drier weather."
        ),
        "visa": (
            "Foreign visitors need a valid Chinese visa or can use the 144-hour "
            "transit visa-free policy if entering via SZX and continuing to a "
            "third country. Hong Kong residents and Chinese nationals enter freely."
        ),
        "neighborhoods": (
            "Futian CBD for business and shopping, Nanshan for tech campuses and "
            "Sea World, Luohu for markets and cross-border shopping, "
            "Shekou for expat dining and the ferry terminal to Hong Kong."
        ),
        "transit": (
            "Shenzhen Metro is extensive and cheap; the Shenzhen Tong card works "
            "across metro, buses, and taxis. High-speed rail (HSR) connects "
            "Shenzhen North to Guangzhou in 30 min and to Hong Kong West Kowloon "
            "in 14 min. Ferries from Shekou to HKG Airport take ~30 min."
        ),
    },
}


def destination_kb_documents() -> list[dict]:
    """Materialise the destination knowledge base as embedding-ready docs."""
    docs: list[dict] = []
    for city, country, _, region in CITIES:
        facts = DESTINATION_FACTS[city]
        page = (
            f"# {city}, {country}\n\n"
            f"## Overview\n{facts['overview']}\n\n"
            f"## Visa\n{facts['visa']}\n\n"
            f"## Neighborhoods\n{facts['neighborhoods']}\n\n"
            f"## Transit\n{facts['transit']}\n"
        )
        docs.append(
            {
                "city": city,
                "country": country,
                "region": region,
                "text": page,
                "metadata": {"city": city, "country": country, "region": region},
            }
        )
    return docs


# ---------------------------------------------------------------------------
# Flights
# ---------------------------------------------------------------------------
AIRLINES = [
    ("United", "UA"),
    ("Delta", "DL"),
    ("ANA", "NH"),
    ("Japan Airlines", "JL"),
    ("Korean Air", "KE"),
    ("Singapore Airlines", "SQ"),
    ("Air France", "AF"),
    ("British Airways", "BA"),
    ("Cathay Pacific", "CX"),
    ("Qantas", "QF"),
    ("HK Express", "UO"),
    ("Shenzhen Airlines", "ZH"),
]

# Base prices by region pair (USD, round-trip economy)
BASE_PRICE = {
    ("Americas", "Asia"): 1200,
    ("Americas", "Europe"): 900,
    ("Americas", "Oceania"): 1500,
    ("Americas", "Americas"): 400,
    ("Europe", "Asia"): 800,
    ("Europe", "Europe"): 250,
    ("Europe", "Oceania"): 1400,
    ("Asia", "Asia"): 350,
    ("Asia", "Oceania"): 900,
    ("Oceania", "Oceania"): 300,
}


def _region_of(city: str) -> str:
    for c, _, _, r in CITIES:
        if c == city:
            return r
    return "Asia"


def _base_price(origin: str, dest: str) -> int:
    a, b = _region_of(origin), _region_of(dest)
    key = (a, b) if (a, b) in BASE_PRICE else (b, a)
    return BASE_PRICE.get(key, 700)


# ---------------------------------------------------------------------------
# HKG / SZX → Japan routes
# Covers: early-morning, daytime, evening, late-night departures
#         economy and business cabin  |  direct and 1-stop variants
# ---------------------------------------------------------------------------

# (origin_city, origin_iata, dest_city, dest_iata,
#  airline_name, airline_code,
#  depart_time, duration_hours, cabin, stops,
#  base_economy_price_usd)
_HKG_SZX_ROUTES: list[tuple] = [
    # ── HKG → Tokyo (HND) ───────────────────────────────────────────────
    ("Hong Kong", "HKG", "Tokyo", "HND", "Cathay Pacific",  "CX", "06:10",  4, "economy",  0,  380),
    ("Hong Kong", "HKG", "Tokyo", "HND", "Cathay Pacific",  "CX", "06:10",  4, "business", 0, 1450),
    ("Hong Kong", "HKG", "Tokyo", "HND", "Japan Airlines",  "JL", "09:30",  4, "economy",  0,  420),
    ("Hong Kong", "HKG", "Tokyo", "HND", "Japan Airlines",  "JL", "09:30",  4, "business", 0, 1680),
    ("Hong Kong", "HKG", "Tokyo", "HND", "HK Express",      "UO", "13:45",  4, "economy",  0,  295),
    ("Hong Kong", "HKG", "Tokyo", "HND", "ANA",             "NH", "17:20",  4, "economy",  0,  410),
    ("Hong Kong", "HKG", "Tokyo", "HND", "ANA",             "NH", "17:20",  4, "business", 0, 1590),
    ("Hong Kong", "HKG", "Tokyo", "HND", "Cathay Pacific",  "CX", "22:55",  4, "economy",  0,  360),
    ("Hong Kong", "HKG", "Tokyo", "HND", "Cathay Pacific",  "CX", "22:55",  4, "business", 0, 1380),
    # ── HKG → Osaka (KIX) ───────────────────────────────────────────────
    ("Hong Kong", "HKG", "Osaka", "KIX", "Cathay Pacific",  "CX", "06:50",  3, "economy",  0,  340),
    ("Hong Kong", "HKG", "Osaka", "KIX", "Cathay Pacific",  "CX", "06:50",  3, "business", 0, 1320),
    ("Hong Kong", "HKG", "Osaka", "KIX", "HK Express",      "UO", "10:15",  3, "economy",  0,  260),
    ("Hong Kong", "HKG", "Osaka", "KIX", "Japan Airlines",  "JL", "14:30",  3, "economy",  0,  390),
    ("Hong Kong", "HKG", "Osaka", "KIX", "Japan Airlines",  "JL", "14:30",  3, "business", 0, 1550),
    ("Hong Kong", "HKG", "Osaka", "KIX", "HK Express",      "UO", "18:40",  3, "economy",  0,  275),
    ("Hong Kong", "HKG", "Osaka", "KIX", "ANA",             "NH", "23:10",  3, "economy",  0,  310),
    ("Hong Kong", "HKG", "Osaka", "KIX", "ANA",             "NH", "23:10",  3, "business", 0, 1200),
    # ── HKG → Kyoto (ITM, nearest major airport) ────────────────────────
    ("Hong Kong", "HKG", "Kyoto", "ITM", "Cathay Pacific",  "CX", "07:20",  3, "economy",  0,  355),
    ("Hong Kong", "HKG", "Kyoto", "ITM", "Cathay Pacific",  "CX", "07:20",  3, "business", 0, 1390),
    ("Hong Kong", "HKG", "Kyoto", "ITM", "HK Express",      "UO", "11:00",  3, "economy",  0,  270),
    ("Hong Kong", "HKG", "Kyoto", "ITM", "Japan Airlines",  "JL", "15:45",  3, "economy",  0,  400),
    ("Hong Kong", "HKG", "Kyoto", "ITM", "Japan Airlines",  "JL", "15:45",  3, "business", 0, 1580),
    ("Hong Kong", "HKG", "Kyoto", "ITM", "ANA",             "NH", "23:30",  3, "economy",  0,  320),
    ("Hong Kong", "HKG", "Kyoto", "ITM", "ANA",             "NH", "23:30",  3, "business", 0, 1250),
    # ── SZX → Tokyo (HND) ───────────────────────────────────────────────
    ("Shenzhen", "SZX", "Tokyo", "HND", "Shenzhen Airlines","ZH", "06:00",  4, "economy",  0,  370),
    ("Shenzhen", "SZX", "Tokyo", "HND", "Shenzhen Airlines","ZH", "06:00",  4, "business", 0, 1420),
    ("Shenzhen", "SZX", "Tokyo", "HND", "ANA",              "NH", "08:45",  4, "economy",  0,  430),
    ("Shenzhen", "SZX", "Tokyo", "HND", "ANA",              "NH", "08:45",  4, "business", 0, 1700),
    ("Shenzhen", "SZX", "Tokyo", "HND", "Shenzhen Airlines","ZH", "12:30",  4, "economy",  0,  350),
    ("Shenzhen", "SZX", "Tokyo", "HND", "Japan Airlines",   "JL", "16:55",  4, "economy",  0,  415),
    ("Shenzhen", "SZX", "Tokyo", "HND", "Japan Airlines",   "JL", "16:55",  4, "business", 0, 1640),
    ("Shenzhen", "SZX", "Tokyo", "HND", "Shenzhen Airlines","ZH", "21:40",  4, "economy",  0,  340),
    ("Shenzhen", "SZX", "Tokyo", "HND", "Shenzhen Airlines","ZH", "21:40",  4, "business", 0, 1310),
    ("Shenzhen", "SZX", "Tokyo", "HND", "ANA",              "NH", "23:50",  4, "economy",  1,  305),
    # ── SZX → Osaka (KIX) ───────────────────────────────────────────────
    ("Shenzhen", "SZX", "Osaka", "KIX", "Shenzhen Airlines","ZH", "06:30",  3, "economy",  0,  330),
    ("Shenzhen", "SZX", "Osaka", "KIX", "Shenzhen Airlines","ZH", "06:30",  3, "business", 0, 1280),
    ("Shenzhen", "SZX", "Osaka", "KIX", "ANA",              "NH", "10:00",  3, "economy",  0,  395),
    ("Shenzhen", "SZX", "Osaka", "KIX", "ANA",              "NH", "10:00",  3, "business", 0, 1540),
    ("Shenzhen", "SZX", "Osaka", "KIX", "Shenzhen Airlines","ZH", "14:20",  3, "economy",  0,  315),
    ("Shenzhen", "SZX", "Osaka", "KIX", "Japan Airlines",   "JL", "18:10",  3, "economy",  0,  380),
    ("Shenzhen", "SZX", "Osaka", "KIX", "Japan Airlines",   "JL", "18:10",  3, "business", 0, 1490),
    ("Shenzhen", "SZX", "Osaka", "KIX", "Shenzhen Airlines","ZH", "22:45",  3, "economy",  0,  290),
    ("Shenzhen", "SZX", "Osaka", "KIX", "Shenzhen Airlines","ZH", "22:45",  3, "business", 0, 1130),
    # ── SZX → Kyoto (ITM) ───────────────────────────────────────────────
    ("Shenzhen", "SZX", "Kyoto", "ITM", "Shenzhen Airlines","ZH", "07:05",  3, "economy",  0,  345),
    ("Shenzhen", "SZX", "Kyoto", "ITM", "Shenzhen Airlines","ZH", "07:05",  3, "business", 0, 1330),
    ("Shenzhen", "SZX", "Kyoto", "ITM", "ANA",              "NH", "11:30",  3, "economy",  0,  405),
    ("Shenzhen", "SZX", "Kyoto", "ITM", "ANA",              "NH", "11:30",  3, "business", 0, 1580),
    ("Shenzhen", "SZX", "Kyoto", "ITM", "Shenzhen Airlines","ZH", "15:50",  3, "economy",  0,  325),
    ("Shenzhen", "SZX", "Kyoto", "ITM", "Japan Airlines",   "JL", "19:25",  3, "economy",  0,  390),
    ("Shenzhen", "SZX", "Kyoto", "ITM", "Japan Airlines",   "JL", "19:25",  3, "business", 0, 1530),
    ("Shenzhen", "SZX", "Kyoto", "ITM", "Shenzhen Airlines","ZH", "23:15",  3, "economy",  0,  300),
    ("Shenzhen", "SZX", "Kyoto", "ITM", "Shenzhen Airlines","ZH", "23:15",  3, "business", 0, 1160),
]

# Price multipliers applied across a 30-day window to simulate demand
# Index i → apply multiplier _PRICE_JITTER[i % len(_PRICE_JITTER)]
_PRICE_JITTER = [1.00, 0.95, 1.05, 0.90, 1.10, 0.98, 1.03, 0.88, 1.12, 1.00,
                 0.93, 1.07, 0.97, 1.08, 0.91, 1.04, 0.96, 1.11, 0.89, 1.02,
                 1.06, 0.94, 1.09, 0.92, 1.01, 0.99, 1.13, 0.87, 1.05, 0.96]


def _hkg_szx_flight_documents(start: date, days: int) -> list[dict]:
    """Generate HKG/SZX → Japan (Tokyo/Osaka/Kyoto) flights for every day in
    the window, covering early-morning, daytime, evening, and late-night
    departure slots in both economy and business cabins."""
    docs: list[dict] = []
    for d_offset in range(days):
        depart = start + timedelta(days=d_offset)
        jitter = _PRICE_JITTER[d_offset % len(_PRICE_JITTER)]
        for i, (
            origin, origin_iata, dest, dest_iata,
            airline, code,
            depart_time, duration, cabin, stops, base_price,
        ) in enumerate(_HKG_SZX_ROUTES):
            price = int(base_price * jitter)
            # Weekend slight premium
            if depart.weekday() >= 5:
                price = int(price * 1.08)
            docs.append({
                "flight_no":        f"{code}{500 + (i * 7 + d_offset * 3) % 400}",
                "airline":          airline,
                "origin":           origin,
                "origin_iata":      origin_iata,
                "destination":      dest,
                "destination_iata": dest_iata,
                "depart_date":      depart.isoformat(),
                "depart_time":      depart_time,
                "duration_hours":   duration,
                "price_usd":        price,
                "cabin":            cabin,
                "stops":            stops,
            })
    return docs


def flight_documents(start: date | None = None, days: int = 30) -> list[dict]:
    """Generate flights spanning a 30-day window.

    Includes:
    - ~300 procedurally-generated flights between the 15 original cities
    - Dedicated HKG/SZX → Tokyo/Osaka/Kyoto routes with multiple departure
      times (early-morning, daytime, evening, late-night) and both economy
      and business cabin options.
    """
    if start is None:
        start = date(2026, 5, 1)
    docs: list[dict] = []
    for d_offset in range(days):
        depart = start + timedelta(days=d_offset)
        for o_idx, (origin, _, origin_iata, _) in enumerate(CITIES):
            # Each city offers flights to a small rotating set of destinations
            for shift in (3, 6, 9):
                dest_idx = (o_idx + shift) % len(CITIES)
                dest, _, dest_iata, _ = CITIES[dest_idx]
                if dest == origin:
                    continue
                airline_idx = (o_idx + dest_idx + d_offset) % len(AIRLINES)
                airline_name, airline_code = AIRLINES[airline_idx]
                base = _base_price(origin, dest)
                # Vary price by ±20%
                jitter = ((d_offset * 37 + o_idx * 13 + dest_idx * 7) % 41) - 20
                price = int(base * (1 + jitter / 100))
                docs.append(
                    {
                        "flight_no": f"{airline_code}{100 + (o_idx * 17 + dest_idx) % 900}",
                        "airline": airline_name,
                        "origin": origin,
                        "origin_iata": origin_iata,
                        "destination": dest,
                        "destination_iata": dest_iata,
                        "depart_date": depart.isoformat(),
                        "depart_time": f"{(7 + (d_offset + o_idx) % 12):02d}:{(d_offset * 7) % 60:02d}",
                        "duration_hours": 2 + ((o_idx + dest_idx) % 13),
                        "price_usd": price,
                        "cabin": "economy",
                        "stops": 0 if (o_idx + dest_idx) % 3 else 1,
                    }
                )

    # Append dedicated HKG / SZX → Japan routes
    docs.extend(_hkg_szx_flight_documents(start, days))
    return docs


# ---------------------------------------------------------------------------
# Hotels
# ---------------------------------------------------------------------------
HOTEL_TYPES = [
    ("Capsule", 60, 3.7, ["wifi", "shared_bath"]),
    ("Hostel", 45, 3.9, ["wifi", "breakfast", "shared_kitchen"]),
    ("Business", 140, 4.1, ["wifi", "breakfast", "gym"]),
    ("Boutique", 240, 4.5, ["wifi", "breakfast", "bar", "design"]),
    ("Ryokan", 320, 4.7, ["wifi", "breakfast", "onsen", "traditional"]),
    ("Luxury", 520, 4.8, ["wifi", "breakfast", "spa", "pool", "concierge"]),
]


def hotel_documents() -> list[dict]:
    docs: list[dict] = []
    for city_idx, (city, _, _, _) in enumerate(CITIES):
        for h_idx, (htype, base_price, base_rating, amenities) in enumerate(HOTEL_TYPES):
            # 2 hotels per type per city, slight price/rating variation
            for variant in range(2):
                price = base_price + (city_idx * 5 + variant * 15) % 80
                # Ryokan only in Japan
                if htype == "Ryokan" and city not in ("Tokyo", "Kyoto", "Osaka"):
                    continue
                docs.append(
                    {
                        "name": f"{city} {htype} {variant + 1}",
                        "city": city,
                        "type": htype.lower(),
                        "price_per_night": price,
                        "rating": round(base_rating + (variant * 0.1), 1),
                        "amenities": amenities,
                        "description": (
                            f"A {htype.lower()} property in {city}. "
                            + ", ".join(amenities)
                            + "."
                        ),
                    }
                )
    return docs


# ---------------------------------------------------------------------------
# Activities
# ---------------------------------------------------------------------------
ACTIVITIES_PER_CITY: dict[str, list[tuple[str, list[str], int, str]]] = {
    "Tokyo": [
        ("Tsukiji Outer Market food tour", ["food", "culture", "walking"], 80, "3h guided"),
        ("Senso-ji Temple & Asakusa stroll", ["culture", "history", "free"], 0, "self-guided"),
        ("teamLab Planets immersive art", ["art", "indoor", "family"], 35, "2h"),
        ("Shibuya Sky observatory", ["views", "sunset"], 25, "1.5h"),
        ("Day trip to Mt. Fuji & Hakone", ["nature", "day-trip"], 180, "full day"),
        ("Robot Restaurant cabaret", ["entertainment", "nightlife"], 90, "2h"),
    ],
    "Kyoto": [
        ("Fushimi Inari shrine hike", ["culture", "outdoor", "free"], 0, "2-3h"),
        ("Arashiyama bamboo grove & monkey park", ["nature", "culture"], 18, "half day"),
        ("Tea ceremony in Gion", ["culture", "indoor"], 65, "1h"),
        ("Kinkaku-ji golden pavilion", ["culture", "history"], 5, "1h"),
        ("Nishiki market food crawl", ["food", "walking"], 60, "2h"),
        ("Geisha district evening walk", ["culture", "evening"], 0, "1h"),
    ],
    "Osaka": [
        ("Dotonbori street food tour", ["food", "nightlife"], 55, "2.5h"),
        ("Osaka Castle & park", ["history", "outdoor"], 8, "2h"),
        ("Universal Studios Japan", ["family", "theme-park"], 95, "full day"),
        ("Sumiyoshi-taisha shrine", ["culture", "free"], 0, "1h"),
    ],
    "Seoul": [
        ("Gyeongbokgung Palace + hanbok rental", ["culture", "photo"], 25, "3h"),
        ("Gwangjang night market food", ["food", "nightlife"], 40, "2h"),
        ("DMZ day tour", ["history", "day-trip"], 90, "full day"),
        ("Bukchon hanok village walk", ["culture", "free"], 0, "2h"),
        ("Han River bike ride", ["outdoor", "free"], 5, "2h"),
    ],
    "Bangkok": [
        ("Grand Palace + Wat Pho", ["culture", "history"], 18, "half day"),
        ("Chao Phraya river dinner cruise", ["food", "evening"], 45, "3h"),
        ("Floating market half-day", ["culture", "shopping"], 35, "half day"),
        ("Thai cooking class", ["food", "hands-on"], 60, "4h"),
        ("Muay Thai live match", ["sport", "evening"], 50, "3h"),
    ],
    "Singapore": [
        ("Gardens by the Bay (incl. domes)", ["nature", "family"], 25, "half day"),
        ("Hawker centre food crawl", ["food"], 30, "3h"),
        ("Sentosa island day", ["family", "beach"], 55, "full day"),
        ("Night safari", ["nature", "evening", "family"], 45, "3h"),
    ],
    "Paris": [
        ("Louvre highlights tour", ["art", "history"], 65, "3h"),
        ("Seine river cruise", ["views", "romantic"], 25, "1h"),
        ("Montmartre walking tour", ["culture", "free"], 0, "2h"),
        ("Versailles day trip", ["history", "day-trip"], 80, "full day"),
        ("Pastry & chocolate tour", ["food", "walking"], 70, "3h"),
    ],
    "Rome": [
        ("Colosseum + Forum guided tour", ["history"], 55, "3h"),
        ("Vatican Museums + Sistine Chapel", ["art", "history"], 75, "3h"),
        ("Trastevere food crawl", ["food", "evening"], 70, "3h"),
        ("Pasta-making class", ["food", "hands-on"], 65, "3h"),
    ],
    "Barcelona": [
        ("Sagrada Família tour", ["art", "architecture"], 35, "1.5h"),
        ("Park Güell visit", ["art", "outdoor"], 15, "2h"),
        ("Tapas + wine walking tour", ["food", "evening"], 75, "3.5h"),
        ("Gothic Quarter free walk", ["culture", "free"], 0, "2h"),
        ("Day trip to Montserrat", ["nature", "day-trip"], 60, "full day"),
    ],
    "London": [
        ("West End theatre evening", ["entertainment"], 90, "3h"),
        ("Tower of London tour", ["history"], 40, "3h"),
        ("Borough Market food", ["food"], 30, "2h"),
        ("British Museum highlights", ["art", "history", "free"], 0, "2h"),
        ("Hampton Court Palace day", ["history", "day-trip"], 35, "full day"),
    ],
    "New York": [
        ("Broadway show", ["entertainment"], 130, "2.5h"),
        ("Statue of Liberty + Ellis Island", ["history"], 25, "half day"),
        ("Met Museum visit", ["art"], 30, "3h"),
        ("Brooklyn food tour", ["food"], 75, "3.5h"),
        ("High Line + Hudson Yards walk", ["urban", "free"], 0, "2h"),
    ],
    "San Francisco": [
        ("Alcatraz night tour", ["history", "evening"], 55, "3h"),
        ("Bike across Golden Gate Bridge", ["outdoor"], 45, "half day"),
        ("Mission food tour", ["food"], 70, "3h"),
        ("Muir Woods + Sausalito day", ["nature", "day-trip"], 90, "full day"),
    ],
    "Vancouver": [
        ("Capilano suspension bridge", ["nature"], 55, "half day"),
        ("Granville Island market", ["food", "free"], 0, "2h"),
        ("Stanley Park bike loop", ["outdoor"], 20, "3h"),
        ("Whistler day trip", ["nature", "day-trip"], 120, "full day"),
    ],
    "Mexico City": [
        ("Teotihuacán pyramids", ["history", "day-trip"], 60, "full day"),
        ("Frida Kahlo museum + Coyoacán", ["art"], 25, "3h"),
        ("Roma/Condesa food tour", ["food"], 65, "3h"),
        ("Xochimilco trajinera", ["culture", "fun"], 30, "half day"),
    ],
    "Sydney": [
        ("Sydney Opera House tour", ["culture"], 50, "1h"),
        ("Bondi to Coogee coast walk", ["outdoor", "free"], 0, "2h"),
        ("Blue Mountains day trip", ["nature", "day-trip"], 130, "full day"),
        ("BridgeClimb Sydney", ["adventure", "views"], 220, "3h"),
    ],
}


def activity_documents() -> list[dict]:
    docs: list[dict] = []
    for city, items in ACTIVITIES_PER_CITY.items():
        for name, tags, price, duration in items:
            docs.append(
                {
                    "city": city,
                    "name": name,
                    "tags": tags,
                    "price_usd": price,
                    "duration": duration,
                    "description": f"{name} in {city} — {duration}, ${price}.",
                }
            )
    return docs


# ---------------------------------------------------------------------------
# Restaurants
# ---------------------------------------------------------------------------
RESTAURANTS_PER_CITY: dict[str, list[tuple[str, str, list[str], int, float]]] = {
    # (name, cuisine, dietary_tags, price_level 1-4, rating)
    "Tokyo": [
        ("T's Tantan Vegan Ramen", "japanese", ["vegan", "vegetarian"], 2, 4.4),
        ("Sushi Saito Omakase", "japanese", ["seafood", "pescatarian"], 4, 4.9),
        ("Afuri Ramen Ebisu", "japanese", ["vegetarian-option"], 2, 4.3),
        ("Ain Soph Ginza", "fusion", ["vegan", "vegetarian", "gluten-free"], 2, 4.5),
        ("Kozasa Yakitori", "japanese", [], 3, 4.4),
        ("Tsukiji Outer Market Sushi", "japanese", ["seafood", "pescatarian"], 2, 4.6),
        ("Hamacho Sushi Nakamura", "japanese", ["seafood", "pescatarian"], 3, 4.5),
        ("Tempura Kondo", "japanese", ["seafood", "pescatarian", "vegetarian-option"], 4, 4.7),
    ],
    "Kyoto": [
        ("Shigetsu Shojin Ryori", "japanese", ["vegan", "vegetarian"], 3, 4.6),
        ("Issen Yoshoku Gion", "japanese", ["vegetarian-option"], 1, 4.2),
        ("Mumokuteki Cafe", "fusion", ["vegan", "vegetarian"], 2, 4.3),
        ("Hyotei Kaiseki", "japanese", ["seafood", "pescatarian"], 4, 4.8),
        ("Nishiki Market Seafood Stalls", "japanese", ["seafood", "pescatarian"], 1, 4.4),
        ("Yasaka Kappa Kappo", "japanese", ["seafood", "pescatarian", "vegetarian-option"], 3, 4.5),
    ],
    "Osaka": [
        ("Paprika Shokudo Vegan", "japanese", ["vegan", "vegetarian"], 2, 4.4),
        ("Mizuno Okonomiyaki", "japanese", ["seafood", "pescatarian"], 2, 4.5),
        ("Kushikatsu Daruma", "japanese", [], 2, 4.3),
        ("Dotonbori Kani Doraku", "japanese", ["seafood", "pescatarian"], 3, 4.4),
        ("Endo Sushi Honten", "japanese", ["seafood", "pescatarian"], 2, 4.7),
    ],
    "Seoul": [
        ("Plant Cafe Itaewon", "korean", ["vegan", "vegetarian"], 2, 4.3),
        ("Gwangjang Bindaetteok stalls", "korean", ["vegetarian-option"], 1, 4.4),
        ("Mingles", "korean", ["seafood", "pescatarian"], 4, 4.8),
        ("Noryangjin Fish Market", "korean", ["seafood", "pescatarian"], 2, 4.5),
        ("Haenyeo Restaurant", "korean", ["seafood", "pescatarian", "gluten-free"], 3, 4.6),
    ],
    "Bangkok": [
        ("May Veggie Home", "thai", ["vegan", "vegetarian"], 2, 4.4),
        ("Bo.lan", "thai", ["vegetarian-option", "gluten-free"], 4, 4.7),
        ("Jay Fai", "thai", ["seafood", "pescatarian"], 3, 4.7),
        ("Broccoli Revolution", "thai", ["vegan", "vegetarian", "gluten-free"], 2, 4.5),
        ("Savoey Seafood", "thai", ["seafood", "pescatarian"], 3, 4.5),
        ("Krua Apsorn", "thai", ["seafood", "pescatarian", "vegetarian-option"], 2, 4.4),
    ],
    "Singapore": [
        ("Whole Earth", "fusion", ["vegetarian", "vegan-option"], 2, 4.4),
        ("Tian Tian Hainanese Chicken", "chinese", [], 1, 4.6),
        ("Burnt Ends", "international", [], 4, 4.8),
        ("Original Sin Mediterranean", "mediterranean", ["vegetarian"], 3, 4.5),
        ("Long Beach Seafood", "chinese", ["seafood", "pescatarian"], 3, 4.5),
        ("Jumbo Seafood Clarke Quay", "chinese", ["seafood", "pescatarian"], 3, 4.6),
        ("Ola Beach Club", "mediterranean", ["seafood", "pescatarian", "vegetarian-option"], 3, 4.4),
    ],
    "Paris": [
        ("Le Potager du Marais", "french", ["vegan", "vegetarian"], 2, 4.4),
        ("Septime", "french", ["vegetarian-option", "seafood", "pescatarian"], 4, 4.7),
        ("Le Comptoir du Relais", "french", [], 3, 4.5),
        ("Hank Burger", "american", ["vegan", "vegetarian"], 1, 4.3),
        ("Huîtrerie Régis", "french", ["seafood", "pescatarian", "gluten-free"], 3, 4.6),
        ("Le Dôme Montparnasse", "french", ["seafood", "pescatarian"], 4, 4.5),
    ],
    "Rome": [
        ("Romeow Cat Bistrot", "italian", ["vegan", "vegetarian"], 2, 4.4),
        ("Pizzarium", "italian", ["vegetarian-option"], 1, 4.6),
        ("La Pergola", "italian", ["seafood", "pescatarian"], 4, 4.9),
        ("Trapizzino Trastevere", "italian", ["vegetarian-option"], 1, 4.4),
        ("Pierluigi", "italian", ["seafood", "pescatarian"], 3, 4.6),
        ("Osteria dell'Anima", "italian", ["seafood", "pescatarian", "vegetarian-option"], 2, 4.4),
    ],
    "Barcelona": [
        ("Teresa Carles", "spanish", ["vegan", "vegetarian"], 2, 4.4),
        ("Disfrutar", "spanish", ["seafood", "pescatarian"], 4, 4.8),
        ("Bar del Pla", "spanish", ["vegetarian-option"], 2, 4.5),
        ("La Cova Fumada", "spanish", ["seafood", "pescatarian"], 2, 4.6),
        ("Espai Mescladís", "mediterranean", ["seafood", "pescatarian", "vegetarian-option"], 2, 4.4),
    ],
    "London": [
        ("Mildreds Soho", "international", ["vegan", "vegetarian"], 2, 4.4),
        ("Dishoom Covent Garden", "indian", ["vegetarian", "vegan-option"], 2, 4.6),
        ("The Ledbury", "british", ["seafood", "pescatarian"], 4, 4.8),
        ("Padella", "italian", ["vegetarian-option"], 1, 4.5),
        ("Bonnie Gull Seafood Cafe", "british", ["seafood", "pescatarian"], 3, 4.5),
        ("Wright Brothers Soho", "british", ["seafood", "pescatarian", "gluten-free"], 3, 4.6),
    ],
    "New York": [
        ("Superiority Burger", "american", ["vegan", "vegetarian"], 1, 4.5),
        ("Eleven Madison Park", "american", ["vegan", "vegetarian"], 4, 4.7),
        ("Joe's Pizza", "italian", ["vegetarian-option"], 1, 4.4),
        ("Xi'an Famous Foods", "chinese", ["vegetarian-option"], 1, 4.4),
        ("Le Bernardin", "french", ["seafood", "pescatarian"], 4, 4.8),
        ("Pearl Oyster Bar", "american", ["seafood", "pescatarian"], 2, 4.6),
        ("Cull & Pistol", "american", ["seafood", "pescatarian", "gluten-free"], 3, 4.5),
    ],
    "San Francisco": [
        ("Greens Restaurant", "american", ["vegetarian", "vegan-option"], 3, 4.5),
        ("Tartine Manufactory", "american", ["vegetarian-option"], 2, 4.4),
        ("State Bird Provisions", "american", [], 3, 4.6),
        ("Shizen Vegan Sushi", "japanese", ["vegan", "vegetarian"], 3, 4.5),
        ("Swan Oyster Depot", "american", ["seafood", "pescatarian"], 2, 4.7),
        ("Hog Island Oyster Co", "american", ["seafood", "pescatarian", "gluten-free"], 2, 4.6),
        ("Waterbar", "american", ["seafood", "pescatarian", "vegetarian-option"], 3, 4.5),
    ],
    "Vancouver": [
        ("MeeT in Gastown", "american", ["vegan", "vegetarian"], 2, 4.4),
        ("Vij's Indian", "indian", ["vegetarian"], 3, 4.6),
        ("Miku Sushi", "japanese", ["seafood", "pescatarian"], 4, 4.6),
        ("The Salmon n' Bannock", "canadian", ["seafood", "pescatarian", "gluten-free"], 3, 4.5),
        ("Blue Water Cafe", "canadian", ["seafood", "pescatarian"], 4, 4.7),
    ],
    "Mexico City": [
        ("Por Siempre Vegana", "mexican", ["vegan", "vegetarian"], 1, 4.5),
        ("Pujol", "mexican", ["seafood", "pescatarian", "vegetarian-option"], 4, 4.8),
        ("El Califa", "mexican", [], 1, 4.4),
        ("Forever Vegano", "mexican", ["vegan", "vegetarian", "gluten-free"], 2, 4.4),
        ("Contramar", "mexican", ["seafood", "pescatarian"], 3, 4.7),
        ("El Pescadito", "mexican", ["seafood", "pescatarian"], 1, 4.5),
    ],
    "Sydney": [
        ("Bodhi Restaurant Bar", "asian", ["vegan", "vegetarian"], 2, 4.4),
        ("Quay Restaurant", "australian", ["seafood", "pescatarian"], 4, 4.8),
        ("Three Blue Ducks", "australian", ["vegetarian-option"], 3, 4.5),
        ("Fish at the Rocks", "australian", ["seafood", "pescatarian"], 3, 4.6),
        ("The Boathouse Palm Beach", "australian", ["seafood", "pescatarian", "gluten-free"], 3, 4.5),
    ],
}


def restaurant_documents() -> list[dict]:
    docs: list[dict] = []
    for city, items in RESTAURANTS_PER_CITY.items():
        for name, cuisine, dietary, price_level, rating in items:
            docs.append(
                {
                    "city": city,
                    "name": name,
                    "cuisine": cuisine,
                    "dietary_tags": dietary,
                    "price_level": price_level,  # 1 cheap → 4 fine-dining
                    "rating": rating,
                    "description": (
                        f"{name} ({cuisine.title()}) in {city}. "
                        f"Price level {price_level}, rating {rating}."
                    ),
                }
            )
    return docs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def all_cities() -> Iterable[str]:
    return (c[0] for c in CITIES)
