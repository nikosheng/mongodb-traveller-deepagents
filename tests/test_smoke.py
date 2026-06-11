"""Import-level smoke tests — do not require network or credentials."""

from __future__ import annotations


def test_seed_data_self_consistent():
    from deepagents_mongodb.knowledge import seed_data

    cities = {c[0] for c in seed_data.CITIES}
    # Every city has a KB entry
    assert cities == set(seed_data.DESTINATION_FACTS.keys())

    # Activities reference only known cities
    for city in seed_data.ACTIVITIES_PER_CITY:
        assert city in cities

    # Restaurants reference only known cities
    for city in seed_data.RESTAURANTS_PER_CITY:
        assert city in cities

    # Flight doc generation produces something
    docs = seed_data.flight_documents(days=2)
    assert len(docs) > 0
    assert all("origin" in d and "destination" in d for d in docs)

    # Hotel doc generation produces something
    docs = seed_data.hotel_documents()
    assert any(d["type"] == "ryokan" for d in docs)
    assert all(d["price_per_night"] > 0 for d in docs)


def test_modules_import():
    # Pure-import sanity (no Mongo connection required because we don't call
    # load_settings).
    import deepagents_mongodb.tools.flights  # noqa: F401
    import deepagents_mongodb.tools.hotels  # noqa: F401
    import deepagents_mongodb.tools.activities  # noqa: F401
    import deepagents_mongodb.tools.restaurants  # noqa: F401
    import deepagents_mongodb.tools.budget  # noqa: F401
    import deepagents_mongodb.tools.memory  # noqa: F401
    import deepagents_mongodb.subagents  # noqa: F401
