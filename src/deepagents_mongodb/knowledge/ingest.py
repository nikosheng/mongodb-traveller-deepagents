"""Load mocked data into MongoDB Atlas.

This module is invoked by ``scripts/bootstrap_atlas.py``. Each ``ingest_*``
function is idempotent — it wipes its collection and re-inserts the seed.

The destination knowledge base is embedded with voyage-4 and inserted via
``MongoDBAtlasVectorSearch``; everything else is inserted as plain documents
because they don't need semantic search.
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_mongodb.vectorstores import MongoDBAtlasVectorSearch
from pymongo import MongoClient

from ..config import Settings
from ..llm import make_embeddings
from . import seed_data


def ingest_operational(client: MongoClient, settings: Settings) -> dict[str, int]:
    """Insert flights, hotels, activities, restaurants.

    Returns a per-collection count for logging.
    """
    db = client[settings.mongo.db]
    counts: dict[str, int] = {}

    def replace(coll_name: str, docs: list[dict]) -> None:
        coll = db[coll_name]
        coll.delete_many({})
        if docs:
            coll.insert_many(docs)
        counts[coll_name] = len(docs)

    replace(settings.mongo.flights, seed_data.flight_documents())
    replace(settings.mongo.hotels, seed_data.hotel_documents())
    replace(settings.mongo.activities, seed_data.activity_documents())
    replace(settings.mongo.restaurants, seed_data.restaurant_documents())

    return counts


def ingest_knowledge_base(client: MongoClient, settings: Settings) -> int:
    """Embed and store the destination knowledge base.

    Uses ``MongoDBAtlasVectorSearch`` so the data is queryable by the RAG
    subagent immediately after ingestion (assuming the index is queryable).
    """
    db = client[settings.mongo.db]
    coll = db[settings.mongo.destinations_kb]
    coll.delete_many({})

    docs = seed_data.destination_kb_documents()
    langchain_docs = [
        Document(
            page_content=d["text"],
            metadata={
                "city": d["city"],
                "country": d["country"],
                "region": d["region"],
            },
        )
        for d in docs
    ]

    vector_store = MongoDBAtlasVectorSearch(
        collection=coll,
        embedding=make_embeddings(settings),
        index_name=settings.mongo.destinations_kb_index,
        text_key="text",
        embedding_key="embedding",
        relevance_score_fn="cosine",
    )
    vector_store.add_documents(langchain_docs)
    return len(langchain_docs)
