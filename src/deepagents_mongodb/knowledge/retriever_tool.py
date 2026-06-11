"""RAG retriever tool over the destinations knowledge base.

The result is a LangChain ``Tool`` instance that the deepagent (or the
``destination_researcher`` subagent specifically) can call. Under the hood
it runs ``$vectorSearch`` against ``destinations_kb`` with voyage-4 embeddings.
"""

from __future__ import annotations

try:
    # langchain 1.x re-organised the package; the canonical helper lives in
    # ``langchain_classic`` now. We try the new location first.
    from langchain_classic.tools.retriever import create_retriever_tool
except ImportError:  # pragma: no cover - fallback for older langchain
    from langchain.tools.retriever import create_retriever_tool
from langchain_core.tools import BaseTool
from langchain_mongodb.vectorstores import MongoDBAtlasVectorSearch
from pymongo import MongoClient

from ..config import Settings
from ..llm import make_embeddings


def make_destination_retriever_tool(
    client: MongoClient, settings: Settings, *, k: int = 4
) -> BaseTool:
    coll = client[settings.mongo.db][settings.mongo.destinations_kb]
    vector_store = MongoDBAtlasVectorSearch(
        collection=coll,
        embedding=make_embeddings(settings),
        index_name=settings.mongo.destinations_kb_index,
        text_key="text",
        embedding_key="embedding",
        relevance_score_fn="cosine",
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    return create_retriever_tool(
        retriever,
        name="search_destination_knowledge",
        description=(
            "Semantic search over the curated destination knowledge base in "
            "MongoDB Atlas. Use this to look up visa rules, neighborhoods, "
            "best months to visit, and transit tips for a city. Input is a "
            "natural-language question about the destination."
        ),
    )
