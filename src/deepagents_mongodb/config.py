"""Centralised configuration — env vars, collection names, vector dims.

All other modules import from here so that database names, collection names,
and index names are defined in exactly one place.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import certifi
from dotenv import load_dotenv

load_dotenv()

# macOS Python installs often ship without a system trust store; certifi
# provides one. Setting these env vars before any client library imports its
# transport stack means MongoDB, requests, urllib3, etc. all find the bundle.
os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())


def _require(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"Environment variable {name!r} is required. "
            "Did you forget to create your .env from .env.example?"
        )
    return value


def _optional(name: str, default: str) -> str:
    return os.environ.get(name) or default


# --- Voyage embedding model ---------------------------------------------------
EMBEDDING_MODEL: str = "voyage-4"
EMBEDDING_DIMS: int = 1024  # voyage-4 default output dimensions


@dataclass(frozen=True)
class MongoConfig:
    uri: str
    db: str

    # Collections
    checkpoints: str = "checkpoints"           # MongoDBSaver auto-managed
    checkpoint_writes: str = "checkpoint_writes"
    agent_store: str = "agent_store"           # MongoDBStore (long-term memory)
    semantic_cache: str = "semantic_cache"     # MongoDBAtlasSemanticCache
    destinations_kb: str = "destinations_kb"   # RAG knowledge base
    flights: str = "flights"
    hotels: str = "hotels"
    activities: str = "activities"
    restaurants: str = "restaurants"

    # Vector index names
    agent_store_index: str = "agent_store_vector_index"
    semantic_cache_index: str = "semantic_cache_vector_index"
    destinations_kb_index: str = "destinations_kb_vector_index"


@dataclass(frozen=True)
class AzureOpenAIConfig:
    endpoint: str
    api_key: str
    api_version: str
    deployment: str


@dataclass(frozen=True)
class Settings:
    mongo: MongoConfig
    azure: AzureOpenAIConfig | None
    voyage_api_key: str
    voyage_base_url: str = ""  # Override if using a custom Voyage endpoint


def load_settings(*, require_azure: bool = False) -> Settings:
    """Build the immutable Settings object from the environment.

    Azure OpenAI credentials are only needed when running the agent itself;
    the bootstrap and reset scripts can succeed without them. When
    ``require_azure`` is False and the env vars are missing, the ``azure``
    field is left as ``None`` and the caller can still build the embeddings,
    Store, and operational data.
    """
    azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    azure_key = os.environ.get("AZURE_OPENAI_API_KEY")
    if require_azure or (azure_endpoint and azure_key):
        azure = AzureOpenAIConfig(
            endpoint=_require("AZURE_OPENAI_ENDPOINT"),
            api_key=_require("AZURE_OPENAI_API_KEY"),
            api_version=_optional("AZURE_OPENAI_API_VERSION", "2024-10-21"),
            deployment=_optional("AZURE_OPENAI_DEPLOYMENT", "gpt-5.4-mini"),
        )
    else:
        azure = None
    return Settings(
        mongo=MongoConfig(
            uri=_require("MONGODB_URI"),
            db=_optional("MONGODB_DB", "travel_planner"),
        ),
        azure=azure,
        voyage_api_key=_require("VOYAGE_API_KEY"),
        voyage_base_url=_optional("VOYAGE_API_BASE", ""),
    )
