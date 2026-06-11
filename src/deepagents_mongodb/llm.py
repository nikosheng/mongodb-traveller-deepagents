"""LLM and embedding model factories.

We use Azure OpenAI (`gpt-5.4-mini`) for the chat model and Voyage AI
(`voyage-4`, 1024 dims) for embeddings — for the vector store, the
long-term `MongoDBStore`, and the `MongoDBAtlasSemanticCache`.

The embeddings are wrapped with a small in-memory LRU cache + a
rate-limit-aware retry so that we tolerate Voyage's free-tier 3 RPM /
10K TPM cap without crashing the agent. In production you'd add a paid
plan, but for a demo this lets the multi-subagent flow complete.
"""

from __future__ import annotations

import logging
import os
import time
from collections import OrderedDict
from typing import Any

from langchain_core.embeddings import Embeddings
from langchain_openai import AzureChatOpenAI
from langchain_voyageai import VoyageAIEmbeddings

from .config import EMBEDDING_MODEL, Settings

logger = logging.getLogger(__name__)


def make_chat_model(settings: Settings, *, temperature: float = 0.2) -> AzureChatOpenAI:
    if settings.azure is None:
        raise RuntimeError(
            "Azure OpenAI credentials are not configured. Set "
            "AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY in your .env."
        )
    return AzureChatOpenAI(
        azure_endpoint=settings.azure.endpoint,
        api_key=settings.azure.api_key,
        api_version=settings.azure.api_version,
        azure_deployment=settings.azure.deployment,
        temperature=temperature,
    )


class ResilientVoyageEmbeddings(Embeddings):
    """Voyage embeddings with LRU cache + 429-backoff retries.

    Tolerates Voyage's free-tier 3 RPM / 10K TPM limits without bringing
    the whole agent down. Strictly a transport-layer concern; downstream
    components (the Store, the semantic cache, the vector store) see a
    normal ``Embeddings`` interface.
    """

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        cache_size: int = 1024,
        max_retries: int = 6,
        initial_backoff_seconds: float = 12.0,
        backoff_multiplier: float = 1.6,
    ) -> None:
        kwargs: dict = {"model": model}
        if api_key:
            kwargs["voyage_api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        self._inner = VoyageAIEmbeddings(**kwargs)
        self._cache: "OrderedDict[str, list[float]]" = OrderedDict()
        self._cache_size = cache_size
        self._max_retries = max_retries
        self._initial_backoff = initial_backoff_seconds
        self._backoff_multiplier = backoff_multiplier

    # --- cache helpers -----------------------------------------------------
    def _cache_get(self, text: str) -> list[float] | None:
        vec = self._cache.get(text)
        if vec is not None:
            # Touch for LRU.
            self._cache.move_to_end(text)
        return vec

    def _cache_put(self, text: str, vec: list[float]) -> None:
        self._cache[text] = vec
        self._cache.move_to_end(text)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

    # --- retry wrapper -----------------------------------------------------
    def _with_retry(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        backoff = self._initial_backoff
        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                return fn(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001 — narrow check below
                msg = str(exc).lower()
                is_rate_limited = (
                    "ratelimit" in exc.__class__.__name__.lower()
                    or "rate limit" in msg
                    or "rpm" in msg
                    or "tpm" in msg
                    or "429" in msg
                )
                if not is_rate_limited:
                    raise
                last_exc = exc
                logger.warning(
                    "Voyage rate-limited (attempt %d/%d); sleeping %.1fs",
                    attempt + 1,
                    self._max_retries,
                    backoff,
                )
                time.sleep(backoff)
                backoff *= self._backoff_multiplier
        if last_exc is not None:
            raise last_exc
        raise RuntimeError("retry budget exhausted")  # pragma: no cover

    # --- Embeddings interface ---------------------------------------------
    def embed_query(self, text: str) -> list[float]:
        cached = self._cache_get(text)
        if cached is not None:
            return cached
        vec = self._with_retry(self._inner.embed_query, text)
        self._cache_put(text, vec)
        return vec

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = [None] * len(texts)  # type: ignore[list-item]
        misses: list[tuple[int, str]] = []
        for i, t in enumerate(texts):
            cached = self._cache_get(t)
            if cached is not None:
                out[i] = cached
            else:
                misses.append((i, t))
        if misses:
            miss_texts = [t for _, t in misses]
            vecs = self._with_retry(self._inner.embed_documents, miss_texts)
            for (i, t), v in zip(misses, vecs, strict=True):
                self._cache_put(t, v)
                out[i] = v
        return out


def make_embeddings(settings: Settings) -> Embeddings:
    return ResilientVoyageEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=settings.voyage_api_key,
        base_url=settings.voyage_base_url or None,
    )
