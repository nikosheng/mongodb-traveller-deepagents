"""Session-scoped semantic LLM cache backed by MongoDB Atlas Vector Search.

Key design decisions vs. the vanilla ``MongoDBAtlasSemanticCache``:

1. **Cache key = last user message only.**
   LangChain's default cache key is the full serialized message list
   (system prompt + history + user text).  That makes paraphrase matching
   nearly impossible because the system prompt dominates the embedding.
   ``SessionScopedSemanticCache`` strips everything except the last
   ``HumanMessage`` content before embedding, so semantically equivalent
   questions produce similar vectors regardless of conversation history.

2. **Session isolation via ``session_id`` filter.**
   Every stored document carries a ``session_id`` field equal to the
   LangGraph ``thread_id`` for the current run.  Both ``lookup`` and
   ``update`` filter by this field, so Session A's cache is invisible to
   Session B and vice-versa.  The Atlas vector index must declare
   ``session_id`` as a filter field (see ``mongo.py``).

3. **Explicit session cleanup.**
   Call ``cache.clear_session(thread_id)`` to delete all entries for a
   finished session.  The CLI path does this automatically in its
   ``finally`` block; the ``langgraph dev`` UI path leaves entries in
   MongoDB (they are session-scoped so they never pollute other sessions).
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, Optional

from langchain_core.globals import set_llm_cache
from langchain_core.outputs import ChatGeneration
from langchain_mongodb.cache import (
    MongoDBAtlasSemanticCache,
    _dumps_generations,
    _loads_generations,
)

from .config import Settings
from .llm import make_embeddings

logger = logging.getLogger(__name__)

_NO_SESSION = "_no_session"


class SessionScopedSemanticCache(MongoDBAtlasSemanticCache):
    """MongoDB Atlas semantic cache scoped to a single LangGraph session.

    Inherits all Atlas Vector Search wiring from
    ``MongoDBAtlasSemanticCache``; overrides ``lookup`` and ``update`` to
    (a) embed only the last user message and (b) scope every query to the
    current ``thread_id``.
    """

    SESSION = "session_id"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _hash_llm_string(llm_string: str) -> str:
        """Return a short SHA-256 hex digest of *llm_string*.

        Atlas Vector Search pre-filters silently return no results when a
        filter string value exceeds ~1 KB.  The LangChain ``llm_string`` for
        a tool-bound model is typically 20–25 KB (it embeds the full tool
        schema).  Hashing it to 64 hex chars makes the filter reliable while
        still uniquely identifying the model configuration.
        """
        return hashlib.sha256(llm_string.encode()).hexdigest()

    def _get_session_id(self) -> str:
        """Return the current LangGraph thread_id, or a fallback sentinel."""
        try:
            from langgraph.config import get_config  # local import — optional dep
            cfg = get_config()
            return cfg.get("configurable", {}).get("thread_id", _NO_SESSION)
        except Exception:
            # Called outside a LangGraph runnable context (e.g. unit tests,
            # the demo script).  Use a shared fallback bucket so the cache
            # still works functionally.
            return _NO_SESSION

    def _extract_user_text(self, prompt: str) -> str:
        """Extract the last HumanMessage content from the serialized prompt.

        LangChain passes ``dumps(messages)`` as *prompt* to the cache.
        We parse that JSON and return just the last human turn as plain text
        so the embedding reflects the user's actual question, not the full
        conversation blob.

        Falls back to the raw *prompt* string if parsing fails (e.g. for
        simple string-based LLM calls).
        """
        try:
            msgs = json.loads(prompt)
            if not isinstance(msgs, list):
                return prompt
            for msg in reversed(msgs):
                if not isinstance(msg, dict):
                    continue
                kwargs = msg.get("kwargs", {})
                # Detect HumanMessage by the 'type' kwarg or the class id.
                msg_type = kwargs.get("type", "")
                if not msg_type:
                    class_id = msg.get("id", [])
                    msg_type = class_id[-1].lower() if class_id else ""
                if "human" in msg_type.lower():
                    content = kwargs.get("content", "")
                    if isinstance(content, str) and content.strip():
                        return content.strip()
                    # content may be a list of blocks (multimodal); flatten.
                    if isinstance(content, list):
                        parts = [
                            p.get("text", "") if isinstance(p, dict) else str(p)
                            for p in content
                        ]
                        return " ".join(parts).strip() or prompt
        except Exception:
            pass
        return prompt

    # ------------------------------------------------------------------
    # Internal helpers (continued)
    # ------------------------------------------------------------------

    @staticmethod
    def _has_tool_calls(generations: Any) -> bool:
        """Return True if any generation in *generations* contains tool calls.

        The LangChain LLM cache stores ``ChatGeneration`` objects that wrap
        ``AIMessage``.  If the cached ``AIMessage`` has ``tool_calls``, replaying
        it inside the agent loop causes an infinite loop: the tool call IDs
        already have matching ``ToolMessage``s in the history, so
        ``pending_tool_calls`` is always empty and the graph keeps routing back
        to the model node via case 6 of ``_make_model_to_tools_edge``.

        We treat any cached response that contains tool calls as a MISS so the
        model is called fresh and produces a new response with new tool call IDs.
        """
        if not isinstance(generations, list):
            return False
        for gen in generations:
            if isinstance(gen, ChatGeneration):
                msg = gen.message
                tool_calls = getattr(msg, "tool_calls", None)
                if tool_calls:
                    return True
        return False

    @staticmethod
    def _clear_message_ids(generations: Any) -> None:
        """Clear the ``id`` field on every ``AIMessage`` in *generations* in-place.

        When a cached ``AIMessage`` carries its original ``.id`` from the first
        run, the deep-agents-ui stream and state snapshot end up with mismatched
        message IDs:

        * ``_areplay_v2_events_for_cache_hit`` emits a ``message-start`` event
          with ``message_id = "lc-{run_manager.run_id}"`` (a fresh UUID).
        * ``StreamMessagesHandlerV2.on_llm_end`` stamps that same synthetic UUID
          onto ``gen.message.id`` and adds it to ``self.seen``.
        * The state-update ``AIMessage`` still carries the *original* cached
          ``.id`` (a different UUID), so ``on_chain_end._emit(dedupe=True)``
          does not suppress it — a second stream entry is emitted with the
          stale ID.  The UI receives two competing entries and cannot correlate
          either with the final state snapshot → the result is not rendered.

        Clearing ``.id`` before returning from ``lookup()`` lets LangChain
        assign a brand-new UUID during ``on_llm_end``.  That single UUID is then
        shared by both the stream events and the state-update message, so the
        UI can deduplicate correctly and render the cached response.
        """
        if not isinstance(generations, list):
            return
        for gen in generations:
            if isinstance(gen, ChatGeneration):
                msg = gen.message
                if hasattr(msg, "id"):
                    msg.id = None

    # ------------------------------------------------------------------
    # Overridden cache interface
    # ------------------------------------------------------------------

    def lookup(self, prompt: str, llm_string: str):
        """Look up a cached response for the current session."""
        session_id = self._get_session_id()
        user_text = self._extract_user_text(prompt)
        llm_hash = self._hash_llm_string(llm_string)

        post_filter = (
            [{"$match": {"score": {"$gte": self.score_threshold}}}]
            if self.score_threshold
            else None
        )

        try:
            results = self.similarity_search_with_score(
                user_text,
                1,
                pre_filter={
                    self.LLM: {"$eq": llm_hash},
                    self.SESSION: {"$eq": session_id},
                },
                post_filter_pipeline=post_filter,
            )
        except Exception as exc:
            # Defensive: if the index isn't ready yet, treat as a miss.
            logger.warning("SessionScopedSemanticCache.lookup failed: %s", exc)
            return None

        if results:
            return_val = results[0][0].metadata.get(self.RETURN_VAL)
            response = _loads_generations(return_val) or return_val

            # Never return a cached response that contains tool calls.
            # Replaying an AIMessage with tool_calls inside the agent loop
            # causes an infinite loop because the tool call IDs already have
            # matching ToolMessages in the conversation history, making
            # pending_tool_calls empty and routing back to model forever.
            if self._has_tool_calls(response):
                logger.info(
                    "SemanticCache SKIP (has tool_calls) session=%s score=%.4f text=%r",
                    session_id,
                    results[0][1],
                    user_text[:60],
                )
                return None

            # Clear message IDs so LangChain assigns fresh UUIDs during
            # on_llm_end.  This ensures the stream events and the state-update
            # AIMessage share the same ID, allowing the deep-agents-ui to
            # correlate and render the cached response correctly.
            self._clear_message_ids(response)

            logger.info(
                "SemanticCache HIT  session=%s score=%.4f text=%r",
                session_id,
                results[0][1],
                user_text[:60],
            )
            return response

        logger.info("SemanticCache MISS session=%s text=%r", session_id, user_text[:60])
        return None

    def update(
        self,
        prompt: str,
        llm_string: str,
        return_val: Any,
        wait_until_ready: Optional[float] = None,
    ) -> None:
        """Store a new response in the cache for the current session.

        Responses that contain tool calls are never stored.  Caching them
        would be wasteful (they are always skipped on lookup to avoid the
        infinite-loop bug) and would pollute the cache collection.
        """
        # Skip storing responses with tool calls — they would always be
        # discarded on lookup anyway (see _has_tool_calls / lookup).
        if self._has_tool_calls(return_val):
            logger.info(
                "SemanticCache SKIP STORE (has tool_calls) session=%s text=%r",
                self._get_session_id(),
                self._extract_user_text(prompt)[:60],
            )
            return

        session_id = self._get_session_id()
        user_text = self._extract_user_text(prompt)
        llm_hash = self._hash_llm_string(llm_string)

        try:
            self.add_texts(
                [user_text],
                [
                    {
                        self.LLM: llm_hash,
                        self.SESSION: session_id,
                        self.RETURN_VAL: _dumps_generations(return_val),
                    }
                ],
            )
            logger.info(
                "SemanticCache STORE session=%s text=%r",
                session_id,
                user_text[:60],
            )
        except Exception as exc:
            # Never crash the agent because the cache write failed.
            logger.warning("SessionScopedSemanticCache.update failed: %s", exc)

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    def clear_session(self, thread_id: str) -> int:
        """Delete all cache entries for *thread_id*.

        Returns the number of documents deleted.
        """
        result = self.collection.delete_many({self.SESSION: thread_id})
        logger.info(
            "Cleared %d cache entries for session %s",
            result.deleted_count,
            thread_id,
        )
        return result.deleted_count


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def enable_semantic_cache(
    settings: Settings,
    *,
    score_threshold: float = 0.92,
) -> SessionScopedSemanticCache:
    """Install the session-scoped MongoDB semantic cache as LangChain's global LLM cache.

    Returns the cache instance so callers can invoke ``clear_session`` at
    the end of a conversation thread.

    The underlying vector search index must already exist and must declare
    both ``llm_string`` and ``session_id`` as filter fields.  Run
    ``scripts/bootstrap_atlas.py`` (or ``uv run python scripts/bootstrap_atlas.py``)
    to create/update the index.
    """
    cache = SessionScopedSemanticCache(
        connection_string=settings.mongo.uri,
        embedding=make_embeddings(settings),
        database_name=settings.mongo.db,
        collection_name=settings.mongo.semantic_cache,
        index_name=settings.mongo.semantic_cache_index,
        score_threshold=score_threshold,
    )
    set_llm_cache(cache)
    return cache
