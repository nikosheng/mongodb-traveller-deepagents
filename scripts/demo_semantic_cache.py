"""Demonstrate MongoDB Atlas Semantic Cache — no LLM call on cache hit.

Run:
    uv run python scripts/demo_semantic_cache.py

What it shows
-------------
Call 1 — cold cache: LLM is invoked (~3-4s). The prompt is embedded with voyage-4
          and stored alongside the response in MongoDB (`semantic_cache` collection).

Call 2 — same prompt: Atlas Vector Search finds the cached embedding (score ~1.0),
          threshold check passes, the stored response is returned immediately (~0.02s).
          The LLM API is never called.

Call 3 — paraphrase:  A differently-worded question is embedded; Atlas finds the
          cached doc with score ~0.985 (JSON-wrapped voyage-4 cosine), threshold
          check passes, cache hit returned.  LLM bypassed again.

Score calibration note
----------------------
MongoDBAtlasSemanticCache stores the full LangChain message-list serialized as JSON,
e.g. '[{"lc":1,"type":"constructor",...,"content":"What visa..."}]'.
The voyage-4 embedding of that JSON string is what is indexed.

Atlas $vectorSearch returns normalized cosine scores.  Observed values:
  - Exact same JSON string : ~1.0   → always a hit at any threshold
  - JSON paraphrase        : ~0.985 → hits at threshold ≤ 0.98
  - Semantically related   : ~0.85  → needs threshold ≤ 0.84

The production agent (cache.py enable_semantic_cache) uses score_threshold=0.92 to
avoid serving cached answers for questions that are merely *topically* related.
This demo uses 0.98 to allow paraphrase hits without false positives.

Atlas ANN caveat
----------------
Atlas approximate nearest-neighbor (ANN) search requires enough candidates to be
accurate.  With a very small cache (1-2 docs) and numCandidates=10 (k=1 × 10),
results can be inconsistent.  In production with a warmed cache the paraphrase hit
is reliable; in this demo we seed a few documents first to stabilise it.
"""

from __future__ import annotations

import sys
import time

sys.path.insert(0, "src")

from deepagents_mongodb.config import load_settings
from deepagents_mongodb.llm import make_chat_model
from deepagents_mongodb.cache import enable_semantic_cache

from langchain_core.messages import HumanMessage
from langchain_mongodb import MongoDBAtlasSemanticCache


# ── Prompts ───────────────────────────────────────────────────────────────────
PROMPT_A = "What are the visa requirements for Japan for China passport holders?"
PROMPT_B = "Do China citizens need a visa to enter Japan?"        # paraphrase of A
SEED_PROMPTS = [
    "Best time to visit Tokyo",
    "Budget tips for traveling in Europe",
    "Hotel recommendations in Paris",
    "Is tap water safe to drink in Singapore?",
]

DEMO_THRESHOLD = 0.98   # catches paraphrases (~0.985) without false positives


def _count(cache: MongoDBAtlasSemanticCache) -> int:
    return cache.collection.count_documents({})


def _call(model, prompt: str) -> tuple[str, float]:
    t0 = time.perf_counter()
    resp = model.invoke([HumanMessage(content=prompt)])
    return resp.content, time.perf_counter() - t0


def _sep():
    print("=" * 65)


def main() -> None:
    settings = load_settings(require_azure=True)
    cache = enable_semantic_cache(settings, score_threshold=DEMO_THRESHOLD)
    model = make_chat_model(settings)

    # ── Clear and seed to give Atlas enough candidates ────────────────────────
    cache.clear()
    print(f"Cache cleared.  score_threshold={DEMO_THRESHOLD}")
    print("Seeding cache with unrelated prompts so Atlas ANN has enough candidates...")
    for p in SEED_PROMPTS:
        model.invoke([HumanMessage(content=p)])
    print(f"  Docs after seed: {_count(cache)}\n")

    # ── CALL 1: cold cache for our target prompt ──────────────────────────────
    _sep()
    print(f"CALL 1 — Cold cache\n  Prompt : {PROMPT_A!r}")
    before = _count(cache)
    answer_a, t_a = _call(model, PROMPT_A)
    after = _count(cache)
    wrote = after > before
    print(f"  Latency: {t_a:.2f}s")
    print(f"  Cache  : {'MISS — LLM invoked, embedding + response stored in MongoDB' if wrote else 'Unexpected HIT'}")
    print(f"  Docs   : {before} → {after}")
    print(f"  Answer : {answer_a[:120].strip()}...\n")

    # ── CALL 2: exact same prompt → cache HIT ────────────────────────────────
    _sep()
    print(f"CALL 2 — Exact repeat\n  Prompt : {PROMPT_A!r}")
    before2 = _count(cache)
    answer_b, t_b = _call(model, PROMPT_A)
    after2 = _count(cache)
    hit2 = after2 == before2
    print(f"  Latency: {t_b:.2f}s  (was {t_a:.2f}s)")
    print(f"  Cache  : {'HIT  — LLM bypassed, answer served from MongoDB' if hit2 else 'MISS'}")
    print(f"  Docs   : {before2} → {after2}")
    if hit2:
        print(f"  Speedup: {t_a/t_b:.0f}x faster\n")

    # ── CALL 3: paraphrase → semantic cache HIT ───────────────────────────────
    _sep()
    print(f"CALL 3 — Paraphrase (different wording, same intent)\n  Prompt : {PROMPT_B!r}")
    before3 = _count(cache)
    answer_c, t_c = _call(model, PROMPT_B)
    after3 = _count(cache)
    hit3 = after3 == before3
    print(f"  Latency: {t_c:.2f}s")
    print(f"  Cache  : {'HIT  — LLM bypassed, semantic match in MongoDB' if hit3 else 'MISS (Atlas ANN needs more candidates — re-run or lower threshold)'}")
    print(f"  Docs   : {before3} → {after3}\n")

    # ── Summary ───────────────────────────────────────────────────────────────
    _sep()
    print("SUMMARY")
    print(f"  Call 1  cold cache  : MISS  ({t_a:.2f}s — LLM called)")
    print(f"  Call 2  exact-repeat: {'HIT ' if hit2 else 'MISS'} ({t_b:.2f}s{' — LLM bypassed' if hit2 else ''})")
    print(f"  Call 3  paraphrase  : {'HIT ' if hit3 else 'MISS'} ({t_c:.2f}s{' — LLM bypassed' if hit3 else ''})")
    print(f"\n  Open Atlas UI → travel_planner → semantic_cache")
    print(f"  Total docs: {_count(cache)}  (seeds + 1 target doc, no extra from cache hits)")


if __name__ == "__main__":
    main()
