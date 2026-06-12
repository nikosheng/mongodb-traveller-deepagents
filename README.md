# deepagents-mongodb — Travel Planner Demo

End-to-end demonstration of [LangChain **deepagents**](https://github.com/langchain-ai/deepagents) integrated with **MongoDB Atlas** for a multi-subagent **AI Travel Planner**.

This project shows, in one runnable codebase, how to use MongoDB Atlas as the backbone for an agentic application:

| MongoDB Atlas capability | Component used |
| --- | --- |
| Short-term memory (per-thread state, resumable) | `MongoDBSaver` checkpointer (`langgraph-checkpoint-mongodb`) |
| Long-term shared memory across subagents & sessions | `MongoDBStore` with `voyage-4` vector index (`langgraph-store-mongodb`) |
| Domain knowledge retrieval (RAG) | `MongoDBAtlasVectorSearch` + `create_retriever_tool` |
| Semantic LLM cache | `MongoDBAtlasSemanticCache` set as the global LangChain LLM cache |
| Operational data store (flights/hotels/etc.) | Regular MongoDB collections with B-tree indexes |

The travel planner uses **6 specialized subagents** orchestrated by a deepagent:

1. `destination_researcher` — RAG over `destinations_kb`
2. `flight_researcher`
3. `hotel_researcher`
4. `activity_planner`
5. `restaurant_recommender`
6. `itinerary_compiler`

---

## 1. Prerequisites

- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/) (recommended) or `pip`
- A MongoDB Atlas cluster (free M0 works) — needed for `$vectorSearch`
- Azure OpenAI deployment for `gpt-5.4-mini` (or any chat model — see `src/deepagents_mongodb/llm.py`)
- Voyage AI API key (for `voyage-4` embeddings, 1024 dims)

If you do not have `uv` installed yet:

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Verify installation:

```bash
uv --version
```

## 2. Install

```bash
uv sync                 # or: pip install -e .
cp .env.example .env    # then fill in Azure OpenAI credentials
```

`MONGODB_URI` and `VOYAGE_API_KEY` are already set in the provided `.env`.

## 3. Bootstrap MongoDB Atlas

Creates collections, vector search indexes, and loads all mocked travel data:

```bash
uv run python scripts/bootstrap_atlas.py
```

This is **idempotent** — safe to re-run. Vector indexes can take 1–2 minutes to become queryable; the script waits for them.

## 4. Run the demo (UI path)

Start the LangGraph server and the UI as described in Section 5, then follow
the full workshop flow (with copy/paste prompts) in Section 9.

If you need a clean reset before a workshop run:

```bash
uv run python scripts/reset_demo.py
uv run python scripts/bootstrap_atlas.py
```

## 5. Interactive UI (deep-agents-ui)

For an interactive chat experience over the same agent, plug in the official
[deep-agents-ui](https://github.com/langchain-ai/deep-agents-ui) — a Next.js
front-end that speaks to any LangGraph deployment. We expose the travel
planner via `langgraph dev` so the UI can drive it without code changes.

### One-time setup

```bash
uv sync --extra ui          # installs langgraph-cli[inmem]
```

You also need **Node 18+** and **yarn** on `PATH` for the UI side.

### Terminal 1 — start the LangGraph server

```bash
uv run langgraph dev
# → API:    http://127.0.0.1:2024
# → Studio: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
```

This loads `langgraph.json`, which points at
`src/deepagents_mongodb/graph.py:agent`. The graph is the same travel
planner implementation used in this repo, but **without** the `MongoDBSaver` checkpointer
(langgraph dev injects its own in-memory checkpointer). The
**`MongoDBStore`** (long-term memory) and **semantic cache** are still wired
in, so recall / preference tools and cache hits all demo against Atlas.

### Terminal 2 — start the UI

Use the customized UI that is already in this repository:

```bash
cd ui
yarn install
yarn dev
```

This starts the local customized UI on <http://localhost:3000>.

### Connect the UI to the agent

Open <http://localhost:3000>, click **Settings**, and fill in:

| Field            | Value                     |
| ---------------- | ------------------------- |
| Deployment URL   | `http://127.0.0.1:2024`   |
| Assistant ID     | `travel_planner`          |
| User ID          | `(e.g. alice)`          |
| LangSmith API key | *(optional)*             |

Send a prompt such as the demo query above. As the agent runs you will see:

- the orchestrator's `write_todos` plan,
- each sub-agent's messages and tool calls (`task` tool spawns),
- the virtual filesystem updating live — click `itinerary.md` to view it.

### What's persisted in UI mode

| Layer                                | Storage                              |
| ------------------------------------ | ------------------------------------ |
| Conversation state / threads          | In-memory (`langgraph dev`)         |
| Long-term memory (prefs, past trips)  | **MongoDB Atlas** (`MongoDBStore`)  |
| Destination knowledge (RAG)           | **MongoDB Atlas** (`$vectorSearch`) |
| LLM semantic cache                    | **MongoDB Atlas**                   |

Note: in UI mode with `langgraph dev`, conversation threads are in-memory and
reset when the dev server restarts.

## 6. Architecture

```
                 ┌──────────────────────────────────────────────────┐
                 │            Travel Orchestrator (deepagent)        │
                 │     write_todos │ task │ save/recall preference   │
                 └──────────────┬───────────────┬───────────────────┘
                                │  task() spawns isolated sub-threads
   ┌────────────────────────────┼────────────────────┐
   ▼                ▼           ▼           ▼        ▼          ▼
destination_     flight_     hotel_      activity_  restaurant_  itinerary_
researcher       researcher  researcher  planner    recommender  compiler
   │                │           │            │           │           │
   │ vector_search  │ MongoDB   │ MongoDB    │ MongoDB   │ MongoDB   │ writes
   │ destinations_kb│ flights   │ hotels     │ activities│ restaurants│ itinerary.md
   ▼                ▼           ▼            ▼           ▼           ▼
            ┌──────────────────── MongoDB Atlas ──────────────────────┐
            │ checkpoints  │ agent_store (vec)  │ semantic_cache (vec)│
            └──────────────────────────────────────────────────────────┘
```

## 7. File layout

See the repository tree — every module is documented inline with what MongoDB feature it demonstrates.

## 8. Operational notes

### Atlas Vector Search quota
M0 / M2 / M5 clusters allow at most **3 vector search indexes per cluster**. The demo creates exactly 3 (`agent_store`, `semantic_cache`, `destinations_kb`). If the bootstrap fails with `The maximum number of FTS indexes has been reached`, list and drop unused ones:

```bash
uv run python scripts/list_search_indexes.py
uv run python scripts/list_search_indexes.py --drop <db>.<coll>::<index_name>
```

### Voyage AI rate limits
The free Voyage tier is **3 RPM / 10 K TPM**. Every long-term memory recall and every semantic cache lookup costs one embedding call, so a single plan triggers ~3–6 embed requests. `src/deepagents_mongodb/llm.py` wraps `voyage-4` with:

- an in-memory LRU cache (so repeated identical queries don't re-embed), and
- exponential-backoff retries on 429s.

This is enough to push a full plan through the free tier in 60–90 s. For production, add a Voyage payment method and the limits jump dramatically.

### Architecture decision: memory recall happens once, on the orchestrator
To minimise embedding calls, subagents do **not** hold the memory tools. The orchestrator:

1. Recalls preferences from MongoDBStore once at the start.
2. Passes the relevant facts to each subagent via the `task` description.
3. Saves any newly-learned preferences once at the end.

This is the recommended deepagents pattern: keep subagent tool-sets minimal and let the orchestrator do the cross-cutting state work.

## 9. Workshop runbook (full demo flow)

This section consolidates the full live-demo script so you can run the workshop
end to end from one file.

### 9.1 One-time setup

```bash
cp .env.example .env
uv sync --extra ui
```

Populate `.env` with:

- `MONGODB_URI`
- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_API_VERSION`
- `AZURE_OPENAI_DEPLOYMENT`
- `VOYAGE_API_KEY`

### 9.2 Bootstrap Atlas

```bash
uv run python scripts/bootstrap_atlas.py
```

Notes:

- Safe to rerun (idempotent)
- Vector indexes can take 1-3 minutes to become queryable

### 9.3 Launch LangGraph + UI

Open two terminals.

Terminal 1:

```bash
uv run langgraph dev
```

Expected endpoints:

- API: `http://127.0.0.1:2024`
- Studio: `https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024`

Terminal 2:

```bash
./scripts/run_ui.sh
```

Then open `http://localhost:3000` and set:

- Deployment URL: `http://127.0.0.1:2024`
- Assistant ID: `travel_planner`
- LangSmith API key: optional

### 9.4 User identity and thread strategy (UI)

Use this message format in Studio/UI:

```text
user_id: alice
request: <your prompt>
```

Recommended thread IDs for the demo:

- `thread-alice-001` for Act 1 and Act 2
- `thread-alice-002` for Act 3 (new session, cross-thread memory)

### 9.5 Full demo flow (EN + 中文)

#### Act 1 - Cold start

1) Initial plan (EN)

```text
user_id: alice
request: I'd like to plan a 5-day trip to Tokyo for two people in May 2026. We're both vegetarian and flying from San Francisco. Total budget around $4,000. We love culture, food markets, and a bit of art. Please suggest boutique or ryokan-style hotels.
```

1) 初始规划（中文）

```text
user_id: alice
request: 我想规划一次 2026 年 5 月去东京的 5 天双人旅行。我们两人都是素食者，从旧金山出发。总预算大约 4000 美元。我们喜欢文化体验、美食市场和一些艺术活动。请推荐精品酒店或日式旅馆（ryokan）风格的住宿。
```

2) Preference recall (EN)

```text
user_id: alice
request: What dietary preferences do you have on file for me?
```

2) 偏好回顾（中文）

```text
user_id: alice
request: 你目前记录的我的饮食偏好是什么？
```

#### Act 2 - Update preference + replan

1) Update diet and budget (EN)

```text
user_id: alice
request: Actually, I've decided I'm okay with seafood now - I'm pescatarian, not strictly vegetarian. Also, we're happy to go up to $5,000 total. Can you redo the restaurant recommendations and update the budget?
```

1) 更新偏好和预算（中文）

```text
user_id: alice
request: 我改变主意了，我现在可以吃海鲜了，也就是偏海鲜素（pescatarian），不再是严格素食。另外，我们总预算可以提高到 5000 美元。请重新给出餐厅推荐并更新预算方案。
```

2) Verify updated profile (EN)

```text
user_id: alice
request: What's my current diet preference and budget on file?
```

2) 验证更新后的用户画像（中文）

```text
user_id: alice
request: 你当前记录的我的饮食偏好和预算分别是什么？
```

#### Act 3 - Cross-thread memory (new thread)

Create a new thread: `thread-alice-002`.

1) New city without repeating preferences (EN)

```text
user_id: alice
request: Hi! I'd like to plan a 4-day trip to Kyoto in June 2026, also from San Francisco. Same two travelers. Please put together a full plan.
```

1) 新城市规划（不重复偏好）（中文）

```text
user_id: alice
request: 你好！我想规划一次 2026 年 6 月去京都的 4 天旅行，也从旧金山出发，还是两位旅行者。请直接给我一个完整方案。
```

2) Compare with prior trip (EN)

```text
user_id: alice
request: How does this Kyoto trip compare to the Tokyo plan we did before?
```

2) 与之前行程对比（中文）

```text
user_id: alice
request: 这次京都行程和我们之前做的东京方案相比，有哪些差异？
```

#### Semantic cache

Create a third thread (for example, `thread-alice-003`) and run the following
three calls in order.

Expected pattern:

- Call 1: `MISS` (LLM response is generated and cached)
- Wait about 2-5 seconds for Atlas vector indexing
- Call 2: `HIT` (same request, much faster)
- Call 3: `HIT` (paraphrase request, still matches semantically)

Reference timings (example):

- Call 1: ~3.45s (MISS)
- Call 2: ~0.03s (exact-hit, ~115x faster)
- Call 3: ~0.41s (paraphrase-hit)

1) Cold call (EN)

```text
user_id: alice
request: Best time to go to Osaka
```

1) 首次请求（中文）

```text
user_id: alice
request: 去大阪最佳的時間
```

2) Repeat the same request (expect cache HIT)

```text
user_id: alice
request: Best time to go to Osaka
```

```text
user_id: alice
request: 去大阪最佳的時間
```

3) Rephrase the request (expect semantic HIT)

```text
user_id: alice
request: When is the best season to go to Osaka?
```

```text
user_id: alice
request: 什麼季節最適合去大阪
```

Check the `langgraph dev` console for cache evidence. You should see lines like:

```text
SemanticCache MISS ... text='Best time to go to Osaka'
SemanticCache HIT  ... score=1.0000 ... text='去大阪最佳的時間'
SemanticCache HIT  ... score=0.95xx ... text='什麼季節最適合去大阪'
```

### 9.6 Suggested facilitation sequence

1. Validate services are up (`langgraph dev`, UI reachable, Atlas reachable)
2. Run Act 1 in `thread-alice-001`
3. Run Act 2 in the same thread
4. Start `thread-alice-002` and run Act 3
5. Optionally show cache speedup with exact prompt replay

### 9.7 Verification and reset

Reset and reseed if needed:

```bash
uv run python scripts/reset_demo.py
uv run python scripts/bootstrap_atlas.py
```

### 9.8 Common troubleshooting

- Empty memory in Act 3: rerun Act 1/2 and confirm preference save calls complete
- Startup/auth errors: re-check `.env` values for MongoDB, Azure OpenAI, Voyage
- No tool traces in Studio: ensure assistant is `travel_planner` and deployment URL is correct
- Cache demo not faster: prompt must be near-identical to prior run

