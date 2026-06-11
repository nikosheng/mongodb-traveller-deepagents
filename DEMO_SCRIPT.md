# Workshop Runbook: DeepAgents + MongoDB Travel Planner Demo

This runbook is a practical, copy-paste guide for running the workshop end to end.

## 1) Prerequisites

- Python 3.11+
- `uv`
- Node.js 18+
- `yarn`
- MongoDB Atlas cluster (M0+)
- Azure OpenAI deployment (default model in this repo: `gpt-5.4-mini`)
- Voyage AI API key (`voyage-4` embeddings)

## 2) One-Time Setup

From repo root:

```bash
cp .env.example .env
uv sync --extra ui
```

Populate `.env` with valid credentials:

- `MONGODB_URI`
- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_API_VERSION`
- `AZURE_OPENAI_DEPLOYMENT`
- `VOYAGE_API_KEY`

## 3) Bootstrap MongoDB Atlas

Creates operational indexes, vector indexes, seeded travel data, and destination KB embeddings:

```bash
uv run python scripts/bootstrap_atlas.py
```

Notes:

- Safe to rerun (idempotent).
- Vector indexes can take 1-3 minutes to become queryable.

## 4) Launch LangGraph + UI

Open two terminals.

### Terminal 1: LangGraph server

```bash
uv run langgraph dev
```

Expected endpoints:

- API: `http://127.0.0.1:2024`
- Studio: `https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024`

### Terminal 2: UI

```bash
./scripts/run_ui.sh
```

Then open `http://localhost:3000` and in **Settings** set:

- Deployment URL: `http://127.0.0.1:2024`
- Assistant ID: `travel_planner`
- LangSmith API key: optional

## 5) How to Set Username / Workshop Parameters

### Option A: CLI (recommended for explicit user scoping)

```bash
uv run deepagents-mongodb plan \
  --user alice \
  --query "Plan a 5-day Tokyo trip for two vegetarians in May 2026, budget under $4000."
```

Useful flags:

- `--thread-id <id>` to continue the same session
- `--no-cache` to disable semantic cache

### Option B: Studio / UI message format

For workshop prompts, prefix each message with:

```text
user_id: alice
request: <your prompt>
```

Use thread IDs for demo structure:

- `thread-alice-001` for Act 1 and Act 2
- `thread-alice-002` for Act 3 (new session, persistent memory demo)

## 6) Demo Flow Questions (Copy/Paste)

### Act 1 - Cold Start

#### 1.1 Initial plan

```text
user_id: alice
request: I'd like to plan a 5-day trip to Tokyo for two people in May 2026. We're both vegetarian and flying from San Francisco. Total budget around $4,000. We love culture, food markets, and a bit of art. Please suggest boutique or ryokan-style hotels.
```

#### 1.2 Preference recall

```text
user_id: alice
request: What dietary preferences do you have on file for me?
```

### Act 2 - Update Preference + Replan

#### 2.1 Update diet and budget

```text
user_id: alice
request: Actually, I've decided I'm okay with seafood now - I'm pescatarian, not strictly vegetarian. Also, we're happy to go up to $5,000 total. Can you redo the restaurant recommendations and update the budget?
```

#### 2.2 Verify updated profile

```text
user_id: alice
request: What's my current diet preference and budget on file?
```

### Act 3 - Cross-Thread Memory (New Thread)

Create a new thread: `thread-alice-002`.

#### 3.1 New city without repeating preferences

```text
user_id: alice
request: Hi! I'd like to plan a 4-day trip to Kyoto in June 2026, also from San Francisco. Same two travelers. Please put together a full plan.
```

#### 3.2 Compare with prior trip

```text
user_id: alice
request: How does this Kyoto trip compare to the Tokyo plan we did before?
```

### Optional Bonus - Semantic Cache

Create a third thread and resend 1.1 exactly:

```text
user_id: alice
request: I'd like to plan a 5-day trip to Tokyo for two people in May 2026. We're both vegetarian and flying from San Francisco. Total budget around $4,000. We love culture, food markets, and a bit of art. Please suggest boutique or ryokan-style hotels.
```

Expected: materially faster response due to semantic cache hit.

## 7) Workshop Facilitation Sequence (Suggested)

1. Validate services are up (`langgraph dev`, UI reachable, Atlas reachable).
2. Run Act 1 in `thread-alice-001`.
3. Run Act 2 in the same thread.
4. Start `thread-alice-002` and run Act 3.
5. Optional: show cache speedup with exact prompt replay.

## 8) Verification Commands

Inspect saved profile/trips:

```bash
uv run deepagents-mongodb inspect --user alice
```

Reset and reseed demo data if needed:

```bash
uv run python scripts/reset_demo.py
uv run python scripts/bootstrap_atlas.py
```

## 9) Common Troubleshooting

- Empty memory in Act 3: rerun Act 1/2 and confirm preference save calls complete.
- Startup/auth errors: re-check `.env` values for MongoDB, Azure OpenAI, Voyage.
- No tool traces in Studio: ensure assistant is `travel_planner` and deployment URL is correct.
- Cache demo not faster: prompt must be near-identical to prior run.