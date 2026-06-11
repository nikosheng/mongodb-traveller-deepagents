# Instructor Demo Guide
## MongoDB Multi-Agent Travel Planner

---

## Overview

**Total runtime:** ~20–25 minutes  
**Difficulty:** Beginner-friendly  
**Tools on screen:** LangGraph Studio (`http://127.0.0.1:2024`)

### What This Demo Teaches

| Concept | MongoDB Feature Used |
|---|---|
| Agents query live data | Operational collections: `flights`, `hotels`, `activities`, `restaurants` |
| Agents reason over documents | Vector Search RAG on `destinations_kb` |
| Agents remember users | Long-term memory in `agent_store` (exact key lookup) |
| Memory survives new sessions | Cross-thread persistence — `agent_store` outlives any thread |
| Memory evolves over time | Single-document preference overwrite — no stale copies |
| Past trips are semantically searchable | Vector Search on `agent_store` past trips |
| Repeated prompts cost nothing | Semantic LLM cache on `semantic_cache` |

### Demo User Persona

> **Alice** — a traveler based in San Francisco.
> She initially says she's vegetarian, later changes to pescatarian.
> She prefers boutique or ryokan-style hotels and has a budget of ~$4,000.

---

## Pre-Demo Checklist

Run through this before presenting:

- [ ] `langgraph dev` is running — Studio loads at `http://127.0.0.1:2024`
- [ ] Assistant dropdown shows **`travel_planner`**
- [ ] MongoDB Atlas cluster is reachable (check `.env` for `MONGODB_URI`)
- [ ] All seed data is in place — run `uv run python scripts/bootstrap_atlas.py` if unsure
- [ ] Optional: open Atlas UI in a second browser tab to show collections live

---

## Act 1 — Cold Start: Generate the First Travel Plan

**Duration:** ~8 minutes  
**Thread:** Create a new thread — name it `thread-alice-001`

### Setup (say this out loud)

> "Alice is a brand-new user. She has no history with this system — no saved preferences,
> no past trips. Let's watch what the agent does the very first time it meets her."

---

### Step 1.1 — Paste and send this message

```
user_id: alice
request: I'd like to plan a 5-day trip to Tokyo for two people in May 2026. We're both vegetarian and flying from San Francisco. Total budget around $4,000. We love culture, food markets, and a bit of art. Please suggest boutique or ryokan-style hotels.
```

**While it runs — narrate the tool calls appearing in the Studio UI:**

| Tool call you'll see | What to say |
|---|---|
| `recall_traveler_preferences` | "First thing the orchestrator does — check if we know anything about Alice. We don't yet." |
| `destination_researcher` | "Now it searches our knowledge base in MongoDB — visa rules, best neighborhoods, transit tips for Tokyo." |
| `flight_researcher` | "Queries the flights collection — 1,350 real flight records. Filters SFO to Tokyo in May." |
| `hotel_researcher` | "156 hotels in our database. Agent filters for boutique and ryokan types under the budget." |
| `activity_planner` | "Finds activities tagged with culture, food, art. All stored in MongoDB." |
| `restaurant_recommender` | "Filters restaurants by the vegetarian tag Alice mentioned." |
| `itinerary_compiler` | "Pulls all the pieces together, computes the final budget, writes the itinerary." |
| `save_traveler_preference` | "Alice mentioned her diet, hotel style, and budget. The agent saves these to MongoDB Atlas — permanently." |
| `save_past_trip` | "One-sentence summary of this trip saved for future recall." |

**Point out in the response:**
- Hotel: Tokyo Ryokan 1 or Tokyo Boutique — ~$240–$320/night
- Flights: ~$1,200 SFO→HND Economy
- Restaurants: T's Tantan Vegan Ramen, Ain Soph Ginza, Afuri Ramen (all vegetarian-friendly)
- Grand total lands around $3,800 — under the $4,000 budget

---

### Step 1.2 — Ask about saved preferences (same thread)

```
user_id: alice
request: What dietary preferences do you have on file for me?
```

**Say:**
> "The conversation is still open — same thread. But notice the agent doesn't replan.
> It just reads Alice's profile back from MongoDB and confirms what it saved."

**Expected answer:** Agent confirms `diet: vegetarian`, explains it used that to filter restaurants.

**Talking point:**
> "This is long-term memory — not in the conversation window, but in MongoDB Atlas.
> Even if this conversation ended and a new one started, that preference is still there."

---

## Act 2 — Update a Preference and Replan

**Duration:** ~5 minutes  
**Thread:** Continue in `thread-alice-001`

### Setup (say this)

> "Preferences change. Alice just told us she's actually pescatarian — she'll eat seafood.
> She's also willing to spend a bit more. Let's watch the agent update its understanding."

---

### Step 2.1 — Paste and send

```
user_id: alice
request: Actually, I've decided I'm okay with seafood now — I'm pescatarian, not strictly vegetarian. Also, we're happy to go up to $5,000 total. Can you redo the restaurant recommendations and update the budget?
```

**While it runs — say:**
> "Watch for `save_traveler_preference` to fire twice — once for diet, once for budget.
> In MongoDB, this overwrites the same document field. There's no old copy sitting around."

**Point out in the response:**
- New restaurants appear: Sushi Saito Omakase (4.9 stars), Kozasa Yakitori
- Budget recalculated — still under the new $5,000 ceiling

---

### Step 2.2 — Confirm the overwrite

```
user_id: alice
request: What's my current diet preference and budget on file?
```

**Expected answer:** `diet: pescatarian`, `budget: up to $5,000 for two`

**Say:**
> "Old value is gone. No stale data. MongoDB stores one profile document per user —
> update the field, and the old value is replaced. Clean, no duplicates."

---

## Act 3 — Cross-Session Memory: New City, New Thread

**Duration:** ~7 minutes

### Setup (say this)

> "Now we simulate Alice coming back a week later. Different session, different destination.
> She shouldn't have to repeat herself."

**Action — in LangGraph Studio:**
1. Click **New Thread** (or change the thread ID to `thread-alice-002`)
2. The previous conversation is gone from the UI

> "The chat history is gone. But Alice's preferences are not — they're in MongoDB Atlas,
> not in the conversation window."

---

### Step 3.1 — Paste and send (new thread)

```
user_id: alice
request: Hi! I'd like to plan a 4-day trip to Kyoto in June 2026, also from San Francisco. Same two travelers. Please put together a full plan.
```

**While it runs — narrate:**

| What you'll see | What to say |
|---|---|
| `recall_traveler_preferences` fires immediately | "First call — check what we know about Alice. Watch what comes back." |
| Profile loads: `pescatarian`, `boutique/ryokan`, `$5,000` | "Everything she told us in the previous session is here. New thread, same memory." |
| `recall_past_trips` fires | "Agent also checks past trips — finds the Tokyo plan from before." |
| Restaurants filtered by pescatarian | "No extra instruction needed. The agent already knows." |
| Kyoto Ryokan or Boutique recommended | "Hotel style preference carried over automatically." |

**Point out in the response:**
- Agent greets Alice as a returning user
- Preferences applied without Alice repeating them — diet, hotel, budget all correct
- Restaurants include seafood options (Hyotei Kaiseki) because diet is now pescatarian

---

### Step 3.2 — Ask to compare trips

```
user_id: alice
request: How does this Kyoto trip compare to the Tokyo plan we did before?
```

**Say:**
> "Alice never mentioned Tokyo in this thread. The agent has to search its memory."

**While it runs:**
> "Watch `recall_past_trips` fire. It runs a vector similarity search over past trip summaries
> stored in MongoDB. It finds the Tokyo record and uses it to answer the question."

**Expected answer:** Agent compares Tokyo (~$3,800, 5 nights, vegetarian focus) with Kyoto (~$3,280, 4 nights, pescatarian options) — references both trips accurately.

---

## Bonus — Semantic Cache (Optional, ~2 minutes)

**Say:**
> "One more thing worth showing — what happens the second time you ask the same question."

**Action:** Create a third thread. Paste **the exact same message as Step 1.1** and send it.

```
user_id: alice
request: I'd like to plan a 5-day trip to Tokyo for two people in May 2026. We're both vegetarian and flying from San Francisco. Total budget around $4,000. We love culture, food markets, and a bit of art. Please suggest boutique or ryokan-style hotels.
```

**What to watch for:** Response arrives much faster — roughly 5–10s vs. ~40s the first time.

**Say:**
> "MongoDB Atlas is also acting as a semantic cache for LLM calls. When the prompt is
> similar enough to one it's seen before, it skips the model entirely and returns the
> cached response. Same quality, fraction of the cost and latency."

---

## Closing Talking Points

> "Let's recap what MongoDB did in this demo:"

1. **Operational data** — flights, hotels, activities, restaurants queried in real time with standard indexes
2. **Vector Search (RAG)** — destination knowledge retrieved by semantic similarity, not keyword matching
3. **Long-term memory** — Alice's preferences stored in Atlas, survive any thread or session boundary
4. **Single-document profile** — one update overwrites in place, no stale preference copies
5. **Past trip recall** — vector search over a user's history to answer open-ended questions
6. **Semantic LLM cache** — repeated prompts hit the cache; no redundant model calls
7. **Checkpointing** — every conversation turn is persisted; conversations are resumable

> "One database. Seven distinct roles. That's what MongoDB Atlas enables for an agentic AI system."

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `recall_traveler_preferences` returns empty in Act 3 | Preferences weren't saved in Act 1 — rerun from Step 1.1 |
| Restaurants still show vegetarian in Act 3 | `save_traveler_preference` may have failed in Step 2.1 — check tool call output in Studio |
| Agent errors on startup | Check `MONGODB_URI` and `AZURE_OPENAI_*` vars in `.env` |
| Semantic cache not hitting | Cache only hits on near-identical prompts — copy-paste exactly from Step 1.1 |
| Studio shows no tool calls | Confirm assistant is set to `travel_planner`, not the default |

---

## Message Copy-Paste Reference

### Step 1.1
```
user_id: alice
request: I'd like to plan a 5-day trip to Tokyo for two people in May 2026. We're both vegetarian and flying from San Francisco. Total budget around $4,000. We love culture, food markets, and a bit of art. Please suggest boutique or ryokan-style hotels.
```

### Step 1.2
```
user_id: alice
request: What dietary preferences do you have on file for me?
```

### Step 2.1
```
user_id: alice
request: Actually, I've decided I'm okay with seafood now — I'm pescatarian, not strictly vegetarian. Also, we're happy to go up to $5,000 total. Can you redo the restaurant recommendations and update the budget?
```

### Step 2.2
```
user_id: alice
request: What's my current diet preference and budget on file?
```

### Step 3.1 — NEW THREAD
```
user_id: alice
request: Hi! I'd like to plan a 4-day trip to Kyoto in June 2026, also from San Francisco. Same two travelers. Please put together a full plan.
```

### Step 3.2
```
user_id: alice
request: How does this Kyoto trip compare to the Tokyo plan we did before?
```

### Bonus (optional)
```
user_id: alice
request: I'd like to plan a 5-day trip to Tokyo for two people in May 2026. We're both vegetarian and flying from San Francisco. Total budget around $4,000. We love culture, food markets, and a bit of art. Please suggest boutique or ryokan-style hotels.
```
