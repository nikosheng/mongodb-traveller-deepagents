Demo Flow (4 Acts, 7 Messages)
Act 1 — Cold Start: Tokyo Plan (Thread 1)
Message 1.1 — Initial plan request:
user_id: alice
request: I'd like to plan a 5-day trip to Tokyo for two people in May 2026. We're both vegetarian and flying from San Francisco. Total budget around $4,000. We love culture, food markets, and a bit of art. Please suggest boutique or ryokan-style hotels.

我希望能安排一次在2026年7月的兩人東京5日遊。我們兩個都是素食者，將從舊金山出發。總預算大約在4,000美元左右。我們熱愛文化、美食市場和一點藝術。請推薦一些評分高的精品酒店或日式旅館風格的住宿。

Fires all 6 subagents. Saves diet=vegetarian, hotel=boutique/ryokan, budget=~$4,000 to agent_store.

Message 1.2 — In-thread recall verification:
user_id: alice
request: What dietary preferences do you have on file for me?
您記錄了我哪些飲食偏好
Demonstrates recall_traveler_preferences reading back from MongoDB.

Act 2 — Preference Update + Replan (Same Thread)
Message 2.1 — Update preferences:
user_id: alice
request: Actually, I've decided I'm okay with seafood now — I'm pescatarian, not strictly vegetarian. Also, we're happy to go up to $5,000 total. Can you redo the restaurant recommendations and update the budget?
Overwrites diet to pescatarian in the single-document profile. Adds Sushi Saito Omakase, Kozasa Yakitori.

事實上，我決定我現在可以吃海鮮了, 不是嚴格的素食者。另外，我們很高興能將總預算提高到 $5,000。你可以重新做餐廳推薦並更新預算嗎？

Message 2.2 — Confirm overwrite worked:
user_id: alice
request: What's my current diet preference and budget on file?

Act 3 — Cross-Thread Memory: Kyoto (New Thread)
Message 3.1 — Close Thread 1, open thread-alice-002, new city:

user_id: alice
request: Hi! I'd like to plan a 4-day trip to Kyoto in October 2026, from Hong Kong. Same two travelers. Please put together a full plan.
嗨！我想計劃一個 2026 年 10 月從香港出發的 4 天京都之旅。同樣是兩位旅客。請幫忙制定一個完整的計劃
Agent recalls pescatarian + boutique/ryokan + $5,000 automatically from MongoDB — Alice never repeats herself.

Message 3.2 — Cross-trip comparison via semantic search:
user_id: alice
request: How does this Kyoto trip compare to the Tokyo plan we did before?
京都行程和我們之前排的東京計劃相比怎麼樣
Hits recall_past_trips vector search on agent_store.

Bonus — Semantic Cache
Re-send Message 1.1 verbatim in a third thread. Latency drops from ~40s to ~5–10s — cache hit in semantic_cache.

上次您幫我安排的京都行程能不能再發我一下

The script also includes a table showing which MongoDB feature fires at each step (Vector Search RAG, operational B-tree queries, long-term memory, cross-thread persistence, semantic cache, checkpointing).

去大阪最佳的時間
什麼季節最適合去大阪

Call 1  "Best time to visit Osaka"                  → MISS  3.45s  (LLM called, stored)
Wait ~3s  (Atlas indexes the new vector)
Call 2  "Best time to visit Osaka"                  → HIT   0.03s  (115x faster)
Call 3  "When is the best season to go to Osaka?"   → HIT   0.41s  (paraphrase matched)
Session end → clear_session() → 1 doc deleted