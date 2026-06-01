# Request Coalescing in LLM Inference: From First Principles

---

# 1. The Problem: Redundant Work

In a high-QPS inference serving system, many requests are often **similar or identical**:

- Multiple users asking the same question ("What's the weather?", "Summarize this article")
- RAG pipelines where many queries retrieve the same top documents and share the same augmented prompt prefix
- Agent systems where every request starts with the same 4K-token system prompt
- Batch processing where thousands of items share the same prompt template with minor variations

Without coalescing, the server treats each request as fully independent — running a separate prefill, allocating separate KV cache, and generating separate outputs, even when much of the computation is identical.

**Request coalescing** recognizes and exploits this redundancy.

---

# 2. First Principles: What Can Be Shared?

In autoregressive LLM inference, there are several levels of potential sharing:

### 2.1 Prompt Prefix Sharing (Most Common)

If two requests share the same first N tokens, their KV cache entries for those N tokens are **identical** (assuming deterministic prefill, which is standard).

```
Request A: "You are a helpful assistant. Summarize: [doc A]"
Request B: "You are a helpful assistant. Summarize: [doc B]"
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            shared prefix (system prompt)
```

The KV cache for "You are a helpful assistant. Summarize:" is computed once and shared.

### 2.2 Full Prompt Deduplication

If two requests have the **exact same prompt**, the entire prefill and KV cache can be shared. The outputs will differ only if sampling is non-deterministic (temperature > 0).

### 2.3 Partial Result Sharing

If the model is generating deterministic output (greedy decoding, temperature=0) for identical prompts, even the **output tokens** are identical. The system can return a cached result instead of running inference at all.

---

# 3. Coalescing Mechanisms

### 3.1 Prefix Caching (KV Cache Level)

This is the most widely implemented form of request coalescing.

**How it works**:

1. Compute a hash of the prompt token sequence (or a prefix of it)
2. Check if this hash exists in a KV cache store
3. If yes → reuse the cached KV blocks, skip prefill for the shared prefix
4. If no → compute prefill normally, store KV blocks in the cache for future reuse

```
Cache key: hash(token_ids[0:N])
Cache value: KV blocks for layers 0..L, positions 0..N
```

**Integration with PagedAttention**: vLLM's automatic prefix caching (APC) stores KV blocks keyed by their token content. When a new request shares a prefix with a cached one, it gets **read-only references** to the same physical KV blocks. No data is copied — it's a pointer, just like virtual memory page sharing via copy-on-write.

```
Request A: pages [P1, P2, P3, P4_a, P5_a]  ← P1-P3 are shared prefix
Request B: pages [P1, P2, P3, P4_b, P5_b]  ← P1-P3 reused (same physical blocks)
```

**Savings**: For a 4K-token system prompt, prefix caching saves:
- ~100% of prefill compute for the shared prefix
- ~100% of KV cache memory for the shared portion
- TTFT drops from seconds (full 4K prefill) to milliseconds (only process the unique suffix)

### 3.2 Request Deduplication (API Level)

Before requests even reach the inference engine, an API-level deduplication layer can:

1. Hash the full request (prompt + sampling parameters)
2. Check if an identical request is **currently being processed** or has a **cached result**
3. If in-flight: attach the new caller to the same generation stream (fan-out the result)
4. If cached: return the cached result immediately

```
Request A arrives: hash=abc123 → start inference
Request B arrives: hash=abc123 → attach to A's stream (no new inference)
Request C arrives: hash=abc123 → attach to A's stream

When A's generation completes:
  → stream result to A, B, and C simultaneously
```

This is especially powerful for:
- Popular queries (trending topics, common questions)
- Health checks or status endpoints that invoke the LLM
- Template-based generation where the template is fixed

### 3.3 Semantic Coalescing (Approximate)

A more advanced form: requests that are semantically similar (but not identical) might benefit from shared computation:

- Embed the prompt → find nearest neighbors in currently processing requests
- If similarity > threshold, share the KV cache prefix up to the point where the prompts diverge

This requires embedding computation overhead and is not widely deployed, but is an active research direction.

---

# 4. Concrete Example: RAG Pipeline

A RAG system retrieves documents and builds augmented prompts:

```
10 users query within 1 second:
  User 1: "What is X?" → retrieves [doc_A, doc_B] → prompt: system + doc_A + doc_B + "What is X?"
  User 2: "Explain X"  → retrieves [doc_A, doc_B] → prompt: system + doc_A + doc_B + "Explain X"
  User 3: "Define X"   → retrieves [doc_A, doc_C] → prompt: system + doc_A + doc_C + "Define X"
  ...
```

**Without coalescing**:
- 10 full prefills, each processing ~8K tokens
- 10 separate KV caches, each ~2.6 GB (for 70B model)
- Total: ~26 GB KV cache memory, ~10× prefill compute

**With prefix caching**:
- System prompt KV (2K tokens): computed once, shared by all 10
- doc_A KV (2K tokens): computed once, shared by users who retrieved it
- doc_B, doc_C KV: computed once each, shared by respective groups
- Only the unique query suffix (~50 tokens) needs per-request prefill

- Total KV cache: ~5 GB (unique portions) + ~2.6 GB (shared, stored once) ≈ 7.6 GB
- Prefill compute: ~70% reduction (only unique suffixes computed)

**With full deduplication** (if Users 1 and 2 have temperature=0):
- If their full prompts differ only in the query suffix, they still get separate decode streams
- But if identical: a single generation serves both

---

# 5. When Coalescing Helps vs. Doesn't Help

| Scenario | Coalescing value | Why |
|---|---|---|
| API with shared system prompt | ✅ Very high | Same prefix on every request — cache hit rate near 100% |
| RAG with overlapping document retrieval | ✅ High | Popular documents retrieved by many queries |
| Multi-turn chat with same conversation history | ✅ High | Each new turn shares the previous turns' KV cache |
| Unique, diverse prompts (creative writing, personal queries) | ❌ Low | Little prefix overlap between requests |
| Batch processing with template + variable | ✅ High | Template is fixed, only the variable changes |
| High-temperature creative sampling | ⚠️ Moderate | Prefixes can be shared, but outputs diverge (can't deduplicate results) |

---

# 6. The Future of Request Coalescing

### 1. Cross-Session KV Cache Persistence
Today, prefix caches are ephemeral (lost when a server restarts or memory is reclaimed). Future systems will persist cached KV blocks to durable storage (SSD, CXL memory pools), allowing hot prefixes to survive across deployments and even across servers.

### 2. KV Cache CDNs
For popular system prompts used by millions of API calls (e.g., OpenAI's ChatGPT system prompt, Anthropic's Claude system prompt), the pre-computed KV cache could be distributed to inference servers like a CDN distributes static assets. Every server in the fleet has the prefix KV pre-loaded.

### 3. Learned Coalescing Routers
A lightweight model predicts which incoming requests will share prefixes and routes them to the same GPU — maximizing cache hit rates. This is prefix-aware load balancing, moving intelligence from hash tables to learned routing.

### 4. Prompt Normalization
Before coalescing, a normalization step rewrites semantically-equivalent prompts into a canonical form to increase cache hit rates. "Summarize this:" and "Please summarize the following:" become the same prefix after normalization.

---

# 7. The Investor Lens (Aligned with the Inference Framework)

Request coalescing sits in the **Serving / Runtime Layer** and directly impacts the economics of high-QPS inference APIs.

### Value Drivers

- **Cost reduction for API providers**: Prefix caching can reduce per-request compute cost by 50–90% for workloads with shared system prompts (which is the vast majority of production deployments). This is one of the largest single-optimization cost reductions available in inference serving.
- **The commoditization cascade applies fast**: vLLM's automatic prefix caching is already open-source. SGLang has RadixAttention (even more granular prefix sharing). Within 12 months, prefix caching is table-stakes for every serving runtime. Early adopters capture margin temporarily; latecomers face cost disadvantages.
- **TTFT improvement drives product differentiation**: When prefix caching eliminates 90% of prefill compute, TTFT drops from seconds to sub-200ms. This UX improvement is directly visible to users and drives preference for products that feel "instant."
- **The "system prompt tax" disappears**: Today, every API call pays for processing the system prompt (often 1–4K tokens). Prefix caching makes this cost amortized to near-zero at scale. This shifts the cost structure from "pay per prompt token" to "pay per unique content token" — a fundamental change in inference economics.

### The Broader Strategic Implication

Request coalescing is one mechanism in the broader Jevons paradox cycle:
- Coalescing reduces cost-per-request → providers can lower API prices
- Lower prices → more applications integrate LLM calls → more total requests
- More requests → higher cache hit rates (more shared prefixes) → even lower per-request cost
- This positive feedback loop accelerates adoption while maintaining or growing total inference demand

### Risk Factor

If model architectures shift toward SSMs (which don't have KV caches), the entire prefix caching infrastructure becomes irrelevant. The coalescing logic would need to adapt to whatever state representation replaces KV cache — likely a simpler problem (fixed-size recurrent state), but the current engineering investment in PagedAttention-based prefix sharing would be stranded.

### Summary Signal

> Request coalescing — especially prefix caching — is the single highest-ROI serving optimization for production LLM APIs today. It reduces compute and memory costs by 50–90% for common workloads, dramatically improves TTFT, and creates a positive feedback loop between cost reduction and adoption growth. Every inference provider that doesn't implement it faces a structural cost disadvantage. But it commoditizes fast — within 12 months of open-source availability, it becomes table-stakes.
