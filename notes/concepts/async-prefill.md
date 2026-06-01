# Asynchronous Prefill in LLM Inference: From First Principles

---

## 1. The Problem: Prefill Blocks Everything

To understand asynchronous prefill, you first need to understand why **synchronous prefill** is a bottleneck.

### How a standard inference step works

Recall the two phases of LLM inference:

1. **Prefill phase**: Process the entire input prompt (e.g., 2,000 tokens) in a single forward pass. This computes all Q, K, V vectors for every prompt token, populates the KV cache, and produces the first output token. This is **compute-heavy** — it involves full matrix multiplications across the entire prompt length.

2. **Decode phase**: Generate tokens one at a time, autoregressively. Each step processes only the new token's Q vector against the cached K, V from all previous tokens. This is **memory-bandwidth-heavy** — each step reads the full model weights and KV cache from HBM but does relatively little compute per token.

### The blocking problem in continuous batching

In a continuous batching system, the GPU runs a decode step for all active sequences every iteration. When a new request arrives:

1. The scheduler pulls the new request from the waiting queue
2. The GPU must **prefill** this new request — process its entire prompt
3. **During prefill, all active decode sequences are stalled.** The GPU can't decode existing sequences and prefill a new one at the same time (in the synchronous model)
4. After prefill completes, the new sequence joins the decode batch and normal iteration-level scheduling resumes

The cost of this stall depends on the prompt length:

| Prompt length | Approximate prefill time (70B model, H100) | Decode steps missed |
|---|---|---|
| 100 tokens | ~2 ms | ~0 (negligible) |
| 1,000 tokens | ~15 ms | ~1 step |
| 10,000 tokens | ~150 ms | ~10–15 steps |
| 100,000 tokens | ~1,500 ms | ~100+ steps |

For a 100K-token prompt, **every existing user in the decode batch experiences a 1.5-second pause** — no tokens generated for any of them. In a system serving 50 concurrent users, that's 50 users seeing a visible latency spike because one new request arrived with a long prompt.

This is the **prefill interference problem**: long prefills degrade the latency of all other active sequences.

---

## 2. The Solution: Asynchronous Prefill

Asynchronous prefill decouples the prefill computation from the decode loop so that they no longer block each other. There are two main approaches:

### Approach A: Chunked Prefill (Interleaved Prefill)

Instead of processing the entire prompt in one monolithic forward pass, break it into smaller chunks and **interleave** them with decode steps:

```
Standard (synchronous):
  [========== PREFILL (10K tokens) ==========][decode][decode][decode]...
  ↑ all decode sequences stalled for ~150ms

Chunked (async-style):
  [prefill chunk 1 (1K tokens)][decode][prefill chunk 2 (1K)][decode][prefill chunk 3]...
  ↑ decode sequences get a step between every chunk
```

#### How it works from first principles:

1. The scheduler splits the incoming prompt into chunks of size C (e.g., C = 512 or 1,024 tokens)
2. Each iteration, the GPU processes **one prefill chunk** plus **one decode step for all active sequences** in the same forward pass
3. The prefill chunk's Q, K, V are computed and its KV cache is incrementally extended
4. After all chunks are processed, the sequence transitions to decode phase and joins the batch normally

**Why this works mechanically:**

The key insight is that prefill chunks and decode tokens are both just "tokens being processed through the transformer." The GPU can pack them into the same batch:

- Active decode sequences contribute 1 token each
- The prefill chunk contributes C tokens
- Total tokens in the forward pass = B_decode + C

The GPU processes all of them together. The prefill chunk computes attention only against its own prior chunks (causal mask), while decode tokens attend to their full KV caches. These are independent attention computations that can be batched.

**The trade-off:**

| Chunk size C | Prefill throughput | Decode interference |
|---|---|---|
| Very large (= full prompt) | Maximum (one pass) | Maximum (long stall) |
| 1,024 tokens | High | Low (~5 ms per chunk) |
| 256 tokens | Moderate | Very low (~1.5 ms) |
| 64 tokens | Low (many iterations to finish) | Negligible |

Smaller chunks → less interference but more total iterations to complete prefill (overhead from repeated kernel launches, reduced compute efficiency of small matmuls). Larger chunks → faster prefill but more disruption to decode latency.

In practice, **C = 512–2,048** is the sweet spot, giving <5ms per-chunk overhead while maintaining high prefill throughput.

#### What systems implement this:

- **vLLM**: Chunked prefill is available and configurable via `--enable-chunked-prefill` with tunable `max_num_batched_tokens`
- **SGLang**: Default chunked prefill with adaptive chunk sizing
- **TensorRT-LLM**: Inflight batching with mixed prefill/decode tokens per iteration

### Approach B: Disaggregated Prefill (True Async)

The more radical approach: run prefill on **entirely separate hardware** from decode. The prefill cluster and decode cluster operate independently and asynchronously.

```
PREFILL CLUSTER (compute-optimized GPUs):
  Request arrives → prefill entire prompt → produce KV cache → send KV cache to decode cluster

DECODE CLUSTER (bandwidth-optimized GPUs):
  Receive KV cache → add sequence to decode batch → generate tokens → stream to user
```

#### How it works from first principles:

1. **Request arrives** at the router/scheduler
2. Router sends the prompt to the **prefill cluster** — a pool of GPUs optimized for compute-heavy workloads (high FLOP/s, less HBM needed since no long-running KV caches accumulate)
3. The prefill cluster processes the entire prompt, producing the KV cache (all K, V vectors for every layer)
4. The KV cache is **transferred** over the network (NVLink, InfiniBand, or PCIe) to the **decode cluster**
5. The decode cluster receives the KV cache, adds the sequence to its continuous batching loop, and begins generating tokens
6. **At no point does prefill interfere with decode** — they run on different hardware

#### Why separate hardware makes sense:

Prefill and decode have fundamentally different hardware requirements:

| Property | Prefill | Decode |
|---|---|---|
| **Bottleneck** | Compute (FLOP/s) | Memory bandwidth (GB/s) |
| **Batch behavior** | Single large sequence per GPU | Many small sequences per GPU |
| **KV cache memory** | Transient (produced, then sent away) | Persistent (must hold all active sequences) |
| **Ideal hardware** | Compute-dense (high FLOP/s, moderate HBM) | Bandwidth-dense (high HBM capacity and bandwidth) |
| **GPU utilization pattern** | Bursty (one request at a time, high intensity) | Steady (continuous decode loop) |

Putting both on the same GPU forces a compromise: the GPU must have enough FLOP/s for prefill AND enough HBM for all active decode sequences' KV caches. Disaggregation lets each cluster be right-sized.

#### The KV cache transfer problem:

The main engineering challenge is transferring the KV cache between clusters. For a 70B model with GQA and context length 4,096:

$$
\text{KV cache size} = 2 \times 80 \text{ layers} \times 8 \text{ KV heads} \times 128 \text{ dim} \times 4096 \text{ tokens} \times 2 \text{ bytes} = 1.34 \text{ GB}
$$

Transfer times:

| Interconnect | Bandwidth | Transfer time for 1.34 GB |
|---|---|---|
| InfiniBand HDR (200 Gbps) | ~25 GB/s | ~54 ms |
| InfiniBand NDR (400 Gbps) | ~50 GB/s | ~27 ms |
| NVLink 4 (intra-node) | 900 GB/s | ~1.5 ms |
| PCIe Gen 5 | ~64 GB/s | ~21 ms |

For NVLink (same node, different GPU), transfer is nearly free. For cross-node (InfiniBand), there's a 27–54 ms overhead per request. This must be weighed against the latency savings from eliminating prefill interference.

**When disaggregation wins:** High-throughput serving with many concurrent users AND long prompts. The prefill interference cost on shared hardware exceeds the KV cache transfer cost.

**When it doesn't win:** Low-traffic scenarios, very short prompts (prefill is fast enough not to interfere), or systems without high-bandwidth interconnects.

---

## 3. Chunked Prefill: The Mechanics in Detail

### 3.1 The Scheduling Algorithm

The scheduler must decide, every iteration, what to include in the batch:

```python
def schedule_iteration(active_sequences, prefilling_sequences, waiting_queue, max_tokens):
    batch = []
    token_budget = max_tokens  # e.g., 2048 total tokens per iteration

    # 1. Always include active decode sequences (1 token each)
    for seq in active_sequences:
        batch.append((seq, 'decode', 1))
        token_budget -= 1

    # 2. Continue any in-progress prefills
    for seq in prefilling_sequences:
        chunk_size = min(seq.remaining_prefill_tokens, token_budget)
        if chunk_size > 0:
            batch.append((seq, 'prefill_chunk', chunk_size))
            token_budget -= chunk_size
            seq.remaining_prefill_tokens -= chunk_size

    # 3. Start new prefills if budget remains
    while token_budget > 0 and waiting_queue:
        new_seq = waiting_queue.pop()
        chunk_size = min(new_seq.prompt_length, token_budget)
        batch.append((new_seq, 'prefill_chunk', chunk_size))
        token_budget -= chunk_size
        new_seq.remaining_prefill_tokens = new_seq.prompt_length - chunk_size
        if new_seq.remaining_prefill_tokens > 0:
            prefilling_sequences.append(new_seq)
        else:
            active_sequences.append(new_seq)  # Prefill complete, joins decode

    return batch
```

The `max_tokens` budget controls the trade-off:
- More tokens allocated to prefill chunks → faster prefill, but decode batch has less room
- More tokens allocated to decode → more concurrent users, but new requests wait longer

### 3.2 Attention Masking Complexity

When prefill chunks and decode tokens share a forward pass, the attention computation needs careful masking:

- **Decode tokens**: Each decode token attends to its full KV cache (all past tokens). Standard causal attention.
- **Prefill chunk tokens**: Each token in the chunk attends to (a) all previous chunks' KV cache (already stored), and (b) tokens earlier in the *current* chunk. They do NOT attend to decode tokens from other sequences.

The attention mask becomes a block-diagonal matrix plus the causal triangle within each prefill chunk. This is more complex than standard batched attention, but modern kernels (FlashAttention variants, FlashInfer) handle it efficiently using ragged/variable-length attention APIs.

### 3.3 KV Cache Incrementally Built

With chunked prefill, the KV cache for the new sequence is built over multiple iterations:

```
Iteration 1: Process tokens 1–512    → KV cache has entries for tokens 1–512
Iteration 2: Process tokens 513–1024 → KV cache extended to tokens 1–1024
Iteration 3: Process tokens 1025–1536 → KV cache extended to tokens 1–1536
...
Iteration N: Process final chunk → KV cache complete → sequence starts decode
```

The PagedAttention block manager allocates new physical blocks as each chunk extends the KV cache. This is seamless — it's the same mechanism used when decode tokens extend the cache one token at a time, just done in chunks.

---

## 4. Concrete Example: The Difference Async Prefill Makes

### Scenario: 30 active decode sequences, new request arrives with 8,192-token prompt

**Without async prefill (synchronous):**

```
Timeline:
  t=0ms    : New request arrives
  t=0-120ms: GPU runs prefill for all 8,192 tokens (single forward pass)
             → All 30 decode sequences STALLED for 120ms
             → Each user sees a ~120ms gap in token streaming
  t=120ms  : Decode resumes with 31 sequences in batch
```

Impact: 30 users × 120ms = **3,600 user-milliseconds of added latency.**

At 30ms per decode step (ITL), that's 4 missed tokens per user. If they're watching a streaming response, they notice a visible pause.

**With chunked prefill (C=1024):**

```
Timeline:
  t=0ms   : New request arrives
  t=0-5ms : Decode step for 30 sequences + prefill chunk 1 (tokens 1–1024)
  t=5-10ms: Decode step for 30 sequences + prefill chunk 2 (tokens 1025–2048)
  t=10-15ms: Decode step for 30 sequences + prefill chunk 3 (tokens 2049–3072)
  ...
  t=35-40ms: Decode step for 30 sequences + prefill chunk 8 (tokens 7169–8192)
  t=40ms   : Prefill complete, sequence 31 joins decode batch
```

Impact: Each decode iteration takes ~5ms instead of ~3ms (the extra 2ms is the prefill chunk's compute). No user experiences a pause longer than ~2ms above normal ITL. Total prefill takes ~40ms (vs. 120ms synchronous, because of reduced compute efficiency from smaller chunks), but **no user notices any interruption.**

**With disaggregated prefill:**

```
Timeline (decode cluster):
  t=0ms   : Business as usual — decode step for 30 sequences every 3ms
  t=27ms  : KV cache for new request arrives from prefill cluster (transferred via InfiniBand)
  t=27ms  : Sequence 31 added to decode batch
  t=30ms  : Decode step for 31 sequences
```

Impact: **Zero interference.** Decode sequences never see any delay. The only "cost" is the 27ms network transfer, which the user experiences as slightly higher TTFT. But no other user is affected at all.

---

## 5. The Future of Asynchronous Prefill

### 5.1 Adaptive Chunk Sizing

Current systems use a fixed chunk size. Future schedulers will dynamically adjust the chunk size based on system state:

- **Many active decode sequences, high load**: Smaller chunks (256 tokens) to minimize interference
- **Few active decode sequences, low load**: Larger chunks (4,096 tokens) to maximize prefill throughput since there's less to interfere with
- **Latency-sensitive requests in the decode batch**: Even smaller chunks or defer prefill entirely
- **Prefill-only requests (batch processing)**: No chunking needed — run full prefill at maximum throughput

The scheduler learns the optimal chunk-size policy from real traffic patterns, treating it as a multi-objective optimization (minimize decode ITL degradation, minimize prefill TTFT, maximize total throughput).

### 5.2 Priority-Aware Prefill Scheduling

Not all requests are equal. A premium-tier user's prefill should preempt a free-tier user's decode:

- Requests annotated with priority levels (real-time, batch, background)
- The prefill scheduler allocates larger chunks to high-priority requests, even if it slightly degrades low-priority decode sequences
- Background/batch requests are prefilled only during low-traffic windows, using whatever token budget remains after serving active decode sequences

### 5.3 Speculative Prefill

Predict which requests will arrive next and prefill them preemptively:

- In agentic systems, the orchestrator knows which tool calls it will make next
- Pre-prefill the system prompt + likely context before the actual request arrives
- When the request does arrive, the KV cache is already populated → TTFT drops to near-zero

This requires tight integration between the application layer and the serving layer — the app must signal "I'm about to send a request with this prefix" so the serving system can pre-warm the KV cache.

### 5.4 Full Disaggregation Becomes Default

As models grow larger and context windows extend to millions of tokens, the case for disaggregated prefill strengthens:

- Prefill for a 10M-token prompt takes minutes on a single GPU — it must be parallelized across many GPUs (context parallelism on the prefill cluster)
- The decode cluster never sees these massive prefills — it only receives the finalized KV cache
- The KV cache transfer is pipelined: the prefill cluster sends blocks as they're computed, and the decode cluster starts decoding early tokens while later chunks are still being prefilled

This "streaming prefill" pattern — where decode begins before prefill is fully complete — collapses TTFT further by overlapping the two phases across the network boundary.

---

## 6. The Investor Lens

Asynchronous prefill sits in the **Serving/Runtime Layer** of the inference stack. It is a scheduling innovation that directly affects latency SLAs, user experience, and therefore pricing power.

### Core Thesis

> **Asynchronous prefill eliminates the last major source of latency unpredictability in LLM serving. In a world where inference is priced on latency tiers, the ability to guarantee consistent ITL (inter-token latency) — regardless of what other requests are arriving — is a direct margin lever. Prefill interference is the reason users experience "stuttering" in streaming responses; async prefill is what makes streaming feel like a premium product.**

### Primary Investment Implications

#### 1. Latency Consistency Is Pricing Power

The difference between "average ITL = 30ms" and "guaranteed ITL ≤ 35ms at p99" is enormous for enterprise buyers:

- **Without async prefill**: Average ITL might be 30ms, but p99 ITL spikes to 150ms+ whenever a long-context request triggers a synchronous prefill. The provider cannot offer a tight SLA.
- **With async prefill**: p99 ITL stays within 5–10ms of the average, because prefill never blocks decode. The provider can offer — and charge for — a guaranteed latency SLA.

Enterprise customers pay premiums for SLAs. A provider that can guarantee p99 ITL ≤ 40ms commands 2–3× the per-token price of one that can only promise "best effort." Async prefill is what makes that guarantee deliverable.

**Investor takeaway**: When evaluating inference providers, ask about p99 latency guarantees, not just average throughput. Providers offering tight SLAs almost certainly have chunked or disaggregated prefill. Those that don't are leaving premium pricing on the table.

#### 2. Disaggregated Prefill Enables Hardware Specialization

When prefill and decode run on the same GPU, hardware selection is a compromise. Disaggregation lets each cluster use purpose-built hardware:

| Cluster | Optimal hardware | Cost optimization |
|---|---|---|
| **Prefill** | High FLOP/s, moderate HBM (e.g., dense GPUs or TPUs) | Right-size for compute, don't overpay for memory capacity |
| **Decode** | High HBM bandwidth and capacity, moderate FLOP/s (e.g., HBM-heavy GPUs, bandwidth-optimized ASICs) | Right-size for bandwidth, don't overpay for compute |

This specialization can reduce total infrastructure cost by 20–40% compared to using identical GPUs for both phases, because you're not paying for capabilities each phase doesn't need.

**Investor takeaway**: Disaggregated serving creates demand for **heterogeneous GPU/ASIC fleets** — not just "buy more H100s." Watch for inference providers announcing mixed-hardware deployments (e.g., compute-optimized GPUs for prefill + bandwidth-optimized GPUs or ASICs for decode). This is a signal of mature infrastructure optimization and likely superior unit economics.

#### 3. TTFT vs. ITL Trade-Off Creates Pricing Tiers

Async prefill introduces a natural trade-off that maps to pricing tiers:

- **Low TTFT, low ITL (premium)**: Disaggregated prefill with NVLink transfer, large token budget for prefill chunks, priority scheduling. Expensive infrastructure, high price.
- **Moderate TTFT, low ITL (standard)**: Chunked prefill with moderate chunk sizes. Decode barely affected, prefill takes a few extra iterations. Reasonable cost.
- **Best-effort TTFT, best-effort ITL (economy)**: Synchronous prefill during low-load windows. Prefill interference accepted. Low cost.

This tiering is structurally similar to cloud compute tiers (on-demand vs. spot pricing) and creates predictable revenue segmentation:

- Premium tier: Real-time applications (voice assistants, co-pilots, live customer support)
- Standard tier: Interactive chat, developer tools
- Economy tier: Batch processing, background analysis, non-latency-sensitive workflows

**Investor takeaway**: Inference providers that implement latency-tiered pricing backed by async prefill infrastructure can extract maximum revenue across the demand curve. The premium tier has high gross margins; the economy tier fills spare capacity. This is the "yield management" model applied to inference — the same principle that makes airlines and hotels profitable.

#### 4. Async Prefill Makes Continuous Batching Actually Work at Scale

Without async prefill, continuous batching has a hidden failure mode: at high load with long-context requests, prefill interference degrades ITL for everyone, creating a negative feedback loop:

```
High load → more requests → more prefills → ITL spikes → SLA violations
→ load balancer routes to other instances → remaining instance even more loaded
→ cascading degradation
```

Async prefill breaks this loop. No matter how many long-context requests arrive, decode sequences maintain consistent ITL. This makes continuous batching predictably scalable, which is essential for capacity planning.

**Investor takeaway**: Async prefill is a prerequisite for reliable large-scale inference. Any provider serving >1,000 concurrent users without chunked or disaggregated prefill will hit unpredictable latency spikes that damage reliability metrics and customer trust. When evaluating infrastructure at scale, async prefill is table-stakes, not a differentiator.

#### 5. The Commoditization Timeline

| Technique | Stage (2025) | Time to commodity |
|---|---|---|
| Chunked prefill (fixed chunk size) | Fully commoditized (vLLM, SGLang, TRT-LLM) | Already there |
| Adaptive chunk sizing | Early production | 6–12 months |
| Disaggregated prefill/decode | Production at frontier providers (Databricks, some hyperscalers) | 12–18 months |
| KV cache streaming (prefill→decode pipelining) | Research/early prototypes | 18–24 months |
| Speculative prefill (pre-warming) | Research | 24+ months |

**Investor takeaway**: Basic chunked prefill is already commoditized — it's not a moat. Disaggregated prefill is the current competitive frontier (12–18 months from commodity). The durable advantage isn't any single technique but the **orchestration depth** — the ability to dynamically balance prefill throughput, decode latency, and total system throughput across heterogeneous hardware under unpredictable traffic.

### Risk Factors

**Risk 1 — Short prompts dominate.** If the dominant use case remains short-prompt chatbots (100–500 token prompts), prefill interference is negligible and async prefill adds complexity without meaningful benefit. The investment in disaggregated infrastructure wouldn't pay off.

**Risk 2 — KV cache transfer becomes the new bottleneck.** In disaggregated systems, the network transfer of KV caches can become the latency bottleneck, especially for long-context requests. Without high-bandwidth interconnects (NVLink, fast InfiniBand), disaggregation can actually increase TTFT. This limits the approach to well-connected clusters.

**Risk 3 — Non-autoregressive models eliminate the decode phase.** If future models (diffusion-based, parallel decoding architectures) generate all tokens simultaneously rather than autoregressively, the prefill/decode split disappears entirely. Async prefill optimizes a bottleneck that might not exist in future architectures.

### Summary Signal for Investors

> **Asynchronous prefill is the engineering that makes latency SLAs possible. Without it, inference providers cannot guarantee consistent token delivery, which limits their ability to charge premium prices. With it, they unlock latency-tiered pricing, hardware specialization (heterogeneous fleets for prefill vs. decode), and predictable scaling behavior. The immediate competitive frontier is disaggregated prefill — providers that ship it get 12–18 months of infrastructure cost advantage before it commoditizes. The durable moat is the full orchestration stack that adaptively balances prefill, decode, and transfer across heterogeneous hardware under real-world traffic.**
