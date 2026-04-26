# Disaggregated Prefill and Decode: From First Principles

---

# 1. The two phases of LLM inference

Every autoregressive LLM request goes through exactly two phases. Understanding their differences is the entire foundation.

## Phase 1: Prefill

The model processes the entire input prompt **in parallel**. Every token in your prompt is fed through the transformer at once.

If your prompt is 4,000 tokens:

```
Input:  [token_1, token_2, ..., token_4000]
         ↓ all processed in parallel
Output: KV cache entries for all 4000 tokens
        + logits for the first generated token
```

### What makes prefill special

- **Compute-bound**: The GPU is doing massive matrix multiplications across all prompt tokens simultaneously. This looks like a training step — high arithmetic intensity, high FLOP utilization.
- **One-shot**: It runs once per request.
- **Duration scales with prompt length**: A 16K token prompt takes roughly 4× longer to prefill than a 4K token prompt.

---

## Phase 2: Decode

The model generates output tokens **one at a time**. Each step:

1. Take the last generated token
2. Compute its Q vector
3. Attend to the entire KV cache (all previous tokens)
4. Produce logits for the next token
5. Sample the next token
6. Append its K/V to the cache
7. Repeat

```
Step 1:  generate token_4001  (attend to 4000 cached KV entries)
Step 2:  generate token_4002  (attend to 4001 cached KV entries)
Step 3:  generate token_4003  (attend to 4002 cached KV entries)
...
Step N:  generate token_{4000+N}
```

### What makes decode special

- **Memory-bandwidth-bound**: Each step does a tiny amount of compute (one token's worth of matrix multiplies) but must read the *entire* KV cache from GPU memory. The bottleneck is how fast you can move data from HBM to the compute units, not how many FLOPs you can do.
- **Sequential**: You cannot parallelize across steps — each token depends on the previous one.
- **Duration scales with output length**: Generating 500 tokens means 500 sequential decode steps.

---

## The fundamental asymmetry

| Property | Prefill | Decode |
|---|---|---|
| **Bottleneck** | Compute (FLOPS) | Memory bandwidth (GB/s) |
| **Arithmetic intensity** | High (many FLOPs per byte loaded) | Low (few FLOPs per byte loaded) |
| **GPU utilization** | High (saturates SMs) | Low (SMs idle, waiting for memory) |
| **Parallelism** | Fully parallel across prompt tokens | Strictly sequential |
| **Duration** | One burst, proportional to input length | Many steps, proportional to output length |
| **Analogous to** | Training forward pass | Inference generation loop |

This asymmetry is the entire reason disaggregation exists. These two phases want **fundamentally different hardware characteristics**.

---

# 2. The problem: co-located prefill and decode

In a standard serving system (e.g., baseline vLLM), both phases run on the **same pool of GPUs**.

## What the GPU sees

Continuous batching means the GPU is running a mixed workload at every step:

```
Step T:
  - Sequence A: decode (generating its 50th token)
  - Sequence B: decode (generating its 200th token)
  - Sequence C: PREFILL (processing a new 8K token prompt)  ← just arrived
  - Sequence D: decode (generating its 12th token)
```

### The interference problem

Sequence C's prefill is compute-bound. It wants to saturate all the GPU's streaming multiprocessors (SMs) with heavy matrix multiplies. Meanwhile, sequences A, B, and D are in decode — they're doing tiny matrix ops but need fast memory reads.

What happens:

1. **C's prefill hogs compute**: The large prompt's prefill uses most of the SMs, starving decode sequences of compute cycles.
2. **Decode latency spikes**: Sequences A, B, D were generating tokens every ~30ms. During C's prefill, their decode step gets delayed to 100–200ms+ because the GPU is busy.
3. **Users see stuttering**: The humans waiting for A, B, D's responses see the token stream freeze momentarily, then resume in a burst.

---

## Concrete example

Imagine a chat API serving 4 concurrent users on a single A100:

```
Timeline (ms):
0    A: decode   B: decode   C: decode   D: decode     ← each takes ~30ms
30   A: decode   B: decode   C: decode   D: decode
60   A: decode   B: decode   C: decode   D: decode
90   ──── NEW REQUEST E ARRIVES (8K token prompt) ────
90   E: PREFILL (takes ~150ms because 8K tokens)
     A, B, C, D: BLOCKED — waiting for E's prefill to finish
240  A: decode   B: decode   C: decode   D: decode   E: decode
```

Between t=90 and t=240, users A through D see **150ms of silence**. Their inter-token latency (ITL) jumped from 30ms to 180ms+ for that step. This is the **P99 latency spike** that kills user experience.

---

## Why continuous batching doesn't fully solve this

Continuous batching allows new sequences to join mid-flight. But:

- The prefill of a new sequence still runs **on the same GPU** as active decodes
- The scheduler can chunk the prefill (process it in pieces across steps), but this increases prefill latency for the new request and still causes partial interference
- At high QPS, prefills are frequent — the interference is constant, not occasional

> The core insight: **prefill and decode want different things from the GPU. Forcing them to share hardware creates an irreconcilable scheduling conflict.**

---

# 3. The solution: disaggregation

## Core idea

Split the two phases onto **physically separate hardware pools**:

```
                    ┌─────────────────────┐
Client request ───→ │   PREFILL CLUSTER   │  (compute-optimized GPUs)
                    │                     │
                    │  Process the prompt  │
                    │  Generate KV cache   │
                    └────────┬────────────┘
                             │
                    KV cache transfer
                    (NVLink / PCIe / RDMA / network)
                             │
                    ┌────────▼────────────┐
                    │   DECODE CLUSTER    │  (memory-bandwidth-optimized GPUs/ASICs)
                    │                     │
                    │  Generate tokens    │
                    │  Stream to client   │
                    └─────────────────────┘
```

### What each cluster is optimized for

**Prefill cluster:**
- Needs raw FLOPS — dense matrix multiplies across thousands of tokens
- Can use fewer, more powerful GPUs
- Doesn't need to worry about sustained low-latency streaming
- Workload is bursty: process a prompt, emit KV blocks, pick up next prompt

**Decode cluster:**
- Needs maximum memory bandwidth — reading large KV caches every step
- Benefits from high HBM capacity (to hold many concurrent sequences' KV caches)
- Needs consistent, low-latency execution — every decode step must finish quickly
- Workload is sustained: steady stream of small operations

---

## Why this works (from first principles)

### Resource utilization

In a co-located system, the GPU is constantly context-switching between two modes:
- High-compute burst (prefill)
- Low-compute, high-bandwidth steady-state (decode)

Neither mode runs optimally because they share resources.

In a disaggregated system:
- Prefill GPUs run at **near-peak FLOP utilization** — they only do compute-heavy work
- Decode GPUs run at **near-peak memory bandwidth utilization** — they only do bandwidth-heavy work

### The analogy

Think of a restaurant kitchen:
- **Co-located**: One chef does both prep (chopping, mixing — labor-intensive) and plating (arranging, garnishing — precision and speed). When a big prep job comes in, plating stops, and food gets cold at the pass.
- **Disaggregated**: Prep cooks handle chopping and mixing. Line cooks handle plating and serving. Prep work never blocks service. Each team is optimized for their task.

---

# 4. The KV cache transfer problem

The critical engineering challenge: **how do you move the KV cache from the prefill cluster to the decode cluster?**

## What gets transferred

After prefill, each sequence has a KV cache:

```
Per sequence:
  K tensors: [num_layers × num_heads × seq_len × head_dim]
  V tensors: [num_layers × num_heads × seq_len × head_dim]
```

For a model like Llama-3-70B with a 4K token prompt (FP16):

```
layers = 80
heads = 8 (GQA)
seq_len = 4096
head_dim = 128
dtype = FP16 (2 bytes)

KV cache size = 2 × 80 × 8 × 4096 × 128 × 2 bytes
             = 2 × 80 × 8 × 4096 × 256
             ≈ 1.34 GB per sequence
```

That's 1.34 GB that must move from one GPU (or set of GPUs) to another.

---

## Transfer mechanisms

| Method | Bandwidth | Latency | Use case |
|---|---|---|---|
| **NVLink (intra-node)** | 900 GB/s (H100) | ~μs | Prefill and decode on different GPUs in the same server |
| **PCIe 5.0** | ~64 GB/s | ~μs | Cross-GPU within a node (slower than NVLink) |
| **RDMA / InfiniBand** | 400 Gb/s (~50 GB/s) | ~10μs | Cross-node within a cluster |
| **Ethernet (RoCE)** | 100–400 Gb/s | ~50-100μs | Cross-rack, data center scale |

### Transfer time for our 1.34 GB KV cache

| Method | Transfer time |
|---|---|
| NVLink | ~1.5 ms |
| PCIe 5.0 | ~21 ms |
| InfiniBand | ~27 ms |
| 100Gb Ethernet | ~107 ms |

> **Key insight**: The transfer overhead must be less than the latency savings from eliminating prefill-decode interference. NVLink makes this nearly free. Ethernet makes it expensive.

---

## Why this favors co-location within a node

In practice, disaggregation works best when:

- Prefill and decode GPUs are **in the same server** (connected via NVLink)
- Or in adjacent servers connected via InfiniBand

Cross-data-center disaggregation doesn't make sense — the KV transfer latency would dominate.

---

# 5. Scheduling in a disaggregated system

The scheduler becomes significantly more complex.

## Co-located scheduler (baseline)

```
while True:
    check_for_new_requests()
    add_to_batch()
    run_one_step()         # handles both prefill and decode
    remove_finished()
```

## Disaggregated scheduler

Now you need to coordinate two separate execution loops:

```
PREFILL SCHEDULER:
  while True:
      request = pick_next_from_queue()
      prefill(request)
      send_kv_cache_to_decode_cluster(request)

DECODE SCHEDULER:
  while True:
      check_for_incoming_kv_caches()
      add_new_sequences_to_decode_batch()
      run_one_decode_step()
      remove_finished()
      stream_tokens_to_clients()
```

### New scheduling decisions

1. **Prefill queue ordering**: Which request to prefill first? Short prompts are cheaper → process them first to reduce average TTFT? Or long prompts first to keep decode GPUs fed?

2. **Decode cluster capacity management**: How many concurrent sequences can the decode cluster hold? When it's full, do you queue prefilled sequences (adding latency) or preempt low-priority ones?

3. **Load balancing across prefill GPUs**: If you have 4 prefill GPUs and 8 decode GPUs, how do you route to balance prefill throughput with decode capacity?

4. **KV cache staging**: Do you pipeline the KV transfer (start sending early layers while later layers are still computing)?

---

# 6. Concrete example: disaggregated system in action

Setup:
- 2 GPUs for prefill (P0, P1)
- 4 GPUs for decode (D0, D1, D2, D3)
- Connected via NVLink within a single DGX node

```
Timeline:

t=0ms     Request A (2K prompt) → assigned to P0
          D0, D1, D2, D3 all running existing decode sequences smoothly

t=15ms    P0 finishes prefilling A
          KV cache (0.7 GB) transferred to D2 via NVLink (~0.8ms)

t=16ms    D2 begins decoding A → tokens start streaming to client
          NO INTERFERENCE with sequences on D0, D1, D3

t=20ms    Request B (16K prompt) → assigned to P1
          Decode cluster continues uninterrupted

t=140ms   P1 finishes prefilling B (16K tokens takes ~120ms)
          KV cache (5.4 GB) transferred to D0 via NVLink (~6ms)

t=146ms   D0 begins decoding B → tokens start streaming
```

### Compare to co-located (all 6 GPUs shared)

```
t=0ms     Request A (2K prompt) → prefill on GPU pool
          ALL 6 GPUs share compute; decode sequences stutter during prefill

t=20ms    Request B (16K prompt) → prefill on GPU pool
          MASSIVE interference: all decode sequences on all GPUs
          stall for ~120ms while B's prefill runs
```

The disaggregated system keeps decode latency **constant and predictable** regardless of incoming prefill load.

---

# 7. Where things get complex (and interesting)

## 7.1 Heterogeneous hardware for each phase

Since prefill and decode have different bottlenecks, you can use **different hardware** for each:

| Phase | Ideal hardware characteristics | Example |
|---|---|---|
| Prefill | High FLOPS, high compute density | NVIDIA H100 SXM, Blackwell B200 |
| Decode | High memory bandwidth, large HBM | NVIDIA L40S, custom ASICs like Groq LPU |

This is a natural fit for Groq in particular: their LPU architecture has extreme memory bandwidth and low latency per token, but isn't cost-efficient for large matrix multiplies. In a disaggregated system, Groq handles decode while commodity GPUs handle prefill.

---

## 7.2 Interaction with other optimizations

### With speculative decoding

Speculative decoding proposes K draft tokens and verifies them in one pass. In a disaggregated system:
- The **draft model** runs on the decode cluster (it generates tokens sequentially, same as regular decode)
- The **verification pass** is like a mini-prefill (processing K tokens in parallel) → runs on the prefill cluster?

This creates a question: does verification belong on prefill or decode hardware? If K is small (5–10 tokens), verification is too small to justify a cross-GPU transfer. Most implementations keep verification on the decode GPU to avoid the overhead.

### With prefix caching

Prefix caching stores pre-computed KV blocks for shared system prompts. In a disaggregated system:
- Cached KV blocks can be **pre-loaded on the decode cluster**, skipping prefill entirely for repeated prefixes
- The prefill cluster only processes the **unique portion** of each request
- This dramatically reduces KV transfer volume for high-QPS APIs with shared system prompts

### With KV cache quantization

Quantizing the KV cache (FP16 → FP8) before transfer:
- Halves the transfer bandwidth requirement
- But introduces quantization error that propagates through all subsequent decode steps

The trade-off is sharper in disaggregated systems because the transfer is a concrete bottleneck. FP8 KV transfer is often the right choice — the bandwidth savings outweigh the quality loss for most applications.

---

## 7.3 Chunked prefill as a middle ground

Before full disaggregation, many systems implement **chunked prefill**: break the prompt into chunks (e.g., 512 tokens each) and interleave prefill chunks with decode steps:

```
Step 1: decode A, B, C + prefill chunk 1 of D (tokens 1-512)
Step 2: decode A, B, C + prefill chunk 2 of D (tokens 513-1024)
Step 3: decode A, B, C + prefill chunk 3 of D (tokens 1025-1536)
...
```

This reduces interference (each prefill chunk is small) at the cost of increased TTFT for the new request (prefill is spread across many steps instead of one burst).

**Chunked prefill is the "software disaggregation" approach** — you don't need separate hardware, but you don't get the full benefit either.

---

# 8. Systems that implement disaggregation

| System | Approach | Status |
|---|---|---|
| **Splitwise** (Microsoft Research) | Full prefill/decode separation across GPUs | Research paper (2023) |
| **DistServe** (Peking University) | Goodput-optimal placement of prefill and decode | Research paper (2024) |
| **TetriInfer** | Packing prefill and decode onto different SMs within a single GPU | Research paper (2024) |
| **vLLM v0.4+** | Disaggregated prefill as an experimental feature | Production-ready, evolving |
| **SGLang** | Disaggregated prefill support | In development |
| **TensorRT-LLM** | NVIDIA's serving framework with disaggregated support | Production |
| **DeepSeek** | Reportedly uses disaggregated serving in production | Production (internal) |

---

# 9. When to use disaggregation vs. co-located serving

| Scenario | Best approach | Why |
|---|---|---|
| Low QPS, single GPU | Co-located | Not enough load to justify the complexity |
| High QPS, short prompts | Co-located (with chunked prefill) | Prefill interference is minimal |
| High QPS, mixed prompt lengths | **Disaggregated** | Long prefills would spike P99 for all users |
| Long prompts + latency SLAs | **Disaggregated** | Only way to guarantee stable ITL |
| Multi-GPU node (DGX) | **Disaggregated** | NVLink makes KV transfer nearly free |
| Cross-data-center | Co-located | KV transfer latency over network is too high |
| Reasoning models (long output, short input) | Decode-heavy allocation | Most compute is in decode; prefill cluster can be small |
| RAG / long-context (long input, short output) | Prefill-heavy allocation | Most compute is in prefill; need fast prefill throughput |

---

# 10. The math: why disaggregation improves throughput

## Co-located throughput model

In a co-located system with continuous batching, the effective throughput is limited by whichever phase is currently bottlenecking:

```
Effective throughput = min(prefill_throughput, decode_throughput)
                     × interference_penalty
```

The interference penalty comes from shared resources. Empirically, co-located systems achieve 60–80% of theoretical peak throughput due to scheduling conflicts.

## Disaggregated throughput model

```
Prefill throughput = prefill_cluster_FLOPS / FLOPS_per_prompt
Decode throughput  = decode_cluster_bandwidth / bytes_per_decode_step × batch_size

Effective throughput = min(prefill_throughput, decode_throughput)
                     × transfer_overhead_penalty
```

The transfer overhead penalty is small (1–5% with NVLink) compared to the interference penalty eliminated. Disaggregated systems achieve 85–95% of theoretical peak.

### Example

8-GPU DGX H100 node:

**Co-located** (all 8 GPUs shared):
- Mixed prefill/decode interference → ~65% utilization
- Effective: ~5.2 GPU-equivalents of useful work

**Disaggregated** (2 prefill + 6 decode):
- Prefill GPUs: ~95% compute utilization → 1.9 GPU-equivalents of prefill work
- Decode GPUs: ~90% bandwidth utilization → 5.4 GPU-equivalents of decode work
- Effective: 7.3 GPU-equivalents of useful work

That's a **40% throughput improvement** from the same hardware, just by separating the workloads.

---

# 11. The decision framework

```
Do you have multiple GPUs available?
├── NO → Use chunked prefill (software-only mitigation)
└── YES
    ├── Are they connected via NVLink?
    │   ├── YES → Strong candidate for disaggregation
    │   └── NO  → Measure KV transfer latency; if < 30ms, still viable
    ├── Is your workload mixed (varied prompt lengths)?
    │   ├── YES → Disaggregate. The P99 improvement alone justifies it.
    │   └── NO (uniform short prompts) → Co-located is fine
    └── Do you have latency SLAs on ITL?
        ├── YES → Disaggregate. It's the only way to decouple prefill spikes from decode.
        └── NO  → Measure. If prefill interference < 10% of decode time, skip.
```

---

# 12. The Future of Disaggregated Serving

Disaggregation is not a niche optimization — it is becoming the default architecture for high-scale LLM serving. The trajectory is clear and accelerating.

### 1. Full Hardware Heterogeneity Becomes Standard

Today, most disaggregated deployments use the same GPU model for both clusters (e.g., H100 for prefill and H100 for decode). The future is purpose-built hardware per phase:

- **Prefill**: Dense compute chips optimized for large matrix multiplies. Current GPUs are already good at this, but future prefill-specific ASICs could strip out the memory bandwidth silicon that prefill doesn't need, reducing cost per prefill FLOP.
- **Decode**: Memory-bandwidth-optimized chips. This is where Groq's LPU, Cerebras, and custom hyperscaler silicon have a structural advantage. A decode ASIC needs massive SRAM/HBM bandwidth, low-latency access patterns, and modest compute — almost the inverse of a training GPU.

The implication: **buying a single GPU SKU for all inference work will become as outdated as buying a single server type for all cloud workloads.**

### 2. Prefill-as-a-Service

As disaggregation becomes standard, prefill and decode may be operated by **different providers**:

- A company with cheap compute-dense hardware handles prefill
- A company with bandwidth-optimized hardware handles decode
- KV cache is transferred between them over high-speed interconnects

This would create a new API boundary in the inference stack — a "KV cache interchange format" that standardizes how prefill outputs are represented and transferred. Early signs of this exist in vLLM's disaggregated mode, which already defines an internal KV transfer protocol.

### 3. Dynamic Ratio Adjustment

The optimal prefill:decode GPU ratio depends on traffic patterns. A system serving mostly RAG queries (long prefill, short output) needs more prefill capacity. A system serving coding agents (short prefill, long output) needs more decode capacity.

Future orchestrators will **dynamically reassign GPUs** between prefill and decode pools based on real-time traffic analysis. This is analogous to how cloud auto-scalers adjust instance counts, but applied at the intra-node GPU level.

### 4. Convergence with Speculative Decoding

In a disaggregated system, speculative decode's verification pass is a mini-prefill. Future architectures may route verification back to the prefill cluster when K is large enough to justify the transfer, creating a **three-way pipeline**: draft (decode cluster) → verify (prefill cluster) → accept/reject → resume (decode cluster). This requires extremely low-latency interconnects but could push effective tokens-per-second much higher.

### 5. KV Cache as a First-Class Distributed Object

As KV caches move between GPUs, nodes, and potentially providers, they become **distributed stateful objects** with their own lifecycle:

- Creation (prefill)
- Transfer (prefill → decode)
- Replication (prefix caching across decode GPUs)
- Compression (quantization during transfer)
- Eviction (memory pressure)
- Persistence (saved to disk for session resumption)

Expect the KV cache to evolve from an implementation detail into a managed resource with its own APIs, quotas, and pricing — similar to how object storage emerged from being "just files on disk."

---

# 13. The Investor Lens (Aligned with the Inference Framework)

Disaggregated prefill/decode sits at the intersection of the **Serving / Runtime Layer** and the **Hardware Layer** of the inference stack. It is one of the few infrastructure innovations that simultaneously reshapes both layers. This dual impact makes it unusually important for investment positioning.

## Why This Is a Bigger Deal Than Previous Serving Optimizations

Previous serving innovations (continuous batching, PagedAttention) improved utilization on existing hardware. Disaggregation goes further: it **changes what hardware you buy**. This makes it a capex-level decision, not just a software upgrade, and capex decisions have much longer half-lives and higher switching costs.

## Primary Value Drivers

### 1. The Hardware Procurement Signal

Disaggregation is the forcing function for hardware heterogeneity in inference. When operators begin deploying different GPU/ASIC types for prefill and decode, it creates a structural shift in AI hardware demand:

- **NVIDIA's position**: NVIDIA GPUs are well-suited for prefill (high FLOPS) but over-provisioned for decode (too much compute, not enough bandwidth per dollar). Disaggregation means operators buy **fewer NVIDIA GPUs** for decode and potentially substitute with cheaper, bandwidth-optimized alternatives.
- **Custom ASIC opportunity**: Decode clusters are the entry point for companies like Groq (LPU), Cerebras, and hyperscaler custom silicon (Google TPU, AWS Inferentia). These chips have struggled to compete with NVIDIA on general training/inference but become competitive when the workload is narrowed to decode-only.
- **Signal to watch**: When a major cloud provider or inference startup publicly deploys a disaggregated system with different hardware for each phase, it validates the thesis that the inference silicon market is bifurcating. Track hardware purchase orders, not research papers.

### 2. Margin Expansion for Inference API Providers

Disaggregation delivers a ~30–40% throughput improvement on existing hardware. For inference API businesses (Together AI, Fireworks, Anyscale), this translates directly into gross margin expansion:

```
Before disaggregation:
  8× H100 serving 1000 tokens/sec → cost per million tokens = $X

After disaggregation (same hardware):
  8× H100 serving 1400 tokens/sec → cost per million tokens = $0.71X

Margin improvement: ~29% cost reduction per token
```

This is pure infrastructure-level margin, independent of model quality or pricing power. Investors should monitor which inference providers adopt disaggregation first — they will have a temporary margin advantage until competitors follow (typically 6–12 months, per the commoditization cascade).

### 3. The Jevons Paradox at Work

Disaggregation makes inference cheaper per token and more latency-predictable. The result:
- Use cases that were latency-blocked (real-time voice, interactive agents with long system prompts) become viable
- Use cases that were cost-blocked (high-QPS consumer products at mass scale) become economical
- Total token volume increases faster than cost-per-token decreases

This is the Jevons loop in action. Disaggregation doesn't shrink the GPU market — it expands the addressable use-case space, sustaining or increasing total hardware demand.

### 4. Operational Complexity as a Moat Filter

Disaggregated serving is significantly harder to operate than co-located serving:
- Requires KV cache transfer infrastructure
- Requires coordinated scheduling across GPU pools
- Requires real-time monitoring of prefill:decode ratio vs. traffic patterns
- Requires dynamic rebalancing logic

This operational complexity acts as a **temporary moat** for teams with deep systems engineering talent. Self-hosting an open-weight model with naive vLLM is easy. Self-hosting with production-grade disaggregation, dynamic ratio management, and multi-tier KV caching is hard. This widens the gap between managed inference platforms and DIY deployments, partially counteracting the open-source commoditization pressure.

## Risk Factors

### Risk 1: Architecture Changes That Eliminate the Prefill/Decode Asymmetry

If model architectures shift toward linear attention (SSMs like Mamba) or hybrid architectures where prefill and decode have similar compute profiles, the rationale for disaggregation weakens. In an SSM, there is no KV cache — state is a fixed-size recurrence, and both "prefill" and "decode" are the same operation (a matrix multiply updating the hidden state). If SSMs reach transformer-quality, disaggregation infrastructure becomes stranded investment.

**Probability assessment**: Low-medium in the next 2–3 years. Transformer-based architectures dominate frontier models, and hybrid approaches (transformer + SSM layers) still have transformer-style prefill/decode asymmetry. But monitor closely.

### Risk 2: NVLink/Interconnect as a Bottleneck

Disaggregation only works well when KV transfer is fast. Current NVLink (900 GB/s on H100) makes intra-node transfer nearly free. But:
- Cross-node disaggregation requires InfiniBand/Ethernet, which adds 10–100ms of transfer latency
- As models grow larger (and KV caches grow proportionally), even NVLink bandwidth may become a constraint
- NVIDIA controls NVLink pricing and availability, creating a dependency

### Risk 3: The Commoditization Timeline

Disaggregated serving is currently a differentiator. But the commoditization cascade applies:
1. Research papers (Splitwise, DistServe) — 2023–2024 ✓
2. Frontier lab adoption — 2024–2025 ✓ (DeepSeek, likely others)
3. Open-source integration — 2025–2026 (vLLM, SGLang adding support now)
4. Table-stakes — 2026–2027

By 2027, disaggregation will not be a competitive advantage — it will be expected. Companies whose pitch is "we do disaggregated serving" will face the same margin compression as companies whose pitch was "we do continuous batching" in 2024.

## Sector Positioning Map (Disaggregation-Specific)

| Sector | Effect | Signal |
|---|---|---|
| **NVIDIA** | Mixed — sells more prefill-optimized GPUs but loses decode share to ASICs | Watch decode-specific SKU launches (e.g., bandwidth-optimized Blackwell variants) |
| **Inference ASIC startups** (Groq, Cerebras) | **Strong beneficiary** — decode cluster is their optimal insertion point | Watch production deployment announcements, not benchmarks |
| **Cloud hyperscalers** | Beneficiary — disaggregation improves utilization of their massive GPU fleets | Watch internal serving architecture disclosures at ML conferences |
| **Inference API providers** | Temporary margin advantage for early adopters; levels out within 12–18 months | Monitor who claims disaggregation support first; early mover captures margin window |
| **Interconnect/networking** (Broadcom, Arista, Mellanox/NVIDIA) | Beneficiary — KV cache transfer drives demand for high-bandwidth intra-cluster networking | Watch InfiniBand/NVLink deployment volumes in inference clusters specifically |

## The Key Investment Question

> If disaggregated serving becomes the standard architecture (high probability by 2027), which companies are positioned to capture the **decode cluster hardware market**?

This is a large, nascent market. Decode-specific hardware doesn't exist at scale today. The first company to deliver a purpose-built decode chip that is materially cheaper (cost per generated token) than an NVIDIA GPU while matching quality and reliability will capture significant share of the fastest-growing segment of AI infrastructure spend.

---

# 14. Summary

The one-sentence version:

> **Prefill is a sprint; decode is a marathon. Making the same hardware do both is like asking a sprinter and a distance runner to share the same legs. Disaggregation gives each phase the hardware it deserves.**

The investment version:

> **Disaggregation is the architectural shift that bifurcates the inference hardware market into compute-optimized (prefill) and bandwidth-optimized (decode) segments. The decode segment is where ASIC startups can finally compete with NVIDIA. The first mover in purpose-built decode silicon captures a structurally advantaged position in the fastest-growing compute market.**
