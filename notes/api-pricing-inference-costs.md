# Deducing Inference Costs from API Pricing: From First Principles

Source: Reiner Pope podcast discussion (~01:33:02), public API pricing pages (Google, OpenAI, Anthropic, April 2026)

---

## 1. The Core Idea: API Prices Are a Window into Hardware Costs

When a provider like Google, OpenAI, or Anthropic publishes per-token prices, they are (with margin) revealing the **dominant hardware cost** of serving that token. By studying the *structure* of pricing — not just the absolute numbers — you can reverse-engineer what's happening on the GPU.

The key pricing patterns that reveal inference physics:

| Pattern | What it reveals |
|---|---|
| Output tokens cost 3–5× more than input tokens | Decode is memory-bound; prefill is compute-bound |
| Price jumps above a context-length threshold | KV cache loading becomes the bottleneck |
| Cached input tokens are 75–90% cheaper | Prefill compute is the dominant input cost, and it's eliminated by caching |
| Batch API is ~50% cheaper | Relaxed latency allows higher batch sizes → better GPU utilization |

This note walks through each pattern from first principles.

---

## 2. Real-World API Pricing (April 2026)

### Gemini

| Model | Context | Input (per 1M tokens) | Output (per 1M tokens) |
|---|---|---|---|
| **Gemini 2.5 Pro** | ≤ 200K | $1.25 | $10.00 |
| | > 200K | $2.50 | $15.00 |
| **Gemini 3.1 Pro** | ≤ 200K | $2.00 | $12.00 |
| | > 200K | $4.00 | $18.00 |
| **Gemini 2.5 Flash** | Any | $0.30 | $2.50 |

### OpenAI

| Model | Input (per 1M tokens) | Output (per 1M tokens) |
|---|---|---|
| **GPT-4o** | $2.50 | $10.00 |
| **GPT-4.1** | $2.00 | $8.00 |
| **GPT-4.1 Mini** | $0.40 | $1.60 |

### Anthropic

| Model | Input (per 1M tokens) | Output (per 1M tokens) |
|---|---|---|
| **Claude Opus 4.7** | $5.00 | $25.00 |
| **Claude Sonnet 4.6** | $3.00 | $15.00 |
| **Claude Haiku 4.5** | $1.00 | $5.00 |

Notice the universal pattern: **output tokens are 3–5× more expensive than input tokens** across every provider and every model tier. This is not coincidence — it reflects a fundamental asymmetry in the inference pipeline.

---

## 3. Why Output Tokens Cost 3–5× More Than Input Tokens

### The two phases of inference, revisited

Every LLM API call has two phases:

1. **Prefill** (processes input tokens): The entire input prompt is processed in a single forward pass. All input tokens are computed **in parallel**.
2. **Decode** (generates output tokens): Output tokens are generated **one at a time**, each requiring a separate forward pass through the model.

### The hardware utilization asymmetry

From the roofline model (see [reiner-pope-podcast.md](reiner-pope-podcast.md)):

**During prefill**, the GPU processes all $S$ input tokens at once. The compute cost is:

$$
\text{FLOPs}_{\text{prefill}} = S \times 2 \times N_{\text{active}}
$$

This is a massive matrix multiplication — exactly what GPUs are designed for. The GPU's Tensor Cores are fully utilized. **Prefill is compute-bound.**

The time is dominated by:

$$
t_{\text{prefill}} \approx \frac{S \times 2 \times N_{\text{active}}}{\text{FLOPs}_{\text{peak}}}
$$

**During decode**, the GPU generates one token per forward pass. The compute cost per token is:

$$
\text{FLOPs}_{\text{decode}} = 1 \times 2 \times N_{\text{active}}
$$

But the GPU still has to **load all model weights from HBM** for each forward pass. With $B$ sequences batched together:

$$
t_{\text{decode}} = \max\left(\frac{B \times 2 \times N_{\text{active}}}{\text{FLOPs}_{\text{peak}}}, \frac{N_{\text{total}} \times \text{bytes}_{w}}{\text{Bandwidth}}\right)
$$

At typical interactive batch sizes ($B \ll B^*$), decode is **deeply memory-bound**. The GPU spends most of its time waiting for weights to load and its Tensor Cores are idle.

### The MFU gap

**Model FLOPs Utilization (MFU)** measures what fraction of the GPU's peak compute is actually being used:

| Phase | Typical MFU | Why |
|---|---|---|
| **Prefill** | 50–70% | Large matrix multiplications fill the Tensor Cores efficiently |
| **Decode** | 5–15% | Memory-bound — Tensor Cores idle while waiting for weight loads |

Decode MFU is roughly **$\frac{1}{5}$ to $\frac{1}{10}$** of prefill MFU. This means:

> **To generate one output token, you occupy the GPU for approximately 5× longer than it takes to process one input token.**

The GPU-seconds per token are ~5× higher for decode. Since the provider is renting GPU time, the per-token cost for output is ~5× higher. The 3–5× price ratio directly reflects this MFU gap.

### Why the weight fetch is the bottleneck

During prefill, the weight fetch cost is **amortized** across all $S$ input tokens. You load each weight matrix once and multiply it against $S$ token vectors simultaneously. The cost per input token of loading weights is:

$$
\text{Weight load cost per input token} = \frac{N_{\text{total}} \times \text{bytes}_{w}}{S \times \text{Bandwidth}}
$$

As $S$ grows, this term shrinks — it becomes negligible for long prompts. The per-token cost converges to pure compute cost.

During decode, you load **all** the weights for **every single token**:

$$
\text{Weight load cost per output token} = \frac{N_{\text{total}} \times \text{bytes}_{w}}{\text{Bandwidth}}
$$

No amortization. Every output token pays the full weight-loading toll.

### Concrete example

For Gemini 2.5 Pro (assume ~100B active parameters, H100 cluster):

- Weight load time (FP8, 100 GB): $\frac{100 \times 10^9}{3.35 \times 10^{12}} \approx 30 \text{ ms}$
- Compute time per token: $\frac{2 \times 100 \times 10^9}{990 \times 10^{12}} \approx 0.2 \text{ ms}$

Each decode step takes ~30 ms (memory-bound). Processing 1,000 input tokens in prefill takes roughly the same ~30 ms weight load time, but computes all 1,000 tokens in that window.

**Effective cost per token:**
- Input: 30 ms of GPU time / 1,000 tokens = **0.03 ms per token**
- Output: 30 ms of GPU time / 1 token = **30 ms per token**

The output token is **1,000× more expensive in raw GPU-ms for a 1K-token prompt**. The 4–8× price ratio in the API dramatically undercharges for output relative to raw hardware cost — this is because batching multiple sequences during decode (continuous batching) amortizes the weight load across $B$ sequences, bringing the effective cost down. But the asymmetry remains.

---

## 4. Why Gemini Charges ~50% More Above 200K Context

### The observation

Gemini 2.5 Pro pricing:
- ≤ 200K context: $1.25 input, $10.00 output per 1M tokens
- \> 200K context: $2.50 input, $15.00 output per 1M tokens

That's a **2× jump in input cost** and a **1.5× jump in output cost** at the 200K boundary.

### The explanation: regime change from compute-bound to memory-bound

Recall the two time components from [reiner-pope-podcast.md](reiner-pope-podcast.md):

**For input tokens (prefill)**, the two clocks are:

1. **Compute time**: $t_{\text{compute}} = \frac{S \times 2 \times N_{\text{active}}}{\text{FLOPs}}$ — scales linearly with sequence length $S$
2. **KV cache write time**: The KV cache for $S$ tokens must be computed and stored. This is part of the compute — it scales with $S$ and is included in the above.

At moderate context lengths (< 200K), prefill is **compute-bound** — the cost per token is flat because you're just doing more matrix multiplications, and the GPU handles them efficiently.

**For output tokens (decode)**, there are now two memory-loading costs per step:

1. **Weight loading**: $t_{\text{weights}} = \frac{N_{\text{total}} \times \text{bytes}_{w}}{\text{Bandwidth}}$ — constant regardless of context
2. **KV cache loading**: $t_{\text{KV}} = \frac{B \times \text{len}_{\text{ctx}} \times \text{bytes/token}}{\text{Bandwidth}}$ — **grows linearly with context length**

At short contexts, weight loading dominates: $t_{\text{weights}} \gg t_{\text{KV}}$. The per-token decode cost is roughly constant.

At long contexts, KV cache loading catches up and eventually dominates: $t_{\text{KV}} > t_{\text{weights}}$.

### Visualizing the crossover

```
  Time per token
     │
     │                                    ╱ t_KV (grows with context)
     │                                 ╱
     │                              ╱
     │  t_weights ──────────────╱────── (constant)
     │  ─────────────────────╱
     │                     ╱
     │                  ╱
     │               ╱
     │            ╱    t_compute (grows, but slower for decode)
     │         ╱
     │──────┼───────────────────────── Context length
            200K (crossover)
```

Below 200K: The weight loading time dominates. Adding more context tokens to the KV cache costs a little extra memory bandwidth, but it's a small fraction of the total. **Cost per token is roughly flat.**

Above 200K: The KV cache has grown so large that the time to load it exceeds the time to load the weights. **Every additional context token now makes every subsequent decode step more expensive**, because the attention computation must scan more KV entries. The cost per output token now **grows linearly with context length**.

Google reflects this regime change by bumping the price at 200K tokens. The precise crossover point depends on the model architecture and hardware — for Gemini's ~100B active parameter model on their TPU infrastructure, it happens around 200K tokens.

### Why input price also doubles

Even for prefill (input tokens), very long contexts create problems:

1. The **attention computation** in prefill is $O(S^2)$ in standard attention (or $O(S \log S)$ with techniques like ring attention). At 200K+ tokens, the quadratic attention cost starts becoming significant even for the compute-bound prefill phase.
2. The **KV cache must be written** to HBM. At 200K tokens with ~1.7 KB per token (derived below), that's ~340 GB of KV cache — which may exceed a single device's HBM, requiring cross-device communication.
3. **Scheduling complexity** increases — a 200K+ request monopolizes significant HBM, reducing the number of other requests that can be batched alongside it.

---

## 5. Deriving KV Cache Bytes-per-Token from Gemini's Pricing Crossover

This is a beautiful example of reverse-engineering hardware constraints from public pricing data.

### The setup

We know:
- Gemini's pricing crossover happens at $\text{len}_{\text{ctx}} = 200\text{K}$ tokens
- At the crossover, the time to load KV cache equals some other time component
- Assume $N_{\text{active}} \approx 100\text{B}$ parameters

### The derivation

At the crossover point, the two time components for decode are equal:

$$
t_{\text{compute}} = t_{\text{KV fetch}}
$$

The compute time per decode step (for batch size $B$):

$$
t_{\text{compute}} = \frac{B \times 2 \times N_{\text{active}}}{\text{FLOPs}}
$$

The KV cache fetch time per decode step:

$$
t_{\text{KV fetch}} = \frac{B \times \text{len}_{\text{ctx}} \times \text{bytes/token}}{\text{mem\_bw}}
$$

Setting them equal:

$$
\frac{B \times 2 \times N_{\text{active}}}{\text{FLOPs}} = \frac{B \times \text{len}_{\text{ctx}} \times \text{bytes/token}}{\text{mem\_bw}}
$$

The $B$ cancels (both scale identically with batch size):

$$
\frac{2 \times N_{\text{active}}}{\text{FLOPs}} = \frac{\text{len}_{\text{ctx}} \times \text{bytes/token}}{\text{mem\_bw}}
$$

Solve for bytes/token:

$$
\text{bytes/token} = \frac{\text{mem\_bw}}{\text{FLOPs}} \times \frac{2 \times N_{\text{active}}}{\text{len}_{\text{ctx}}}
$$

### The hardware ratio

For most modern accelerators, $\frac{\text{mem\_bw}}{\text{FLOPs}} \approx \frac{1}{300}$ (this is approximately the inverse of $B^*$ for dense FP16 models). This holds roughly for H100, TPUv5, and similar chips.

### Plugging in numbers

$$
\text{bytes/token} = \frac{1}{300} \times \frac{2 \times 100 \times 10^9}{200 \times 10^3}
$$

$$
= \frac{1}{300} \times \frac{200 \times 10^9}{200 \times 10^3}
$$

$$
= \frac{1}{300} \times 10^6
$$

$$
= \frac{10^6}{300} \approx 3{,}333 \text{ bytes}
$$

Hmm — but the podcast gives ~1.7 KB. The discrepancy comes from the factor of 2 in the numerator. If we use the simplified Pope notation where FLOPs per token ≈ $N_{\text{active}}$ (absorbing the factor of 2), then:

$$
\text{bytes/token} = \frac{1}{300} \times \frac{N_{\text{active}}}{\text{len}_{\text{ctx}}} = \frac{1}{300} \times \frac{100 \times 10^9}{200 \times 10^3} = \frac{100 \times 10^9}{60 \times 10^6} \approx 1{,}667 \text{ bytes} \approx 1.7 \text{ KB}
$$

### What does 1.7 KB per token mean physically?

The KV cache stores Key and Value vectors for every token at every layer:

$$
\text{bytes/token} = 2 \times n_{\text{layers}} \times n_{\text{kv\_heads}} \times d_{\text{head}} \times \text{bytes}_{\text{kv}}
$$

For a model with 1.7 KB per token:
- If using FP16 (2 bytes) with 80 layers and GQA (say 8 KV heads, 128 head dim):
  - $2 \times 80 \times 8 \times 128 \times 2 = 327{,}680$ bytes = 320 KB — way too high!
- If using FP8 (1 byte) with MQA or aggressive GQA (1 KV head, 128 head dim):
  - $2 \times 80 \times 1 \times 128 \times 1 = 20{,}480$ bytes = 20 KB — still higher

The 1.7 KB figure implies aggressive KV cache compression — likely **Multi-Latent Attention (MLA)** (as used in DeepSeek V3) or heavy quantization (INT4 KV cache with very few KV heads). This gives us a concrete signal about Gemini's architecture: they are likely using **very aggressive KV cache compression** to enable long-context serving at reasonable cost.

### Why this reverse-engineering matters

From a single public pricing page, we deduced:
1. The approximate ratio of hardware memory bandwidth to compute
2. The per-token KV cache footprint of Gemini's model
3. Evidence of aggressive KV cache compression in the architecture

This is the kind of analysis that turns public pricing data into architectural intelligence.

---

## 6. The Pricing Crossover Visualized

### Compute and memory time as context grows

```
  Cost per output token
     │
     │         Region A          │      Region B
     │     (weight-bound)        │   (KV-cache-bound)
     │                           │
     │                           │         ╱
     │                           │       ╱
     │                           │     ╱  ← total cost
     │                           │   ╱
     │  ─────────────────────────┼─╱──────── ← weight loading (flat)
     │                           ╱
     │                         ╱ │
     │                       ╱   │
     │                     ╱     │
     │                   ╱       │  ← KV cache loading
     │                 ╱         │
     │               ╱           │
     │             ╱             │
     │───────────┼───────────────┼────────── Context length
                               200K
```

### How pricing maps to this

```
  API Price per output token
     │
     │                           │
     │                           │      ╱ (ideally tracks cost)
     │                           │    ╱
     │  ──────── $10/1M ─────────│──╱
     │                           │╱
     │                           ├── $15/1M (stepped approximation)
     │                           │
     │───────────────────────────┼────────── Context length
                               200K
```

The stepped pricing (flat below 200K, flat but higher above 200K) is a **discrete approximation** of the continuous cost curve. In reality, the cost per output token grows linearly with context length beyond the crossover. But a two-tier pricing model is simpler for customers to understand.

---

## 7. Why Cached Input Tokens Are 75–90% Cheaper

Both Google and Anthropic offer **prompt caching** with dramatic discounts:

| Provider | Standard input price | Cached input price | Discount |
|---|---|---|---|
| Gemini 2.5 Pro | $1.25 / 1M | ~$0.16 / 1M | ~87% |
| Claude Sonnet 4.6 | $3.00 / 1M | ~$0.30 / 1M | ~90% |

### What caching eliminates

When you cache a prompt prefix, the provider stores the **precomputed KV cache** from a previous prefill. On a cache hit, the server skips the prefill computation entirely for the cached portion.

The standard input token cost is dominated by **prefill compute** — running the full forward pass to generate KV cache entries for each input token. When the KV cache already exists:

- **Compute cost**: eliminated (no forward pass needed for cached tokens)
- **Memory cost**: only the cost of storing and loading the cached KV blocks

The 10–13% residual price (after the discount) represents the **storage and memory-bandwidth cost** of maintaining and loading the cached KV blocks. This confirms that the vast majority of input token cost (~90%) is compute, and only ~10% is memory.

### What this tells us about margin structure

If cached tokens cost 10% of standard tokens, and the provider is still profitable on cached tokens (they must be, or they wouldn't offer the feature), then the **total infrastructure cost of serving an input token is less than or equal to 10% of the standard price**. The remaining 90% is split between compute cost and margin. This bounds the provider's gross margin on standard input tokens.

---

## 8. Why Batch API Is ~50% Cheaper

Both OpenAI and Google offer ~50% discounts for batch processing (24-hour turnaround):

### What batch processing enables

When latency doesn't matter, the provider can:

1. **Wait for more requests** to accumulate → push batch size $B$ closer to $B^*$
2. **Fill pipeline bubbles** with batch requests during off-peak hours
3. **Use cheaper/older hardware** that has lower utilization demand
4. **Schedule optimally** to maximize GPU utilization across the entire fleet

From the roofline model: below $B^*$, throughput scales linearly with batch size while step time stays constant. Going from $B = 50$ to $B = 200$ quadruples throughput without changing per-step latency. The GPU does 4× more useful work per unit time → the cost per token drops proportionally.

The ~50% discount suggests that interactive serving operates at roughly **50% GPU utilization** for compute — meaning the average interactive batch size is about $B^*/2$. Batch processing fills the remaining utilization gap.

---

## 9. Cross-Provider Price Ratios and What They Reveal

### The output/input ratio

| Provider | Model | Output/Input ratio |
|---|---|---|
| Google | Gemini 2.5 Pro | 8.0× |
| OpenAI | GPT-4o | 4.0× |
| OpenAI | GPT-4.1 | 4.0× |
| Anthropic | Claude Sonnet 4.6 | 5.0× |
| Anthropic | Claude Opus 4.7 | 5.0× |

**Gemini's 8× ratio is notably higher** than the industry norm (4–5×). This could indicate:
- Gemini's decode phase is particularly memory-bound (possibly due to a larger model or less optimized batching)
- Google's prefill is unusually efficient (TPU architecture excels at large matrix multiplications, reducing compute cost)
- Different margin structures (Google may be pricing input tokens below cost to attract usage, making up margin on output tokens)

### The model-size-to-price relationship

| Model | Approx. active params | Input price | Output price | Price per B params (output) |
|---|---|---|---|---|
| Claude Haiku 4.5 | ~20B (est.) | $1.00 | $5.00 | $0.25/B |
| GPT-4.1 Mini | ~30B (est.) | $0.40 | $1.60 | $0.05/B |
| Gemini 2.5 Flash | ~40B (est.) | $0.30 | $2.50 | $0.06/B |
| GPT-4o | ~100B (est.) | $2.50 | $10.00 | $0.10/B |
| Claude Sonnet 4.6 | ~70B (est.) | $3.00 | $15.00 | $0.21/B |
| Claude Opus 4.7 | ~200B+ (est.) | $5.00 | $25.00 | $0.13/B |

The "price per billion parameters" for output tokens is remarkably consistent at **$0.05–$0.25 per billion active parameters per million output tokens**. This is a rough sanity check: the cost per output token scales roughly linearly with model size (as predicted by $t_{\text{memory}} \propto N_{\text{total}}$).

---

## 10. Summary: Reading Hardware Physics from a Pricing Page

| Pricing signal | Hardware physics revealed |
|---|---|
| **Output 3–5× more than input** | Decode is memory-bound (MFU ~$\frac{1}{5}$ of prefill) |
| **Price jumps at 200K context** | KV cache loading exceeds weight loading at this length |
| **Gemini: bytes/token ≈ 1.7 KB** | Implies aggressive KV cache compression (MLA or INT4 KV) |
| **Cached input 90% cheaper** | ~90% of input cost is compute (eliminated by caching) |
| **Batch API 50% cheaper** | Interactive serving runs at ~50% GPU utilization |
| **Output/input ratio varies by provider** | Reflects prefill vs. decode efficiency of different hardware (GPU vs. TPU) |
| **Price scales ~linearly with model size** | Confirms $t_{\text{memory}} \propto N_{\text{total}}$ — larger models cost proportionally more per token |

### The investor lens

1. **Track the output/input ratio over time.** A declining ratio signals improving decode efficiency (better batching, speculative decoding, or decode-optimized hardware). This is a leading indicator of cost structure improvement.
2. **Watch for context-length pricing tiers.** The crossover point reveals the provider's KV cache strategy. A higher crossover point = better KV cache compression = competitive advantage for long-context workloads.
3. **Compare cached vs. standard pricing.** The gap bounds the compute-vs-memory cost split. If cached prices drop further, compute costs are falling faster than memory costs (suggesting hardware evolution like higher-bandwidth HBM).
4. **The 50% batch discount constrains utilization.** If batch discounts deepen (e.g., 70% off), real-time utilization is worsening — a bearish signal for the provider's infrastructure efficiency.
