# t_mem: The Memory Time Equation From First Principles

Source: Pope et al., *"Efficiently Scaling Transformer Inference"* (2023)

---

# The Equation

$$
t_{\text{mem}} = \frac{N_{\text{total}} + B \cdot \text{len}_{\text{ctx}} \cdot \text{KV}_{\text{bytes/token}}}{\text{mem\_bw}}
$$

Where:
- **N_total** = total model parameters (in bytes, i.e., param count × bytes per param)
- **B** = batch size (number of concurrent sequences)
- **len_ctx** = context length (prompt + generated tokens so far)
- **KV_bytes/token** = bytes of KV cache stored per token
- **mem_bw** = HBM memory bandwidth (bytes/sec)

The hint in the slide is the key: t_mem has **two contributions** — one from weights, one from KV cache. Let's derive each from first principles.

---

# 1. First Contribution: Weight Loading

## What happens physically

During every decode step, the GPU executes each layer of the transformer sequentially:

```
Layer 1: load Q/K/V/O projection weights → multiply against activations
Layer 2: load FFN up/gate/down weights → multiply against activations
...
Layer N: same
```

Each layer's weights must be read from HBM into the compute units (L2 → L1 → registers). The weights are far too large to stay in cache — a 70B model in FP16 has 140 GB of weights, while L2 cache is ~50 MB.

**Every single decode step reads the entire model from HBM.** Regardless of batch size. Whether you're generating a token for 1 user or 100 users, the same weights get loaded once and multiplied against all B activation vectors.

## The math

$$
t_{\text{weights}} = \frac{N_{\text{total}} \times \text{bytes per param}}{\text{mem\_bw}}
$$

In Pope's notation, N_total already includes the bytes-per-param factor (i.e., N_total is in bytes), so:

$$
t_{\text{weights}} = \frac{N_{\text{total}}}{\text{mem\_bw}}
$$

### Example: Llama 3 70B, FP16, on H100

$$
t_{\text{weights}} = \frac{70 \times 10^9 \times 2 \text{ bytes}}{3.35 \times 10^{12} \text{ bytes/s}} = \frac{140 \text{ GB}}{3.35 \text{ TB/s}} \approx 41.8 \text{ ms}
$$

**Key property**: This term is **constant with respect to B and len_ctx**. It doesn't matter how many users are in the batch or how long their conversations are — the same weights get loaded every step.

---

# 2. Second Contribution: KV Cache Loading

## What happens physically

The KV cache stores the Key and Value vectors for every past token in every layer. During each decode step, the attention mechanism must:

1. Compute the new token's Query vector (cheap — just a matmul with the Q projection weight, already counted in the weights term)
2. **Read all past K and V vectors from HBM** to compute attention scores and weighted values

This means: for each sequence in the batch, the GPU reads the entire KV cache for that sequence from HBM.

## How big is the KV cache per token?

For each transformer layer, each token stores:
- One K vector of size `num_kv_heads × head_dim`
- One V vector of size `num_kv_heads × head_dim`

Total per token per layer:

$$
\text{KV per token per layer} = 2 \times \text{num\_kv\_heads} \times \text{head\_dim} \times \text{bytes per element}
$$

Across all layers:

$$
\text{KV}_{\text{bytes/token}} = 2 \times n_{\text{layers}} \times n_{\text{kv\_heads}} \times d_{\text{head}} \times \text{bytes}
$$

### Example: Llama 3 70B (FP16)

- 80 layers
- 8 KV heads (GQA — grouped query attention)
- 128-dim head
- 2 bytes per element (FP16)

$$
\text{KV}_{\text{bytes/token}} = 2 \times 80 \times 8 \times 128 \times 2 = 327{,}680 \text{ bytes} \approx 320 \text{ KB per token}
$$

For comparison, Llama 3 70B with full MHA (64 KV heads instead of 8) would be:

$$
\text{KV}_{\text{bytes/token}} = 2 \times 80 \times 64 \times 128 \times 2 = 2{,}621{,}440 \text{ bytes} \approx 2.5 \text{ MB per token}
$$

**GQA reduces KV cache by 8× in this case.** This is why every modern model uses GQA — it directly reduces t_mem.

## The total KV cache that must be read

For **one sequence** with context length len_ctx:

$$
\text{KV cache per sequence} = \text{len}_{\text{ctx}} \times \text{KV}_{\text{bytes/token}}
$$

For the **full batch** of B sequences:

$$
\text{Total KV cache to read} = B \times \text{len}_{\text{ctx}} \times \text{KV}_{\text{bytes/token}}
$$

Divide by bandwidth:

$$
t_{\text{KV}} = \frac{B \times \text{len}_{\text{ctx}} \times \text{KV}_{\text{bytes/token}}}{\text{mem\_bw}}
$$

### Example: Llama 3 70B (GQA), B=32, context=4096, H100

$$
t_{\text{KV}} = \frac{32 \times 4096 \times 327{,}680}{3.35 \times 10^{12}} = \frac{42.9 \text{ GB}}{3.35 \text{ TB/s}} \approx 12.8 \text{ ms}
$$

---

# 3. Putting Them Together

$$
t_{\text{mem}} = t_{\text{weights}} + t_{\text{KV}} = \frac{N_{\text{total}} + B \cdot \text{len}_{\text{ctx}} \cdot \text{KV}_{\text{bytes/token}}}{\text{mem\_bw}}
$$

The two terms are additive because the GPU must read **both** the weights **and** the KV cache from HBM during each decode step. They share the same memory bus, so their bytes add up.

### Concrete example: Llama 3 70B, B=32, context=4096, H100

$$
t_{\text{mem}} = \frac{140 \text{ GB} + 42.9 \text{ GB}}{3.35 \text{ TB/s}} = \frac{182.9 \text{ GB}}{3.35 \text{ TB/s}} \approx 54.6 \text{ ms}
$$

Breaking it down:
- Weight loading: 41.8 ms (77% of memory time)
- KV cache loading: 12.8 ms (23% of memory time)

---

# 4. How Each Variable Shifts the Balance

## Batch Size (B)

| B | t_weights | t_KV | t_mem | KV share of t_mem |
|---|---|---|---|---|
| 1 | 41.8 ms | 0.4 ms | 42.2 ms | 1% |
| 16 | 41.8 ms | 6.4 ms | 48.2 ms | 13% |
| 64 | 41.8 ms | 25.6 ms | 67.4 ms | 38% |
| 256 | 41.8 ms | 102.4 ms | 144.2 ms | 71% |

At small batches, weights dominate t_mem. At large batches, **KV cache takes over** and becomes the bottleneck. This is the regime shift that makes large-batch, long-context serving so hard.

## Context Length (len_ctx)

| len_ctx | t_KV (B=32) | t_mem | KV share |
|---|---|---|---|
| 512 | 1.6 ms | 43.4 ms | 4% |
| 4,096 | 12.8 ms | 54.6 ms | 23% |
| 32,768 | 102.4 ms | 144.2 ms | 71% |
| 131,072 | 409.6 ms | 451.4 ms | 91% |

At 128K context, the KV cache loading **dominates everything**. The weight loading (41.8 ms, always constant) becomes almost irrelevant. This is why long-context inference requires fundamentally different optimization strategies (KV cache quantization, eviction, sliding windows, sparse attention).

## Quantization (bytes per param / KV)

Quantizing weights from FP16 → INT4:

$$
t_{\text{weights}} = \frac{70 \times 10^9 \times 0.5}{3.35 \times 10^{12}} \approx 10.4 \text{ ms}
$$

That's a **4× reduction** in weight loading time. Quantizing KV cache from FP16 → FP8:

$$
t_{\text{KV}} = \frac{32 \times 4096 \times 163{,}840}{3.35 \times 10^{12}} \approx 6.4 \text{ ms}
$$

**2× reduction** in KV loading time.

Combined: t_mem drops from 54.6 ms → 16.8 ms — a **3.3× speedup** from quantization alone. This is why quantization is the single highest-impact inference optimization.

---

# 5. The Modified Critical Batch Size (With KV Cache)

In the previous notes (reiner-pope-podcast.md), we derived B* assuming t_memory was constant. With the KV cache term, t_memory itself grows with B, which changes the crossover point.

Setting t_compute = t_mem:

$$
\frac{B \times 2 \times N_{\text{active}}}{\text{FLOPs}} = \frac{N_{\text{total}} + B \times \text{len}_{\text{ctx}} \times \text{KV}_{\text{bytes/token}}}{\text{mem\_bw}}
$$

Solving for B:

$$
B \times \left(\frac{2 \times N_{\text{active}}}{\text{FLOPs}} - \frac{\text{len}_{\text{ctx}} \times \text{KV}_{\text{bytes/token}}}{\text{mem\_bw}}\right) = \frac{N_{\text{total}}}{\text{mem\_bw}}
$$

$$
B^* = \frac{N_{\text{total}} / \text{mem\_bw}}{\frac{2 \times N_{\text{active}}}{\text{FLOPs}} - \frac{\text{len}_{\text{ctx}} \times \text{KV}_{\text{bytes/token}}}{\text{mem\_bw}}}
$$

**Key insight**: The denominator decreases as context length grows. At some critical context length, the denominator hits zero — meaning **t_mem grows faster than t_compute** and the system is **always memory-bound regardless of batch size**.

This happens when:

$$
\frac{2 \times N_{\text{active}}}{\text{FLOPs}} = \frac{\text{len}_{\text{ctx}} \times \text{KV}_{\text{bytes/token}}}{\text{mem\_bw}}
$$

$$
\text{len}_{\text{ctx}}^{\text{critical}} = \frac{2 \times N_{\text{active}} \times \text{mem\_bw}}{\text{FLOPs} \times \text{KV}_{\text{bytes/token}}}
$$

### Example: Llama 3 70B, GQA, FP16, H100

$$
\text{len}_{\text{ctx}}^{\text{critical}} = \frac{2 \times 70 \times 10^9 \times 3.35 \times 10^{12}}{990 \times 10^{12} \times 327{,}680} \approx 1{,}444
$$

**At context lengths > ~1,444 tokens, the KV cache memory traffic grows faster per added batch element than the compute does.** Beyond this point, increasing B always makes the system more memory-bound, not less. You can never reach compute-bound operation.

This is the mathematically precise reason why long-context inference is fundamentally harder to make efficient. The "free batching" region from the simple model (reiner-pope-podcast.md) doesn't just shrink — it can **disappear entirely**.

---

# 6. The Two Regimes of Memory Bottleneck

This gives us a more nuanced view than the simple "memory-bound vs. compute-bound" dichotomy:

| Regime | Condition | Dominant bottleneck | Character |
|---|---|---|---|
| **Weight-bound** | Small B, short context | Weight loading (N_total) | Classic decode bottleneck. Batch is "free." |
| **KV-bound** | Large B, long context | KV cache loading (B × len × KV) | Memory still bottleneck, but now batch is costly. |
| **Compute-bound** | B > B*, short context | FLOPs | Rare in production decode. Common in prefill. |

Most production serving operates in the **transition zone** between weight-bound and KV-bound. This is why the most impactful optimizations target both:

- **Weight quantization** (INT4/FP8): Attacks the N_total term
- **KV cache quantization** (FP8/INT4): Attacks the B × len × KV term
- **GQA/MQA**: Reduces KV_bytes/token by 4–8×
- **KV cache eviction / sliding window**: Reduces effective len_ctx
- **PagedAttention**: Doesn't reduce bytes loaded, but eliminates memory fragmentation so you can actually fit more sequences (higher B)
- **FlashAttention**: Doesn't reduce HBM reads (the KV cache must still be loaded), but optimizes the intermediate memory traffic between HBM and SRAM during the attention computation itself

---

# 7. Summary: Reading the Equation

$$
t_{\text{mem}} = \frac{\underbrace{N_{\text{total}}}_{\text{weights}} + \underbrace{B \cdot \text{len}_{\text{ctx}} \cdot \text{KV}_{\text{bytes/token}}}_{\text{KV cache}}}{\text{mem\_bw}}
$$

**Left term (weights)**: Fixed cost per step. Every decode step reads the whole model. Independent of batch size, context length, or anything else. Reduced by weight quantization.

**Right term (KV cache)**: Variable cost per step. Scales with batch size × context length. This is the term that makes long-context, high-concurrency serving expensive. Reduced by GQA, KV quantization, eviction, and shorter contexts.

**Denominator (mem_bw)**: The pipe. Everything divides by this. Wider pipe = faster. This is why HBM bandwidth (not TFLOP/s) is the primary hardware metric for decode, and why each HBM generation (HBM3 → HBM3e → HBM4) directly improves decode throughput.

The entire inference optimization stack can be understood as efforts to shrink the numerator (fewer bytes to move) or grow the denominator (faster memory) of this single fraction.
