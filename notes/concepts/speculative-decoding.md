# Speculative Decoding: A Product Lens

## What It Is (30-Second Recap)

Standard autoregressive decoding generates one token per forward pass through the target model. Speculative decoding uses a **small, fast draft model** to propose `K` candidate tokens, then the **large target model verifies all K in a single forward pass**. Accepted tokens are kept; the first rejected token is resampled from an adjusted distribution. The output distribution is **mathematically identical** to the target model — speculation is a pure latency optimization, not an approximation.

Key insight: verification is cheap because the target model processes all K draft tokens in parallel (like a prefill), so you pay roughly the cost of **one** target forward pass to potentially get **multiple** tokens.

---

## When Does It Help a Chatbot?

Chatbots are **latency-bound**. The user is staring at a blinking cursor. Time-to-first-token (TTFT) matters, but **inter-token latency (ITL)** dominates the perceived experience for longer responses. Speculative decoding directly attacks ITL.

### Scenarios Where It Shines

| Scenario | Why It Helps |
|---|---|
| **Low QPS, large model** | GPU compute is underutilized during decode (memory-bandwidth-bound). The draft model's forward passes fit in the "spare" compute. Verification is nearly free. |
| **Highly predictable outputs** | Code generation, structured JSON, boilerplate text, templated answers — the draft model's acceptance rate `α` is high (0.7–0.9+), so you get close to `K` tokens per target pass. |
| **Single-user / low-concurrency** | With one request in flight, the GPU has idle SMs. The draft model exploits this slack. |
| **Long outputs** | The latency savings compound. Saving 50ms/token × 500 tokens = 25 seconds off a response. |

### Product Impact

- **Perceived speed**: Users report higher satisfaction when text streams 2–3× faster, even if total generation time is similar.
- **Enables larger models**: If your target model is too slow for interactive use, speculation can bring ITL into the acceptable range (~30–80ms) without switching to a weaker model.
- **No quality regression**: Unlike quantization or distillation, there is zero accuracy trade-off. The output distribution is identical.

---

## When Does It Hurt a Batch Job?

Batch inference is **throughput-bound**. You want to maximize tokens-per-second-per-dollar across thousands of requests. The GPU should be saturated. This changes the calculus completely.

### Why It Can Backfire

| Issue | Explanation |
|---|---|
| **Draft model steals compute** | In a throughput-optimized serving system (continuous batching, large batch sizes), the GPU is already saturated. Running a draft model **competes** for the same compute/memory bandwidth that could serve more requests. |
| **Wasted work on rejection** | Every rejected draft token is a forward pass of the draft model that produced nothing. At low acceptance rates (creative tasks, diverse distributions), this waste is significant. |
| **Memory pressure** | The draft model's weights, KV cache, and activations consume GPU memory. That memory could instead hold a larger batch of requests for the target model, improving throughput via amortization. |
| **Complexity in continuous batching** | Requests in a batch have different acceptance rates and are at different stages. Speculation makes scheduling harder — some requests advance K tokens, others advance 1, fragmenting the batch. |
| **Diminishing returns at high batch size** | At large batch sizes, decode is already compute-bound (not memory-bandwidth-bound). The arithmetic intensity is high enough that there's no "free" compute for the draft model. |

### The Throughput Tax

Think of it this way: in a batch job, every FLOP spent on the draft model is a FLOP **not** spent on processing another request through the target model. The question becomes:

> Does the speculative speedup per request exceed the throughput loss from reduced batch capacity?

For most high-throughput offline workloads (data labeling, bulk summarization, synthetic data generation), the answer is **no**. You're better off running the target model with maximum batch size and continuous batching.

### When Batch + Speculation *Can* Work

- **Strict per-request latency SLAs** even in batch mode (e.g., "process 10K requests but each must complete within 5s").
- **Very high acceptance rate** tasks (structured extraction, code transpilation) where almost no draft work is wasted.
- **Small batch sizes** where the GPU isn't fully utilized anyway.

---

## Draft Model Selection

Choosing the draft model is the single most important design decision. It controls the **acceptance rate** `α`, which determines whether speculation is net-positive.

### The Core Trade-Off

```
                    ┌─────────────────────────────────────────┐
  Acceptance Rate   │  Larger draft → higher α → more tokens  │
       α            │  per verify pass                        │
                    └────────────────┬────────────────────────┘
                                     │
                              vs.    │
                                     │
                    ┌────────────────┴────────────────────────┐
  Draft Latency     │  Larger draft → slower K proposals →    │
       t_d          │  higher overhead per speculation round  │
                    └─────────────────────────────────────────┘
```

The expected speedup is roughly:

```
Speedup ≈ E[accepted tokens + 1] / (1 + K × t_draft / t_target)
```

You want `α` high and `t_draft / t_target` low. These goals are in tension.

### Practical Selection Criteria

#### 1. Architecture Compatibility

| Approach | Pros | Cons |
|---|---|---|
| **Same-family smaller model** (e.g., Llama-3-8B drafting for Llama-3-70B) | Shared vocab/tokenizer, easy to deploy, good α for in-distribution text | Still a separate model to load |
| **Pruned/distilled version** of the target | Highest α (trained to mimic target), shared tokenizer guaranteed | Requires training effort |
| **Self-speculative** (early exit from target) | No extra model, no extra memory for weights | Lower α, requires architectural support (e.g., early exit heads) |
| **N-gram / retrieval draft** | Near-zero compute cost | Low α except for highly repetitive/templated outputs |
| **Medusa-style multi-head** | Single model, parallel candidate generation | Requires fine-tuning extra heads, couples draft to target |

#### 2. Size Ratio Rule of Thumb

- Draft should be **5–20× smaller** than target (in parameters).
- Below 5×: the draft is too slow relative to savings.
- Above 20×: the draft is too weak, α drops, and you waste work.
- Sweet spot depends on task — structured outputs tolerate smaller drafts.

#### 3. Vocabulary & Tokenizer Match

The draft and target **must share the same tokenizer**. Mismatched vocabularies make token-level verification impossible without expensive detokenize–retokenize steps. This is a hard constraint, not a preference.

#### 4. Acceptance Rate Benchmarking

Before deploying, measure `α` on realistic traffic:

- `α > 0.8`: Speculation is a clear win. Expect 2–3× ITL reduction.
- `α ≈ 0.5–0.7`: Marginal. Wins for latency-sensitive, loses for throughput-sensitive.
- `α < 0.5`: Speculation is likely net-negative. Draft is too weak or task is too creative/diverse.

#### 5. Hardware-Aware Placement

- **Same GPU**: Simplest. Draft model shares GPU with target. Works when target doesn't saturate compute (low batch, large model).
- **Separate smaller GPU**: Draft runs on a cheaper accelerator (e.g., T4 drafting for an A100 target). Eliminates compute contention but adds network latency and system complexity.
- **CPU draft**: For tiny draft models (n-gram, small transformer). Eliminates GPU contention entirely but caps draft speed.

---

## Decision Framework: Should You Use Speculative Decoding?

```
Is your workload latency-sensitive (interactive)?
├── YES
│   ├── Is your GPU underutilized during decode? (low batch size, large model)
│   │   ├── YES → ✅ Strong candidate. Pick a same-family draft 10-15× smaller.
│   │   └── NO  → ⚠️  Measure carefully. Draft may steal useful compute.
│   └── Is output highly structured/predictable?
│       ├── YES → ✅ Even better. Expect α > 0.8.
│       └── NO  → ⚠️  Expect α ~ 0.5–0.6. Benchmark before committing.
└── NO (batch / throughput job)
    ├── Do you have per-request latency SLAs?
    │   ├── YES → ⚠️  Consider speculation, but benchmark throughput impact.
    │   └── NO  → ❌ Skip speculation. Maximize batch size instead.
    └── Is your batch size small (GPU not saturated)?
        ├── YES → ⚠️  Speculation might help. Measure.
        └── NO  → ❌ Definitely skip. Throughput loss > latency gain.
```

---

## The Future of Speculative Decoding

Speculative decoding is rapidly evolving from a niche trick into a standard optimization in inference engines (like vLLM, TensorRT-LLM, TGI). The frontier of research is focused on pushing acceptance rates higher and removing the need for a separate draft model.

### Emerging Trends

1.  **Draft-Free / Self-Speculation approaches:** Methods like Medusa or structured early-exit heads add extra prediction heads to the target model itself. This simplifies deployment (one model to load, one KV cache to maintain) but requires fine-tuning the base model's weights. Because it leverages the target model's internal representations, `α` is usually much higher than a separate small model.
2.  **Tree Attention:** Instead of proposing a single sequence of `K` tokens, the draft model proposes a *tree* of candidate sequences. Verification then validates multiple paths simultaneously using a specialized attention mask. This significantly boosts the expected number of accepted tokens per step at a slight compute cost during verification.
3.  **Heterogeneous Speculation:** Offloading the draft model entirely to the CPU or edge devices, while the target model sits in the cloud or on a powerful GPU. This is particularly relevant for "AI PC" setups where local compute handles drafting and cloud compute handles verification.
4.  **Continuous Speculation integration:** The challenge of scheduling specular requests in continuous batching (fragmentation) is being solved by systems that dynamically adjust `K` based on batch size and GPU utilization in real-time. When throughput goes up, `K` scales down to zero.

---

## The Investor Lens (Aligned with the Inference Innovation Framework)

Speculative decoding sits squarely in the **Algorithmic Efficiency** layer of the inference stack. By doing more with the same hardware, it shifts value across the ecosystem and acts as a persistent deflationary force on infrastructure pricing.

### Primary Drivers & Value Capture

*   **Direct Beneficiaries (Margin Expansion):** Cloud inference providers (e.g., Together AI, Fireworks) capture immediate margin expansion. They can serve the same token volume using fewer GPU-hours because the draft model exploits memory-bandwidth bottlenecks that were previously wasted compute. 
*   **Pressure Point on GPU Vendors:** When the same output requires fewer GPU-hours, revenue-per-model-deployed for hardware vendors like NVIDIA declines unless volume compensates. Speculative decoding increases the effective throughput of existing hardware for latency-bound tasks.
*   **The Jevons Paradox:** As speculative decoding reduces the cost and latency of inference, it triggers the Jevons paradox. Lower token costs expand the population of viable use cases (e.g., real-time voice agents), which ultimately increases total token demand faster than efficiency reduces it.

### Moat and Commoditization Risks

*   **The Commoditization Cascade:** Speculative decoding is rapidly moving through the commoditization cascade. What started in Google/DeepMind papers is now being integrated into open-source runtimes like vLLM and TensorRT-LLM. *No single inference optimization technique is a durable moat.* Companies claiming a moat based purely on speculative speeds will see it erode within 12–24 months as the technique becomes table-stakes.
*   **Enterprise "Build vs. Buy" Shift:** As open-source serving runtimes perfect continuous speculative decoding, self-hosting a 70B parameter model becomes much more cost-competitive (on latency and QPS) with managed APIs. This narrows the operational advantage of managed providers and accelerates open-weight model enterprise adoption.
*   **Risk: Throughput-Heavy Revenue Models:** If an inference provider's primary revenue comes from batch processing rather than interactive latency-sensitive APIs, speculation offers marginal value. Investors must ensure the optimization technique aligns with the company's core customer workloads.

---

## TL;DR

| Dimension | Chatbot (Latency) | Batch Job (Throughput) |
|---|---|---|
| **Primary metric** | Inter-token latency (ms/token) | Tokens/second/$ |
| **GPU utilization during decode** | Low (memory-bound) | High (compute-bound) |
| **Speculation value** | High — exploits idle compute | Low — competes for saturated compute |
| **Draft model cost** | Amortized by latency savings | Direct throughput tax |
| **Best draft strategy** | Same-family, 10–15× smaller | Don't speculate; or self-speculative if forced |
| **Quality impact** | None (mathematically identical) | None (mathematically identical) |

