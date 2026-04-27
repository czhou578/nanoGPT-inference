---
title: "LLM Inference Latency: TTFT, ITL, E2E Explained"
source: "https://grok.com/c/a971581f-bcd5-4cc2-90d2-132667f71787?rid=8307b62e-caac-4ab2-b35b-108155ec57c8"
author:
published:
created: 2026-04-25
description: "Grok is an AI assistant built by xAI. Chat, create images, write code, and get real-time answers from the web and X."
tags:
  - "clippings"
---
could you explain ttft, e23 latency, itl, and in which situations they matter? chatbot vs rag vs batch pipelines? go from first principles

**LLM inference from first principles**

An LLM doesn't "think" in one shot like a human. It works in two distinct computational phases every single time it answers a question:

1. **Prefill (prompt processing)**: You send a prompt → the model tokenizes it → it runs the entire prompt through all its layers *in parallel*. This builds the key-value (KV) cache for every token in the input. This phase is **compute-bound** and highly parallelizable on GPUs. The longer the prompt (or context), the more time it takes, but it only happens once.
2. **Decode (token generation)**: The model now generates tokens **one at a time**, autoregressively. For each new token:
	- Take the last token + the existing KV cache
		- Run a single forward pass (much smaller than prefill)
		- Sample the next token This phase is **memory-bound** (KV cache access dominates) and inherently sequential — you literally cannot compute token #47 until you have token #46. This is why generation speed feels "chunky."

All the metrics you asked about are simply different slices of these two phases.

### The three key latency metrics

| Metric | Full name | What it actually measures | Formula (simplified) | Typical good target (2025–2026 hardware) |
| --- | --- | --- | --- | --- |
| **TTFT** | Time To First Token | Time from the moment the user hits "send" until the **first token** appears in the UI. | Prefill time + time to decode the very first output token | < 300–800 ms for great UX |
| **ITL** | Inter-Token Latency | Average time between each **subsequent** token after the first one. | Time per decode step (after TTFT) | < 30–60 ms/token for buttery streaming |
| **E2E** | End-to-End latency | Total time until the **entire response** is finished (user sees the last token). | TTFT + (output length – 1) × ITL | Varies by use case |

- **TTFT** is almost entirely prefill-dominated. If your prompt is 10k tokens or you have slow retrieval, TTFT explodes.
- **ITL** is decode-dominated. This is what makes streaming feel "fast" or "laggy" after the first token.
- **E2E** is what users actually experience when they wait for a complete answer. For a 500-token response, even a great TTFT of 400 ms becomes terrible if ITL is 200 ms/token (E2E ≈ 400 ms + 499 × 200 ms = ~100 seconds).

### When each metric matters (real-world impact)

**TTFT matters most when:**

- The user is waiting for the AI to "start talking" (chat, customer support, coding assistants, voice).
- You have very long contexts or heavy pre-processing (RAG, long documents, tool calls).
- First-impression UX is critical — humans notice delays >1 second as "slow."

**ITL matters most when:**

- Responses are long (>200–300 tokens).
- You stream the answer live (most chat UIs).
- The user is reading along in real time (they hate stuttering after the first sentence).

**E2E latency matters when:**

- The user needs the full answer before they can act (summarization, analysis, batch reports).
- You're measuring overall system performance or comparing models/services.

### Chatbot vs RAG vs Batch pipelines — how the phases change

| Pipeline type | Typical flow | Dominant phase for latency | What usually hurts TTFT | What usually hurts ITL | Primary metric that decides "feels fast" | Optimization focus |
| --- | --- | --- | --- | --- | --- | --- |
| **Chatbot** (simple) | Prompt → model (maybe short history) | Prefill + decode | Prompt length + model size | KV cache size & memory bandwidth | **TTFT** (first token must feel instant) | Speculative decoding, small models, quantization, prefix caching |
| **RAG** (retrieval-augmented) | Query → retrieve docs → build big prompt → model | Prefill (big time) | **Retrieval + long augmented prompt** | Same as chatbot | **TTFT** (retrieval can add 200–1000 ms) | Fast vector DB, reranking, prompt compression, caching embeddings |
| **Batch pipelines** | Thousands of independent prompts → processed together (no user waiting) | Decode (but parallelized) | Almost irrelevant | Throughput per GPU | **Throughput** (tokens/second across batch) not per-request latency | Continuous batching, vLLM/PagedAttention, larger batch sizes, higher quantization |

**Key differences explained:**

- **Chatbot (conversational)**: The user is staring at the screen. They expect the cursor to start typing *immediately*. → Prioritize **low TTFT** above everything. Once the first token arrives, decent ITL keeps the illusion of thinking live. Example: Grok, ChatGPT, Claude in chat mode.
- **RAG**: Same as chatbot *except* you insert an extra expensive step before prefill: retrieval (vector search + reranking + prompt stuffing). This makes TTFT the biggest pain point. A perfect 200 ms model can still feel sluggish if retrieval takes 600 ms. Many RAG systems hide this by streaming "Searching documents..." but the real generation still waits.
- **Batch pipelines** (offline / bulk processing): No human is waiting in real time. You can process 1000 prompts at once. Here TTFT and per-request ITL become almost meaningless. What matters is **total tokens per second across the entire batch** (throughput). You use continuous batching, dynamic batch sizing, and can tolerate higher per-request latency because you're maximizing GPU utilization instead of responsiveness.

### Quick mental model to remember

Think of it like a restaurant:

- **TTFT** = time until the waiter brings the first dish (appetizer arrives).
- **ITL** = time between each subsequent dish (how fast the main course comes out).
- **E2E** = time until your entire table is done eating.
- **Chatbot** = fine-dining experience where speed of first bite matters a lot.
- **RAG** = fine-dining but the chef first has to run to the market to fetch ingredients → delays the first bite.
- **Batch** = catering 500 people at once → you don't care when the first plate comes out, you care how many plates you finish per hour.

These metrics are why different serving frameworks (vLLM, TGI, SGLang, TensorRT-LLM, etc.) and hardware choices exist — they trade off prefill speed vs decode speed vs throughput depending on whether you're building a real-time chatbot or a nightly batch analytics job.
---

# The Future of Latency Optimization

The three metrics â€” TTFT, ITL, E2E â€” are not static targets. The demands placed on each are shifting as AI applications evolve from simple chatbots toward real-time agents, voice interfaces, and reasoning systems. The optimization frontier is moving in several directions simultaneously.

## 1. TTFT Is Under Pressure From Two Directions

**Upward pressure (making TTFT worse):**
- Prompts are getting longer. RAG systems now stuff 32Kâ€“128K tokens of retrieved context. Agentic systems prepend extensive tool schemas, memory, and multi-turn history. Each doubling of prompt length roughly doubles prefill time.
- Reasoning models (o1/o3-class) add "thinking" tokens before the visible output. The user sees nothing while the model reasons internally, making perceived TTFT much worse even if technical TTFT (first token emitted by the model) is unchanged.

**Downward pressure (making TTFT better):**
- **Prefix caching** â€” shared system prompts are pre-computed and reused. For APIs with a fixed system prompt (95%+ of production deployments), this eliminates 50â€“90% of prefill compute.
- **Disaggregated prefill/decode** â€” dedicated prefill GPUs process prompts without interfering with active decode streams. Prefill throughput improves and queuing delays drop.
- **Speculative prefill** â€” newer architectures can begin streaming draft tokens before prefill fully completes, giving the user something to read while the full KV cache is still being built.

**Net trajectory**: For simple chatbots, TTFT is rapidly improving (sub-200ms is achievable with prefix caching + optimized serving). For RAG and agent workloads with long, dynamic prompts, TTFT will remain a major UX bottleneck for the foreseeable future.

## 2. ITL Is Becoming the Differentiator

As TTFT gets solved by caching and disaggregation, ITL becomes the primary axis of competition for interactive applications:

- **Voice/real-time applications** need ITL < 15â€“20ms to avoid perceptible delays in speech synthesis pipelines. Current best-in-class (Groq) achieves ~6â€“10ms ITL on smaller models. This is the threshold that unlocks real-time voice AI agents.
- **Speculative decoding** directly attacks ITL by generating 2â€“4Ã— more tokens per target-model forward pass, effectively dividing ITL by the acceptance rate multiplier.
- **SRAM-heavy decode ASICs** (Groq LPU, Cerebras) achieve ultra-low ITL by eliminating the HBM bandwidth bottleneck during decode. This is a hardware-level solution to what has been a physics-level problem.

**The emerging metric**: **Token throughput at a given ITL SLA** â€” how many concurrent users can you serve while guaranteeing every user gets < 30ms ITL? This composite metric better captures the production reality than either metric alone.

## 3. E2E Latency Is Being Reshaped by Reasoning Models

Inference-time compute scaling (chain-of-thought reasoning, best-of-N sampling, tree search) fundamentally redefines E2E latency:

- A simple completion: 200 output tokens, E2E â‰ˆ 0.5s TTFT + 199 Ã— 30ms ITL â‰ˆ 6.5 seconds.
- A reasoning completion: 2,000 "thinking" tokens + 200 visible tokens, E2E â‰ˆ 0.5s + 2,199 Ã— 30ms â‰ˆ 66 seconds.

The same model, same hardware, same ITL â€” but 10Ã— longer E2E because the model is doing more cognitive work. This shifts pricing from per-token (commodity) to per-task or per-outcome, where the value delivered justifies the extended compute time.

**Streaming UX innovation**: Systems are beginning to show partial reasoning traces to the user during the "thinking" phase (similar to how DeepSeek-R1 shows its chain-of-thought). This doesn't reduce E2E but dramatically improves perceived latency by keeping the user engaged.

## 4. New Metrics Emerging

As the field matures, the three classic metrics are being supplemented:

| Emerging Metric | Definition | Why It Matters |
|---|---|---|
| **Time to Last Token (TTLT)** | E2E latency minus TTFT â€” pure decode phase duration | Isolates decode efficiency for benchmarking |
| **P99 ITL** | 99th percentile inter-token latency (worst-case stutter) | Captures latency spikes from prefill interference, garbage collection, or scheduling contention â€” what users actually complain about |
| **Tokens per Second per Dollar (TPS/$)** | Throughput normalized by infrastructure cost | The metric that actually determines inference economics |
| **Time to Useful Output (TTUO)** | Time until enough tokens have streamed for the user to begin acting | More relevant than TTFT for code completion (need a full line) or RAG (need a full sentence) |

---

# The Investor Lens (Aligned with the Inference Framework)

Latency metrics sit at the intersection of the **Serving / Runtime Layer** and the **Application Layer** of the inference stack. They are the translation layer between raw hardware capability and user-perceived product quality. Understanding which metrics matter for which workloads is essential for evaluating any AI infrastructure or application investment.

## The Core Thesis

> **Latency is not a single number â€” it is a multi-dimensional surface. Companies that optimize the wrong latency dimension for their workload will lose to competitors who optimize the right one. Investors must understand which metric drives value for each business model.**

## Primary Value Drivers

### 1. ITL as the New Hardware Battleground

TTFT is increasingly solvable through software (prefix caching, disaggregated serving, prompt compression). ITL, however, is fundamentally bound by the memory-bandwidth ceiling of the decode hardware. This makes ITL the axis on which hardware differentiation matters most:

- **NVIDIA GPUs**: ITL is limited by HBM bandwidth. An H100 with 3.35 TB/s HBM bandwidth imposes a floor on ITL for large models. Software optimization (speculative decoding, quantization) can improve it 2â€“3Ã—, but the hardware ceiling is fixed per GPU generation.
- **SRAM-heavy ASICs (Groq, Cerebras)**: Can achieve 3â€“10Ã— lower ITL than GPUs by eliminating the HBM bottleneck. This is not a software advantage â€” it is a physics advantage that GPUs cannot match at the same price point.

**Signal to watch**: When a major consumer AI product (voice assistant, real-time translation, coding copilot) requires < 15ms ITL to function, it creates structural hardware demand that only SRAM-heavy ASICs can satisfy cost-effectively. This is the inflection point for decode-specific silicon.

**Investor takeaway**: Companies building products that require ultra-low ITL (real-time voice, interactive agents) are natural customers for non-NVIDIA inference hardware. The "ITL floor" is a hardware-selection forcing function. Groq's commercial thesis depends on ITL-sensitive applications reaching sufficient scale to justify dedicated decode infrastructure.

### 2. TTFT Optimization Is Deflationary for Cloud API Revenue

Every TTFT optimization â€” prefix caching, disaggregated prefill, prompt compression â€” reduces the compute cost of serving a request. For cloud API providers:

- Prefix caching can reduce prefill compute by 50â€“90% for repeat system prompts. This is pure margin for the provider *until* competition forces price reductions.
- The commoditization cascade applies: vLLM and SGLang already implement prefix caching. Within 12 months it becomes table-stakes. Providers who don't implement it face cost disadvantages; providers who do see margin advantages competed away.

**Investor takeaway**: TTFT optimizations are margin-expanding in the short term (6â€“12 months) and margin-neutral in the long term (as competitors adopt). Do not model sustained margin advantage from any single TTFT optimization. The durable advantage belongs to teams that consistently ship the next optimization fastest â€” it is an execution moat, not a technology moat.

### 3. Reasoning Model E2E Latency Sustains GPU Demand

Inference-time compute scaling (reasoning models generating thousands of thinking tokens) is the single most important counter-trend to efficiency-driven cost deflation:

- A reasoning query consumes 10â€“100Ã— more tokens than a simple completion
- Each thinking token requires a full decode step â†’ more GPU-seconds per query
- Users are willing to pay premium pricing for reasoning quality (e.g., OpenAI o1 is priced significantly higher than GPT-4o)

This creates a structural demand floor for inference compute that is resistant to efficiency deflation. Even if ITL improves 3Ã— through speculative decoding, the number of tokens per reasoning query grows faster.

**Signal to watch**: The proportion of API revenue from reasoning-tier models vs. standard completions. When reasoning models exceed 30â€“40% of inference API revenue at any major provider, it confirms that inference-time compute expansion is the dominant demand driver.

**Investor takeaway**: Long GPU infrastructure through the reasoning model era. Reasoning models are the mechanism by which AI chip demand stays robust even as per-token costs fall. Companies with revenue tied to token volume (cloud inference providers, GPU makers) benefit directly from this trend.

### 4. The Latency-Throughput Pricing Frontier

A key market dynamic: cloud inference providers are beginning to offer **tiered pricing based on latency SLAs**, not just model choice:

| Tier | Typical SLA | Pricing Premium | Target Workload |
|---|---|---|---|
| **Real-time** | TTFT < 200ms, ITL < 20ms | 3â€“5Ã— base price | Voice, real-time agents |
| **Interactive** | TTFT < 500ms, ITL < 50ms | 1â€“2Ã— base price | Chat, copilots |
| **Standard** | TTFT < 2s, ITL < 100ms | Base price | General API |
| **Batch** | No latency SLA | 0.3â€“0.5Ã— base price | Offline processing |

This pricing structure means the **same model, same hardware** generates 3â€“10Ã— different revenue per token depending on the latency tier. The marginal cost difference between tiers is modest (serving infrastructure, not model), but the price differential is large.

**Investor takeaway**: Inference providers with the ability to offer credible real-time latency SLAs (requiring disaggregated serving, speculative decoding, and potentially ASIC decode hardware) can capture 3â€“5Ã— revenue premium over batch-only providers on the same model. This is the most under-appreciated margin lever in inference pricing. Evaluate providers on their latency tier diversity, not just per-token pricing.

### 5. The Build-vs-Buy Decision Hinges on Latency Expertise

For enterprises evaluating self-hosting vs. API:

- **Batch workloads**: Self-hosting is increasingly viable. Open-source serving stacks (vLLM + continuous batching) achieve near-optimal throughput with modest engineering effort.
- **Interactive workloads**: Self-hosting with production-grade latency (stable P99 ITL, low TTFT under load) requires significant operational expertise â€” disaggregated serving setup, speculative decoding tuning, prefix cache management, load balancing.
- **Real-time workloads**: Self-hosting at < 20ms ITL SLA is extremely difficult without specialized hardware (Groq, Cerebras, or carefully tuned NVIDIA clusters).

The harder the latency requirement, the more likely enterprises are to pay premium pricing to managed providers. This is the structural dynamic that sustains managed inference API margins against the open-source commoditization pressure.

**Investor takeaway**: The value of managed inference API businesses is proportional to the difficulty of the latency SLA they can guarantee. Batch-tier inference is a commodity with razor-thin margins. Real-time-tier inference is a premium service with defensible margins. Investors should evaluate inference providers on their tightest achievable SLA, not their cheapest per-token price.

## Risk Factors

**Risk 1 â€” SSM architectures collapse the TTFT/ITL distinction.** State Space Models have no prefill/decode asymmetry â€” every step is the same operation. If SSMs reach transformer quality, TTFT and ITL converge to a single metric, and the serving stack simplifies dramatically. This would commoditize disaggregated serving infrastructure and reduce the value of ASIC decode hardware.

**Risk 2 â€” Latency becomes irrelevant for dominant use cases.** If the highest-revenue AI applications turn out to be batch-oriented (synthetic data generation, code review, document processing) rather than real-time, then latency optimization confers no pricing premium. The market would compete purely on throughput-per-dollar 

**Risk 3 â€” Client-side inference eliminates cloud latency entirely.** As quantized models become capable enough to run on-device (Apple Silicon, Qualcomm NPUs), latency-sensitive applications migrate to edge inference where TTFT and ITL are sub-10ms by default. This removes the highest-margin latency tier from cloud providers' revenue.

## Summary Signal for Investors

> **Latency tiers are the new pricing power in inference.** The ability to guarantee SLAs on TTFT, ITL, and P99 stability is the primary margin differentiator between commodity inference ($0.10/M tokens, batch, no SLA) and premium inference ($0.50â€“3.00/M tokens, real-time, guaranteed). Invest in companies that can credibly serve the tightest latency tiers â€” they capture 3â€“10Ã— revenue per token and face structurally less competition because operational difficulty filters out most competitors.

