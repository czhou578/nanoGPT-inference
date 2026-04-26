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