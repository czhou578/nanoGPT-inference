# 🧠 NanoGPT Inference Engine — From First Principles

A deep-dive systems project that **implements 10 production inference optimizations from scratch** on top of Andrej Karpathy's NanoGPT, each paired with automated benchmark suites and interactive browser-based visualizations. The goal is to demonstrate a first-principles understanding of how modern LLM inference engines (like vLLM) work internally — not by reading about them, but by building each component by hand.

> **TL;DR** — Trained a character-level GPT on Shakespeare, then progressively added KV caching, continuous batching, paged attention, chunked prefill, prefix caching, scheduling, speculative decoding, interleaved prefill-decode, and INT8 quantization. Every optimization is benchmarked for throughput and latency, and the key concepts are visualized in a React-based interactive simulation frontend.

---

## 📐 Architecture Overview

```
┌───────────────────────────────────────────────────────────────────────┐
│                          Repository Structure                        │
├───────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  nanogpt.py                    ← Baseline: Karpathy's NanoGPT        │
│  nanogpt-kv-cache.py           ← + KV Cache (prefill/decode split)   │
│  nanogpt-continuous-batching.py← + Continuous Batching + Scheduler    │
│  nanogpt-chunked-prefill.py    ← + Chunked Prefill                   │
│  nanogpt-paged-attention.py    ← + PagedAttention (block allocator)  │
│  nanogpt-prefix-caching.py     ← + Content-hashed prefix caching     │
│  nanogpt-scheduling.py         ← + FCFS / Priority scheduling        │
│  nanogpt-interleaving.py       ← + Fused prefill-decode batches      │
│  nanogpt-spec-decode.py        ← + Speculative decoding (bigram)     │
│  nanogpt-trigram-spec-decode.py← + Trigram draft model variant        │
│  nanogpt-quantize.py           ← + Dynamic & Static INT8 quantization│
│                                                                       │
│  benchmarks/                   ← Automated benchmark suites (22 files)│
│  results/                      ← Raw benchmark output (10 files)     │
│  notes/                        ← 30+ research notes & writeups       │
│  nanogpt-notebooks/            ← Jupyter notebooks for exploration   │
│  frontend/                     ← React + Vite interactive visualizer │
│                                                                       │
└───────────────────────────────────────────────────────────────────────┘
```

---

## 🔧 Implemented Optimizations

Each optimization is a standalone Python file that extends the base NanoGPT model. The implementations are intentionally self-contained — you can read each file top-to-bottom and understand exactly how the technique works at the tensor level.

### 1. KV Cache — `nanogpt-kv-cache.py`
**Problem:** During autoregressive decoding, the vanilla transformer recomputes keys and values for *all* previous tokens at every step — O(n²) redundant work.

**Implementation:**
- Added `key_cache` and `value_cache` tensors to each attention head
- Split inference into a **prefill phase** (process the full prompt once) and a **decode phase** (feed only the new token, attend over the cached K/V)
- Added `start_pos` parameter to the forward pass for correct positional embeddings during single-token decoding
- Implemented `clear_kv_cache()` for proper cache lifecycle management

**Benchmark result:** Up to **2.6× throughput improvement** over no-cache generation on medium-length sequences.

---

### 2. Continuous Batching — `nanogpt-continuous-batching.py`
**Problem:** Static batching wastes GPU cycles — when one request finishes early, its slot sits idle until the entire batch completes.

**Implementation:**
- Designed a `Request` dataclass tracking per-request state: prompt tokens, generated tokens, KV cache, prefill cursor, and lifecycle status (`waiting` → `prefilling` → `active` → `done`)
- Built `assemble_batch_cache()` / `disassemble_batch_cache()` to left-pad variable-length per-request KV caches into batched tensors with attention masks
- Implemented `scheduled_generate()` — an event-loop that dynamically admits new requests into empty batch slots at every decode step
- Token budget enforcement prevents over-scheduling

---

### 3. Chunked Prefill — `nanogpt-chunked-prefill.py`
**Problem:** Long prompts create a compute-heavy prefill phase that blocks all decode work. Active requests stall while a new prompt is being processed.

**Implementation:**
- Split prefill into fixed-size chunks (configurable `token_budget`)
- Each scheduler step processes one prefill chunk *and* one decode step for all active requests
- Prefill cursor tracks partial progress through the prompt
- Explicit position embeddings (`pos` parameter) ensure chunks get correct positional encoding regardless of their offset in the prompt

---

### 4. PagedAttention — `nanogpt-paged-attention.py`
**Problem:** Contiguous KV cache allocation wastes memory through fragmentation and requires pre-allocating for the maximum possible sequence length.

**Implementation:**
- Built a `KVBlockPool` — pre-allocated GPU memory organized as `(num_blocks, block_size, head_size)` tensors per layer/head pair
- Implemented `BlockAllocator` with `allocate_one()`, `allocate_n()`, and `free_blocks_for_request()` for dynamic block management
- Each `Request` maintains a `block_table` (list of physical block indices) that maps logical positions to physical memory
- `write_kv_to_pool()` and `gather_kv_from_pool()` translate between logical token positions and physical block/slot addresses
- `assemble_paged_cache()` / `disassemble_paged_cache()` gather/scatter between the block pool and batched model input

**Key insight:** The block table indirection is the same principle as OS virtual memory — logical addresses (token positions) map to physical addresses (block indices) through a page table.

---

### 5. Prefix Caching — `nanogpt-prefix-caching.py`
**Problem:** When multiple requests share the same system prompt or common prefix, each one redundantly recomputes the KV cache for that prefix.

**Implementation:**
- Content-addressed caching using chained MD5 hashes: `hash(parent_hash, token_ids)` — each block's hash depends on all preceding blocks, forming a Merkle-like chain
- `BlockCache` with LRU eviction policy (`max_blocks` configurable)
- `find_cached_prefix()` walks the prompt left-to-right in block-sized chunks, looking up each block's hash
- `load_cached_blocks()` restores cached KV data into a request's cache, advancing `prefill_cursor` past the cached portion
- `commit_completed_blocks()` inserts newly computed full blocks into the global cache after prefill

---

### 6. Scheduling — `nanogpt-scheduling.py`
**Problem:** A naive FIFO queue can't handle mixed priorities, memory pressure, or preemption.

**Implementation:**
- `Scheduler` class with configurable policies: **FCFS** (first-come-first-served) and **Priority** (lower number = higher priority)
- Heap-based waiting queue using `heapq` with composite sort keys `(priority, arrival_time, id)`
- `_maybe_admit()` checks KV memory budget before promoting requests from waiting → prefilling
- `_maybe_preempt()` evicts the lowest-priority active request when memory usage exceeds `max_kv_tokens`, clearing its KV cache and re-enqueuing it
- Request lifecycle: `waiting` → `prefilling` → `active` → `done` (with possible re-entry to `waiting` after preemption)

---

### 7. Prefill-Decode Interleaving — `nanogpt-interleaving.py`
**Problem:** Chunked prefill processes prefill and decode sequentially in each step. Fusing them into a single forward pass maximizes hardware utilization.

**Implementation:**
- `assemble_fused_batch()` packs decode requests (1 token each) and one prefill request (N tokens from the chunk) into a single `(B, T_max)` batch tensor with left-padding
- Each row gets its own position embeddings and attention mask
- `disassemble_fused_cache()` strips the model's output back into per-request KV caches, correctly handling the variable number of new tokens per row
- Integrates prefix caching: cached blocks are loaded before the fused pass, and newly completed blocks are committed afterward

---

### 8. Speculative Decoding (Bigram Draft) — `nanogpt-spec-decode.py`
**Problem:** Autoregressive decoding is inherently sequential — each token requires a full forward pass through the target model.

**Implementation:**
- `BigramDraftModel` — a zero-parameter statistical model that computes `P(next | current)` from training data bigram counts with Laplace smoothing
- `draft_tokens()` samples K candidate tokens autoregressively from the draft model
- `verify_candidates()` runs all K+1 tokens (current + K candidates) through the target model in a **single batched forward pass**
- `accept_reject()` implements the standard rejection sampling algorithm:
  - Accept token `i` with probability `min(1, p_target[token] / q_draft[token])`
  - On rejection, resample from `clamp(p_target - q_draft, min=0)` (the "residual" distribution)
  - If all K tokens accepted, sample a bonus token from the target's distribution at position K+1
- `trim_kv_cache()` rolls back the KV cache to only keep entries for accepted tokens

**Integrations:** Full PagedAttention support with block allocation, scheduling with preemption, and prefix caching.

---

### 9. Speculative Decoding (Trigram Draft) — `nanogpt-trigram-spec-decode.py`
**Problem:** Bigram models have limited predictive power because they only condition on one token of context.

**Implementation:**
- `TrigramDraftModel` — conditions on `P(next | prev, current)` using a `(vocab_size, vocab_size, vocab_size)` count tensor
- Automatic fallback to the bigram model when a `(prev, current)` pair has fewer than `min_context_count` observations
- Temperature-scaled sampling for controlling draft diversity
- Same verification pipeline as the bigram variant

---

### 10. INT8 Quantization — `nanogpt-quantize.py`
**Problem:** Full FP32 model weights consume excessive memory and limit batch sizes.

**Implementation:**
- **Dynamic Quantization:** Applied `torch.quantization.quantize_dynamic()` to all `nn.Linear` layers, converting weights to INT8 at runtime
- **Static Quantization:** Full quantization-aware pipeline:
  - Fused `Linear + ReLU` in feed-forward blocks using `torch.ao.quantization.fuse_modules()`
  - Wrapped every Linear layer (K/Q/V projections, attention output, FFN layers, LM head) with `QuantStub`/`DeQuantStub`
  - Calibrated on 100 validation batches to determine per-tensor scale/zero-point
  - Converted to fully quantized INT8 model

---

## 📊 Benchmark Suites

Every optimization includes an automated benchmark harness in `benchmarks/` that measures:

| Metric | Description |
|--------|-------------|
| **Wall-clock time** | End-to-end generation latency |
| **Tokens/second** | Throughput (total generated tokens / wall time) |
| **TTFT** | Time-to-first-token (prefill latency) |
| **Speedup** | Ratio vs. baseline (no-cache or naive approach) |

### Benchmark Coverage

| Implementation | Benchmark File | Key Scenarios |
|---|---|---|
| KV Cache | `kv_cache_baseline_benchmark_runs.py` | Varying prompt/generation lengths, generation length sweep |
| Continuous Batching | `continuous_batching_benchmark_runs.py` | Multi-request throughput, batch utilization |
| Chunked Prefill | `chunked_prefill_benchmark_runs.py` | Long prompt chunking, interleave ratio |
| PagedAttention | `paged_attention_benchmark_runs.py` | Block allocation overhead, memory fragmentation |
| Prefix Caching | `prefix_caching_benchmark_runs.py` | Shared prefix hit rates, cache eviction |
| Scheduling | `scheduling_benchmark_runs.py` | FCFS vs. priority, preemption under memory pressure |
| Interleaving | `interleaving_benchmark_runs.py` | Fused vs. sequential prefill-decode |
| Speculative Decoding (Bigram) | `speculative_decoding_benchmark_runs.py` | Acceptance rates, K-value sweep |
| Speculative Decoding (Trigram) | `trigram_speculative_decoding_benchmark_runs.py` | Trigram vs. bigram acceptance rates |
| Simulation Traces | `simulation_benchmark_runs.py` | End-to-end request simulation |

All raw results are stored in `results/` with detailed per-scenario breakdowns.

---

## 🖥️ Interactive Frontend

A React + Vite web application that provides interactive, step-by-step visualizations of each inference technique.

### Features

- **6 Interactive Simulations** with play/pause/step-forward/step-back transport controls and adjustable playback speed (0.5×–4×):
  - **KV Cache** — Watch the cache grow token-by-token as the model decodes
  - **Continuous Batching** — See requests dynamically enter and leave batch slots
  - **Chunked Prefill** — Observe how long prompts are split and interleaved with decode
  - **PagedAttention** — Visualize block allocation, block tables, and physical memory layout
  - **Scheduling** — Compare FCFS vs. priority scheduling with live preemption
  - **Speculative Decoding** — Step through draft/verify/accept-reject rounds

- **Knowledge Base** — 30+ in-depth research articles organized by topic (21 categories), rendered from Markdown with KaTeX math support, syntax highlighting, and source citations

- **Pipeline Visualizer** — End-to-end multimodal inference pipeline showing vision encoding, token sequences, block memory allocation, and decode step logs

- **Bibliography** — Auto-extracted source citations and paper references across all articles

### Tech Stack

| Layer | Technology |
|-------|-----------|
| Framework | React 18 |
| Build Tool | Vite 8 |
| Routing | React Router v7 |
| Markdown | react-markdown + remark-gfm + remark-math |
| Math Rendering | rehype-katex |
| Syntax Highlighting | rehype-highlight |
| Icons | Lucide React |

---

## 📚 Research Notes

The `notes/` directory contains extensive first-principles research across three areas:

### Concept Deep-Dives (`notes/concepts/` — 31 articles)

In-depth explorations of inference infrastructure topics, including:

- **Hardware Fundamentals** — HBM vs. SRAM memory hierarchies, roofline model analysis, arithmetic intensity
- **Inference Optimizations** — Streaming generation, pipeline parallelism, CUDA graphs, speculative decoding, PagedAttention, dynamic batching, prefetch pipelines
- **Memory Management** — KV cache quantization, sliding-window eviction, memory offload strategies, GPU-CPU overlap
- **System Architecture** — Disaggregated prefill/decode, request coalescing, async prefill, token parallelism, FP8 kernels, context window streaming
- **Economics & Strategy** — API pricing models, inference cost analysis, open-source economics, investor frameworks
- **vLLM Internals** — Deep dive into the vLLM codebase architecture and design decisions

### Benchmark Writeups (`notes/benchmark-writeups/` — 10 reports)

Detailed analysis reports for each benchmark suite, including methodology, results tables, performance analysis, and takeaways.

---

## 🛠️ Tech Stack

### Backend (Inference Engine)

- **Python 3** + **PyTorch** — All model implementations and inference loops
- **torch.ao.quantization** — INT8 dynamic and static quantization
- Character-level tokenization on Tiny Shakespeare (~1.1M characters)

### Frontend (Visualizer)

- **React 18** + **Vite 8** — Component-based UI with hot module replacement
- **React Router v7** — Client-side routing for notes, simulations, visualizer, and bibliography views
- **react-markdown** ecosystem — Markdown rendering with math (KaTeX), GFM tables, and syntax highlighting

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- PyTorch 2.0+
- Node.js 18+

### Running an Inference Optimization

Each `nanogpt-*.py` file is self-contained. It trains the model, runs generation, and executes its benchmark suite:

```bash
# Example: Run the KV cache implementation with benchmarks
python nanogpt-kv-cache.py

# Example: Run speculative decoding with trigram draft model
python nanogpt-trigram-spec-decode.py
```

### Running the Frontend

```bash
cd frontend
npm install
npm run dev
```

Navigate to `http://localhost:5173` to explore the interactive simulations and knowledge base.

---

## 📁 Repository Structure

```
.
├── nanogpt.py                        # Baseline NanoGPT (Karpathy)
├── nanogpt-kv-cache.py               # KV cache implementation
├── nanogpt-continuous-batching.py     # Continuous batching + request scheduler
├── nanogpt-chunked-prefill.py         # Chunked prefill with token budgets
├── nanogpt-paged-attention.py         # PagedAttention with block allocator
├── nanogpt-prefix-caching.py          # Content-hashed prefix caching
├── nanogpt-scheduling.py             # FCFS + priority scheduling with preemption
├── nanogpt-interleaving.py           # Fused prefill-decode interleaving
├── nanogpt-spec-decode.py            # Speculative decoding (bigram draft)
├── nanogpt-trigram-spec-decode.py    # Speculative decoding (trigram draft)
├── nanogpt-quantize.py               # Dynamic + static INT8 quantization
├── input.txt                          # Tiny Shakespeare training data
│
├── benchmarks/                        # Automated benchmark suites
│   ├── kv_cache_baseline_benchmark_runs.py
│   ├── continuous_batching_benchmark_runs.py
│   ├── chunked_prefill_benchmark_runs.py
│   ├── paged_attention_benchmark_runs.py
│   ├── prefix_caching_benchmark_runs.py
│   ├── scheduling_benchmark_runs.py
│   ├── interleaving_benchmark_runs.py
│   ├── speculative_decoding_benchmark_runs.py
│   ├── trigram_speculative_decoding_benchmark_runs.py
│   ├── simulation_benchmark_runs.py
│   └── ... (benchmark implementations + plots)
│
├── results/                           # Raw benchmark output
│   ├── kv_cache_results.txt
│   ├── continuous_batching_results.txt
│   ├── paged_attent_results.txt
│   └── ... (10 result files)
│
├── nanogpt-notebooks/                 # Jupyter notebooks for exploration
│   ├── nanogpt-kv-cache.ipynb
│   ├── nanogpt-paged-attention.ipynb
│   ├── nanogpt-speculative-decoding.ipynb
│   └── ... (10 notebooks)
│
├── notes/
│   ├── concepts/                      # 31 in-depth research articles
│   ├── benchmark-writeups/            # 10 benchmark analysis reports
│   └── plans/                         # Implementation planning documents
│
└── frontend/                          # React + Vite interactive visualizer
    └── src/
        ├── App.jsx                    # Router setup
        ├── LandingPage.jsx            # Knowledge base landing
        ├── ArticleView.jsx            # Markdown article renderer
        ├── Bibliography.jsx           # Auto-generated citations
        ├── Sidebar.jsx                # Navigation sidebar
        ├── SimulationPage.jsx         # Simulation controller
        ├── PipelineVisualizer.jsx     # E2E pipeline visualization
        ├── Visualizer.jsx             # Multimodal visualizer
        ├── notesRegistry.js           # Note taxonomy + metadata extraction
        └── simulations/
            ├── KVCacheViz.jsx
            ├── ContinuousBatchingViz.jsx
            ├── ChunkedPrefillViz.jsx
            ├── PagedAttentionViz.jsx
            ├── SchedulingViz.jsx
            ├── SpeculativeDecodingViz.jsx
            ├── simulationData.js      # Scenario definitions
            └── simulations.css
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

MIT
