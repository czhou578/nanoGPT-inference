# Inference Optimization Feasibility for NanoGPT

Current state: ~210K param char-level GPT with KV cache implemented. 4 layers, 4 heads, 64-dim embeddings, block_size=32.

---

## ✅ Highly Feasible (clear educational value, straightforward to implement)

| Optimization | Why Feasible | Notes |
|---|---|---|
| **Continuous Batching** | vLLM concepts already documented. Core idea: maintain a pool of in-flight requests, schedule them into the same forward pass, let requests enter/leave independently. Model already supports batch dim `B`. | Build a request queue + scheduler loop that pads/packs multiple sequences into a single `forward()` call. Requests finish at different times. |
| **Chunked Prefill** | Natural extension of continuous batching. Long prompts get split across steps so decode tokens aren't starved. | Implement a `token_budget` that caps total tokens per forward pass. Partially-prefilled requests resume next step. |
| **Scheduling Policies (FCFS, Priority)** | Pure Python logic on top of a request queue — no model changes needed. | Add priority scores, preemption (evict lowest-priority request when KV memory is exhausted, re-prefill later). |
| **Static Quantization (INT8 weights)** | PyTorch has `torch.quantization.quantize_dynamic` — one-liner for linear layers. | At 210K params you won't see meaningful speedup, but it demonstrates the concept. The real win is on >1B models where weight loading is the bottleneck. |
| **Prefix Caching / Prompt Caching** | Hash KV blocks by token content (vLLM notes already document this thoroughly). If two requests share a prompt prefix, reuse their cached K/V. | Need to change per-`Head` cache into a shared block pool keyed by content hashes. Meaty but very doable. |
| **Token Budget & Decode-Prefill Interleaving** | The scheduler allocates a fixed token budget per step. Running (decode) requests consume 1 token each first; remaining budget goes to prefill. | Core scheduling insight from vLLM. Very teachable at this scale. |

---

## ⚠️ Feasible With Caveats (works conceptually, limited observable benefit at this scale)

| Optimization | Why | Caveat |
|---|---|---|
| **PagedAttention** | Replace contiguous KV cache (`torch.cat`) with a block table mapping logical positions → physical blocks. Eliminates fragmentation and enables prefix sharing. | Significant refactor of the `Head` class. Without a custom CUDA kernel, you'd index into blocks with Python/PyTorch gather ops, which is slower than contiguous attention. Educational value is high; raw perf gain at this scale is negligible. |
| **Speculative Decoding** | Use a smaller "draft" model (or even a simple n-gram model) to guess N tokens, then verify them in one forward pass of the main model. | Need a second, smaller model. At 210K params, the model is already tiny — finding a meaningfully smaller draft model is awkward. But you could use a bigram table as the draft. |

## Only for larger models like GPT-2, 124M

| **Dynamic/Weight-only Quantization (INT4/GPTQ-style)** | `torch.ao.quantization` or manual weight packing. | 64-dim embeddings and 16-dim head sizes mean quantization error is proportionally larger than on 4096-dim models. Demonstrates the mechanics but output quality would degrade noticeably. |
| **KV Cache Quantization (FP8/INT8 KV)** | Store cached K/V in reduced precision, dequantize on the fly during attention. | Same concern — 16-dim head vectors have very few distinct values, so quantization noise is proportionally brutal. Demonstrates the concept but hurts quality. |

---

## Profiling and benchmarking

## ❌ Not Feasible / Not Meaningful at This Scale

| Optimization | Why Not |
|---|---|
| **Tensor Parallelism / Pipeline Parallelism** | Requires multiple GPUs. Model fits in <1 MB. |
| **Flash Attention** | Requires a custom CUDA kernel (`flash_attn` library). Doesn't apply to separate-head architecture (designed for fused multi-head attention). Would need refactor to fused `MultiHeadAttention` first, and even then only helps at large sequence lengths on GPU. |
| **CUDA Graph Capture** | Only useful when Python overhead dominates — i.e., very fast GPU kernels. At 210K params on CPU, Python overhead *is* the model. |
| **Activation Checkpointing** | Training-time optimization, not inference. |
| **MoE Routing / Expert Parallelism** | Architectural change, not an inference optimization on the existing model. |

---

## Recommended Build Order

If the goal is to build toward a **mini inference engine** that mirrors vLLM concepts:

```
1. Continuous Batching          ← multi-request batching with independent completion
2. Token Budget + Scheduling    ← FCFS queue, budget-based admission
3. Chunked Prefill              ← long prompts don't starve decode
4. Prefix Caching               ← hash-based KV block reuse
5. PagedAttention               ← block-table KV management (replaces torch.cat)
6. Speculative Decoding         ← bigram draft + verify
7. INT8 Weight Quantization     ← demonstrate the concept
```

Steps 1–4 are the highest-value and map directly to the vLLM architecture already studied. They also tie in nicely with the frontend visualizer project.

Yes. If you already added batching, chunked prefill, scheduling, prefix caching, paged KV, speculative decoding, and INT8, the next best additions are less “new feature” and more “make this a serious tiny inference engine.”

Highest-value additions:

1. **Benchmark harness**
   Add a repeatable benchmark suite with:
   - tokens/sec
   - time to first token
   - inter-token latency
   - batch throughput vs batch size
   - prefill/decode split
   - KV cache hit rate
   - speculative acceptance rate
   - memory used by KV blocks

   This will make every optimization actually measurable.

2. **Correctness equivalence tests**
   This is probably the most important thing now. Add tests like:
   - full forward logits == cached incremental logits
   - unbatched output == continuously batched output
   - contiguous KV output == paged KV output
   - prefix-cached output == normal prefill output
   - speculative decoding statistically matches target sampling

   These catch almost every subtle bug in inference-engine work.

3. **Refactor attention to fused QKV / fused multi-head tensors**
   Karpathy nanoGPT’s separate `Head` modules are educational, but inefficient. A good next step is one `CausalSelfAttention` module with:

   ```python
   qkv = self.c_attn(x)
   q, k, v = qkv.split(n_embd, dim=2)
   q = q.view(B, T, n_head, head_size).transpose(1, 2)
   ```

   Then use `torch.nn.functional.scaled_dot_product_attention` when possible. This makes the model closer to real GPT implementations and opens the door to Flash Attention-style paths later.

4. **`torch.inference_mode()` and optional AMP**
   Wrap generation/serving in:

   ```python
   with torch.inference_mode():
       ...
   ```

   On CUDA, add optional `torch.autocast(device_type="cuda", dtype=torch.float16)` or `bfloat16` if supported. This is more practically useful than INT8 for a tiny model.

5. **Adaptive speculative decoding**
   Instead of fixed `K`, adjust based on recent acceptance rate:
   - high acceptance: increase `K`
   - low acceptance: decrease `K`
   - cap by remaining context/token budget

   This is very realistic and not too hard.

6. **Better draft model**
   A bigram draft is nice, but a trigram/backoff n-gram draft would likely improve speculative acceptance a lot while staying simple:

   ```text
   use trigram if seen
   else bigram
   else unigram
   ```

   That would make speculative decoding visibly better without training a second transformer.

7. **Streaming API / request simulator**
   If your goal is a mini-vLLM, add a simple request simulator:
   - random arrivals
   - different prompt lengths
   - different max generation lengths
   - priorities
   - cancellation

   Then print or plot scheduler behavior over time. This makes batching/chunked prefill much more meaningful.

8. **Sampling controls**
   Add `temperature`, `top_k`, `top_p`, and maybe repetition penalty. Not an inference optimization, but it makes the generator feel much more complete.

My strongest recommendation: add **benchmarking + correctness tests before any more clever optimization**. At this point, those will make the biggest difference because they let you prove each fancy component still samples from the right model.
