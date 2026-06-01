# KV Cache Benchmark Results Write-up

This note explains the results in [`kv_cache_results.txt`](../kv_cache_results.txt) for the KV-cache baseline benchmark in [`benchmarks/kv_cache_baseline.py`](../benchmarks/kv_cache_baseline.py) and [`benchmarks/kv_cache_baseline_benchmark_runs.py`](../benchmarks/kv_cache_baseline_benchmark_runs.py).

## Executive Summary

The benchmark shows that KV caching consistently improves generation throughput for this NanoGPT-style model. Across the named benchmark cases, cached generation is between **1.11x and 2.60x faster** than the no-cache baseline, with an average speedup of about **1.88x**. In the generation length sweep, the speedup ranges from **1.01x to 1.79x**, averaging about **1.45x**.

The main takeaway is that the KV cache is doing what it is supposed to do: it avoids recomputing keys and values for the entire context on every generated token. Instead, the model processes the prompt once during prefill, stores the previous keys and values, and then feeds only one new token per decode step.

The results are especially meaningful because the model and workload are intentionally small:

- The model has only **0.056769M parameters**.
- The benchmark runs on **CPU**.
- The context length is only **64 tokens**.
- Each timing window is short, usually fractions of a second.

Even under these small conditions, where Python overhead, sampling overhead, and tensor concatenation overhead are large relative to model compute, KV caching still produces clear throughput gains in most cases. On larger models, longer contexts, and GPU-backed inference systems, the same principle usually matters even more.

## What Was Benchmarked

The benchmark compares two generation paths:

| Method | Behavior | Why It Matters |
|---|---|---|
| `no_cache` | Recomputes the active context on every generated token. | This is the simple baseline. It does extra work because old tokens are repeatedly processed. |
| `kv_cache` | Runs the prompt once, stores key/value tensors inside each attention head, then decodes one token at a time. | This is the standard inference optimization used by autoregressive language models. |

The important implementation detail is that the model switches behavior based on mode:

- `model.train()` forces the no-cache attention path.
- `model.eval()` enables the internal KV-cache path.

In the no-cache path, each decode step crops the sequence to the last `block_size` tokens and runs a normal forward pass. In the cached path, the benchmark first performs a **prefill** over the prompt, then each remaining decode step forwards only the latest token while attending over cached keys and values.

## Metrics

The output reports:

| Metric | Meaning |
|---|---|
| `tokens` | Number of new tokens generated. |
| `wall_time_s` | Total elapsed time for the generation run. Lower is better. |
| `tokens_per_s` | Generated tokens divided by wall time. Higher is better. |
| `ttft_ms` | Time to first token in milliseconds. Lower is better. |
| `KV-cache throughput speedup` | `kv_cache tokens_per_s / no_cache tokens_per_s`. Higher is better. |

The most important metric here is **tokens per second**, because this benchmark is focused on decode throughput. TTFT is still useful, but in this particular harness it is affected by the difference between the no-cache first step and the cached prefill step, so it should be interpreted more carefully.

## Training Context

Before the benchmark runs, the file shows a short training log:

| Step | Train Loss | Validation Loss |
|---:|---:|---:|
| 0 | 4.1800 | 4.1795 |
| 20 | 3.6072 | 3.6477 |
| 40 | 3.3308 | 3.3400 |
| 60 | 3.1136 | 3.1371 |
| 80 | 2.9638 | 2.9796 |
| 100 | 2.8327 | 2.8783 |
| 119 | 2.7903 | 2.8249 |

The losses decline steadily, which means the tiny model is learning the character-level dataset. The generated sample after training is still noisy, which is expected for a tiny model trained for only 120 iterations on CPU. For this benchmark, generation quality is not the point. The point is to compare the cost of two inference paths on the same model.

## Named Benchmark Results

These are the main benchmark cases:

| Case | Prompt Len | Generated Tokens | No Cache Tok/s | KV Cache Tok/s | Speedup | Wall Time Reduction |
|---|---:|---:|---:|---:|---:|---:|
| `small_smoke_test` | 8 | 16 | 126.67 | 185.93 | 1.47x | 31.8% |
| `medium_generation` | 16 | 32 | 102.20 | 265.23 | 2.60x | 61.5% |
| `longer_generation` | 16 | 48 | 143.56 | 276.58 | 1.93x | 48.1% |
| `heavier_prompt` | 32 | 32 | 119.99 | 274.34 | 2.29x | 56.3% |
| `near_context_limit` | 8 | 56 | 209.39 | 233.16 | 1.11x | 10.2% |

### Overall Trend

The named runs show a strong throughput advantage for KV caching. The largest speedup appears in `medium_generation`, where cached generation reaches **265.23 tokens/sec** compared with **102.20 tokens/sec** for the no-cache baseline. That is a **2.60x throughput improvement** and cuts wall time by about **61.5%**.

The `heavier_prompt` result is also important. It uses a longer prompt of 32 tokens and generates 32 new tokens. The KV-cache path reaches **274.34 tokens/sec**, while the no-cache path reaches **119.99 tokens/sec**, giving a **2.29x speedup**. This matches the expected behavior: as the prompt gets longer, recomputing the full context becomes more wasteful, so caching previous keys and values becomes more valuable.

### Why KV Cache Helps

Autoregressive generation emits one token at a time. Without a cache, generating token `t` requires the model to reprocess the previous tokens again, even though their keys and values were already computed in earlier steps.

Conceptually, no-cache decoding repeats work like this:

```text
step 1: process prompt
step 2: process prompt + token 1
step 3: process prompt + token 1 + token 2
step 4: process prompt + token 1 + token 2 + token 3
...
```

KV-cache decoding changes the pattern:

```text
prefill: process prompt once and store K/V tensors
step 1: process one new token, attend to cached prompt
step 2: process one new token, attend to cached prompt + token 1
step 3: process one new token, attend to cached prompt + token 1 + token 2
...
```

The cached version still attends over the previous tokens, but it does not recompute all previous key and value projections. That reduces repeated linear-projection work and reduces the amount of token history that must flow through the transformer blocks at every generation step.

## Generation Length Sweep

The generation length sweep holds the prompt length fixed at 8 tokens and varies the number of generated tokens:

| Generated Tokens | No Cache Tok/s | KV Cache Tok/s | Speedup | Wall Time Reduction |
|---:|---:|---:|---:|---:|
| 8 | 222.60 | 275.18 | 1.24x | 18.9% |
| 16 | 109.83 | 193.64 | 1.76x | 43.3% |
| 32 | 225.47 | 227.41 | 1.01x | 0.8% |
| 48 | 130.52 | 233.10 | 1.79x | 44.0% |
| 56 | 188.92 | 275.38 | 1.46x | 31.4% |

### Sweep Trend

The sweep mostly confirms that KV caching improves decode throughput as generation length increases. The strongest sweep result is at **N=48**, where KV caching reaches **233.10 tokens/sec** compared with **130.52 tokens/sec** for no-cache generation, a **1.79x speedup**.

The **N=16** case is also strong, with a **1.76x speedup**. The **N=56** case remains meaningfully faster at **1.46x**.

The unusual row is **N=32**, where KV caching is only **1.01x faster**. That does not mean the KV cache stopped working. It is more likely a measurement artifact caused by the small CPU workload. The total wall times are only about **0.14 seconds** for both methods, so a small amount of scheduler noise, allocator behavior, Python overhead, or CPU frequency variation can move the result noticeably.

## TTFT Interpretation

The TTFT results are mixed:

| Case | No Cache TTFT | KV Cache TTFT | Change |
|---|---:|---:|---:|
| `small_smoke_test` | 5.26 ms | 7.15 ms | KV cache slower by 1.89 ms |
| `medium_generation` | 4.37 ms | 6.56 ms | KV cache slower by 2.19 ms |
| `longer_generation` | 3.84 ms | 3.64 ms | KV cache faster by 0.20 ms |
| `heavier_prompt` | 4.42 ms | 3.61 ms | KV cache faster by 0.81 ms |
| `near_context_limit` | 3.86 ms | 3.58 ms | KV cache faster by 0.28 ms |

This variation is expected. In the cached path, TTFT includes the prefill pass over the prompt plus sampling the first output token. In the no-cache path, the first decode step also processes the prompt, but the exact measured path differs because the benchmark toggles between train and eval behavior and uses different attention branches.

For serving systems, TTFT usually separates into:

1. Queueing and scheduling delay.
2. Prefill time.
3. Time to sample and emit the first token.

This benchmark does not model queueing, batching, network transfer, streaming, or scheduler delay. So TTFT here should be treated as a local implementation sanity check, not a complete user-facing latency measurement.

The throughput numbers are the clearer signal.

## Why The Speedup Is Not Perfectly Monotonic

In theory, KV-cache benefits should become more obvious as the generated sequence gets longer or as the prompt gets longer. The benchmark broadly shows that, but the numbers are not perfectly monotonic. There are several reasons.

### 1. The Model Is Tiny

At **0.056769M parameters**, this model is small enough that framework overhead can be a large fraction of total runtime. In a larger transformer, attention and MLP compute dominate more of the timing, so avoiding repeated work tends to show up more cleanly.

### 2. The Benchmark Runs On CPU

CPU timing is sensitive to operating system scheduling, cache locality, thread behavior, and frequency scaling. Since many runs complete in under half a second, small timing fluctuations can noticeably change tokens/sec.

### 3. The KV Cache Uses Tensor Concatenation

The implementation stores `key_cache` and `value_cache` inside each attention head and appends with `torch.cat` on every decode step. This is simple and educational, but it is not how production inference engines usually manage KV memory.

Each append can allocate and copy tensors. As the cache grows, that overhead can partially offset the benefit of avoiding recomputation. Production systems usually preallocate KV buffers or use paged/block-based KV cache layouts to avoid repeated copying.

### 4. Sequence Length Is Capped At 64

The benchmark uses `block_size=64`, so the no-cache baseline never processes more than 64 tokens per step. This limits how bad the no-cache path can get. With longer context windows, the cost of recomputing the full context would grow, making the KV-cache advantage more dramatic.

### 5. The No-Cache Path Uses A Different Mode

The benchmark uses `model.train()` to force the no-cache path and `model.eval()` to enable the cache path. Dropout is set to `0.0`, so this should not introduce dropout randomness, but the model still takes different branches through the attention implementation. This is fine for demonstrating the optimization, but it is worth remembering when interpreting very small timing differences.

## Significance For LLM Inference

This benchmark demonstrates one of the central facts of LLM serving: **decode is sequential, so avoiding repeated per-token work matters a lot**.

In an autoregressive model, every output token depends on the tokens before it. That dependency means generation cannot be fully parallelized across time in the same way prefill can. The model must produce token 1 before it knows the input for token 2. Because of that, the efficiency of each decode step is critical.

KV caching improves decode by making each step closer to "process just the new token" instead of "process the entire sequence again." This has several practical consequences:

- **Higher throughput:** More generated tokens per second on the same hardware.
- **Lower per-token latency:** Each streamed token can be produced with less repeated computation.
- **Better serving capacity:** A server can handle more concurrent decode work before saturating.
- **Foundation for batching:** Continuous batching relies on efficient cached decode steps.
- **Foundation for memory optimizations:** Techniques like paged attention, prefix caching, and KV eviction build on the same KV-cache concept.

The benchmark is small, but it validates the core mechanism that modern inference stacks depend on.

## Row-by-Row Interpretation

### `small_smoke_test`

This is a short sanity test with an 8-token prompt and 16 generated tokens. KV caching improves throughput from **126.67 tokens/sec** to **185.93 tokens/sec**, a **1.47x speedup**.

This confirms that the cached path is functional. Even on a tiny workload, avoiding repeated context processing saves enough time to show up.

### `medium_generation`

This is the strongest result: a 16-token prompt and 32 generated tokens. KV caching improves throughput from **102.20 tokens/sec** to **265.23 tokens/sec**, a **2.60x speedup**.

This is the clearest evidence that the benchmark is capturing the intended optimization. The generation length is long enough for repeated no-cache recomputation to accumulate, while the cached path can reuse previous keys and values.

### `longer_generation`

With a 16-token prompt and 48 generated tokens, KV caching gives a **1.93x speedup**. Wall time drops from **0.3344s** to **0.1735s**.

The speedup is lower than `medium_generation`, even though generation is longer. That is a reminder that this microbenchmark is noisy. Still, nearly doubling throughput is a strong result.

### `heavier_prompt`

This case increases prompt length to 32 while generating 32 tokens. KV caching gives a **2.29x speedup**.

This result is especially aligned with theory. A longer prompt means the no-cache path has more previous context to recompute at every step. The cached path pays for that prompt once, then reuses the cached keys and values.

### `near_context_limit`

This case generates 56 tokens with an 8-token prompt, filling the 64-token context. KV caching gives only a **1.11x speedup**.

This is lower than expected in a pure compute model, but understandable in this implementation. As generation approaches the context limit, the internal cache grows, and repeated `torch.cat` operations become more expensive. The no-cache path is also capped by `block_size=64`, so it never grows beyond that fixed context size. The result still favors KV caching, but the simple cache implementation leaves performance on the table.

## Main Conclusions

1. **KV caching works in this benchmark.** Every named benchmark case shows cached generation outperforming no-cache generation.

2. **The largest gains appear when repeated context work matters most.** `medium_generation` and `heavier_prompt` both exceed **2x speedup**.

3. **The current cache implementation is educational, not production-optimized.** Appending to cache tensors with `torch.cat` is simple but can introduce copying overhead.

4. **TTFT is not the main story here.** The TTFT numbers are mixed and measured over very short CPU timings. Throughput is the more reliable signal for this benchmark.

5. **The results likely understate the importance of KV caching for real LLMs.** Larger models, longer contexts, and optimized GPU kernels typically make the cost of recomputation much more expensive than it appears in this tiny CPU setup.

## Suggested Follow-up Benchmarks

To make the benchmark more rigorous, the next step would be to add:

- Multiple repetitions per configuration with mean, median, min, max, and standard deviation.
- A warmup phase before measurement.
- Separate prefill time and decode time metrics.
- Inter-token latency measurements for every generated token.
- A preallocated KV-cache implementation to avoid `torch.cat` overhead.
- Longer context windows to show how recomputation scales.
- CUDA runs with explicit synchronization around timed regions.
- Batch-size sweeps to connect this benchmark to continuous batching.

The current results are already useful because they demonstrate the direction and significance of KV caching. These follow-ups would make the results more stable and more representative of real inference serving.
