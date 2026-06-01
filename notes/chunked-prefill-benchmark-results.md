# Chunked Prefill Benchmark Results Write-up

This note explains the results in [`prefill_benchmark_results.txt`](../prefill_benchmark_results.txt) for the normal prefill vs chunked prefill benchmark implemented in [`benchmarks/normal_chunked_prefill.py`](../benchmarks/normal_chunked_prefill.py) and configured by [`benchmarks/chunked_prefill_benchmark_runs.py`](../benchmarks/chunked_prefill_benchmark_runs.py).

## Executive Summary

This benchmark compares two prefill scheduling policies:

- **Normal prefill:** when a request arrives, process its entire prompt in one forward pass.
- **Chunked prefill:** split prompt processing into smaller chunks and interleave those chunks with decode work under a token budget.

In these results, **normal prefill is faster in every configuration**. Chunked prefill achieves only **0.74x to 0.91x** of normal-prefill generated-token throughput, averaging about **0.86x**. It also has much worse average TTFT, ranging from **4.30x to 13.09x** slower than normal prefill.

That may look surprising because chunked prefill is usually introduced as an optimization. The key is that chunked prefill is not primarily about making a tiny CPU microbenchmark faster. It is a serving policy for managing interference between long prompt prefill and active decode streams, especially on larger GPU-backed systems. In this small NanoGPT benchmark, chunking creates extra forward passes and scheduler overhead without getting the production benefits of GPU utilization, batched decode, or many concurrent requests.

The important result is not "chunked prefill is bad." The better reading is:

> In this implementation and workload, chunked prefill trades throughput and first-token latency for slightly smoother decode gaps in some pressure cases.

## What Was Benchmarked

The benchmark uses a mixed workload:

- Short, decode-heavy requests arrive first.
- Long, prefill-heavy requests arrive later.
- Both policies complete the same requests and the same number of tokens in every row.

This makes the comparison cleaner than the current continuous batching benchmark: request counts match, prompt-token counts match, and generated-token counts match.

| Method | Behavior | Intended Effect |
|---|---|---|
| `normal_prefill` | Fully prefills each newly arrived prompt before continuing active decode. | Simple and efficient, but long prompts can stall existing decode streams. |
| `chunked_prefill` | Decodes active requests first, then spends the remaining per-step token budget on prompt chunks. | Reduces large prefill stalls by spreading long prompt work across multiple steps. |

The benchmark intentionally decodes active requests **one by one**, not as a batched decode. That keeps the experiment focused on prefill policy rather than continuous batching.

## Metrics

The output reports:

| Metric | Meaning |
|---|---|
| `reqs` | Number of completed requests. |
| `prompt_tok` | Total prompt tokens processed. |
| `gen_tok` | Total generated tokens emitted. |
| `wall_s` | Total measured wall-clock time. Lower is better. |
| `gen_tok/s` | Generated tokens divided by wall time. Higher is better. |
| `prompt_tok/s` | Prompt tokens divided by wall time. Higher is better. |
| `avg_ttft_ms` | Average time to first generated token. Lower is better. |
| `p95_ttft_ms` | 95th percentile time to first token. Lower is better. |
| `avg_lat_ms` | Average request latency from arrival to completion. Lower is better. |
| `p95_lat_ms` | 95th percentile request latency. Lower is better. |
| `avg_gap_ms` | Average gap between generated tokens for a request. Lower is smoother. |
| `max_gap_ms` | Worst observed gap between generated tokens. Lower is smoother. |
| `forward_s` | Total measured time inside model forward work. |

The most important metrics for this benchmark are:

- **Generated-token throughput**, because it shows total decode production.
- **TTFT**, because chunking can delay a long request's first token.
- **Max inter-token gap**, because chunking is meant to reduce decode stalls for already-active requests.

## Training Context

Before the benchmark, the tiny model is trained briefly:

| Step | Train Loss | Validation Loss |
|---:|---:|---:|
| 0 | 4.1800 | 4.1791 |
| 20 | 3.6074 | 3.6479 |
| 40 | 3.3261 | 3.3321 |
| 60 | 3.1051 | 3.1305 |
| 80 | 2.9561 | 2.9651 |
| 100 | 2.8319 | 2.8681 |
| 119 | 2.7760 | 2.7998 |

The model is learning, but generation quality is not the purpose of this file. The benchmark is measuring inference scheduling behavior.

The setup is intentionally small:

- **0.056769M parameters**
- **CPU**
- **`block_size=64`**
- Short benchmark windows, usually under one second

Because the model is so small, scheduler overhead and extra forward-call overhead are large relative to model computation.

## Benchmark Results

| Case | Requests | Prompt Tokens | Generated Tokens | Chunk Size | Token Budget | Normal Gen Tok/s | Chunked Gen Tok/s | Throughput Ratio | Avg TTFT Ratio | Max Gap Ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `small_smoke_test` | 6 | 96 | 112 | 8 | 16 | 423.55 | 384.01 | 0.91x | 5.62x | 1.25x |
| `more_long_prompts` | 8 | 160 | 160 | 8 | 16 | 399.67 | 296.54 | 0.74x | 9.77x | 1.23x |
| `smaller_chunks_smoother_decode` | 8 | 160 | 160 | 4 | 16 | 395.04 | 310.58 | 0.79x | 13.09x | 0.97x |
| `larger_chunks_less_overhead` | 8 | 160 | 160 | 16 | 32 | 394.13 | 359.99 | 0.91x | 4.30x | 1.16x |
| `decode_heavy_pressure` | 12 | 192 | 416 | 8 | 16 | 456.66 | 412.63 | 0.90x | 10.58x | 0.90x |
| `late_long_prompt_interruptions` | 12 | 224 | 416 | 8 | 16 | 455.18 | 406.72 | 0.89x | 10.87x | 0.94x |

### Overall Trend

Chunked prefill is slower in every row:

- Wall time increases by about **9.5% to 34.8%**.
- Generated-token throughput decreases by about **8.7% to 25.8%**.
- Average latency increases in every configuration.
- Average TTFT gets much worse in every configuration.

The worst throughput case is `more_long_prompts`, where chunked prefill reaches **296.54 generated tokens/sec** compared with **399.67 generated tokens/sec** for normal prefill. That is only **0.74x** of normal-prefill throughput.

The best throughput cases are `small_smoke_test` and `larger_chunks_less_overhead`, both at **0.91x**. Those rows show that larger chunks and larger budgets reduce chunking overhead, but they still do not beat normal prefill in this CPU setup.

## Why Chunked Prefill Is Slower Here

### 1. Chunking Creates More Forward Passes

Normal prefill processes a prompt in one model call:

```text
long prompt: [32 tokens] -> one prefill forward
```

Chunked prefill splits that same prompt:

```text
long prompt: [8 tokens] -> forward
long prompt: [8 tokens] -> forward
long prompt: [8 tokens] -> forward
long prompt: [8 tokens] -> forward
```

That is easier to schedule around active decode, but it adds repeated Python, PyTorch, position setup, cache update, and sampling overhead. On a tiny CPU model, that overhead is very visible.

### 2. The Benchmark Does Not Batch Decode

The benchmark intentionally decodes active requests one at a time. That is useful for isolating prefill policy, but it removes one of the main reasons chunked prefill is useful in production.

In a real serving engine, chunked prefill is often paired with continuous batching. Decode tokens from many requests can be batched together, and leftover token budget can be used for prompt chunks. This benchmark does not get that throughput benefit.

### 3. Decode-First Scheduling Delays Long Requests' First Token

The chunked policy decodes active requests first, then spends remaining budget on prefill chunks. That protects existing streams, but new long-prompt requests may need several scheduler steps before their full prompt is processed and their first token can be emitted.

That is why TTFT gets much worse:

| Case | Normal Avg TTFT | Chunked Avg TTFT | Increase |
|---|---:|---:|---:|
| `small_smoke_test` | 6.13 ms | 34.40 ms | +28.27 ms |
| `more_long_prompts` | 8.48 ms | 82.81 ms | +74.33 ms |
| `smaller_chunks_smoother_decode` | 7.71 ms | 100.92 ms | +93.21 ms |
| `larger_chunks_less_overhead` | 9.84 ms | 42.31 ms | +32.47 ms |
| `decode_heavy_pressure` | 11.27 ms | 119.16 ms | +107.89 ms |
| `late_long_prompt_interruptions` | 11.92 ms | 129.57 ms | +117.65 ms |

This is a real policy tradeoff: protecting existing decode streams can make newly arrived long prompts wait longer for their first token.

### 4. The Model And Context Are Tiny

With only **0.056769M parameters** and `block_size=64`, full-prompt prefill is not expensive enough to create massive stalls. Splitting the prompt into chunks can cost more than it saves.

With larger models, longer prompts, GPU execution, and many concurrent decode streams, the tradeoff can look different.

## Streaming Smoothness: The More Interesting Signal

Chunked prefill is meant to reduce decode interruptions. The best metric for that in this file is `max_gap_ms`, the worst inter-token gap.

| Case | Normal Max Gap | Chunked Max Gap | Ratio | Interpretation |
|---|---:|---:|---:|---|
| `small_smoke_test` | 16.10 ms | 20.09 ms | 1.25x | Chunked worse |
| `more_long_prompts` | 24.46 ms | 29.98 ms | 1.23x | Chunked worse |
| `smaller_chunks_smoother_decode` | 24.02 ms | 23.35 ms | 0.97x | Slightly smoother |
| `larger_chunks_less_overhead` | 25.93 ms | 30.05 ms | 1.16x | Chunked worse |
| `decode_heavy_pressure` | 35.75 ms | 32.25 ms | 0.90x | Smoother |
| `late_long_prompt_interruptions` | 33.94 ms | 31.78 ms | 0.94x | Smoother |

This is where chunked prefill starts to show its intended purpose. In the decode-heavy pressure cases, chunking reduces the worst inter-token gap:

- `decode_heavy_pressure`: max gap improves from **35.75 ms** to **32.25 ms**.
- `late_long_prompt_interruptions`: max gap improves from **33.94 ms** to **31.78 ms**.
- `smaller_chunks_smoother_decode`: max gap improves slightly from **24.02 ms** to **23.35 ms**.

These improvements are modest, but they point in the expected direction: smaller chunks can prevent long prefill work from blocking decode for too long.

The average gap does not improve, though. Chunked prefill has slightly higher average gaps in every row. So in this benchmark, chunking mainly helps the worst stall in some pressure cases, not the overall streaming cadence.

## Chunk Size And Token Budget

The clearest chunk-size comparison is between these two rows:

| Case | Chunk Size | Token Budget | Chunked Gen Tok/s | Chunked Avg TTFT | Chunked Max Gap |
|---|---:|---:|---:|---:|---:|
| `smaller_chunks_smoother_decode` | 4 | 16 | 310.58 | 100.92 ms | 23.35 ms |
| `larger_chunks_less_overhead` | 16 | 32 | 359.99 | 42.31 ms | 30.05 ms |

Smaller chunks produce the smoothest worst-case decode gap, but they are expensive:

- More chunks means more forward calls.
- More forward calls means more overhead.
- More chunks also mean long prompts wait through more steps before first token.

Larger chunks reduce overhead and improve TTFT, but they can create larger decode stalls because each chunk occupies a bigger piece of a scheduler step.

This is the core chunked-prefill tuning problem:

> Smaller chunks protect streaming smoothness. Larger chunks protect throughput and TTFT.

The right value depends on the serving goal. A chat system may prefer smoother streaming. A batch/offline system may prefer throughput.

## Row-by-Row Interpretation

### `small_smoke_test`

This is a light workload: 4 short requests and 2 long requests. Chunked prefill reaches **384.01 generated tokens/sec** compared with **423.55** for normal prefill, a **0.91x ratio**.

This is the least alarming slowdown because the workload is small and the chunk size is moderate. But TTFT is already much worse: **34.40 ms** vs **6.13 ms**.

### `more_long_prompts`

This increases long requests from 2 to 4. Chunked throughput falls to **0.74x** of normal prefill, the worst ratio in the file.

The extra long prompts create more chunked-prefill work, and each long prompt takes several chunks before it can emit its first token. That pushes average TTFT to **82.81 ms**, almost **10x** normal prefill.

### `smaller_chunks_smoother_decode`

This keeps the same workload as `more_long_prompts` but reduces chunk size from 8 to 4. The smaller chunks slightly improve max inter-token gap: **23.35 ms** vs **24.02 ms** for normal prefill.

But the cost is high. Average TTFT rises to **100.92 ms**, the throughput ratio remains only **0.79x**, and total wall time increases by about **27.2%**.

This row captures the smoothness-throughput tradeoff most clearly.

### `larger_chunks_less_overhead`

This increases chunk size to 16 and token budget to 32. Chunked prefill becomes much more competitive, reaching **0.91x** of normal throughput.

Average TTFT is still worse than normal prefill, but much better than the small-chunk case: **42.31 ms** instead of **100.92 ms**. The tradeoff is that max inter-token gap becomes worse than normal prefill: **30.05 ms** vs **25.93 ms**.

### `decode_heavy_pressure`

This workload has 8 short decode-heavy requests and 4 long prompt-heavy requests. Chunked throughput is **0.90x** of normal throughput, but max inter-token gap improves from **35.75 ms** to **32.25 ms**.

This is one of the rows where chunked prefill behaves according to its intended purpose: it reduces the worst decode stall while active decode-heavy streams are running. It still pays for that with worse TTFT and lower total throughput.

### `late_long_prompt_interruptions`

This is similar to `decode_heavy_pressure`, but long prompts arrive later and are longer at 40 tokens. Chunked throughput is **0.89x** of normal throughput, and max gap improves from **33.94 ms** to **31.78 ms**.

The TTFT penalty is the largest in the file: chunked prefill averages **129.57 ms** vs **11.92 ms** for normal prefill. Late long prompts wait behind ongoing decode-first work and require multiple chunks before they emit a first token.

## Main Conclusions

1. **Normal prefill wins on raw throughput in this benchmark.** Chunked prefill is consistently slower, with an average generated-token throughput ratio of about **0.86x**.

2. **Chunked prefill heavily penalizes TTFT here.** Average TTFT is roughly **4.30x to 13.09x** worse because long prompts must complete multiple chunks before emitting their first token.

3. **Chunking can improve worst-case decode stalls in pressure cases.** The max inter-token gap improves in `smaller_chunks_smoother_decode`, `decode_heavy_pressure`, and `late_long_prompt_interruptions`.

4. **Smaller chunks are smoother but more expensive.** Chunk size 4 gives the best max-gap behavior, while chunk size 16 gives better throughput and TTFT.

5. **This benchmark does not include the full production setting where chunked prefill shines.** There is no batched decode, no GPU utilization pressure, no large model, and no very long context.

6. **The current result is still useful.** It demonstrates the real scheduling tradeoff: prefill work can either be processed efficiently in large blocks or spread out to protect streaming decode.

## Suggested Follow-up Benchmarks

To make the chunked-prefill benchmark more representative, the next steps are:

- Separate TTFT for short requests and long requests.
- Report decode gaps for short active requests separately from long prompt requests.
- Add per-step traces for prefill tokens, decode tokens, queue length, and token budget usage.
- Compare normal vs chunked prefill with continuous batched decode enabled.
- Add longer prompts that stress prefill more strongly.
- Run on CUDA with explicit synchronization around timed regions.
- Run multiple repetitions and report mean, median, p95, and standard deviation.
- Sweep `chunk_size` and `token_budget` independently.
- Try a policy that reserves some budget for prefill even when many decode requests are active.
- Try a policy that prioritizes first-token prefill for waiting long requests before returning to decode-first scheduling.

The current results show that chunked prefill is not free. It is a policy knob: it can smooth decode stalls, but the chunk size, token budget, and scheduling priority determine whether that tradeoff is worthwhile.
