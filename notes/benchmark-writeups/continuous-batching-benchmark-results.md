# Continuous Batching Benchmark Results Write-up

This note explains the results in [`continuous_batching_results.txt`](../continuous_batching_results.txt) for the continuous batching benchmark implemented in [`benchmarks/single_req_cont_batching.py`](../benchmarks/single_req_cont_batching.py) and configured by [`benchmarks/continuous_batching_benchmark_runs.py`](../benchmarks/continuous_batching_benchmark_runs.py).

## Executive Summary

The benchmark shows the core tradeoff behind continuous batching:

- **Throughput improves substantially** when multiple active requests share decode forward passes.
- **Average latency and TTFT get worse** in this simple harness because requests wait behind batch formation, batched decode steps, and individually handled prefill.
- **Larger batch sizes produce larger throughput gains**, but also increase user-facing waiting time.

Across the named benchmark cases, continuous batching reports throughput speedups from **1.77x to 3.61x**, averaging about **2.56x**. In the batch-size sweep, speedup grows from **0.68x** at `max_batch_size=1` to **5.99x** at `max_batch_size=16`.

The most important caveat is that the current continuous batching scaffold does **not complete the same number of requests as the sequential baseline**. For example, in `stress_batch_capacity`, the sequential path completes **32 requests**, while the continuous batching path reports only **9 completed requests**. This means the throughput ratios are best read as "tokens/sec while the batched engine is active" rather than as a fully fair end-to-end comparison over the same workload.

That caveat does not invalidate the trend. It does mean the benchmark should be treated as an educational microbenchmark for batched decode, not yet as a production-quality serving benchmark.

## What Was Benchmarked

The benchmark compares two serving strategies:

| Method | Behavior | Why It Matters |
|---|---|---|
| `single_request_sequential` | Serves each request to completion before starting the next request. Each request uses KV-cache generation. | This is the simple baseline. It minimizes per-request interference but cannot share work across concurrent requests. |
| `continuous_batching` | Maintains active requests, admits arrivals by scheduler step, prefills admitted requests, and batches active decode tokens into one forward pass. | This models the core idea used by LLM serving systems: every decode step can process one token for many requests at once. |

The benchmark uses uniform workloads:

- Same prompt length within each run.
- Same generation length within each run.
- `arrival_gap=0` for all runs in the result file, so all requests are immediately available.
- KV-cache decode is used by both strategies.

The uniform setup is intentional. Batched cached decode is easiest when active requests have aligned KV-cache lengths. Mixed prompt lengths, mixed output lengths, and real arrival timing would require padding, masks, or a more complete scheduler.

## Important Harness Caveat: Request Counts Differ

The `reqs` column reveals a major limitation:

| Case | Requested Workload | Sequential Completed | Continuous Completed |
|---|---:|---:|---:|
| `small_smoke_test` | 8 | 8 | 5 |
| `more_requests_small_batch` | 16 | 16 | 5 |
| `more_requests_larger_batch` | 16 | 16 | 9 |
| `longer_generations` | 16 | 16 | 9 |
| `heavier_prompt` | 16 | 16 | 9 |
| `stress_batch_capacity` | 32 | 32 | 9 |

This happens because the continuous batching implementation admits all requests that have arrived for the current scheduler step into a temporary `newly_arrived` list. If the active batch is full, it requeues only the current overflow request and breaks out of the loop. Later requests already removed from `pending` are not requeued.

The practical result is that the continuous batching path often processes only `max_batch_size + 1` requests:

- `max_batch_size=4` tends to complete 5 requests.
- `max_batch_size=8` tends to complete 9 requests.
- `max_batch_size=16` tends to complete 17 requests.

Because of this, the `wall_s` values are not directly comparable as "time to finish the same workload." The safer comparison is the reported **tokens/sec for completed tokens**, while remembering that the batched path is operating on a smaller subset of the intended workload.

Before using this benchmark as a rigorous end-to-end serving comparison, the admission loop should preserve all overflow arrivals.

## Metrics

The output reports:

| Metric | Meaning |
|---|---|
| `reqs` | Number of requests completed by that method. |
| `tokens` | Total generated tokens emitted by completed work. |
| `wall_s` | Total measured wall-clock time for the run. |
| `tok/s` | Generated tokens divided by wall time. Higher is better. |
| `avg_ttft_ms` | Average time to first token. Lower is better. |
| `p95_ttft_ms` | 95th percentile time to first token. Lower is better. |
| `avg_lat_ms` | Average request latency from arrival to completion. Lower is better. |
| `p95_lat_ms` | 95th percentile request latency. Lower is better. |
| `avg_batch` | Average recorded non-empty active batch size. |
| `max_batch` | Largest recorded active batch size. |
| `forward_s` | Total time spent inside measured model forward work. |

The main engine-facing metric is **tokens/sec**. The main user-facing metrics are **TTFT** and **request latency**.

One extra detail: in this implementation, `avg_batch` is computed from the `StepMetrics.batch_size` field, which is recorded after each scheduler step has updated active requests. It is a useful signal, but it is not a perfect trace of the exact batch size passed into every forward call.

## Training Context

Before the benchmark runs, the file shows a short training log:

| Step | Train Loss | Validation Loss |
|---:|---:|---:|
| 0 | 4.1800 | 4.1791 |
| 20 | 3.6074 | 3.6479 |
| 40 | 3.3261 | 3.3321 |
| 60 | 3.1051 | 3.1305 |
| 80 | 2.9561 | 2.9651 |
| 100 | 2.8319 | 2.8681 |
| 119 | 2.7760 | 2.7998 |

The model is learning: both training and validation loss decrease steadily. As with the KV-cache benchmark, generation quality is not the point here. The model is tiny and trained briefly so the benchmark can focus on inference mechanics.

The run uses:

- **0.056769M parameters**
- **CPU**
- **`block_size=64`**
- A very small NanoGPT-style model

These conditions make the benchmark fast and educational, but also noisy. Python overhead, scheduler bookkeeping, cache stacking/unstacking, and CPU timing variation are all large relative to the model's actual compute.

## Named Benchmark Results

| Case | Requests Configured | Max Batch | Seq Tok/s | Batched Tok/s | Speedup | Avg Lat Ratio | Avg TTFT Ratio | Batched Completion |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `small_smoke_test` | 8 | 4 | 271.52 | 524.11 | 1.93x | 1.37x | 1.94x | 5/8 |
| `more_requests_small_batch` | 16 | 4 | 266.90 | 476.62 | 1.79x | 1.25x | 1.74x | 5/16 |
| `more_requests_larger_batch` | 16 | 8 | 241.15 | 850.88 | 3.53x | 1.58x | 2.85x | 9/16 |
| `longer_generations` | 16 | 8 | 265.04 | 956.53 | 3.61x | 1.51x | 3.98x | 9/16 |
| `heavier_prompt` | 16 | 8 | 277.84 | 492.91 | 1.77x | 3.41x | 7.83x | 9/16 |
| `stress_batch_capacity` | 32 | 8 | 246.66 | 671.97 | 2.72x | 1.97x | 3.21x | 9/32 |

### Overall Trend

Continuous batching improves generated-token throughput in every named case. The most favorable named case is `longer_generations`, where throughput increases from **265.04 tokens/sec** to **956.53 tokens/sec**, a **3.61x speedup**.

The second strongest named case is `more_requests_larger_batch`, where increasing the batch cap to 8 raises throughput to **850.88 tokens/sec**, a **3.53x speedup** over sequential serving.

The pattern is exactly what we expect: when the engine can decode several requests in the same forward pass, it amortizes model overhead across more emitted tokens.

## Why Continuous Batching Helps

Autoregressive decode produces one token per request per step. Without batching, a serving loop might do:

```text
request A: decode token 1
request A: decode token 2
request A: decode token 3
...
request B: decode token 1
request B: decode token 2
...
```

Continuous batching instead tries to do:

```text
step 1: decode one token for A, B, C, D together
step 2: decode one token for A, B, C, D together
step 3: decode one token for A, B, C, D together
...
```

Each request still gets only one new token per decode step, but the model forward pass is shared across requests. This is especially important for GPUs, where larger batches can improve utilization. Even on CPU in this tiny benchmark, the effect is visible.

## Latency Tradeoff

The throughput gains come with worse latency metrics:

| Case | Sequential Avg Latency | Batched Avg Latency | Sequential Avg TTFT | Batched Avg TTFT |
|---|---:|---:|---:|---:|
| `small_smoke_test` | 58.93 ms | 80.78 ms | 5.01 ms | 9.70 ms |
| `more_requests_small_batch` | 89.92 ms | 112.57 ms | 5.27 ms | 9.15 ms |
| `more_requests_larger_batch` | 99.52 ms | 157.11 ms | 5.57 ms | 15.84 ms |
| `longer_generations` | 181.10 ms | 272.66 ms | 4.32 ms | 17.21 ms |
| `heavier_prompt` | 115.17 ms | 392.75 ms | 5.17 ms | 40.45 ms |
| `stress_batch_capacity` | 129.73 ms | 255.02 ms | 5.69 ms | 18.29 ms |

This is the central serving tradeoff:

- Sequential serving gives each request exclusive attention, so a single request can finish quickly.
- Continuous batching improves system throughput by making requests share model steps.
- Sharing model steps can increase per-request wait time, especially for TTFT.

The `heavier_prompt` case makes this especially clear. Throughput still improves by **1.77x**, but average latency worsens by **3.41x** and average TTFT worsens by **7.83x**. The reason is that prefill is handled individually in this scaffold. Longer prompts increase the amount of prefill work that must happen before requests can join the decode batch, so first-token latency suffers.

## Batch Size Sweep

The batch-size sweep uses 32 configured requests, prompt length 8, and 32 generated tokens per request. It varies only `max_batch_size`.

| Max Batch Size | Seq Tok/s | Batched Tok/s | Speedup | Avg Batch | Avg Lat Ratio | Avg TTFT Ratio | Batched Completion |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 221.53 | 149.76 | 0.68x | 1.00 | 1.48x | 1.86x | 2/32 |
| 2 | 255.75 | 246.86 | 0.97x | 1.50 | 1.53x | 1.40x | 3/32 |
| 4 | 273.32 | 607.05 | 2.22x | 2.50 | 1.28x | 1.74x | 5/32 |
| 8 | 235.78 | 651.10 | 2.76x | 4.50 | 1.77x | 3.76x | 9/32 |
| 16 | 245.28 | 1469.02 | 5.99x | 8.50 | 1.96x | 5.25x | 17/32 |

### Sweep Trend

The sweep shows a clear throughput scaling curve:

- At `max_batch_size=1`, continuous batching is slower than the sequential baseline: **0.68x**.
- At `max_batch_size=2`, it is roughly break-even: **0.97x**.
- At `max_batch_size=4`, it becomes meaningfully faster: **2.22x**.
- At `max_batch_size=8`, it reaches **2.76x**.
- At `max_batch_size=16`, it reaches **5.99x**.

This is the strongest evidence that the batched decode path is doing real work. With batch size 1, the continuous batching loop adds scheduler and cache-management overhead without getting much batching benefit. As the allowed batch size grows, more requests share each model forward pass, so throughput rises sharply.

The latency trend moves in the opposite direction. At `max_batch_size=16`, throughput is excellent, but average TTFT is **5.25x** worse than sequential serving. This is the classic throughput-latency tension in inference serving.

## Significance For LLM Inference

Continuous batching is one of the most important techniques in LLM serving because decode is inherently step-by-step. Each request can only produce the next token after the previous token is known. That means a server with many concurrent users needs a way to keep hardware busy while respecting each request's sequential dependency.

Continuous batching solves this by batching across requests rather than across time:

- Each request contributes one token position to the current decode step.
- Finished requests leave the batch.
- Newly ready requests can join future steps.
- The engine maintains high utilization even when individual requests start and finish at different times.

In production systems, this is the basis for serving many streaming chat completions concurrently. It also connects directly to:

- **KV caching**, because each request needs its own cached history.
- **Paged attention**, because many concurrent KV caches need efficient memory management.
- **Scheduling**, because the engine must decide which requests enter each step.
- **Chunked prefill**, because long prompts can otherwise delay decode work.
- **Prefix caching**, because shared prompt prefixes can reduce prefill cost.

This benchmark demonstrates the basic throughput mechanism, but it does not yet model all the surrounding serving-engine complexity.

## Row-by-Row Interpretation

### `small_smoke_test`

This run uses 8 configured requests, an 8-token prompt, 16 generated tokens, and `max_batch_size=4`. Continuous batching reports **524.11 tokens/sec** compared with **271.52 tokens/sec** for sequential serving, a **1.93x speedup**.

The batched path completes 5 of the 8 configured requests. Within that subset, the benchmark shows the intended batching effect: average batch size rises to **2.50**, and throughput nearly doubles.

### `more_requests_small_batch`

This run increases the configured workload to 16 requests while keeping `max_batch_size=4`. Throughput improves by **1.79x**, from **266.90 tokens/sec** to **476.62 tokens/sec**.

Because the batch cap is still 4, the completed batched subset remains 5 requests. The result shows that simply adding more queued requests does not help unless the scheduler can keep and admit them correctly.

### `more_requests_larger_batch`

This run raises `max_batch_size` from 4 to 8. Throughput jumps to **850.88 tokens/sec**, a **3.53x speedup**.

This is the cleanest comparison against `more_requests_small_batch`: larger batches create more opportunity to amortize each forward pass. Average latency and TTFT also rise, showing the cost of waiting and sharing.

### `longer_generations`

This run keeps `max_batch_size=8` but doubles generation length to 48 tokens. It produces the strongest named speedup: **3.61x**.

Longer generations make continuous batching more useful because there are more decode steps over which the active batch can remain full. Once prefill is done, the engine gets many chances to run efficient batched decode.

### `heavier_prompt`

This run increases prompt length to 16 and generates 32 tokens. Throughput still improves by **1.77x**, but latency gets much worse.

Average TTFT rises from **5.17 ms** to **40.45 ms**. This is the most important latency warning in the file. The benchmark handles prefill one request at a time, so heavier prompts delay first-token emission and reduce the relative benefit of batched decode.

### `stress_batch_capacity`

This run configures 32 requests with `max_batch_size=8`. Continuous batching reports a **2.72x throughput speedup**, but completes only 9 of 32 configured requests.

The result still shows that batched decode is faster than sequential decode, but this row should not be interpreted as "the batched engine served all 32 requests quickly." It served a subset quickly due to the admission-loop limitation.

## Main Conclusions

1. **Continuous batching increases throughput when the batch size is large enough.** The sweep shows a clear scaling pattern from **0.68x** at batch size 1 to **5.99x** at batch size 16.

2. **Batching has overhead.** With `max_batch_size=1` and `max_batch_size=2`, the continuous batching path is slower or roughly equal to sequential serving because it adds scheduling, KV stacking, and bookkeeping without enough parallel work to amortize that overhead.

3. **Latency and TTFT worsen in this scaffold.** This is expected: the engine is optimizing system throughput, not single-request latency. The effect is especially strong when prompt prefill is heavier.

4. **Prefill handling matters.** The `heavier_prompt` case shows that prefill can dominate first-token latency. This motivates chunked prefill and prefill/decode scheduling.

5. **The benchmark currently under-completes the workload in the continuous path.** The request-count mismatch must be fixed before treating the ratios as a rigorous end-to-end serving comparison.

6. **The result still demonstrates the core principle.** Batched decode can emit many more tokens per second than serving one request at a time.

## Suggested Follow-up Improvements

To make this benchmark more rigorous and more representative of real inference serving, the next steps are:

- Fix the admission loop so all overflow arrivals remain pending.
- Report both configured requests and completed requests.
- Add a correctness check that every configured request completed.
- Record actual decode batch size before each batched forward call.
- Separate prefill time from decode time in the printed table.
- Add per-step emitted token counts and queue length traces.
- Add staggered-arrival workloads after the all-at-once case is correct.
- Add mixed prompt lengths and mixed output lengths with padding or masks.
- Run multiple repetitions and report mean, median, p95, and standard deviation.
- Add CUDA runs with explicit synchronization around measured regions.

The current benchmark is a useful first step because it makes the throughput-latency tradeoff visible. After the request-admission bug is fixed, it can become a much stronger benchmark for comparing real scheduling policies.
