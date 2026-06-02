# Scheduling Benchmark Results Write-up

This note explains the results in [`scheduling_results.txt`](../scheduling_results.txt) for the FCFS vs priority scheduling benchmark implemented in [`benchmarks/scheduling_policy.py`](../benchmarks/scheduling_policy.py) and configured by [`benchmarks/scheduling_benchmark_runs.py`](../benchmarks/scheduling_benchmark_runs.py).

## Executive Summary

The benchmark compares two scheduling policies:

- **FCFS**, which serves requests in first-come, first-served order.
- **Priority scheduling**, which admits lower priority-number requests first and can preempt lower-priority active requests when a higher-priority request is blocked.

The headline result is:

> Priority scheduling greatly improves high-priority request latency, while total throughput stays roughly similar or slightly worse when preemption causes recomputation.

This is exactly the tradeoff a scheduler is supposed to expose. Scheduling is not mainly a model-speed optimization. It is a policy layer that decides **who gets served first** when token budget, batch size, and KV-cache capacity are limited.

Across the benchmark:

- Priority improves high-priority latency in the meaningful priority workloads.
- Priority has little to no effect in the equal-priority control case.
- Priority can slightly reduce throughput when it preempts active work and has to recompute it later.
- All benchmark cases complete the same number of requests and generated tokens under both policies, so the comparisons are clean.

## Benchmark Setup

The run uses:

- **Model size:** `0.056769M` parameters
- **Device:** CUDA
- **Context length:** `block_size=64`
- **Benchmark target:** scheduling behavior, not model quality

The training log before the benchmark shows the model learning:

| Step | Train Loss | Validation Loss |
|---:|---:|---:|
| 0 | 4.1800 | 4.1791 |
| 20 | 3.6074 | 3.6479 |
| 40 | 3.3261 | 3.3321 |
| 60 | 3.1051 | 3.1305 |
| 80 | 2.9561 | 2.9651 |
| 100 | 2.8321 | 2.8684 |
| 119 | 2.7762 | 2.7998 |

The generated sample is still noisy, which is expected for a tiny model trained briefly. The important part is that the same model and same workloads are used for both scheduling policies.

## What Was Measured

The benchmark runs four scheduling scenarios:

| Case | Purpose |
|---|---|
| `priority_inversion_serial` | A high-priority short request arrives behind lower-priority work, with `max_batch_size=1`. This makes priority inversion easy to see. |
| `priority_mix_small_batch` | Several requests with mixed priorities run under a small batched decode setup. |
| `memory_pressure_preemption` | A tight KV-token budget forces priority scheduling to preempt lower-priority requests. |
| `equal_priority_control` | All requests have equal priority. FCFS and priority should behave the same. |

Both policies use cached generation, chunked prefill, and batched decode where the workload allows it. The key difference is admission order and whether priority is allowed to preempt lower-priority active requests.

## Metrics

| Metric | Meaning |
|---|---|
| `reqs` | Total requests in the workload. |
| `done` | Requests completed. This matches `reqs` in every row. |
| `gen_tok` | Generated tokens completed by the policy. |
| `wall_s` | Total wall-clock time for the run. |
| `tok/s` | Generated tokens per second. |
| `avg_ttft_ms` | Average time to first token. |
| `p95_ttft_ms` | 95th percentile time to first token. |
| `avg_lat_ms` | Average request latency from arrival to completion. |
| `p95_lat_ms` | 95th percentile request latency. |
| `hi_lat_ms` | Average latency for high-priority requests. |
| `low_lat_ms` | Average latency for lower-priority requests. |
| `preempt` | Number of recompute preemptions. |
| `avg_batch` | Average decode batch size. |
| `max_batch` | Largest decode batch size. |
| `forward_s` | Time spent in measured model forward work. |

The most important metrics are `hi_lat_ms`, `avg_lat_ms`, `tok/s`, and `preempt`.

## Results Summary

| Case | FCFS Tok/s | Priority Tok/s | Throughput Ratio | FCFS High-Priority Latency | Priority High-Priority Latency | High-Priority Latency Ratio | Preemptions |
|---|---:|---:|---:|---:|---:|---:|---:|
| `priority_inversion_serial` | 83.04 | 79.77 | 0.96x | 308.77 ms | 112.88 ms | 0.37x | 1 |
| `priority_mix_small_batch` | 166.15 | 169.10 | 1.02x | 235.69 ms | 108.55 ms | 0.46x | 0 |
| `memory_pressure_preemption` | 127.67 | 119.76 | 0.94x | 183.62 ms | 120.07 ms | 0.65x | 2 |
| `equal_priority_control` | 165.27 | 165.35 | 1.00x | 210.86 ms | 210.66 ms | 1.00x | 0 |

The strongest and most useful trend is the high-priority latency improvement:

- In `priority_inversion_serial`, high-priority latency drops from **308.77 ms** to **112.88 ms**.
- In `priority_mix_small_batch`, high-priority latency drops from **235.69 ms** to **108.55 ms**.
- In `memory_pressure_preemption`, high-priority latency drops from **183.62 ms** to **120.07 ms**.
- In the equal-priority control, latency is unchanged, which validates that priority ordering is not changing behavior when there is no priority difference.

## Why Priority Scheduling Helps

FCFS is fair in arrival order, but not necessarily fair in user impact. If a long, low-priority request arrives first, a short high-priority request may wait behind it even though serving the high-priority request first would improve responsiveness.

Priority scheduling changes the admission rule:

```text
FCFS:
    serve earlier request first

Priority:
    serve higher-priority request first
    use arrival order only as a tie-breaker
```

When resources are constrained, priority scheduling can also preempt lower-priority active requests. In this benchmark, preemption is recompute-based: the evicted request loses its KV cache and must prefill again later. That improves high-priority latency, but it can hurt throughput and low-priority latency.

## Case-By-Case Interpretation

### `priority_inversion_serial`

Configuration:

```text
max_batch_size=1
token_budget=12
prefill_chunk_size=10
max_kv_tokens=40
```

This is the cleanest priority-inversion test. With `max_batch_size=1`, only one request can be active at a time. FCFS lets the early lower-priority request occupy the engine. Priority scheduling performs **1 preemption**, allowing the high-priority requests to move ahead.

Key results:

| Metric | FCFS | Priority | Interpretation |
|---|---:|---:|---|
| Throughput | 83.04 tok/s | 79.77 tok/s | Priority is slightly slower. |
| Avg TTFT | 199.97 ms | 130.38 ms | Priority improves first-token responsiveness. |
| Avg latency | 323.95 ms | 253.90 ms | Overall average latency improves. |
| High-priority latency | 308.77 ms | 112.88 ms | Major win for important requests. |
| Low-priority latency | 339.14 ms | 394.92 ms | Lower-priority work waits longer. |
| Preemptions | 0 | 1 | Priority pays recompute cost. |

This row captures the main scheduling tradeoff beautifully. Priority scheduling reduces high-priority latency to about **37%** of the FCFS value, but low-priority latency gets worse and throughput drops slightly to **0.96x**.

### `priority_mix_small_batch`

Configuration:

```text
max_batch_size=4
token_budget=16
prefill_chunk_size=8
max_kv_tokens=64
```

This workload has 12 requests with a mix of high-priority and low-priority work. Because batch size and KV budget are less restrictive, priority scheduling does not need to preempt.

Key results:

| Metric | FCFS | Priority | Interpretation |
|---|---:|---:|---|
| Throughput | 166.15 tok/s | 169.10 tok/s | Essentially unchanged, slightly better for priority. |
| Avg TTFT | 178.68 ms | 162.93 ms | Priority improves first-token delay. |
| Avg latency | 344.37 ms | 330.97 ms | Priority slightly improves average latency. |
| High-priority latency | 235.69 ms | 108.55 ms | High-priority requests complete much faster. |
| Low-priority latency | 380.60 ms | 405.12 ms | Low-priority requests pay the cost. |
| Preemptions | 0 | 0 | Admission order alone creates the benefit. |

This is the most attractive result for priority scheduling. It cuts high-priority latency by more than half, with no preemption and no meaningful throughput penalty.

### `memory_pressure_preemption`

Configuration:

```text
max_batch_size=3
token_budget=12
prefill_chunk_size=8
max_kv_tokens=32
```

This workload intentionally creates KV-cache pressure. Priority scheduling performs **2 preemptions**, which lets high-priority work move forward but forces lower-priority requests to recompute.

Key results:

| Metric | FCFS | Priority | Interpretation |
|---|---:|---:|---|
| Throughput | 127.67 tok/s | 119.76 tok/s | Priority is slower due to recompute. |
| Avg TTFT | 75.94 ms | 90.15 ms | Average TTFT gets worse. |
| Avg latency | 225.58 ms | 249.03 ms | Average latency gets worse. |
| High-priority latency | 183.62 ms | 120.07 ms | High-priority latency still improves. |
| Low-priority latency | 253.56 ms | 335.00 ms | Low-priority latency worsens substantially. |
| Preemptions | 0 | 2 | Recompute cost is visible. |

This is the sharpest demonstration of the cost side of priority scheduling. High-priority latency improves to about **65%** of the FCFS value, but total throughput falls to **0.94x** and average latency gets worse.

That is not a bug. It is the expected cost of recompute preemption: the scheduler discards lower-priority KV cache state to make room, then pays to rebuild it later.

### `equal_priority_control`

Configuration:

```text
max_batch_size=4
token_budget=16
prefill_chunk_size=8
max_kv_tokens=64
```

This is the control case. All requests have equal priority, so priority scheduling should reduce to FCFS behavior.

Key results:

| Metric | FCFS | Priority |
|---|---:|---:|
| Throughput | 165.27 tok/s | 165.35 tok/s |
| Avg TTFT | 60.42 ms | 60.22 ms |
| Avg latency | 210.86 ms | 210.66 ms |
| High-priority latency | 210.86 ms | 210.66 ms |
| Preemptions | 0 | 0 |

The results are effectively identical. This is important because it validates the benchmark. Priority scheduling changes behavior only when priority information matters.

## Main Trends

### 1. Priority Scheduling Improves High-Priority Latency

This is the main success criterion. In every workload with meaningful priority differences, high-priority latency improves:

| Case | High-Priority Latency Improvement |
|---|---:|
| `priority_inversion_serial` | 63.4% lower |
| `priority_mix_small_batch` | 53.9% lower |
| `memory_pressure_preemption` | 34.6% lower |

For serving systems, this is often exactly what you want. Interactive requests, paid-tier users, short chat completions, or latency-sensitive requests can avoid getting stuck behind slow background work.

### 2. Throughput Is Not The Main Win

Priority throughput is close to FCFS:

| Case | Priority Throughput Ratio |
|---|---:|
| `priority_inversion_serial` | 0.96x |
| `priority_mix_small_batch` | 1.02x |
| `memory_pressure_preemption` | 0.94x |
| `equal_priority_control` | 1.00x |

The scheduler does not make the model inherently faster. It changes request ordering. Throughput only changes indirectly through batching shape, memory pressure, and recomputation.

### 3. Low-Priority Requests Pay The Cost

Priority scheduling improves important requests by making less important requests wait:

| Case | FCFS Low-Priority Latency | Priority Low-Priority Latency |
|---|---:|---:|
| `priority_inversion_serial` | 339.14 ms | 394.92 ms |
| `priority_mix_small_batch` | 380.60 ms | 405.12 ms |
| `memory_pressure_preemption` | 253.56 ms | 335.00 ms |

This is expected. A scheduler is a policy tool. It does not remove work; it decides which work absorbs waiting time.

### 4. Preemption Is Useful But Expensive

Preemption appears in two priority runs:

| Case | Preemptions | Effect |
|---|---:|---|
| `priority_inversion_serial` | 1 | Big high-priority latency win, slight throughput loss. |
| `memory_pressure_preemption` | 2 | High-priority latency win, average latency and throughput worsen. |

In this benchmark, preemption uses a recompute strategy. When a request is preempted:

1. Its KV cache is discarded.
2. Its generated state is reset.
3. It re-enters the waiting queue.
4. It must prefill again later.

That is simple and memory-efficient, but it wastes compute. Production systems often add more sophisticated options, such as swapping KV cache blocks to CPU memory or using paged KV memory to reduce the need for preemption.

### 5. The Control Case Confirms The Policy Logic

The equal-priority run has nearly identical FCFS and priority results. That is exactly what should happen. If every request has the same priority, the priority key falls back to arrival order, so the behavior becomes FCFS.

## Significance For LLM Serving

Scheduling matters because LLM serving is resource-constrained:

- KV cache memory is finite.
- Decode is sequential per request.
- Batch slots are limited.
- Prefill and decode compete for token budget.
- Not all requests have equal importance.

Without scheduling, a server can accidentally let long, low-value work delay short, high-value work. Priority scheduling fixes that by explicitly encoding service policy.

In real systems, this can support:

- Interactive requests over background batch jobs.
- Paid-tier traffic over free-tier traffic.
- Short latency-sensitive requests over long offline generations.
- System or moderation requests over normal traffic.
- Deadline-aware or SLA-aware request handling.

The benchmark shows that this does not come for free. Better high-priority latency can mean worse low-priority latency, extra recomputation, and occasionally lower throughput. That is the central scheduler tradeoff.

## Caveats

These results are useful, but they are still a small educational benchmark:

- The model is tiny.
- The workloads are synthetic.
- CUDA timings are short and can have noise.
- The benchmark uses recompute preemption, not KV swapping.
- It does not model network latency, real request arrivals, cancellation, streaming clients, or admission control.
- It reports averages over one run, not repeated trials with variance.

Even with those caveats, the qualitative result is strong: priority scheduling changes who waits, and the benchmark makes that visible.

## Suggested Follow-up Benchmarks

Good next steps:

- Run multiple repetitions and report mean, median, p95, min, max, and standard deviation.
- Add per-priority p50/p95/p99 latency breakdowns.
- Track per-request preemption counts and recompute tokens.
- Report wasted prefill tokens caused by recompute preemption.
- Add arrival-rate sweeps instead of only fixed synthetic arrivals.
- Add a deadline-aware policy and compare it to FCFS and priority.
- Add a shortest-job-first or shortest-remaining-processing-time policy.
- Compare recompute preemption with swap-style preemption.
- Combine scheduling with prefix caching and paged attention.
- Add trace logs showing waiting, prefilling, active, and completed request IDs per step.

The current benchmark already demonstrates the core scheduling result: **priority scheduling is valuable because it protects important request latency, not because it universally increases throughput.**
