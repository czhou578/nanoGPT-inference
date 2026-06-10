# Paged Attention Benchmark Results Write-up

This note explains the results in [`paged_attention_results.txt`](../paged_attention_results.txt) for the contiguous KV vs paged KV benchmark implemented in [`benchmarks/paged_attention.py`](../benchmarks/paged_attention.py) and configured by [`benchmarks/paged_attention_benchmark_runs.py`](../benchmarks/paged_attention_benchmark_runs.py).

## Executive Summary

This benchmark compares two KV-cache layouts:

- **`contiguous_kv`**: each request stores its KV cache as normal contiguous tensors.
- **`paged_kv`**: each request owns a block table pointing into a fixed-size physical KV block pool.

The benchmark shows that paged KV is **functionally working**: it completes the same requests and generated tokens as the contiguous baseline, allocates and frees physical KV blocks, and keeps peak block usage within the configured pool.

But paged KV is slower in every row:

- Throughput ranges from **0.60x to 0.73x** of contiguous KV.
- Average latency is **1.37x to 1.65x** higher.
- Average TTFT is roughly **1.7x to 2.3x** higher.

That result is expected for this educational implementation. The benchmark does **not** use a true fused PagedAttention kernel. Instead, it stores KV in a paged block pool, then gathers those blocks back into contiguous tensors before each model forward. That demonstrates the memory-management idea, but adds gather/scatter and allocation bookkeeping overhead.

The key takeaway:

> This benchmark validates paged KV block-table mechanics, but it does not show the speed advantage of production PagedAttention because the model still consumes contiguous KV tensors.

## Benchmark Setup

The run uses:

- **Model size:** `0.056769M` parameters
- **Device:** CUDA
- **Context length:** `block_size=64`
- **Paged KV block size:** `page_block_size=4`
- **Benchmark target:** KV memory layout and block-pool behavior

Before the benchmark, the tiny model trains briefly:

| Step | Train Loss | Validation Loss |
|---:|---:|---:|
| 0 | 4.1800 | 4.1791 |
| 20 | 3.6074 | 3.6479 |
| 40 | 3.3261 | 3.3321 |
| 60 | 3.1051 | 3.1305 |
| 80 | 2.9561 | 2.9651 |
| 100 | 2.8321 | 2.8684 |
| 119 | 2.7762 | 2.7998 |

The generated sample is noisy, as expected for a tiny model trained briefly. Generation quality is not the point of this benchmark.

## What Was Benchmarked

The benchmark compares two serving paths over the same workloads.

| Method | Behavior |
|---|---|
| `contiguous_kv` | Prefill and decode store each request's KV cache as contiguous per-request tensors. Batches are formed by padding and stacking those tensors. |
| `paged_kv` | Prefill writes KV into physical blocks from a block pool. Each request has a block table. Decode gathers the request's blocks, runs the model, writes the new KV slot back into the pool, and frees blocks when the request completes. |

The paged path tracks:

- physical block allocations,
- physical block frees,
- peak physical blocks used,
- average unused slots inside active blocks,
- peak unused slots inside active blocks.

This is useful because PagedAttention is mainly a memory-management technique: it avoids requiring every request to reserve one large contiguous KV allocation.

## Important Implementation Caveat

Production PagedAttention does not gather every request's paged KV cache into contiguous tensors before attention. It uses kernels that read through block tables directly.

This benchmark does gather paged KV back into contiguous tensors because the NanoGPT attention module expects ordinary `past_kvs`. So the paged path pays extra overhead:

```text
physical blocks -> gather into contiguous tensors -> model forward -> scatter new KV back into blocks
```

That overhead is why `paged_kv` is slower here. The result should be interpreted as a benchmark of the educational block-pool mechanism, not as a benchmark of optimized vLLM-style PagedAttention.

## Metrics

| Metric | Meaning |
|---|---|
| `reqs` | Total requests in the workload. |
| `done` | Completed requests. This matches `reqs` in every row. |
| `prompt_tok` | Total prompt tokens processed. |
| `gen_tok` | Total generated tokens emitted. |
| `wall_s` | Total wall-clock time. Lower is better. |
| `tok/s` | Generated tokens per second. Higher is better. |
| `avg_ttft_ms` | Average time to first token. Lower is better. |
| `p95_ttft_ms` | 95th percentile time to first token. |
| `avg_lat_ms` | Average request latency. Lower is better. |
| `p95_lat_ms` | 95th percentile request latency. |
| `avg_batch` | Average decode batch size. |
| `max_batch` | Maximum decode batch size. |
| `peak_blocks` | Maximum physical KV blocks used at once. |
| `allocs` | Number of physical block allocations. |
| `frees` | Number of physical block frees. |
| `avg_waste` | Average unused slots in allocated active blocks. |
| `peak_waste` | Maximum unused slots in allocated active blocks. |
| `forward_s` | Time spent in measured model forward work. |

The most important metrics are `tok/s`, `avg_lat_ms`, `peak_blocks`, `allocs`, `frees`, and waste slots.

## Results Summary

| Case | Requests | Generated Tokens | Contiguous Tok/s | Paged Tok/s | Throughput Ratio | Latency Ratio | Peak Blocks | Pool Size | Avg Waste |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `uniform_short` | 12 | 192 | 248.21 | 149.55 | 0.60x | 1.65x | 28 | 96 | 6.13 |
| `mixed_lengths` | 16 | 188 | 167.24 | 117.33 | 0.70x | 1.43x | 27 | 128 | 5.54 |
| `block_boundary_pressure` | 12 | 144 | 183.31 | 132.94 | 0.73x | 1.37x | 22 | 128 | 5.70 |
| `limited_pool` | 16 | 188 | 149.57 | 103.36 | 0.69x | 1.45x | 20 | 28 | 4.51 |

Paged KV completes all work, but it is slower in every case. Its average throughput ratio is about **0.68x**, and its average latency ratio is about **1.47x**.

## Main Trend: Paged KV Is Correct But Slower

The paged implementation completes the same number of requests and generated tokens in every workload:

- `uniform_short`: 12/12 requests, 192 generated tokens
- `mixed_lengths`: 16/16 requests, 188 generated tokens
- `block_boundary_pressure`: 12/12 requests, 144 generated tokens
- `limited_pool`: 16/16 requests, 188 generated tokens

That is the correctness signal. The block pool and request block tables are functioning.

The performance signal is different. Paged KV is slower because this implementation adds extra work around the model:

- allocate physical blocks,
- gather each request's block-table KV into contiguous tensors,
- pad gathered caches for batching,
- run the model,
- scatter the new KV token back to the physical pool,
- free physical blocks at completion.

For a tiny model, that overhead is large relative to model compute.

## Case-By-Case Interpretation

### `uniform_short`

Configuration:

```text
num_requests=12
prompt_len=12
max_new_tokens=16
max_batch_size=4
page_block_size=4
num_physical_blocks=96
```

This workload has uniform prompt and output lengths, so it is the simplest case.

| Metric | Contiguous KV | Paged KV |
|---|---:|---:|
| Throughput | 248.21 tok/s | 149.55 tok/s |
| Avg TTFT | 2.72 ms | 5.55 ms |
| Avg latency | 239.53 ms | 396.23 ms |
| Avg batch | 4.00 | 4.00 |
| Peak blocks | 0 | 28 |
| Allocations | 0 | 84 |
| Frees | 0 | 84 |

Paged KV is only **0.60x** as fast and has **1.65x** higher average latency. Because batch size is identical, the slowdown comes from paged-cache overhead rather than reduced batching.

The memory behavior is visible: paged KV peaks at **28/96** physical blocks and performs **84 allocations** and **84 frees**.

### `mixed_lengths`

Configuration:

```text
num_requests=16
prompt_lens=(8, 12, 16, 20)
output_lens=(8, 12, 16)
max_batch_size=4
page_block_size=4
num_physical_blocks=128
```

This workload has varied prompt and output lengths, which is where block-based KV management becomes more relevant.

| Metric | Contiguous KV | Paged KV |
|---|---:|---:|
| Throughput | 167.24 tok/s | 117.33 tok/s |
| Avg TTFT | 2.71 ms | 6.12 ms |
| Avg latency | 255.38 ms | 364.72 ms |
| Avg batch | 3.58 | 3.58 |
| Peak blocks | 0 | 27 |
| Avg waste | 0.00 | 5.54 |

Paged KV reaches **0.70x** of contiguous throughput. This is better than `uniform_short`, but still slower.

The important memory signal is that mixed request lengths create internal waste inside fixed-size blocks. With 4-token pages, the paged path averages **5.54 unused slots** across active requests and peaks at **12 unused slots**.

### `block_boundary_pressure`

Configuration:

```text
num_requests=12
max_new_tokens=12
max_batch_size=4
page_block_size=4
num_physical_blocks=128
```

This workload uses prompt lengths near block boundaries: some requests fit exactly into a block, while others leave partially used blocks.

| Metric | Contiguous KV | Paged KV |
|---|---:|---:|
| Throughput | 183.31 tok/s | 132.94 tok/s |
| Avg TTFT | 2.66 ms | 4.56 ms |
| Avg latency | 244.41 ms | 334.42 ms |
| Peak blocks | 0 | 22 |
| Avg waste | 0.00 | 5.70 |
| Peak waste | 0 | 9 |

This is the best paged throughput ratio in the file at **0.73x**. The block-boundary workload still shows overhead, but the relative penalty is smaller than in the uniform case.

The waste metrics are useful here. Fixed-size pages create internal fragmentation: if a request has 5 tokens and the page size is 4, it needs 2 pages and leaves unused slots in the second page.

### `limited_pool`

Configuration:

```text
num_requests=16
prompt_lens=(8, 12, 16, 20)
output_lens=(8, 12, 16)
max_batch_size=3
page_block_size=4
num_physical_blocks=28
```

This run intentionally limits the physical KV block pool.

| Metric | Contiguous KV | Paged KV |
|---|---:|---:|
| Throughput | 149.57 tok/s | 103.36 tok/s |
| Avg latency | 226.69 ms | 328.37 ms |
| Avg batch | 2.92 | 2.92 |
| Peak blocks | 0 | 20 |
| Pool size | N/A | 28 |
| Peak utilization | N/A | 20/28 |

Paged KV completes all requests without exhausting the pool. Peak utilization is **20/28 blocks**, or about **71%**. This row demonstrates the bounded-memory behavior: the engine can operate inside a fixed physical block budget.

It is still slower, with **0.69x** throughput and **1.45x** average latency. But unlike a raw contiguous approach, the paged path gives direct visibility into physical block pressure.

## Memory Behavior

The most significant paged-KV-specific metrics are block usage and waste.

| Case | Peak Blocks Used | Total Blocks | Peak Utilization | Allocations | Frees | Avg Waste | Peak Waste |
|---|---:|---:|---:|---:|---:|---:|---:|
| `uniform_short` | 28 | 96 | 29.2% | 84 | 84 | 6.13 | 12 |
| `mixed_lengths` | 27 | 128 | 21.1% | 103 | 103 | 5.54 | 12 |
| `block_boundary_pressure` | 22 | 128 | 17.2% | 60 | 60 | 5.70 | 9 |
| `limited_pool` | 20 | 28 | 71.4% | 103 | 103 | 4.51 | 9 |

These metrics are the main value of this benchmark. They show:

- blocks are allocated as requests prefill and decode,
- blocks are freed when requests complete,
- peak pool pressure remains within capacity,
- fixed-size pages create internal waste,
- a smaller pool can still handle the workload if requests complete and free blocks over time.

The equal allocation/free counts are a good sanity check. Every physical block allocated during the run is eventually freed.

## Why Paged KV Is Slower Here

Paged KV is slower for several reasons.

### 1. No Fused PagedAttention Kernel

The biggest reason is architectural. The model still expects contiguous `past_kvs`, so the benchmark must gather paged blocks before the forward pass.

Production systems avoid this by using kernels that read KV through block tables directly.

### 2. Small Model, Small Context

The model has only **0.056769M parameters**, and the context length is only 64. The raw attention work is cheap. Extra Python and tensor movement can outweigh the memory-layout benefit.

### 3. Extra Gather/Scatter Work

Paged KV does this on every decode step:

```text
gather physical blocks -> pad/stack cache -> model forward -> scatter new KV slot
```

The contiguous baseline has fewer moving pieces.

### 4. Internal Fragmentation

Paged KV allocates fixed-size blocks. If the last block for a request is only partially filled, the remaining slots are temporarily wasted. This is visible in `avg_waste` and `peak_waste`.

This kind of fragmentation is still usually better than reserving a full maximum-context KV cache for every request, but it is not free.

## Why Paged Attention Matters Anyway

The speed result here is not the main reason PagedAttention exists.

PagedAttention matters because real LLM serving is often constrained by KV-cache memory. If every request reserves contiguous memory for its maximum possible sequence length, the server wastes huge amounts of memory and cannot batch as many requests.

Paged KV solves that by:

- allocating KV memory in small physical blocks,
- giving each request a logical block table,
- growing KV cache as tokens are actually used,
- freeing blocks immediately when requests finish,
- avoiding large contiguous allocations,
- reducing memory fragmentation at serving scale.

In production, this enables higher concurrency and better batching. The performance win often comes indirectly: more requests fit in memory, so the engine can keep larger batches active.

This benchmark demonstrates the block-table mechanics, not that full production payoff.

## Main Conclusions

1. **Paged KV works functionally.** Every workload completes the same number of requests and generated tokens as contiguous KV.

2. **Paged KV is slower in this educational benchmark.** Throughput is **0.60x to 0.73x** of contiguous KV because the implementation gathers/scatters KV around a model that still expects contiguous tensors.

3. **The memory instrumentation is the useful signal.** Peak blocks, allocations, frees, and waste slots show the physical block-pool behavior clearly.

4. **The limited-pool case validates bounded memory use.** Paged KV completes the workload while peaking at **20/28** physical blocks.

5. **This is not a production PagedAttention speed benchmark.** A fused kernel that reads via block tables is required to evaluate the real throughput benefits.

## Caveats

These results are an educational microbenchmark:

- The model is tiny.
- Context length is short.
- Page size is only 4 tokens.
- Paged KV is gathered into contiguous tensors before attention.
- The benchmark does not include a true vLLM-style PagedAttention kernel.
- CUDA timings are short and can be noisy.
- The workloads are synthetic and small.

Even so, the benchmark is useful because it makes physical KV block management visible.

## Suggested Follow-up Benchmarks

Good next steps:

- Add repeated trials with mean, median, p95, min, max, and standard deviation.
- Sweep `page_block_size` to measure the tradeoff between metadata overhead and internal waste.
- Sweep `num_physical_blocks` until admission stalls or memory exhaustion occurs.
- Add per-step traces of active requests, block tables, free blocks, and waste slots.
- Track logical KV tokens vs allocated KV slots.
- Add cancellation or early completion to show fast block reuse.
- Add mixed arrival times instead of all-at-once arrivals.
- Combine paged KV with continuous batching and priority scheduling.
- Implement or integrate a real paged-attention kernel that reads through block tables directly.

The current result answers the first question: **does the paged block-pool mechanism work?** Yes. The next question is whether the implementation can avoid gather/scatter overhead and turn that memory design into serving throughput.
