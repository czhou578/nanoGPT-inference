# Decode/Prefill Interleaving Benchmark Results Write-up

This note explains the results in [`interleaving_results.txt`](../interleaving_results.txt) for the decode/prefill interleaving benchmark implemented in [`benchmarks/interleaving.py`](../benchmarks/interleaving.py) and configured by [`benchmarks/interleaving_benchmark_runs.py`](../benchmarks/interleaving_benchmark_runs.py).

## Executive Summary

This benchmark compares two ways to schedule decode work and chunked prefill work under a shared token budget:

- **`separate_calls`**: decode and prefill are scheduled in the same step, but they run as separate model forward calls when both kinds of work are present.
- **`interleaved_fused`**: active decode rows and one prefill chunk are packed into a single mixed batch and run through one fused model forward call.

The fused interleaving path works mechanically: it completes the same requests, prompt tokens, and generated tokens as the separate-call baseline in every workload. It also achieves the intended structural goal:

- Forward calls drop by **21% to 34%**.
- Mixed decode/prefill steps appear in nearly every scheduler step where overlap is possible.
- Average decode batch size stays identical between methods.

However, the fused path is slower wall-clock in every row:

- Generated-token throughput is **0.91x to 0.96x** of `separate_calls`.
- Average latency is higher in every workload.
- Max inter-token gap is **1.09x to 1.12x** worse.
- Forward time is also higher, which means the slowdown is inside the measured model/packing path rather than just outside instrumentation.

The key takeaway:

> The benchmark validates the scheduling idea of decode/prefill interleaving, but this educational implementation does not yet turn fewer forward calls into faster wall-clock performance.

That is not surprising. The fused path uses Python-side tensor packing, right padding, KV stacking, KV slicing, and KV unstacking around a very small model. In a production engine, the win usually comes from optimized kernels and reducing launch overhead on large GPU workloads. Here, the extra mixed-batch bookkeeping costs more than the saved forward calls.

## Benchmark Setup

The run uses:

- **Model size:** `0.056769M` parameters
- **Device:** CUDA
- **Context length:** `block_size=64`
- **Benchmark target:** decode/prefill scheduling mechanics, not model quality

The model trains briefly before the benchmark:

| Step | Train Loss | Validation Loss |
|---:|---:|---:|
| 0 | 4.1800 | 4.1791 |
| 20 | 3.6074 | 3.6479 |
| 40 | 3.3261 | 3.3321 |
| 60 | 3.1051 | 3.1305 |
| 80 | 2.9561 | 2.9651 |
| 100 | 2.8321 | 2.8684 |
| 119 | 2.7762 | 2.7998 |

The generated text sample is noisy, as expected for a tiny character-level model trained briefly. The benchmark is about serving mechanics: how decode and prefill are packed, how many forward calls are made, and how latency changes.

## What Was Benchmarked

Both policies use the same broad schedule:

1. Decode active requests first.
2. Use the remaining token budget for one prefill chunk.
3. Promote a request to active decoding once its prompt is fully prefilled.
4. Continue until all requests finish.

The difference is how a step runs when both decode and prefill work are available.

### `separate_calls`

The separate-call policy runs decode and prefill independently:

```text
step N:
  model(decode batch)
  model(prefill chunk)
```

This is simpler. Decode rows have one new token each, while a prefill row may have several new prompt tokens.

### `interleaved_fused`

The fused policy packs both kinds of work into one mixed batch:

```text
step N:
  model([decode rows + prefill row])
```

Decode rows contribute one token. The prefill row contributes a chunk of prompt tokens. The benchmark pads rows to a shared length, runs one forward pass, then keeps only each row's real KV entries.

This is the optimization being tested:

> Can one larger mixed forward replace two smaller forwards?

## Important Implementation Detail

The fused path is intentionally compatible with the current NanoGPT model API:

```python
logits, loss, new_kvs = model(idx, pos=pos, past_kvs=past_kvs, attn_mask=attn_mask)
```

The model does not take an `input_mask`, so the benchmark uses right padding for mixed rows and slices out each row's real KV entries after the forward pass.

That keeps the benchmark self-contained, but it is not a production interleaving kernel. It still pays Python and PyTorch overhead for:

- building mixed token and position tensors,
- padding rows to a common shape,
- stacking per-request KV caches,
- applying attention masks,
- slicing and unstacking updated KV caches,
- sampling from per-row logits.

Those costs matter a lot for a tiny model.

## Metrics

| Metric | Meaning |
|---|---|
| `reqs` | Total requests in the workload. |
| `done` | Completed requests. This matches `reqs` in every row. |
| `prompt_tok` | Total prompt tokens processed. |
| `gen_tok` | Total generated tokens emitted. |
| `wall_s` | Total wall-clock time. Lower is better. |
| `gen_tok/s` | Generated tokens per second. Higher is better. |
| `prompt_tok/s` | Prompt tokens per second. Higher is better. |
| `fwd_calls` | Number of model forward calls. Lower is the main goal of fused interleaving. |
| `mixed_steps` | Steps containing mixed work, usually decode rows plus a prefill row. |
| `avg_ttft_ms` | Average time to first token. Lower is better. |
| `p95_ttft_ms` | 95th percentile time to first token. |
| `avg_lat_ms` | Average request latency. Lower is better. |
| `p95_lat_ms` | 95th percentile request latency. |
| `avg_gap_ms` | Average inter-token gap for streaming requests. Lower is smoother. |
| `max_gap_ms` | Worst observed inter-token gap. Lower is smoother. |
| `avg_decode_b` | Average decode batch size. |
| `forward_s` | Time spent in measured model forward work plus synchronized fused-step work. |

The most important metrics are `fwd_calls`, `gen_tok/s`, `avg_lat_ms`, and `max_gap_ms`.

## Results Summary

| Case | Requests | Generated Tokens | Chunk Size | Token Budget | Separate Tok/s | Fused Tok/s | Throughput Ratio | Forward-Call Ratio | Avg TTFT Ratio | Max Gap Ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `small_interleave_smoke` | 6 | 112 | 8 | 16 | 503.51 | 484.05 | 0.96x | 0.74x | 0.93x | 1.09x |
| `decode_prefill_overlap` | 9 | 216 | 8 | 16 | 523.24 | 486.01 | 0.93x | 0.79x | 1.06x | 1.09x |
| `smaller_chunks_more_mixing` | 9 | 216 | 4 | 16 | 459.34 | 429.26 | 0.93x | 0.66x | 1.01x | 1.12x |
| `larger_chunks_less_overhead` | 9 | 216 | 12 | 24 | 523.30 | 482.05 | 0.92x | 0.79x | 1.06x | 1.10x |
| `staggered_prefill_arrivals` | 12 | 352 | 8 | 16 | 583.69 | 529.78 | 0.91x | 0.79x | 1.07x | 1.11x |

The average throughput ratio is about **0.93x**, so fused interleaving is roughly **7% slower** overall in this run.

The average forward-call ratio is about **0.75x**, so fused interleaving uses about **25% fewer model calls** overall.

That contrast is the central result: fewer calls, but slower end-to-end.

## Main Trend: Fewer Forward Calls, Slower Wall Time

The fused path reduces forward calls in every workload:

| Case | Separate Calls | Fused Calls | Calls Saved | Reduction |
|---|---:|---:|---:|---:|
| `small_interleave_smoke` | 43 | 32 | 11 | 25.6% |
| `decode_prefill_overlap` | 81 | 64 | 17 | 21.0% |
| `smaller_chunks_more_mixing` | 100 | 66 | 34 | 34.0% |
| `larger_chunks_less_overhead` | 81 | 64 | 17 | 21.0% |
| `staggered_prefill_arrivals` | 112 | 89 | 23 | 20.5% |

This confirms that the fused path is doing the intended scheduling transformation. When decode and prefill overlap, it can replace two calls with one.

But throughput falls in every case:

| Case | Separate Wall Time | Fused Wall Time | Slowdown |
|---|---:|---:|---:|
| `small_interleave_smoke` | 0.2224 s | 0.2314 s | +4.0% |
| `decode_prefill_overlap` | 0.4128 s | 0.4444 s | +7.7% |
| `smaller_chunks_more_mixing` | 0.4702 s | 0.5032 s | +7.0% |
| `larger_chunks_less_overhead` | 0.4128 s | 0.4481 s | +8.6% |
| `staggered_prefill_arrivals` | 0.6031 s | 0.6644 s | +10.2% |

This means that each fused step is more expensive than expected. The saved call overhead is not enough to offset the cost of the fused path's packing and masking.

## Why The Fused Path Is Slower Here

### 1. Mixed Batches Are More Expensive To Assemble

The separate path can run a normal decode batch and a normal prefill chunk. The fused path has to construct one mixed tensor where rows have different real token counts:

```text
decode row:  1 real token
decode row:  1 real token
prefill row: 8 real tokens
```

Rows are padded to a shared length. Then the benchmark has to track which tokens are real and slice the KV cache back apart afterward.

That extra work happens in Python and PyTorch tensor operations around a tiny model.

### 2. Padding Makes Decode Rows Heavier

When the prefill chunk has length 8, the fused batch has `T_max=8`. Decode rows contain one real token plus padding positions. The benchmark only uses the real decode position afterward, but the model still receives a larger rectangular tensor.

In production engines, this kind of mixed-token scheduling is handled with lower-level kernels and metadata. Here, the simple NanoGPT model sees the padded batch shape.

### 3. The Model Is Very Small

The model has only **0.056769M** parameters. For a model this small, one extra tensor copy or `torch.cat` can matter almost as much as the forward itself.

Interleaving is more compelling when:

- model forward calls are expensive,
- launch overhead is meaningful,
- batches are large enough to utilize the GPU,
- packing/unpacking is highly optimized.

This benchmark has the structure of interleaving, but not production-scale economics.

### 4. Forward Time Tracks Wall Time

The `forward_s` column is nearly equal to `wall_s` in every row, and it gets worse for the fused path:

| Case | Separate Forward Time | Fused Forward Time |
|---|---:|---:|
| `small_interleave_smoke` | 0.2222 s | 0.2311 s |
| `decode_prefill_overlap` | 0.4122 s | 0.4439 s |
| `smaller_chunks_more_mixing` | 0.4697 s | 0.5027 s |
| `larger_chunks_less_overhead` | 0.4122 s | 0.4475 s |
| `staggered_prefill_arrivals` | 0.6023 s | 0.6637 s |

This shows the slowdown is not just logging or outer-loop overhead. The measured synchronized work per fused step is higher.

## Latency And TTFT

Average latency gets worse under fused interleaving in every row:

| Case | Separate Avg Latency | Fused Avg Latency | Change |
|---|---:|---:|---:|
| `small_interleave_smoke` | 197.05 ms | 205.44 ms | +4.3% |
| `decode_prefill_overlap` | 304.65 ms | 329.92 ms | +8.3% |
| `smaller_chunks_more_mixing` | 364.78 ms | 390.15 ms | +7.0% |
| `larger_chunks_less_overhead` | 305.19 ms | 333.16 ms | +9.2% |
| `staggered_prefill_arrivals` | 480.29 ms | 528.38 ms | +10.0% |

This mostly follows wall-clock time. Since both policies complete the same amount of work in the same broad order, slower steps translate into slower request completion.

TTFT is mostly similar:

| Case | Separate Avg TTFT | Fused Avg TTFT | Ratio |
|---|---:|---:|---:|
| `small_interleave_smoke` | 38.15 ms | 35.32 ms | 0.93x |
| `decode_prefill_overlap` | 51.73 ms | 54.98 ms | 1.06x |
| `smaller_chunks_more_mixing` | 107.68 ms | 108.86 ms | 1.01x |
| `larger_chunks_less_overhead` | 52.20 ms | 55.45 ms | 1.06x |
| `staggered_prefill_arrivals` | 63.09 ms | 67.22 ms | 1.07x |

The fused path slightly improves TTFT only in the smallest workload. In the other rows it is slightly worse. This makes sense because interleaving does not change the high-level decode-first policy. It mostly changes how work inside each step is packed.

## Streaming Smoothness

Streaming smoothness is represented by inter-token gaps. Lower gaps mean tokens arrive more evenly for active requests.

The fused path has worse max gaps in every row:

| Case | Separate Max Gap | Fused Max Gap | Ratio |
|---|---:|---:|---:|
| `small_interleave_smoke` | 117.42 ms | 127.59 ms | 1.09x |
| `decode_prefill_overlap` | 213.13 ms | 231.97 ms | 1.09x |
| `smaller_chunks_more_mixing` | 219.95 ms | 245.81 ms | 1.12x |
| `larger_chunks_less_overhead` | 214.30 ms | 234.94 ms | 1.10x |
| `staggered_prefill_arrivals` | 451.24 ms | 500.65 ms | 1.11x |

Average gaps are also worse:

| Case | Separate Avg Gap | Fused Avg Gap |
|---|---:|---:|
| `small_interleave_smoke` | 8.91 ms | 9.52 ms |
| `decode_prefill_overlap` | 10.96 ms | 11.89 ms |
| `smaller_chunks_more_mixing` | 11.13 ms | 12.14 ms |
| `larger_chunks_less_overhead` | 10.95 ms | 12.01 ms |
| `staggered_prefill_arrivals` | 14.66 ms | 16.20 ms |

This does not mean interleaving is conceptually bad for streaming. It means this implementation's fused step is slower, so the time between emitted tokens stretches out.

## Chunk Size And Mixing

The most direct chunk-size comparison is:

| Case | Chunk Size | Token Budget | Separate Calls | Fused Calls | Forward-Call Ratio | Fused Throughput |
|---|---:|---:|---:|---:|---:|---:|
| `decode_prefill_overlap` | 8 | 16 | 81 | 64 | 0.79x | 486.01 tok/s |
| `smaller_chunks_more_mixing` | 4 | 16 | 100 | 66 | 0.66x | 429.26 tok/s |

Smaller chunks create more opportunities to save calls: forward calls drop from 100 to 66, a **34%** reduction. But smaller chunks also increase the number of scheduler steps and prefill fragments. The result is slower throughput for both methods, and fused interleaving still loses wall-clock.

This shows an important tuning tradeoff:

> Smaller chunks can increase mixing opportunities, but they can also increase scheduling and KV-management overhead.

The larger chunk case tells a related story:

| Case | Chunk Size | Token Budget | Separate Tok/s | Fused Tok/s |
|---|---:|---:|---:|---:|
| `larger_chunks_less_overhead` | 12 | 24 | 523.30 | 482.05 |

Larger chunks reduce the number of prefill fragments, but the fused path is still slower because the mixed-batch execution remains heavier.

## Row-by-Row Interpretation

### `small_interleave_smoke`

Configuration:

```text
num_decode_heavy_requests=4
decode_prompt_len=8
decode_max_new_tokens=24
num_prefill_heavy_requests=2
prefill_prompt_len=32
prefill_max_new_tokens=8
prefill_arrival_step=2
max_batch_size=4
token_budget=16
chunk_size=8
```

This is the smallest workload. Fused interleaving saves **11 forward calls**, reducing calls from **43** to **32**. It also slightly improves average TTFT from **38.15 ms** to **35.32 ms**.

But throughput falls from **503.51** to **484.05 tok/s**, a **0.96x** ratio. This is the best fused result in the file, but still not a speedup.

### `decode_prefill_overlap`

This is the main overlap workload:

```text
num_decode_heavy_requests=6
num_prefill_heavy_requests=3
chunk_size=8
token_budget=16
```

Both methods complete **9 requests** and **216 generated tokens**. Fused interleaving saves **17 forward calls**, dropping from **81** to **64**.

Despite that, throughput falls from **523.24** to **486.01 tok/s**. Average latency increases from **304.65 ms** to **329.92 ms**.

This row is the cleanest demonstration of the benchmark's central result: fewer calls do not automatically mean faster execution when the fused batch is expensive to construct and run.

### `smaller_chunks_more_mixing`

This row changes chunk size from 8 to 4 while keeping the same general workload.

Results:

- Separate calls: **100**
- Fused calls: **66**
- Forward-call ratio: **0.66x**
- Fused throughput ratio: **0.93x**
- Max gap ratio: **1.12x**

This is the strongest forward-call reduction in the file, but not the fastest run. Smaller chunks create more prefill fragments and more scheduling overhead. The fused path saves many calls, but the run still slows down.

This row is useful because it proves the fused mechanism is active. It just also proves that call count alone is not enough to predict performance.

### `larger_chunks_less_overhead`

This row increases token budget and chunk size:

```text
token_budget=24
chunk_size=12
prefill_prompt_len=40
```

The larger prefill prompt raises prompt tokens from **144** to **168**, while generated tokens stay at **216**.

Fused interleaving again saves **17 forward calls**, but throughput falls from **523.30** to **482.05 tok/s**. Average latency increases from **305.19 ms** to **333.16 ms**.

This suggests that simply increasing the chunk size and budget is not enough to make this fused implementation profitable.

### `staggered_prefill_arrivals`

This is the largest workload:

```text
num_decode_heavy_requests=8
decode_max_new_tokens=40
num_prefill_heavy_requests=4
stagger_prefill_arrivals=True
```

The fused path completes the same **12 requests** and **352 generated tokens**, and saves **23 forward calls**.

But this is also the worst throughput ratio:

- Separate calls: **583.69 tok/s**
- Fused interleaving: **529.78 tok/s**
- Throughput ratio: **0.91x**

The larger workload creates more opportunities for mixing, but also more accumulated overhead in the fused path. Average latency rises from **480.29 ms** to **528.38 ms**.

## Why Interleaving Still Matters

Even though this benchmark does not show a speedup, the interleaving idea is important.

In production inference servers, decode and prefill have different shapes:

- Decode: many requests, usually one new token each.
- Prefill: fewer requests, often many tokens each.

If they are always run separately, the engine may launch more kernels and leave hardware underutilized. Interleaving lets a scheduler fill a token budget with a mixture of work:

```text
decode rows first
remaining budget goes to prefill chunks
run one mixed batch
```

That can improve throughput and streaming smoothness when implemented with optimized kernels and efficient KV-cache metadata.

This benchmark shows the policy transformation clearly. It just does not have the optimized backend needed to make the transformation faster.

## Caveats

These results should be interpreted as an educational benchmark, not a production performance result.

1. **The model is tiny.** With only `0.056769M` parameters, packing overhead is large relative to model compute.
2. **The fused path is Python-heavy.** It builds mixed batches and unpacks KV caches in ordinary PyTorch/Python code.
3. **Padding can add extra work.** Decode rows are packed into tensors sized by the prefill chunk length.
4. **The benchmark uses a simple NanoGPT attention path.** It does not use production mixed prefill/decode kernels.
5. **The run is single-seed.** Small timing differences can vary between runs, especially for short subsecond benchmarks.
6. **The benchmark measures throughput, not quality.** The generated sample text is not the point.

## Practical Takeaways

1. **The fused interleaving mechanism works.** It completes the same work and reduces forward calls in every workload.
2. **Forward-call count is not enough.** The fused path saves calls but still loses wall-clock time.
3. **Packing overhead dominates here.** Mixed-batch construction and KV unstacking are too expensive for this tiny model.
4. **Smaller chunks increase call savings but not speed.** `smaller_chunks_more_mixing` saves the most calls but remains slower.
5. **Production interleaving needs optimized kernels.** The real win comes when mixed decode/prefill batches are handled efficiently by the serving backend.

## Suggested Follow-up Benchmarks

Useful next experiments would be:

- Run the same suite across multiple seeds and report mean/stddev.
- Compare with larger `n_embd`, `n_layer`, and longer prompts to increase model compute relative to Python overhead.
- Add a variant with left padding plus an explicit `input_mask` to avoid treating padded positions as real new tokens.
- Track time spent in batch construction, model forward, and KV unstacking separately.
- Increase request counts and output lengths to reduce noise from short runs.
- Try a CUDA-graph or compiled path to reduce per-step Python overhead.
- Compare this benchmark against the chunked-prefill benchmark to isolate which cost comes from chunking versus fused packing.

## Bottom Line

Decode/prefill interleaving is structurally successful in this benchmark: it reduces model forward calls by about **25%** on average while completing the same requests and tokens.

But in this small NanoGPT implementation, the fused path is slower. Throughput drops to about **0.93x** of the separate-call baseline because mixed-batch packing, padding, masking, and KV slicing cost more than the saved forward-call overhead.

The right interpretation is:

> This benchmark proves the interleaving scheduler mechanics, but not a production-style interleaving speedup.

To see the expected production benefit, the same scheduling idea would need a larger workload and a backend where mixed decode/prefill execution is cheap enough for fewer forward calls to dominate.
