# Prefix Caching Benchmark Results Write-up

This note explains the results in [`prefix_caching_results.txt`](../prefix_caching_results.txt) for the prefix caching benchmark implemented in [`benchmarks/prefix_caching.py`](../benchmarks/prefix_caching.py) and configured by [`benchmarks/prefix_caching_benchmark_runs.py`](../benchmarks/prefix_caching_benchmark_runs.py).

## Executive Summary

This benchmark compares two prompt-processing paths:

- **No prefix cache:** every request prefills its full prompt from scratch.
- **Prefix cache:** block-aligned prompt prefixes can be reused from earlier requests, so later requests only prefill the uncached suffix.

The benchmark shows that prefix caching is **mechanically working**: in shared-prefix workloads it avoids a large amount of prompt prefill work.

The strongest examples:

- `high_reuse_many_requests` reduces actual prefill from **672 tokens** to **120 tokens**, an **82.1% prefill-token reduction**.
- `shared_prefix_basic` reduces actual prefill from **192 tokens** to **80 tokens**, a **58.3% reduction**.
- `multi_prefix_groups` reduces actual prefill from **576 tokens** to **256 tokens**, a **55.6% reduction**.

However, wall-clock throughput does **not** improve in this tiny CUDA benchmark. Prefix caching reports generated-token throughput ratios between **0.85x and 0.95x**, meaning it is slower end-to-end in every row.

That is not as contradictory as it first looks. The benchmark successfully demonstrates cache reuse, but the model and prompts are small enough that the overhead of cache lookup, KV cloning, slicing, concatenation, insertion, and eviction dominates the compute saved by skipping prefill tokens.

The right reading is:

> Prefix caching clearly reduces repeated prefill work, but this small educational implementation does not yet convert that saved work into faster wall-clock throughput.

## Benchmark Setup

The run uses:

- **Model size:** `0.056769M` parameters
- **Device:** CUDA
- **Context length:** `block_size=64`
- **Prefix block size:** `4` tokens
- **Benchmark target:** prefix-cache reuse behavior, not model quality

The training log before the benchmark shows the tiny model learning:

| Step | Train Loss | Validation Loss |
|---:|---:|---:|
| 0 | 4.1800 | 4.1791 |
| 20 | 3.6074 | 3.6479 |
| 40 | 3.3261 | 3.3321 |
| 60 | 3.1051 | 3.1305 |
| 80 | 2.9561 | 2.9651 |
| 100 | 2.8321 | 2.8684 |
| 119 | 2.7762 | 2.7998 |

The generated text is noisy, which is expected. The benchmark is about inference mechanics: whether shared prompt prefixes reduce repeated prefill work.

## What Was Benchmarked

The benchmark uses two policies:

| Method | Behavior |
|---|---|
| `no_prefix_cache` | Prefills the full prompt for every request. |
| `prefix_cache` | Checks whether block-aligned prefix blocks already exist in a global cache, loads matching KV blocks, and prefills only the remaining suffix. |

Every shared-prefix workload includes a unique suffix. This matters because the benchmark avoids the special case where an entire prompt is cached and no suffix forward pass is available to produce first-token logits.

The prefix cache is block-based:

```text
prompt = [cached prefix blocks] + [unique suffix]
```

For example, with `prefix_block_size=4` and a 16-token shared prefix:

```text
shared prefix = 4 cached blocks
unique suffix = freshly prefilled
```

The cache uses chained hashes so a block is identified by both its own tokens and all previous prefix blocks. This prevents the same 4-token block from being reused incorrectly when it appears under a different prefix.

## Metrics

| Metric | Meaning |
|---|---|
| `reqs` | Number of requests served. |
| `prompt_tok` | Total logical prompt tokens across all requests. |
| `actual_prefill` | Prompt tokens actually passed through model prefill after cache reuse. Lower is better. |
| `cached_tok` | Prompt tokens skipped by loading cached KV blocks. Higher means more reuse. |
| `gen_tok` | Generated tokens emitted. |
| `wall_s` | Total wall-clock time. Lower is better. |
| `gen_tok/s` | Generated tokens per second. Higher is better. |
| `prefill_tok/s` | Actual prefill tokens per second. This uses actual prefill tokens, not logical prompt tokens. |
| `avg_ttft_ms` | Average time to first token. Lower is better. |
| `p95_ttft_ms` | 95th percentile time to first token. Lower is better. |
| `avg_lat_ms` | Average request latency. Lower is better. |
| `hit_rate` | Cache hits divided by cache lookups. |
| `blocks` | Number of blocks left in the cache at the end. |
| `evict` | Number of cache block evictions. |
| `forward_s` | Time spent in measured model forward calls. |

The most important metrics here are `actual_prefill`, `cached_tok`, `hit_rate`, `evict`, and `gen_tok/s`.

## Results Summary

| Case | Requests | Prompt Tokens | Cached Tokens | Actual Prefill | Prefill Reduction | Hit Rate | Throughput Ratio | Evictions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `shared_prefix_basic` | 8 | 192 | 112 | 80 | 58.3% | 77.8% | 0.95x | 0 |
| `high_reuse_many_requests` | 24 | 672 | 552 | 120 | 82.1% | 85.2% | 0.94x | 0 |
| `multi_prefix_groups` | 24 | 576 | 320 | 256 | 55.6% | 76.9% | 0.94x | 0 |
| `low_reuse_control` | 8 | 192 | 0 | 192 | 0.0% | 0.0% | 0.88x | 0 |
| `eviction_pressure` | 24 | 576 | 0 | 576 | 0.0% | 0.0% | 0.85x | 136 |

The first three rows show the cache doing useful work. The last two rows show what happens when there is no reuse or when the cache is too small to retain useful prefixes.

## Main Trend: Cache Reuse Works

In the shared-prefix cases, prefix caching dramatically reduces actual prefill tokens.

### `shared_prefix_basic`

The workload has 8 requests, one shared prefix group, a 16-token shared prefix, and an 8-token unique suffix.

Results:

- Logical prompt tokens: **192**
- Actual prefill tokens with cache: **80**
- Cached tokens: **112**
- Hit rate: **77.8%**
- Prefill reduction: **58.3%**

This is exactly the intended behavior. The first request populates the cache. Later requests reuse shared prefix blocks and only process their unique suffix.

### `high_reuse_many_requests`

This is the strongest cache-reuse result. It has 24 requests sharing one 24-token prefix.

Results:

- Logical prompt tokens: **672**
- Actual prefill tokens with cache: **120**
- Cached tokens: **552**
- Hit rate: **85.2%**
- Prefill reduction: **82.1%**

This is the best demonstration that prefix caching scales with repeated shared context. Once the shared blocks are cached, almost every later request avoids most of its prompt prefill.

### `multi_prefix_groups`

This workload has 24 requests split across 4 shared-prefix groups.

Results:

- Logical prompt tokens: **576**
- Actual prefill tokens with cache: **256**
- Cached tokens: **320**
- Hit rate: **76.9%**
- Prefill reduction: **55.6%**

The reduction is smaller than `high_reuse_many_requests` because reuse is spread across multiple groups. Each group needs to warm its own prefix blocks before later requests can hit.

## The Wall-Clock Surprise

Even though prefix caching reduces prefill work, it does not improve generated-token throughput in this benchmark:

| Case | No Cache Gen Tok/s | Prefix Cache Gen Tok/s | Ratio |
|---|---:|---:|---:|
| `shared_prefix_basic` | 89.22 | 84.44 | 0.95x |
| `high_reuse_many_requests` | 80.64 | 75.48 | 0.94x |
| `multi_prefix_groups` | 81.10 | 76.43 | 0.94x |
| `low_reuse_control` | 89.53 | 78.98 | 0.88x |
| `eviction_pressure` | 81.30 | 68.75 | 0.85x |

This is the most important nuance in the file. The cache is reducing model prefill tokens, but the benchmark still gets slower overall.

The likely reasons are:

1. **The model is tiny.** With only `0.056769M` parameters, prefill compute is cheap. There is not much expensive model work to skip.
2. **Cache operations are expensive relative to model compute.** The benchmark clones KV tensors, slices blocks, concatenates cached blocks, hashes block tokens, and updates cache metadata.
3. **Requests are short.** Prompts are 24 to 28 tokens long. Prefix caching usually matters more when shared prefixes are much longer.
4. **Decode still dominates much of the run.** The benchmark reports generated-token throughput, and every request still decodes the same number of output tokens.
5. **The implementation is educational.** Production systems optimize KV memory layout, cache metadata, hashing, and block movement much more aggressively.

So the benchmark proves the mechanism, but not a wall-clock speedup.

## TTFT Behavior

Prefix caching increases TTFT in the shared-prefix cases:

| Case | No Cache Avg TTFT | Prefix Cache Avg TTFT | Ratio |
|---|---:|---:|---:|
| `shared_prefix_basic` | 2.79 ms | 7.96 ms | 2.85x |
| `high_reuse_many_requests` | 2.70 ms | 9.43 ms | 3.49x |
| `multi_prefix_groups` | 2.78 ms | 6.64 ms | 2.39x |
| `low_reuse_control` | 2.79 ms | 2.71 ms | 0.97x |
| `eviction_pressure` | 2.76 ms | 2.69 ms | 0.98x |

In a larger production system, prefix caching often improves TTFT for repeated long prompts because it avoids recomputing the shared prefix. Here, the opposite happens in the reuse cases because cache loading overhead is larger than the skipped compute.

The control and eviction rows have TTFT ratios near **1.0x**, which suggests the huge TTFT penalty in shared-prefix rows is tied to the cache-load path rather than normal request processing.

## Control Case: `low_reuse_control`

This workload uses unique prompts, so the prefix cache has nothing useful to reuse.

Results:

- Cached tokens: **0**
- Hit rate: **0.0%**
- Prefill reduction: **0.0%**
- Throughput ratio: **0.88x**
- Final blocks: **48**

This is a useful sanity check. The cache fills with blocks, but no later request has the same prefix chain, so there are no hits. The prefix-cache path is slower because it pays cache insertion and lookup overhead without any reuse benefit.

This row teaches an important serving lesson:

> Prefix caching only helps when prompts actually share prefixes.

If traffic has mostly unique prompts, prefix caching can add overhead without reducing prefill.

## Eviction Case: `eviction_pressure`

This workload has shared-prefix groups, but the cache is intentionally tiny:

```text
max_cache_blocks=8
num_groups=6
shared_prefix_len=16
prefix_block_size=4
```

Each shared prefix needs 4 blocks. Six groups need many more blocks than the cache can hold. The result:

- Cached tokens: **0**
- Hit rate: **0.0%**
- Evictions: **136**
- Throughput ratio: **0.85x**

This is the clearest failure mode. The cache constantly evicts blocks before they can be reused. That creates pure overhead: hashing, insertion, eviction, and metadata churn with no prefill-token reduction.

The lesson:

> Prefix caching needs enough capacity to keep hot prefixes resident.

If the cache is too small relative to the number of active prefix groups, it can perform worse than no cache.

## Why `prefill_tok/s` Drops

At first glance, `prefill_tok/s` looks worse for prefix caching. For example, in `high_reuse_many_requests`, it drops from **564.51** to **94.34**.

That metric is based on **actual prefill tokens**, not logical prompt tokens. Since prefix caching intentionally reduces actual prefill tokens from **672** to **120**, the denominator gets much smaller. The lower `prefill_tok/s` is not by itself a failure. It mostly says:

- fewer prefill tokens were actually run,
- but the total run still includes cache operations and decode work,
- so actual prefill tokens per wall second is not the clearest success metric for prefix caching.

For this benchmark, the clearer cache success metric is **actual prefill-token reduction**.

## Row-by-Row Interpretation

### `shared_prefix_basic`

Prefix caching avoids **112** of **192** prompt tokens, a **58.3%** prefill reduction. The cache hit rate is **77.8%**, and no blocks are evicted.

This confirms the basic mechanism works. However, throughput drops slightly from **89.22** to **84.44 generated tokens/sec**, and average TTFT rises from **2.79 ms** to **7.96 ms**. The cache saves prefill work, but the overhead is larger than the compute saved.

### `high_reuse_many_requests`

This is the strongest reuse case. Prefix caching avoids **552** of **672** prompt tokens, an **82.1%** prefill reduction. Hit rate reaches **85.2%**.

Despite that, throughput drops from **80.64** to **75.48 generated tokens/sec**. This shows how small-model benchmarks can hide the benefits of compute-saving optimizations: the skipped prefill is real, but not expensive enough to overcome cache overhead.

### `multi_prefix_groups`

With 4 prefix groups, prefix caching avoids **320** of **576** prompt tokens, a **55.6%** reduction. This is lower than the single-group high-reuse case because each group needs its own cache warmup.

Throughput again drops slightly, from **81.10** to **76.43 generated tokens/sec**. The mechanism works, but the wall-clock result is still overhead-bound.

### `low_reuse_control`

There are no shared prefixes, so prefix caching avoids **0** tokens and has a **0.0%** hit rate.

The prefix-cache path is slower: **78.98** vs **89.53 generated tokens/sec**. This is expected. Cache bookkeeping with no hits is pure overhead.

### `eviction_pressure`

The workload has shared-prefix groups, but the cache is too small. It ends with only **8** blocks and performs **136 evictions**.

The cache hit rate is **0.0%**, cached tokens are **0**, and throughput drops to **0.85x** of the no-cache baseline. This is the most important warning row: insufficient cache capacity can eliminate the benefit of prefix caching.

## Main Conclusions

1. **Prefix caching works mechanically.** In high-reuse workloads, it reduces actual prefill tokens by **55.6% to 82.1%**.

2. **This benchmark does not show wall-clock speedup.** Generated-token throughput is lower in every row, ranging from **0.85x to 0.95x** of the no-cache path.

3. **Overhead dominates at this scale.** The model, prompts, and benchmark windows are too small for saved prefill compute to outweigh cache-management overhead.

4. **Reuse pattern matters.** The low-reuse control gets no hits and becomes slower.

5. **Cache capacity matters.** The eviction-pressure case has shared prefixes but still gets no hits because blocks are evicted too aggressively.

6. **The most meaningful success metric here is prefill-token reduction.** Throughput will become more meaningful after scaling the model, prompt length, and cache implementation.

## Significance For LLM Serving

Prefix caching is important in real LLM serving because many workloads repeat large prompt prefixes:

- System prompts reused across many conversations.
- Tool instructions or policy text prepended to every request.
- Retrieval-augmented prompts with shared document headers.
- Multi-turn conversations that reuse a long chat history.
- Agents repeatedly calling a model with the same instruction scaffold.

In those settings, recomputing the shared prefix for every request wastes compute. Prefix caching stores KV blocks for the shared prefix and lets later requests resume from the cached state.

The benchmark demonstrates the core serving idea:

```text
first request:
    prefill shared prefix + unique suffix
    cache shared prefix blocks

later request:
    load shared prefix KV blocks
    prefill only unique suffix
```

In production, this can reduce TTFT, reduce prefill GPU load, and increase serving capacity. But the cache has to be implemented efficiently and sized for the traffic pattern.

## Caveats

These results should be interpreted as an educational microbenchmark:

- The model is extremely small.
- Prompt lengths are short.
- Prefix blocks are only 4 tokens.
- The cache stores cloned tensors and reconstructs prefixes with Python-level operations.
- The benchmark serves requests sequentially rather than through a full continuous batching scheduler.
- It reports one run, not repeated statistics.
- CUDA timings for tiny workloads can be noisy.

The qualitative result is still useful: cache hits and prefill-token savings are real, but cache overhead must be controlled for prefix caching to improve wall-clock latency.

## Suggested Follow-up Benchmarks

Good next steps:

- Use much longer shared prefixes, such as 256, 512, or 1024 tokens.
- Increase model size so skipped prefill compute is expensive enough to matter.
- Run repeated trials and report mean, median, p95, min, max, and standard deviation.
- Separate cache lookup/load time, cache insert time, prefill forward time, and decode forward time.
- Report logical prompt tokens/sec in addition to actual prefill tokens/sec.
- Add batched decode and continuous batching around prefix caching.
- Sweep `prefix_block_size` independently.
- Sweep `max_cache_blocks` and plot hit rate vs evictions.
- Add an LRU hot/cold workload where some prefixes are popular and others are rare.
- Compare cloned tensor blocks with a more memory-efficient block table or paged KV layout.

The current benchmark already answers the first key question: **does prefix caching reduce repeated prefill work?** Yes. The next benchmark question is: **under what scale and cache implementation does that saved work become a wall-clock speedup?**
