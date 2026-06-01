# NanoGPT Inference Benchmark Harness Hints

This guide is meant to help you design the benchmark harness yourself. It points at the pieces you should measure, the shape of the implementation, and the traps to avoid, but it intentionally avoids handing you a finished solution.

Your current NanoGPT inference stack appears to include:

- KV cache
- continuous batching
- token budget scheduling
- chunked prefill
- FCFS / priority scheduling
- prefix caching
- paged KV / PagedAttention-style block tables
- speculative decoding with a draft model
- INT8 dynamic/static weight quantization experiments

That is enough machinery that casual timing is no longer useful. The benchmark harness should answer one question clearly:

> Which optimization changed latency, throughput, memory, and correctness behavior, and by how much?

## 1. Start With Benchmark Goals

Before writing timing code, decide what comparisons you want the harness to make.

Good first comparisons:

- baseline full-context generation vs KV cached generation
- single request vs continuous batching
- normal prefill vs chunked prefill
- no prefix cache vs prefix cache
- contiguous KV vs paged KV
- no speculative decoding vs speculative decoding
- fp32 weights vs quantized weights
- FCFS scheduling vs priority scheduling

Try not to benchmark everything at once. A good harness lets you toggle exactly one feature while holding the workload fixed.

## 2. Define The Core Metrics

You want both user-facing latency metrics and engine-facing throughput metrics.

Recommended minimum metrics:

| Metric | Meaning | Why It Matters |
|---|---|---|
| total wall time | full benchmark duration | quick sanity check |
| total generated tokens | all output tokens across requests | throughput denominator |
| tokens/sec | generated tokens / wall time | main throughput metric |
| request latency | time from arrival to completion | what users feel |
| time to first token | time from arrival to first generated token | critical for chat UX |
| inter-token latency | time between streamed decode tokens | smoothness of streaming |
| prefill tokens/sec | prompt tokens processed per second | separates prompt cost |
| decode tokens/sec | output tokens processed per second | separates generation cost |
| batch size per step | number of active requests per forward | confirms batching works |
| tokens per forward | total new tokens processed per model call | confirms token budget behavior |
| KV cache tokens | number of tokens stored in KV cache | memory pressure proxy |
| allocated KV blocks | physical blocks in use | paged KV memory metric |
| prefix cache hits | number of reused prompt blocks | verifies prefix caching |
| speculative acceptance rate | accepted draft tokens / proposed draft tokens | tells whether speculation helps |
| target forwards avoided | estimated target calls saved | connects speculation to speed |

Hint: split your metrics into two layers:

- per-request metrics
- per-engine-step metrics

Do not try to compute everything from only final totals.

## 3. Add Lightweight Instrumentation Objects

Avoid scattering timing variables everywhere. Create a small object whose only job is to collect events.

Think in terms of events like:

- request arrived
- request admitted
- prefill started
- prefill chunk finished
- first token emitted
- decode token emitted
- request completed
- forward pass started
- forward pass ended
- KV block allocated
- KV block freed
- prefix cache lookup
- prefix cache hit
- speculative draft proposed
- speculative tokens accepted

You do not need a complex tracing system. A few lists of dictionaries is enough.

Possible shape:

```python
bench.record_step(...)
bench.record_request_event(...)
bench.record_cache_event(...)
bench.record_speculation(...)
```

The important idea: your model and scheduler should not know how final statistics are calculated. They should only report facts.

## 4. Use A Monotonic Timer

Use a monotonic high-resolution clock.

Look into:

```python
time.perf_counter()
```

If you benchmark on CUDA, remember that GPU execution is asynchronous. CPU timers can lie unless you synchronize around timed regions.

Hint:

```python
if device == "cuda":
    torch.cuda.synchronize()
```

Do not synchronize excessively inside the serving loop unless the thing you are measuring requires it. Synchronization itself changes performance.

## 5. Separate Warmup From Measurement

Always run warmup before collecting metrics.

Why:

- PyTorch may initialize kernels lazily.
- CUDA may pay one-time setup costs.
- CPU caches and allocator behavior are noisy at startup.
- Quantized models may have setup overhead.

Suggested harness phases:

1. build model
2. optionally load weights
3. optionally quantize
4. create workload
5. run warmup workload
6. reset metrics
7. run measured workload
8. print / save report

Keep warmup workload smaller but structurally similar to the real one.

## 6. Make Workloads Reproducible

Your scheduler and speculative decoder may involve randomness. Reproducible workloads make comparisons meaningful.

Control:

- random seed
- prompt selection
- arrival times
- max new tokens
- request priorities
- sampling temperature / top-k / top-p
- speculative `K`
- batch size
- token budget
- block size

Create named workload presets instead of manually changing constants.

Examples:

| Workload | Purpose |
|---|---|
| `single_short` | baseline sanity |
| `single_long` | KV cache and context stress |
| `batch_uniform` | easy continuous batching case |
| `batch_mixed_lengths` | realistic scheduling pressure |
| `prefix_shared` | prefix cache stress test |
| `decode_heavy` | many short prompts, long outputs |
| `prefill_heavy` | long prompts, short outputs |
| `priority_mix` | scheduler policy behavior |
| `speculation_friendly` | repeated text where draft model should do well |
| `speculation_hostile` | varied text where draft model should struggle |

## 7. Design A Request Simulator

Since you implemented continuous batching and scheduling, your benchmark should simulate request arrivals rather than just passing one static batch.

Each synthetic request should have at least:

- `id`
- `prompt_tokens`
- `max_new_tokens`
- `arrival_time` or `arrival_step`
- `priority`
- expected feature tags, such as shared prefix group

You can simulate time in two ways:

1. **Real-time arrival simulation**
   - Uses actual wall-clock time.
   - More realistic.
   - Harder to make deterministic.

2. **Step-based arrival simulation**
   - Request arrives at scheduler step `N`.
   - Easier to debug.
   - Better for an educational NanoGPT engine.

For your project, step-based arrival is probably enough.

## 8. Track Per-Request Latency

Each request should record these timestamps or step numbers:

- `created_at`
- `admitted_at`
- `first_prefill_at`
- `prefill_done_at`
- `first_token_at`
- `completed_at`

From those, compute:

- queue wait time
- prefill duration
- time to first token
- decode duration
- total latency

Hint: if you support chunked prefill, a request can have many prefill chunks. Store both:

- first prefill time
- total prefill compute time

## 9. Track Per-Step Engine Metrics

Every scheduler/model step should report something like:

- step index
- number of waiting requests
- number of prefilling requests
- number of active decode requests
- number of completed requests
- batch size
- prefill tokens this step
- decode tokens this step
- total tokens this step
- forward pass duration
- scheduler overhead duration
- sampled tokens emitted
- KV blocks allocated
- KV blocks free
- prefix cache hits/misses

This gives you a timeline. The timeline is often more useful than the final average.

## 10. Separate Model Time From Scheduler Time

Continuous batching adds Python scheduling overhead. Paged KV and prefix cache also add book-keeping overhead.

Measure at least:

- scheduler/select/admit time
- input assembly time
- model forward time
- cache gather/scatter time
- sampling time
- total step time

This will show you whether an "optimization" is actually dominated by Python overhead at NanoGPT scale.

Do not be surprised if some realistic optimizations are slower on a 210K parameter model. That is still a good result if your report explains why.

## 11. Count Tokens Carefully

Decide which tokens belong in which bucket.

Useful buckets:

- prompt tokens processed during prefill
- generated tokens emitted to users
- draft tokens proposed
- draft tokens accepted
- target verification tokens
- padding tokens processed

Padding tokens are especially important. If your fused batching uses padding, count how many padding tokens the model processed. This helps explain wasted compute.

Hint: `tokens/sec` should usually use emitted generated tokens as the numerator, but model efficiency analysis may need a second metric using total processed tokens.

## 12. Add Feature Flags

Make benchmark configurations explicit.

Possible flags:

```python
use_kv_cache=True
use_continuous_batching=True
use_chunked_prefill=True
use_prefix_cache=True
use_paged_kv=True
use_speculative_decoding=False
use_quantization=False
scheduler_policy="fcfs"
token_budget=...
max_batch_size=...
speculative_k=...
```

Avoid changing implementation code between benchmark runs. Change only config.

## 13. Build Reports That Compare Runs

Start with console output, but save structured results too.

Good output formats:

- JSON for raw metrics
- CSV for tables
- Markdown for human summaries

A useful Markdown report might include:

- config summary
- workload summary
- aggregate metrics table
- latency percentiles
- cache stats
- speculative decoding stats
- short interpretation

Suggested latency percentiles:

- p50
- p90
- p95
- p99
- max

Do not rely only on averages. Scheduling changes often improve average throughput while hurting tail latency.

## 14. Suggested First Benchmark Milestone

Do not start by benchmarking every optimization.

First milestone:

- one model
- one workload
- no prefix cache
- no paged KV
- no speculative decoding
- compare:
  - no KV cache generation
  - KV cache generation

Metrics:

- total time
- generated tokens/sec
- time to first token
- average inter-token latency

Once that works, add continuous batching metrics.

## 15. Suggested Build Order

Recommended order for the harness:

1. `BenchmarkConfig`
2. `RequestSpec`
3. `BenchmarkRecorder`
4. single-request benchmark
5. continuous-batching benchmark
6. workload generator
7. JSON/Markdown reporting
8. prefix cache metrics
9. paged KV memory metrics
10. speculative decoding metrics
11. quantization comparison
12. latency percentiles and plots

At each stage, run only one or two comparisons.

## 16. Sanity Checks To Include

Benchmarks are easy to fool. Add simple sanity checks.

Useful checks:

- all requests complete
- no request generates more than `max_new_tokens`
- total emitted tokens equals sum of per-request emitted tokens
- no negative latency
- no request has `first_token_at` after `completed_at`
- KV blocks allocated never exceeds pool size
- prefix cache hit count never exceeds lookup count
- speculative accepted tokens never exceeds proposed tokens plus bonus-token accounting
- total scheduler token budget is not exceeded per step

These checks will catch many bugs before you interpret bogus performance numbers.

## 17. Correctness Guardrails

A benchmark harness should not replace correctness tests, but it should refuse to report performance for obviously invalid runs.

Before timing an optimization, consider running quick equivalence checks:

- cached logits match full forward logits
- paged KV logits match contiguous KV logits
- prefix-cached prefill logits match normal prefill logits
- continuous batching matches independent request generation when sampling is controlled

Hint: for deterministic comparisons, use greedy decoding or pre-sampled random numbers rather than unconstrained multinomial sampling.

## 18. Handling Random Sampling

Sampling makes exact output comparisons tricky.

Options:

- benchmark with greedy decoding for deterministic performance comparisons
- use a fixed `torch.Generator`
- pre-generate uniform random numbers for multinomial decisions
- compare distributions statistically rather than token-for-token

For performance benchmarks, greedy decoding is often enough. For speculative decoding correctness, you need more care because the whole point is to preserve the target distribution.

## 19. Speculative Decoding Metrics

If speculative decoding is enabled, record:

- draft tokens proposed
- draft model time
- target verification time
- accepted draft tokens
- rejected draft tokens
- bonus tokens sampled
- average accepted run length
- acceptance rate by position in the speculative window
- effective target tokens per target forward

Useful derived metrics:

```text
acceptance_rate = accepted_draft_tokens / proposed_draft_tokens
effective_tokens_per_verify = emitted_tokens / target_verify_forwards
```

Be careful with the "bonus" token. Decide whether it counts as accepted, emitted, or separately sampled. Report it consistently.

## 20. Prefix Cache Metrics

For prefix caching, record:

- prefix cache lookups
- block hits
- block misses
- reused tokens
- blocks inserted
- blocks evicted
- cache hit rate
- tokens skipped due to prefix reuse

Possible derived metric:

```text
prefix_token_reuse_rate = reused_prompt_tokens / total_prompt_tokens
```

Design workloads where some prompts share exact prefixes. Random prompts will make prefix caching look useless.

## 21. Paged KV Metrics

For paged KV, record:

- physical blocks allocated
- physical blocks freed
- peak blocks in use
- average block occupancy
- internal fragmentation
- block allocation failures
- preemption events caused by KV pressure

Possible derived metric:

```text
occupancy = filled_slots / (allocated_blocks * block_size)
```

This is more educational than raw speed for NanoGPT. Without custom kernels, paged KV may be slower but still demonstrate memory management clearly.

## 22. Quantization Metrics

For quantization, record both performance and quality-ish checks.

Performance:

- model size
- load/init time
- tokens/sec
- forward pass time

Quality-ish:

- validation loss before/after quantization
- average KL divergence between fp32 logits and quantized logits on a small sample
- sample text side-by-side

Do not judge quantization only by speed on a 210K parameter model. It may be slower or neutral.

## 23. Suggested File Layout

One possible organization:

```text
bench/
  workloads.py
  recorder.py
  report.py
  run_bench.py
  configs.py
```

Or, if you want to keep the project tiny:

```text
benchmark.py
```

Start simple. Split files only when the harness becomes hard to navigate.

## 24. Command-Line Interface Hints

You may eventually want commands like:

```bash
python benchmark.py --workload batch_mixed_lengths --config kv_cache
python benchmark.py --workload prefix_shared --config prefix_cache
python benchmark.py --workload speculation_friendly --config speculative
```

Arguments worth supporting:

- workload name
- seed
- number of requests
- max batch size
- token budget
- scheduler policy
- speculative K
- output path
- device
- dtype / quantization mode

But do not start with a big CLI. Hard-code one config first, then promote stable knobs into arguments.

## 25. Common Benchmarking Mistakes

Watch out for:

- timing GPU code without synchronization
- including model training time in inference benchmark
- changing random prompts between runs
- comparing different generated token counts
- measuring only averages
- accidentally counting padding as generated tokens
- reporting prefix cache performance on workloads with no shared prefixes
- reporting speculative speedup without acceptance rate
- measuring first-run compilation/setup costs as steady-state performance
- forgetting that Python overhead dominates tiny NanoGPT models

## 26. What A Good First Report Looks Like

Your first useful report might answer:

> Does KV caching improve single-request decode speed for this NanoGPT implementation?

It could include:

```text
Model:
  params: ~210K
  device: cpu/cuda
  block_size: 32

Workload:
  requests: 32
  prompt length: 16
  max new tokens: 64
  decoding: greedy

Results:
  baseline tokens/sec: ...
  cached tokens/sec: ...
  speedup: ...
  p50 request latency: ...
  p95 request latency: ...

Notes:
  ...
```

Keep the interpretation short and honest. If something is slower, explain where the time went.

## 27. Final Hint

The benchmark harness itself should become your map of the engine. If you cannot explain a speedup or slowdown from the recorded metrics, add one more event or counter until you can.

For your current project, the highest-value first implementation is:

1. deterministic workload generator
2. per-request timing
3. per-forward timing
4. emitted token counting
5. JSON + Markdown report

Once that exists, every other optimization becomes much easier to reason about.
