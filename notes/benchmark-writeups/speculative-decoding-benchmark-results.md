# Speculative Decoding Benchmark Results Write-up

This note explains the results in [`spec_decode_results.txt`](../spec_decode_results.txt) for the KV-cache decoding vs speculative decoding benchmark implemented in [`benchmarks/speculative_decoding.py`](../benchmarks/speculative_decoding.py) and configured by [`benchmarks/speculative_decoding_benchmark_runs.py`](../benchmarks/speculative_decoding_benchmark_runs.py).

## Executive Summary

This benchmark compares two decoding paths:

- **`kv_decode`**: normal autoregressive decoding with a target-model KV cache. The target model produces one generated token per decode forward pass.
- **`spec_decode`**: a cheap bigram draft model proposes multiple candidate tokens, then the target model verifies those candidates in a larger forward pass. If the candidates are accepted, the request emits multiple tokens from one target-model verification step.

The headline result is strong: speculative decoding is faster in every workload in this file.

- Throughput improves by **1.54x to 1.78x**.
- Average latency drops by roughly **35% to 44%**.
- Target forward calls drop to **40% to 49%** of the KV baseline.
- Average TTFT is slightly better in every row, although TTFT is not the main benefit of speculative decoding.

The most important nuance is that speculative decoding does **not** reduce the number of target tokens evaluated. In fact, it evaluates **13% to 48% more target tokens** than the baseline. The speedup comes from reducing the number of separate target-model forward calls by verifying multiple candidate tokens at once.

The core takeaway:

> Speculative decoding wins here because it trades many tiny one-token decode forwards for fewer larger verification forwards. Even with a simple bigram draft model and imperfect acceptance, fewer target forward calls are enough to improve end-to-end throughput.

## Benchmark Setup

The run uses:

- **Model size:** `0.056769M` parameters
- **Device:** CPU
- **Context length:** `block_size=64`
- **Prompt length:** `24` tokens in every benchmark row
- **Draft model:** cheap bigram model built from training-token transition counts
- **Benchmark target:** speculative decoding mechanics, not output quality

Before the benchmark, the tiny model trains briefly:

| Step | Train Loss | Validation Loss |
|---:|---:|---:|
| 0 | 4.1800 | 4.1791 |
| 20 | 3.6074 | 3.6479 |
| 40 | 3.3261 | 3.3321 |
| 60 | 3.1051 | 3.1305 |
| 80 | 2.9561 | 2.9651 |
| 100 | 2.8321 | 2.8682 |
| 119 | 2.7759 | 2.7995 |

The generated text sample is noisy, which is expected for a tiny character-level model trained briefly. This benchmark is about inference behavior: how many target forwards are needed, how often draft tokens are accepted, and how throughput changes.

## What Was Benchmarked

The benchmark compares `kv_decode` against `spec_decode` over the same request workload.

| Method | Behavior |
|---|---|
| `kv_decode` | Prefill the prompt once, sample the first token, then decode one token per target-model forward call using the KV cache. |
| `spec_decode` | Prefill the prompt once, sample the first token, then repeatedly ask a bigram draft model for `K` candidate tokens and verify those candidates with the target model in one forward call. |

Speculative decoding has three important phases:

1. **Draft:** a small draft model proposes several future tokens.
2. **Verify:** the target model scores the current token plus the proposed candidates in a single forward pass.
3. **Accept or reject:** accepted draft tokens are emitted immediately; if a candidate is rejected, the benchmark samples a corrected token from the target distribution and trims the KV cache to the accepted prefix.

The draft model is intentionally simple. It is not a second neural network. It is a bigram table:

```text
P(next_token | current_token)
```

That simplicity is useful because it makes the benchmark easy to run and reason about. It also means acceptance is limited: the draft model is much weaker than the target model.

## Metrics

| Metric | Meaning |
|---|---|
| `reqs` | Number of requests served. |
| `prompt_tok` | Total prompt tokens processed during prefill. |
| `gen_tok` | Total generated tokens emitted. |
| `wall_s` | Total wall-clock time. Lower is better. |
| `gen_tok/s` | Generated tokens per second. Higher is better. |
| `target_calls` | Number of target-model forward calls. Lower is usually better. |
| `target_tok` | Number of tokens evaluated by the target model, including prompt tokens and speculative verification tokens. |
| `tgt_tok/gen` | Target tokens evaluated per generated token. Lower means less target work per output token. |
| `avg_verify` | Average emitted tokens per speculative verification step. Higher is better. |
| `accept` | Fraction of proposed draft tokens accepted. Higher is better. |
| `draft_tok` | Number of draft tokens proposed. |
| `bonus` | Number of bonus tokens sampled after all candidates in a step were accepted. |
| `resample` | Number of corrected tokens sampled after a rejection. |
| `avg_ttft_ms` | Average time to first token. Lower is better. |
| `p95_ttft_ms` | 95th percentile time to first token. |
| `avg_lat_ms` | Average request latency. Lower is better. |
| `forward_s` | Time spent in measured target-model forward work. |

For this benchmark, the most important metrics are `gen_tok/s`, `target_calls`, `target_tok`, `accept`, `avg_verify`, and `avg_lat_ms`.

## Results Summary

| Case | Requests | Generated Tokens | K | Draft Noise | KV Tok/s | Spec Tok/s | Throughput Ratio | Target Call Ratio | Target Token Ratio | Acceptance |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `k2_bigram_draft` | 8 | 128 | 2 | 0.0 | 896.05 | 1379.86 | 1.54x | 0.49x | 1.13x | 63.6% |
| `k4_bigram_draft` | 8 | 128 | 4 | 0.0 | 895.85 | 1549.35 | 1.73x | 0.43x | 1.29x | 48.1% |
| `k6_bigram_draft` | 8 | 128 | 6 | 0.0 | 938.86 | 1616.56 | 1.72x | 0.43x | 1.48x | 34.7% |
| `k4_noisy_draft` | 8 | 128 | 4 | 0.5 | 957.25 | 1707.07 | 1.78x | 0.40x | 1.26x | 52.2% |
| `longer_outputs_k4` | 6 | 144 | 4 | 0.0 | 893.93 | 1589.45 | 1.78x | 0.41x | 1.40x | 46.2% |

Speculative decoding improves throughput in every case. The average throughput ratio across the five runs is about **1.71x**.

## Main Trend: Fewer Target Calls Beats More Target Tokens

The central pattern is:

- `spec_decode` makes far fewer target-model forward calls.
- `spec_decode` often evaluates more total target tokens.
- Despite evaluating more target tokens, it still runs faster.

That sounds contradictory at first, but it is exactly what speculative decoding is designed to do.

Normal KV decoding is sequential:

```text
target forward -> 1 token
target forward -> 1 token
target forward -> 1 token
...
```

Speculative decoding groups work:

```text
draft proposes K tokens
target verifies K+1 positions in one forward
emit several tokens if accepted
```

Each speculative verification forward is larger, but there are many fewer of them. In this benchmark, target forward calls drop from **128** to **51-63** for the 8-request, 128-token workloads. That is the main source of the speedup.

The target-token count tells the other side of the tradeoff:

| Case | KV Target Tokens | Spec Target Tokens | Increase |
|---|---:|---:|---:|
| `k2_bigram_draft` | 312 | 354 | +13% |
| `k4_bigram_draft` | 312 | 401 | +29% |
| `k6_bigram_draft` | 312 | 461 | +48% |
| `k4_noisy_draft` | 312 | 392 | +26% |
| `longer_outputs_k4` | 282 | 394 | +40% |

Speculative decoding is doing extra target-token work, but it packages that work into fewer calls. On this CPU microbenchmark, reducing Python/PyTorch forward-call overhead and improving per-call work size outweighs the extra token evaluations.

## Speculation Depth: K=2 vs K=4 vs K=6

The cleanest comparison is the three bigram-draft rows with the same workload:

| Case | K | Spec Tok/s | Target Calls | Target Tokens | Avg Verify | Acceptance |
|---|---:|---:|---:|---:|---:|---:|
| `k2_bigram_draft` | 2 | 1379.86 | 63 | 354 | 2.18 | 63.6% |
| `k4_bigram_draft` | 4 | 1549.35 | 55 | 401 | 2.55 | 48.1% |
| `k6_bigram_draft` | 6 | 1616.56 | 55 | 461 | 2.55 | 34.7% |

Increasing `K` from 2 to 4 helps:

- Throughput rises from **1379.86** to **1549.35 tok/s**.
- Target calls fall from **63** to **55**.
- Average emitted tokens per verification rises from **2.18** to **2.55**.

This is the useful part of speculation: propose more than one token, verify them together, and sometimes emit several tokens from one target call.

Increasing `K` from 4 to 6 is more mixed:

- Throughput rises only slightly: **1549.35** to **1616.56 tok/s**.
- Target calls do not improve: both use **55** target calls.
- Target tokens evaluated jump from **401** to **461**.
- Acceptance falls sharply from **48.1%** to **34.7%**.

This shows the classic speculative-decoding tuning curve. Larger `K` creates a chance to accept more tokens per verification, but it also asks the draft model to predict farther into the future. A weak draft model becomes less reliable as the speculative chain gets longer.

The result:

> K=4 is the better balanced setting in this benchmark. K=6 does not meaningfully reduce target calls, but it does increase verification-token work.

## Acceptance Rate Matters

Acceptance rate measures how often draft tokens survive target verification.

| Case | Draft Tokens Proposed | Accepted Draft Tokens | Acceptance |
|---|---:|---:|---:|
| `k2_bigram_draft` | 107 | 68 | 63.6% |
| `k4_bigram_draft` | 162 | 78 | 48.1% |
| `k6_bigram_draft` | 222 | 77 | 34.7% |
| `k4_noisy_draft` | 157 | 82 | 52.2% |
| `longer_outputs_k4` | 197 | 91 | 46.2% |

The acceptance rate falls as speculation depth increases:

- At `K=2`, the draft only needs to be right for short spans.
- At `K=4`, it proposes farther ahead, so more chains break.
- At `K=6`, many later candidates become wasted verification work.

Acceptance does not need to be perfect for speculative decoding to help. The `k4_bigram_draft` row accepts less than half of draft tokens, but still reaches **1.73x** KV throughput. That is because every fully or partially accepted verification step can emit more than one token.

However, acceptance still sets the ceiling. A stronger draft model would likely improve throughput by:

- raising `avg_verify`,
- reducing resampling,
- reducing wasted target-token evaluations,
- allowing larger `K` values to remain useful.

## Bonus And Resampled Tokens

Speculative decoding has two non-obvious token counters:

| Case | Bonus Tokens | Resampled Tokens |
|---|---:|---:|
| `k2_bigram_draft` | 28 | 24 |
| `k4_bigram_draft` | 13 | 29 |
| `k6_bigram_draft` | 6 | 37 |
| `k4_noisy_draft` | 13 | 25 |
| `longer_outputs_k4` | 13 | 34 |

**Bonus tokens** happen when all draft candidates in a verification step are accepted. The target has already produced the next distribution after those candidates, so the benchmark can sample one extra token.

**Resampled tokens** happen when a draft candidate is rejected. The benchmark samples a replacement token from an adjusted target distribution, emits it, and stops accepting further candidates from that speculative batch.

The trend is useful:

- `K=2` has the most bonus tokens because shorter speculative chains are easier to accept completely.
- `K=6` has fewer bonus tokens and more resampled tokens because long chains are more likely to reject somewhere.

This again shows why larger speculation depth is not automatically better.

## Latency And TTFT

Speculative decoding improves average latency in every row:

| Case | KV Avg Latency | Spec Avg Latency | Reduction |
|---|---:|---:|---:|
| `k2_bigram_draft` | 17.86 ms | 11.59 ms | 35.1% |
| `k4_bigram_draft` | 17.86 ms | 10.33 ms | 42.2% |
| `k6_bigram_draft` | 17.04 ms | 9.90 ms | 41.9% |
| `k4_noisy_draft` | 16.71 ms | 9.37 ms | 43.9% |
| `longer_outputs_k4` | 26.85 ms | 15.10 ms | 43.8% |

The latency improvement tracks the reduction in target calls. Each request needs fewer decode iterations before it reaches its output length.

TTFT improves only slightly:

| Case | KV Avg TTFT | Spec Avg TTFT |
|---|---:|---:|
| `k2_bigram_draft` | 1.55 ms | 1.36 ms |
| `k4_bigram_draft` | 1.60 ms | 1.35 ms |
| `k6_bigram_draft` | 1.53 ms | 1.27 ms |
| `k4_noisy_draft` | 1.48 ms | 1.42 ms |
| `longer_outputs_k4` | 1.69 ms | 1.45 ms |

This is expected. Both methods do the same prompt prefill before producing the first sampled token. Speculative decoding mostly accelerates the continuation after the first token, so its biggest benefit appears in request latency and generated-token throughput rather than TTFT.

## The Noisy Draft Result

`k4_noisy_draft` uses the same `K=4` depth as `k4_bigram_draft`, but blends the draft distribution with uniform noise:

```text
draft_noise=0.5
```

Surprisingly, it is the fastest 128-token row:

- Throughput: **1707.07 tok/s**
- Throughput ratio: **1.78x**
- Target calls: **51**, lower than **55** for the clean K=4 run
- Acceptance: **52.2%**, slightly higher than **48.1%**

This should not be over-interpreted as "noise improves drafts." The benchmark is stochastic: token sampling, acceptance draws, and the tiny model's uncertain distributions can make one run favorable. With only 8 requests and 128 generated tokens, random variation can move the result.

The more conservative interpretation is:

> The K=4 speculative path is robust in this small benchmark. Even with a noisier draft distribution, it still reduces target calls and improves throughput.

For a stronger conclusion about draft quality, this benchmark should be repeated over multiple seeds and larger request counts.

## Longer Output Run

The `longer_outputs_k4` row changes the workload to:

```text
num_requests=6
prompt_len=24
max_new_tokens=24
speculation_len=4
```

Compared with the 16-token output rows, this gives each request more decode work after the initial prompt prefill.

Results:

- KV throughput: **893.93 tok/s**
- Speculative throughput: **1589.45 tok/s**
- Throughput ratio: **1.78x**
- Target call ratio: **0.41x**
- Average latency: **26.85 ms** down to **15.10 ms**

This is an important row because speculative decoding is mainly a decode-phase optimization. When outputs get longer, there are more one-token decode forwards for speculation to collapse into fewer verification forwards. The speedup remains strong.

## Forward Time

The `forward_s` column shows time spent in measured target-model forward work:

| Case | KV Forward Time | Spec Forward Time | Reduction |
|---|---:|---:|---:|
| `k2_bigram_draft` | 0.1364 s | 0.0807 s | 40.8% |
| `k4_bigram_draft` | 0.1364 s | 0.0703 s | 48.5% |
| `k6_bigram_draft` | 0.1302 s | 0.0663 s | 49.1% |
| `k4_noisy_draft` | 0.1274 s | 0.0635 s | 50.2% |
| `longer_outputs_k4` | 0.1535 s | 0.0764 s | 50.2% |

This column reinforces the main story. The speculative path is not merely moving time outside the measured forward loop. It substantially reduces measured target-forward time.

One caveat: this is a tiny CPU model. On a larger GPU model, the exact balance can change. Speculative decoding usually matters most when the target model is expensive and the draft model is much cheaper.

## Row-by-Row Interpretation

### `k2_bigram_draft`

This is the conservative speculation setting:

```text
num_requests=8
prompt_len=24
max_new_tokens=16
speculation_len=2
draft_noise=0.0
```

The draft model proposes at most 2 tokens per verification step.

Results:

- Throughput improves from **896.05** to **1379.86 tok/s**.
- Target calls drop from **128** to **63**.
- Acceptance is **63.6%**, the highest in the suite.
- Average latency drops from **17.86 ms** to **11.59 ms**.

This row demonstrates the safest speculative-decoding tradeoff. K=2 has high acceptance and relatively low wasted verification work, but it leaves some potential batching benefit on the table.

### `k4_bigram_draft`

This is the most balanced clean-draft setting:

```text
speculation_len=4
draft_noise=0.0
```

Results:

- Throughput improves to **1549.35 tok/s**, a **1.73x** speedup.
- Target calls drop to **55**, or **43%** of baseline.
- Average emitted tokens per verification rises to **2.55**.
- Acceptance falls to **48.1%**, but the run is still much faster.

This is the strongest clean demonstration that speculative decoding can tolerate imperfect drafts. Less than half of candidate tokens are accepted, but enough verification steps emit multiple tokens to beat one-token-at-a-time decoding.

### `k6_bigram_draft`

This row pushes speculation deeper:

```text
speculation_len=6
draft_noise=0.0
```

Results:

- Throughput is **1616.56 tok/s**, slightly above K=4.
- Target calls stay at **55**, the same as K=4.
- Target tokens rise to **461**, much higher than K=4's **401**.
- Acceptance falls to **34.7%**.

K=6 is not a clear improvement. It asks the target model to verify more speculative positions, but it does not reduce the number of target calls beyond K=4. The draft model is too weak to reliably support the longer chain.

### `k4_noisy_draft`

This row tests whether the speculative path still works with a lower-quality draft distribution:

```text
speculation_len=4
draft_noise=0.5
```

Results:

- Throughput is **1707.07 tok/s**, the fastest 128-token row.
- Target calls drop to **51**.
- Acceptance is **52.2%**.
- Average latency drops to **9.37 ms**.

The result is positive, but likely includes sampling noise. The safest conclusion is not that noisy drafts are better, but that this speculative decoding implementation remains effective even when the draft distribution is imperfect.

### `longer_outputs_k4`

This row keeps `K=4` but increases each request from 16 to 24 generated tokens:

```text
num_requests=6
prompt_len=24
max_new_tokens=24
```

Results:

- Throughput improves from **893.93** to **1589.45 tok/s**.
- Target calls drop from **144** to **59**.
- Average latency drops from **26.85 ms** to **15.10 ms**.
- Acceptance is **46.2%**.

This row confirms the benefit scales into longer decode workloads. More output tokens means more chances to replace one-token target forwards with multi-token verification forwards.

## Why Speculative Decoding Matters

Autoregressive decoding is hard to parallelize because each next token depends on the previous token. KV caching removes repeated attention work over the prefix, but it does not remove the sequential loop:

```text
sample token 1
sample token 2
sample token 3
...
```

Speculative decoding attacks that loop by using a cheap model to guess several future tokens. The target model still controls correctness: it verifies the draft tokens and rejects or corrects them when needed.

That makes speculative decoding attractive because it can preserve the target model's output distribution while reducing the number of expensive target-model calls.

In this benchmark, the effect is visible even with a very small model:

- The target model is called much less often.
- Multiple tokens are often emitted per verification.
- Average latency falls substantially.
- Throughput rises across every workload.

## Caveats

These results are useful, but they are not a production benchmark.

1. **The model is tiny.** With only `0.056769M` parameters, overheads and sampling noise are large relative to model compute.
2. **The draft model is a bigram table.** Real speculative decoding usually uses a smaller neural draft model or specialized draft heads.
3. **The run is CPU-only.** GPU behavior can differ because larger verification forwards may utilize hardware differently.
4. **The sample size is small.** Most rows generate only 128 tokens, so random sampling can affect acceptance and throughput.
5. **The benchmark is single-run.** Multiple seeds would make the `k4_noisy_draft` result easier to interpret.
6. **Quality is not evaluated.** The benchmark measures mechanics and speed, not whether generated text is good.

## Practical Takeaways

1. **Speculative decoding is working in this implementation.** Every run completes the same generated-token count and improves throughput over KV decoding.
2. **The main win is fewer target forwards.** Target calls fall to about **40% to 49%** of baseline.
3. **More speculation is not always better.** K=6 evaluates many more target tokens and has much lower acceptance than K=4.
4. **Acceptance rate is a key health metric.** Higher acceptance means more emitted tokens per verification and less wasted target work.
5. **Longer decode workloads are a better fit.** The `longer_outputs_k4` row shows speculation remains strong when more time is spent in decode.
6. **Draft quality should be tested over multiple seeds.** The noisy-draft row is encouraging, but too small to prove that noise helps.

## Suggested Follow-up Benchmarks

Useful next experiments would be:

- Run each configuration across multiple seeds and report mean/stddev.
- Increase `num_requests` and `max_new_tokens` to reduce sampling noise.
- Compare the bigram draft against a small neural draft model.
- Add a `K=8` row to show where deeper speculation starts losing clearly.
- Run the same suite on CUDA and compare CPU vs GPU behavior.
- Track wall time spent in draft sampling separately from target verification.
- Add output distribution checks to confirm speculative decoding preserves target-model sampling behavior.

## Bottom Line

This benchmark is a clear success for speculative decoding. The draft model is simple and imperfect, but it still lets the target model verify multiple candidate tokens per call. That cuts target forward calls roughly in half or better, reduces request latency, and improves generated-token throughput by about **1.7x** on average.

The best interpretation is not that speculative decoding reduces total target-token work. It usually increases it. The win comes from changing the shape of the work: fewer target-model calls, larger verification batches, and more generated tokens emitted per decode iteration.
